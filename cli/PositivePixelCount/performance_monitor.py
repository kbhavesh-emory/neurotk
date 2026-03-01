from joblib import Parallel, delayed
import psutil
import time
from datetime import datetime
import numpy as np
from collections import deque
import threading
import json
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import asyncio
import itertools

class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.cpu_percentages = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.io_wait = deque(maxlen=100)
        self.cpu_per_core = deque(maxlen=100)
        self.memory_bandwidth = deque(maxlen=100)
        self.io_stats = deque(maxlen=100)
        self.stop_flag = False
        self.process = psutil.Process()
        
    def start(self):
        self.stop_flag = False
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        
    def stop(self):
        self.stop_flag = True
        self.monitor_thread.join()
        
        cpu_stats = {
            'mean': np.mean(self.cpu_percentages) if len(self.cpu_percentages) > 0 else 0,
            'max': np.max(self.cpu_percentages) if len(self.cpu_percentages) > 0 else 0,
            'min': np.min(self.cpu_percentages) if len(self.cpu_percentages) > 0 else 0,
            'per_core_mean': np.mean(self.cpu_per_core, axis=0) if len(self.cpu_per_core) > 0 else np.zeros(psutil.cpu_count()),
            'per_core_max': np.max(self.cpu_per_core, axis=0) if len(self.cpu_per_core) > 0 else np.zeros(psutil.cpu_count()),
            'active_cores_mean': np.mean([sum(1 for x in cores if x > 10) 
                                        for cores in self.cpu_per_core]) if len(self.cpu_per_core) > 0 else 0
        }
        
        mem_stats = {
            'mean': np.mean(self.memory_usage) if len(self.memory_usage) > 0 else 0,
            'max': np.max(self.memory_usage) if len(self.memory_usage) > 0 else 0,
            'min': np.min(self.memory_usage) if len(self.memory_usage) > 0 else 0
        }
        
        io_stats = {
            'wait_mean': np.mean(self.io_wait) if len(self.io_wait) > 0 else 0,
            'wait_max': np.max(self.io_wait) if len(self.io_wait) > 0 else 0,
            'throughput_read': np.mean([sum(disk['read_bytes'] for disk in x['disk'].values())/x['time_delta'] 
                                      for x in self.io_stats])/1024/1024 if len(self.io_stats) > 0 else 0,
            'throughput_write': np.mean([sum(disk['write_bytes'] for disk in x['disk'].values())/x['time_delta'] 
                                       for x in self.io_stats])/1024/1024 if len(self.io_stats) > 0 else 0,
            'network_read': np.mean([x['network']['bytes_recv']/x['time_delta'] 
                                   for x in self.io_stats])/1024/1024 if len(self.io_stats) > 0 else 0,
            'network_write': np.mean([x['network']['bytes_sent']/x['time_delta'] 
                                    for x in self.io_stats])/1024/1024 if len(self.io_stats) > 0 else 0
        }
        
        return {
            'cpu': cpu_stats,
            'memory': mem_stats,
            'io': io_stats
        }
        
    def _monitor(self):
        last_io = psutil.disk_io_counters(perdisk=True)
        last_net = psutil.net_io_counters()
        last_time = time.time()
        
        while not self.stop_flag:
            per_core = psutil.cpu_percent(percpu=True)
            self.cpu_percentages.append(np.mean(per_core))
            self.cpu_per_core.append(per_core)
            
            mem = self.process.memory_info()
            self.memory_usage.append(mem.rss / 1024 / 1024)
            
            cpu_times = psutil.cpu_times_percent()
            self.io_wait.append(cpu_times.iowait)
            
            current_io = psutil.disk_io_counters(perdisk=True)
            current_net = psutil.net_io_counters()
            current_time = time.time()
            
            delta_time = current_time - last_time
            io_delta = {
                'disk': {
                    disk: {
                        'read_bytes': current_io[disk].read_bytes - last_io[disk].read_bytes,
                        'write_bytes': current_io[disk].write_bytes - last_io[disk].write_bytes
                    } for disk in current_io
                },
                'network': {
                    'bytes_recv': current_net.bytes_recv - last_net.bytes_recv,
                    'bytes_sent': current_net.bytes_sent - last_net.bytes_sent
                },
                'time_delta': delta_time
            }
            self.io_stats.append(io_delta)
            
            last_io = current_io
            last_net = current_net
            last_time = current_time
            
            time.sleep(self.interval)

def save_performance_stats(stats, batch_size, input_file, tile_size, total_cores, memory_per_tile, filename='performance_history.json'):
    """Save performance statistics to a JSON file, appending new results"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    new_entry = {
        'timestamp': timestamp,
        'input_file': input_file,
        'system_config': {
            'tile_size': tile_size,
            'total_cores': total_cores,
            'memory_per_tile': memory_per_tile,
            'batch_size': batch_size
        },
        'timing': {
            'total_time': stats['total_time'],
            'tiles_per_second': stats['tiles_per_second'],
            'avg_tile_fetch': stats['avg_fetch'],
            'avg_processing': stats['avg_processing'],
            'avg_total': stats['avg_total']
        },
        'cpu': {
            'mean': float(stats['system']['cpu']['mean']),
            'active_cores': float(stats['system']['cpu']['active_cores_mean']),
            'per_core': stats['system']['cpu']['per_core_mean'].tolist()
        },
        'memory': {
            'mean_gb': float(stats['system']['memory']['mean']/1024),
            'max_gb': float(stats['system']['memory']['max']/1024)
        },
        'io': {
            'wait_mean': float(stats['system']['io']['wait_mean']),
            'disk_read': float(stats['system']['io']['throughput_read']),
            'disk_write': float(stats['system']['io']['throughput_write']),
            'network_read': float(stats['system']['io']['network_read']),
            'network_write': float(stats['system']['io']['network_write'])
        }
    }
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    
    data.append(new_entry)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nPerformance stats appended to: {filename}")

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timings = {}
        start = time.time()
        
        # Get tile data
        tile_start = time.time()
        img = args[0]["tile"]
        timings['tile_fetch'] = time.time() - tile_start
        
        # Process image
        process_start = time.time()
        result = func(*args, **kwargs)
        timings['processing'] = time.time() - process_start
        
        timings['total'] = time.time() - start
        return result, timings
    return wrapper

def prefetch_tiles(tile_iterator, max_prefetch=20):
    """Prefetch tiles in a separate thread pool"""
    def fetch_tile(tile):
        try:
            return (tile, tile["tile"])
        except Exception as e:
            print(f"Error fetching tile: {e}")
            return None
    
    # Create a thread pool for prefetching
    with ThreadPoolExecutor(max_workers=max_prefetch) as executor:
        # Submit initial smaller batch to start quickly
        futures = []
        initial_batch = min(max_prefetch // 4, 20)  # Start with smaller batch
        
        # Convert to iterator if it's a list
        if isinstance(tile_iterator, list):
            tile_iterator = iter(tile_iterator)
        
        print(f"Prefetching initial {initial_batch} tiles...")
        for tile in itertools.islice(tile_iterator, initial_batch):
            futures.append(executor.submit(fetch_tile, tile))
        
        # Yield results while ramping up prefetching
        try:
            while True:
                # Get the next completed tile
                result = futures.pop(0).result()
                if result:
                    yield result
                
                # Submit next tile and gradually increase prefetch queue
                next_tile = next(tile_iterator)
                futures.append(executor.submit(fetch_tile, next_tile))
                
                # Add more to prefetch queue if we haven't reached max
                if len(futures) < max_prefetch:
                    try:
                        next_tile = next(tile_iterator)
                        futures.append(executor.submit(fetch_tile, next_tile))
                    except StopIteration:
                        break
        except StopIteration:
            pass
        
        # Yield remaining results
        for future in futures:
            result = future.result()
            if result:
                yield result

async def fetch_tiles_async(tiles):
    """Fetch tiles asynchronously"""
    async def fetch_one(tile):
        loop = asyncio.get_event_loop()
        # Run tile fetch in thread pool since it's blocking I/O
        return await loop.run_in_executor(None, lambda: tile["tile"])
    
    return await asyncio.gather(*[fetch_one(t) for t in tiles])

def process_batch_async(batch, process_func, *args):
    """Process a batch of tiles with async fetching
    
    Args:
        batch: List of tiles to process
        process_func: Function to process each tile
        *args: Additional arguments to pass to process_func
    """
    batch_start = time.time()
    
    # Fetch tiles asynchronously
    fetch_start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        tile_data = loop.run_until_complete(fetch_tiles_async(batch))
    finally:
        loop.close()
    fetch_time = time.time() - fetch_start
    
    # Process tiles
    process_start = time.time()
    results = []
    for tile, img in zip(batch, tile_data):
        result = process_func(tile, *args)
        results.append(result)
    process_time = time.time() - process_start
    
    return results, {
        'tile_fetch': fetch_time / len(batch),
        'processing': process_time / len(batch),
        'total': (time.time() - batch_start) / len(batch)
    } 