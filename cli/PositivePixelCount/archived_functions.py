"""
Archived functions from PositivePixelCountV2.py that are no longer in active use.
Kept for reference and potential future use.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from datetime import datetime

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

def log_performance_stats(stats, batch_size, filename=None):
    """Log performance statistics to a file"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'performance_stats_batch{batch_size}_{timestamp}.json'
    
    log_data = {
        'batch_size': batch_size,
        'timing': {
            'total_time': stats['total_time'],
            'tiles_per_second': stats['tiles_per_second'],
            'avg_tile_fetch': stats['avg_fetch'],
            'avg_processing': stats['avg_processing'],
            'avg_total': stats['avg_total']
        },
        'cpu': {
            'mean': stats['system']['cpu']['mean'],
            'active_cores': stats['system']['cpu']['active_cores_mean'],
            'per_core': stats['system']['cpu']['per_core_mean'].tolist()
        },
        'memory': {
            'mean_gb': stats['system']['memory']['mean']/1024,
            'max_gb': stats['system']['memory']['max']/1024
        },
        'io': {
            'wait_mean': stats['system']['io']['wait_mean'],
            'disk_read': stats['system']['io']['throughput_read'],
            'disk_write': stats['system']['io']['throughput_write'],
            'network_read': stats['system']['io']['network_read'],
            'network_write': stats['system']['io']['network_write']
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nPerformance stats logged to: {filename}")

def save_performance_stats(stats, batch_size, input_file, tile_size, total_cores, memory_per_tile, filename='performance_history.json'):
    """Save performance statistics to a JSON file, appending new results"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare new stats entry
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
    
    # Load existing data or create new list
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    
    # Append new entry
    data.append(new_entry)
    
    # Save updated data
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nPerformance stats appended to: {filename}") 