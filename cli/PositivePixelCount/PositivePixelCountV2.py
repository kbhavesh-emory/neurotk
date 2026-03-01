from joblib import Parallel, delayed
import psutil
import time
from datetime import datetime
import os

# Set temp directory to location with more space BEFORE importing pyvips/large_image
os.environ['TMPDIR'] = '/nashome/bhavesh/neurotk/cli/output/tmp'
os.environ['TEMP'] = '/nashome/bhavesh/neurotk/cli/output/tmp'
os.environ['TMP'] = '/nashome/bhavesh/neurotk/cli/output/tmp'
os.makedirs('/nashome/bhavesh/neurotk/cli/output/tmp', exist_ok=True)

import numpy as np
from collections import deque
import threading
import cv2
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import itertools
import json
import random
import asyncio
from pathlib import Path
from pprint import pprint
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser
import histomicstk.segmentation.positive_pixel_count as ppc
from histomicstk.segmentation.positive_pixel_count import Parameters, Labels, OutputTotals
import large_image
from large_image.tilesource import loadTileSources
from performance_monitor import (
    SystemMonitor, save_performance_stats, profile_function,
    prefetch_tiles, fetch_tiles_async, process_batch_async
)

def profile_count_image(img, ppc_params, mask):
    """Profile the count_image function specifically"""
    def optimized_count_image(image, params, mask=None):
        # Convert to HSV and split channels in one step
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        del hsv  # Free memory early
        
        # Pre-compute thresholds as uint8 scalars
        hue_value_hsv = np.uint8(params.hue_value * 180)
        hue_width_hsv = np.uint8(params.hue_width * 180 / 2)
        sat_min = np.uint8(params.saturation_minimum * 255)
        int_lower = np.uint8(params.intensity_lower_limit * 255)
        int_upper = np.uint8(params.intensity_upper_limit * 255)
        
        # Fast hue difference using numpy operations
        hdiff = np.abs(h - hue_value_hsv)
        hdiff = np.minimum(hdiff, 180 - hdiff)
        
        # Create masks using numpy operations
        mask_all_positive = (
            (hdiff <= hue_width_hsv) &
            (s >= sat_min) &
            (v >= int_lower) &
            (v < int_upper)
        )
        
        if mask is not None:
            mask_all_positive &= mask
            
        # Get positive pixels and normalize
        v_positive = v[mask_all_positive]
        if v_positive.size == 0:  # No positive pixels
            return OutputTotals(0, 0, 0, h.size, 0.0, 0.0, 0.0), np.zeros_like(h, dtype=np.uint8)
        
        v_normalized = v_positive.astype(np.float32) * (1.0/255.0)
        
        # Create classification masks
        weak_mask = v_normalized >= params.intensity_weak_threshold
        strong_mask = v_normalized < params.intensity_strong_threshold
        plain_mask = ~(weak_mask | strong_mask)
        
        # Calculate statistics
        nw = np.count_nonzero(weak_mask)
        ns = np.count_nonzero(strong_mask)
        np_ = np.count_nonzero(plain_mask)
        
        # Sum intensities
        iw = np.sum(v_normalized[weak_mask])
        is_ = np.sum(v_normalized[strong_mask])
        ip = np.sum(v_normalized[plain_mask])
        
        # Create label mask efficiently
        label_mask = np.zeros_like(h, dtype=np.uint8)
        pos_pixels = np.where(mask_all_positive)
        
        # Set labels using direct indexing
        label_mask[pos_pixels] = Labels.PLAIN
        label_mask[mask_all_positive][weak_mask] = Labels.WEAK
        label_mask[mask_all_positive][strong_mask] = Labels.STRONG
        
        total = OutputTotals(
            NumberWeakPositive=nw,
            NumberPositive=np_,
            NumberStrongPositive=ns,
            NumberTotalPixels=(np.count_nonzero(mask) if mask is not None else h.size),
            IntensitySumWeakPositive=iw,
            IntensitySumPositive=ip,
            IntensitySumStrongPositive=is_,
        )
        
        return total, label_mask
    
    return optimized_count_image(img, ppc_params, mask)

@profile_function
def process_tile(tile, ppc_params, color_map, useAlpha, region_polygons):
    """Process a single tile with timing information"""
    try:
        img = tile["tile"]
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1):
            img = np.broadcast_to(img[..., None], img.shape + (3,))

        mask = None
        if region_polygons:
            mask = utils.polygons_to_binary_mask(
                region_polygons, tile["x"], tile["y"], tile["width"], tile["height"]
            )

        result, ppcmask = profile_count_image(img, ppc_params, mask)
        
        ppcimg = color_map[ppcmask]
        if not useAlpha:
            ppcimg = ppcimg[:, :, :3]

        return result, ppcimg, tile["x"], tile["y"], mask
    except Exception as e:
        print(f"Error processing tile: {e}")
        raise

def main(args):
    """Main function with detailed resource monitoring."""
    # Debug print arguments
    print("\nCommand line arguments:")
    print("-" * 80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 80)

    # Ensure output directory exists
    if hasattr(args, 'outputAnnotationFile'):
        output_dir = os.path.dirname(args.outputAnnotationFile)
        if output_dir:
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory exists: {os.path.exists(output_dir)}")

    # Initial setup remains the same...
    ts = large_image.getTileSource(args.inputImageFile)
    tiparams = utils.get_region_dict(args.region, None, ts) if args.region else {}
    region_polygons = utils.get_region_polygons(args.region) if args.region else None
    
    tileSize = 4096
    useAlpha = len(args.region) > 6 if args.region else False
    
    color_map = np.array([
        [255, 255, 255, 255],  # NEGATIVE - white background
        [60, 78, 194, 255],    # WEAK - blue
        [60, 78, 194, 255],    # PLAIN - blue
        [60, 78, 194, 255],    # STRONG - blue
    ], dtype=np.uint8)
    
    ppc_params = ppc.Parameters(
        **{k: getattr(args, k) for k in ppc.Parameters._fields},
    )
    
    tiparams["format"] = large_image.constants.TILE_FORMAT_NUMPY
    tiparams["tile_size"] = dict(width=tileSize, height=tileSize)
    
    # Determine optimal number of jobs
    total_cores = args.num_workers

    # if total_cores is negative, use all cores, default is -1 in the xml
    if total_cores < 0:
        total_cores = psutil.cpu_count()

    # Calculate memory per tile - reduce workers to avoid OOM
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
    memory_per_tile = 2.5  # Increased from 1 to account for actual memory usage
    max_jobs_memory = int(available_memory / memory_per_tile)
    # Cap at 16 workers to prevent memory issues with large tiles
    n_jobs = min(total_cores, max_jobs_memory, 16)
    
    print(f"\nSystem Configuration:")
    print(f"Total cores: {total_cores}")
    print(f"Available memory: {available_memory:.2f} GB")
    print(f"Memory-based max workers: {max_jobs_memory}")
    print(f"Using {n_jobs} workers (capped for stability)")
    
    try:
        # Get tiles and shuffle them
        tiles = list(ts.tileIterator(**tiparams)) ##[:500]
        random.shuffle(tiles)
        
        # Create batches
        batch_size = 2
        batches = [tiles[i:i + batch_size] 
                  for i in range(0, len(tiles), batch_size)]
        
        print(f"\nProcessing {len(tiles)} tiles in {len(batches)} batches with async fetching")
        
        monitor = SystemMonitor(interval=0.5)
        monitor.start()
        start_time = time.time()
        
        results = Parallel(
            n_jobs=n_jobs,
            verbose=10,
            timeout=600,
           # timeout=None,
            #max_nbytes='100M',
            prefer="processes",
            pre_dispatch='1.5*n_jobs',
            temp_folder='/nashome/bhavesh/tmp',
            backend='loky',
            mmap_mode='r'
        )(
            delayed(process_batch_async)(
                batch, process_tile, ppc_params, color_map, useAlpha, region_polygons
            ) for batch in batches
        )
        
        # Stop monitoring first
        system_stats = monitor.stop()
        
        # Then collect timing statistics
        timings = []
        for batch_results, batch_timing in results:
            timings.append(batch_timing)
            
        total_time = time.time() - start_time
        tiles_per_second = len(tiles) / total_time
        
        # Print all statistics
        print("\nPerformance Results:")
        print("-" * 80)
        print(f"Total time: {total_time:.2f}s")
        print(f"Tiles per second: {tiles_per_second:.2f}")
        
        print("\nTiming Analysis:")
        print(f"Average tile fetch time: {np.mean([t['tile_fetch'] for t in timings]):.3f}s")
        print(f"Average processing time: {np.mean([t['processing'] for t in timings]):.3f}s")
        print(f"Average total time: {np.mean([t['total'] for t in timings]):.3f}s")
        
        print("\nCPU Usage:")
        print(f"Overall mean: {system_stats['cpu']['mean']:.1f}%")
        print(f"Active cores: {system_stats['cpu']['active_cores_mean']:.1f}")
        print("Per-core utilization:")
        per_core_stats = system_stats['cpu']['per_core_mean']
        for i, usage in enumerate(per_core_stats):
            if usage > 5:  # Only show active cores
                print(f"  Core {i}: {usage:.1f}%")
        
        print("\nMemory Usage:")
        print(f"Mean: {system_stats['memory']['mean']/1024:.1f} GB")
        print(f"Max:  {system_stats['memory']['max']/1024:.1f} GB")
        
        print("\nIO Statistics:")
        print(f"Wait time: {system_stats['io']['wait_mean']:.1f}%")
        print(f"Disk read:  {system_stats['io']['throughput_read']:.1f} MB/s")
        print(f"Disk write: {system_stats['io']['throughput_write']:.1f} MB/s")
        print(f"Network read:  {system_stats['io']['network_read']:.1f} MB/s")
        print(f"Network write: {system_stats['io']['network_write']:.1f} MB/s")
        
        # After collecting all statistics, prepare the stats dictionary
        stats = {
            'total_time': total_time,
            'tiles_per_second': tiles_per_second,
            'avg_fetch': np.mean([t['tile_fetch'] for t in timings]),
            'avg_processing': np.mean([t['processing'] for t in timings]),
            'avg_total': np.mean([t['total'] for t in timings]),
            'system': system_stats
        }

        # Create and save the annotation file first
        print("Creating annotation file...")
        
        # Convert numpy arrays to regular Python types in stats
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        # Convert stats to JSON-serializable format
        stats_json = convert_numpy(stats)
        
        ann_doc = {
            "name": args.docname,
            "elements": [
                {
                    "type": "image",
                    "girderId": "outputLabelImage",
                    "hasAlpha": useAlpha,
                    "transform": {
                        "xoffset": tiparams.get("region", {}).get("left", 0),
                        "yoffset": tiparams.get("region", {}).get("top", 0),
                    },
                }
            ],
            "attributes": {
                "params": vars(args),
                "stats": stats_json,
                "cli": Path(__file__).stem,
            },
        }

        # Create directory for annotation file if it doesn't exist
        output_dir = os.path.dirname(args.outputAnnotationFile)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the annotation dictionary to the output annotation file
        with open(args.outputAnnotationFile, "w") as fh:
            json.dump(ann_doc, fh, separators=(",", ":"), sort_keys=False)
        print(f"Annotation file saved to {args.outputAnnotationFile}")

        # After parallel processing, combine all results
        print("Combining results, time from start %s" % (utils.disp_time_hms(time.time() - start_time)))
        
        # Reload tile sources after parallel processing (joblib can affect global state)
        # This is needed before creating the sink
        loadTileSources()
        
        # Create a sink for the output label image if specified
        sink = None
        if getattr(args, 'outputLabelImage', None):
            try:
                print(f"[DEBUG] Attempting to create sink for output: {args.outputLabelImage}")
                sink = large_image.new()
                print(f"[DEBUG] Sink created: {type(sink)}")
                print(f"Output label image enabled: {args.outputLabelImage}")
            except Exception as e:
                import traceback
                print(f"[ERROR] Output label image disabled due to exception:")
                print(f"[ERROR] Exception type: {type(e).__name__}")
                print(f"[ERROR] Exception message: {e}")
                print(f"[ERROR] Traceback:")
                traceback.print_exc()
                sink = None
        else:
            print("[DEBUG] No outputLabelImage requested")

        # If sink exists and region has 'left', set the crop
        if sink and "left" in tiparams.get("region", {}):
            sink.crop = (
                tiparams["region"]["left"],
                tiparams["region"]["top"],
                tiparams["region"]["width"],
                tiparams["region"]["height"],
            )

        # Extract results and write tiles
        tile_results = []
        for batch_result, timing in results:  # Unpack (batch_results, timing)
            for result_tuple in batch_result:
                # Unpack the nested tuple structure we see in the debug output
                (stats, ppcimg, x, y, mask), _ = result_tuple
                tile_results.append(stats)
                
                if sink:
                    sink.addTile(ppcimg, x, y, mask=mask)

        # After all tiles are processed, write the output file if sink exists
        if sink:
            print(
                "Outputting file, time from start %s"
                % (utils.disp_time_hms(time.time() - start_time))
            )
            sink_start_time = time.time()
            # Write with optimized settings: simple compression, fast
            print(f"[DEBUG] Writing output to: {args.outputLabelImage}")
            sink.write(args.outputLabelImage, lossy=False, vips_kwargs={'compression': 'deflate', 'level': 1})
            print(
                "Output file, time %s (%s from start)"
                % (
                    utils.disp_time_hms(time.time() - sink_start_time),
                    utils.disp_time_hms(time.time() - start_time),
                )
            )
            if not os.path.exists(args.outputLabelImage):
                raise RuntimeError(f"Output label image not created: {args.outputLabelImage}")
        elif getattr(args, 'outputLabelImage', None):
            print(f"[ERROR] sink is None, but outputLabelImage was requested: {args.outputLabelImage}")
            print(f"[ERROR] sink value: {sink}")
            print(f"[ERROR] This means sink creation failed. Check debug messages above.")
            raise RuntimeError("Output label image requested but sink could not be created.")

        return results
            
    except Exception as e:
        print(f"Error in processing: {e}")
        raise
    finally:
        # Make sure monitor is always stopped
        if 'monitor' in locals():
            monitor.stop()

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
