"""
Batch Processing Pipeline for Spectral Features (FIXED)
=======================================================

Production-ready pipeline for processing entire dataset with:
- Parallel processing
- Error handling and recovery
- Progress tracking
- Resource monitoring
- Automatic checkpointing

Author: AI Research Team
Date: October 2025
"""

import numpy as np
import torch
from pathlib import Path
import json
import argparse
import multiprocessing as mp
from functools import partial
import time
import psutil
import logging
from tqdm import tqdm
from typing import List, Dict, Optional

from spectral_features import MagneticLaplacianExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# WORKER FUNCTION (Defined at module level for multiprocessing safety)
# ==============================================================================
def process_single_graph(
    json_file: Path,
    extractor: MagneticLaplacianExtractor, 
    output_dir: Path,
    normalization: str,
    k: int
) -> Dict:
    """
    Worker function to process a single graph file.
    Must be at module level to be picklable.
    """
    instance_id = json_file.stem
    output_file = output_dir / f"{instance_id}_magnetic.npz"
    result = {
        'instance_id': instance_id,
        'status': 'success',
        'error': None,
        'time': 0.0
    }
    
    try:
        start_time = time.time()
        
        with open(json_file, 'r') as f:
            graph_data = json.load(f)
        
        # --- ROBUST EDGE LOADING ---
        edges_data = graph_data.get('edges', [])
        num_nodes = len(graph_data.get('nodes', []))
        
        # Handle cases where nodes exist but edges might be None
        if num_nodes > 0 and not edges_data:
            # Single node graph or disconnected graph is valid!
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_types = None
        else:
            # Handle list-of-dicts format (common in JSON)
            if edges_data and isinstance(edges_data[0], dict):
                edge_list = [[e['src'], e['dst']] for e in edges_data]
                # Extract types
                type_map = {'unknown': 0, 'head': 1, 'body': 2}
                edge_types_list = [type_map.get(e.get('etype', 'unknown'), 0) for e in edges_data]
                edge_types = torch.tensor(edge_types_list, dtype=torch.long)
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                # Fallback
                if edges_data:
                    edge_index = torch.tensor(edges_data, dtype=torch.long).t()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_types = None

        # --- END ROBUST LOADING ---

        # Extract features
        # Note: extractor.extract_features handles magnetic laplacian logic
        features = extractor.extract_features(
            edge_index,
            num_nodes,
            edge_types=edge_types,
            validate=True
        )
        
        # Explicitly set status to success even if validation warns
        result['status'] = 'success'
        
        # Save features
        np.savez_compressed(
            output_file,
            eigenvalues=features['eigenvalues'],
            eigenvectors_real=features['eigenvectors_real'],
            eigenvectors_imag=features['eigenvectors_imag'],
            num_nodes=num_nodes,
            normalization=normalization,
            k=k,
            instance_id=instance_id
        )
        
        result['time'] = time.time() - start_time
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        # Only log actual crashes
        # logger.error(f"CRASH processing {instance_id}: {e}") # avoid spamming main process logs
    
    return result


class SpectralBatchProcessor:
    """
    Batch processor for computing spectral features with parallel processing.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        k: int = 16,
        normalization: str = 'random_walk',
        adaptive_k: bool = True,
        num_workers: int = None,
        checkpoint_interval: int = 100,
        q: float = 0.25, 
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.k = k
        self.normalization = normalization
        self.adaptive_k = adaptive_k
        self.num_workers = num_workers or mp.cpu_count()
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor (will be passed to worker processes)
        self.extractor = MagneticLaplacianExtractor(
            k=k, q=q, normalize=True, adaptive_k=adaptive_k
        )
        
        # Load checkpoint if exists
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.processed_files = self._load_checkpoint()
        
        logger.info(f"Initialized with {self.num_workers} workers")
        logger.info(f"Already processed: {len(self.processed_files)} files")
    
    def _load_checkpoint(self) -> set:
        """Load checkpoint of already processed files."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        return set()
    
    def _save_checkpoint(self, processed_files: set):
        """Save checkpoint of processed files."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed_files': list(processed_files),
                'timestamp': time.time(),
                'total': len(processed_files)
            }, f, indent=2)
    
    def process_batch(
        self,
        max_files: Optional[int] = None,
        force_recompute: bool = False
    ) -> Dict:
        """
        Process entire dataset in parallel.
        """
        # Find all JSON files
        json_files = list(self.data_dir.glob("**/*.json"))        
        if not force_recompute:
            # Filter out already processed files
            json_files = [
                f for f in json_files 
                if f.stem not in self.processed_files
            ]
        
        if max_files is not None:
            json_files = json_files[:max_files]
        
        logger.info(f"Processing {len(json_files)} graphs...")
        
        if len(json_files) == 0:
            logger.info("No files to process!")
            return {'total': 0}
        
        # Statistics
        stats = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'empty_graph': 0,
            'validation_failed': 0,
            'times': [],
            'errors': []
        }
        
        # Process with progress bar
        processed_count = 0
        
        # FIX: Use partial to bind the static arguments (extractor, output_dir, etc.)
        # json_file will be passed by the Pool map function as the first argument (implicitly)
        # or we need to arrange partial correctly.
        
        # partial(func, arg1, arg2) -> new_func(arg3) calls func(arg1, arg2, arg3)
        # Our worker is process_single_graph(json_file, extractor, output_dir, ...)
        # So we want partial to fill everything EXCEPT json_file.
        # But map passes the iterable item as the FIRST argument. 
        # So we refactor worker to take json_file first. (Done above).
        
        process_fn = partial(
            process_single_graph,
            extractor=self.extractor,
            output_dir=self.output_dir,
            normalization=self.normalization,
            k=self.k
        )
        
        # Use multiprocessing pool
        with mp.Pool(processes=self.num_workers) as pool:
            # Use imap for progress tracking
            # imap applies process_fn(json_file)
            results_iter = pool.imap(process_fn, json_files)
            
            for result in tqdm(results_iter, total=len(json_files), desc="Processing"):
                # Update statistics
                stats[result['status']] = stats.get(result['status'], 0) + 1
                
                if result['status'] == 'success':
                    stats['times'].append(result['time'])
                    self.processed_files.add(result['instance_id'])
                else:
                    stats['errors'].append({
                        'instance_id': result['instance_id'],
                        'error': result['error']
                    })
                
                processed_count += 1
                
                # Save checkpoint periodically
                if processed_count % self.checkpoint_interval == 0:
                    self._save_checkpoint(self.processed_files)
                    logger.info(f"Checkpoint saved: {processed_count}/{len(json_files)}")
        
        # Final checkpoint
        self._save_checkpoint(self.processed_files)
        
        # Compute timing statistics
        if stats['times']:
            stats['timing'] = {
                'mean': float(np.mean(stats['times'])),
                'std': float(np.std(stats['times'])),
                'min': float(np.min(stats['times'])),
                'max': float(np.max(stats['times'])),
                'total': float(np.sum(stats['times']))
            }
        
        # Save final statistics
        stats_file = self.output_dir / 'batch_processing_stats.json'
        with open(stats_file, 'w') as f:
            # Remove times array (too large)
            stats_save = {k: v for k, v in stats.items() if k != 'times'}
            json.dump(stats_save, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        
        if stats['times']:
            logger.info(f"  Average time: {stats['timing']['mean']:.3f}s per graph")
        
        return stats


def monitor_resources():
    """Monitor system resources during processing."""
    process = psutil.Process()
    return {
        'cpu_percent': process.cpu_percent(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'num_threads': process.num_threads()
    }


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(description="Batch process spectral features")
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--normalization', type=str, default='random_walk')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--adaptive-k', action='store_true', default=False)
    parser.add_argument('--q', type=float, default=0.25, help='Magnetic charge q')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SpectralBatchProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k=args.k,
        normalization=args.normalization,
        adaptive_k=args.adaptive_k,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        q=args.q
    )
    
    logger.info("System resources before processing:")
    logger.info(monitor_resources())
    
    start_time = time.time()
    stats = processor.process_batch(
        max_files=args.max_files,
        force_recompute=args.force
    )
    elapsed = time.time() - start_time
    
    logger.info(f"Total elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()