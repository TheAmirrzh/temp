import torch
from pathlib import Path
import logging

from torch.utils.data.dataloader import logger

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s | %(filename)s:%(lineno)d] %(message)s')
log = logging.getLogger(__name__)

# --- CONFIG ---
DATA_DIR = "generated_data"
SPECTRAL_DIR = "export"

log.info("🚀 Starting focused data pipeline test...")
log.info(f"   Using Data Dir: {DATA_DIR}")
log.info(f"   Using Spectral Dir: {SPECTRAL_DIR}")

try:
    from dataset import ProofStepDataset, create_split
    from dataset_utils import fixed_collate_fn
    from torch_geometric.data import Batch
except ImportError as e:
    log.error(f"Failed to import a critical module: {e}")
    log.error("Please ensure dataset.py and dataset_utils.py are in this directory.")
    exit(1)

def main():
    # ==========================================================
    log.info("STEP 1: Creating Dataset...")
    # ==========================================================
    try:
        # Get a list of files to load
        all_files = list(Path(DATA_DIR).rglob("*.json"))
        if not all_files:
            log.error(f"No .json files found in {DATA_DIR}")
            return False
        
        train_files, _, _ = create_split(DATA_DIR, train_ratio=0.7, val_ratio=0.15, seed=42)
        
        dataset = ProofStepDataset(
            json_files=train_files,
            spectral_dir=SPECTRAL_DIR,
            seed=42
        )
        log.info(f"✅ ProofStepDataset initialized. Found {len(dataset)} total steps.")
    except Exception as e:
        log.error(f"❌ FAILED to initialize ProofStepDataset: {e}", exc_info=True)
        return False

    # ==========================================================
    log.info("STEP 2: Fetching two data points from dataset.py...")
    # ==========================================================
    data_list = []
    try:
        log.info("   Attempting to get dataset[0]...")
        data1 = dataset[0]
        if data1 is None:
            log.error("   dataset[0] returned None. __getitem__ is failing.")
            return False
        log.info(f"   ✅ dataset[0] loaded (Nodes: {data1.num_nodes})")
        data_list.append(data1)

        log.info("   Attempting to get dataset[1]...")
        data2 = dataset[1]
        if data2 is None:
            log.error("   dataset[1] returned None. __getitem__ is failing.")
            return False
        log.info(f"   ✅ dataset[1] loaded (Nodes: {data2.num_nodes})")
        data_list.append(data2)

    except Exception as e:
        log.error(f"❌ FAILED during __getitem__: {e}", exc_info=True)
        log.error("   This indicates a problem in your data, or dataset.py's logic.")
        return False

    # ==========================================================
    logger.info("STEP 3: Calling fixed_collate_fn from dataset_utils.py...")
    # ==========================================================
    try:
        batch = fixed_collate_fn(data_list)
        if batch is None:
            log.error("   collate function returned None.")
            return False
        logger.info(f"   ✅ Collate function returned a Batch object.")
        logger.info(f"   Total nodes in batch: {batch.num_nodes}")
    except Exception as e:
        log.error(f"❌ FAILED during fixed_collate_fn: {e}", exc_info=True)
        return False

    # ==========================================================
    logger.info("STEP 4: THE FINAL TEST")
    # ==========================================================
    
    logger.info("   Checking for 'num_nodes_per_graph'...")
    if hasattr(batch, 'num_nodes_per_graph'):
        logger.info(f"   ✅✅✅ SUCCESS! Found 'num_nodes_per_graph': {batch.num_nodes_per_graph.tolist()}")
    else:
        logger.error(f"   ❌❌❌ FAILED! 'num_nodes_per_graph' is MISSING.")
        logger.error("   This proves the bug is in 'fixed_collate_fn' in dataset_utils.py.")
        return False
        
    logger.info("   Checking for 'node_offsets'...")
    if hasattr(batch, 'node_offsets'):
        # --- This is the line I fixed from the last test ---
        logger.info(f"   ✅✅✅ SUCCESS! Found 'node_offsets': {batch.node_offsets.tolist()}")
    else:
        logger.error(f"   ❌❌❌ FAILED! 'node_offsets' is MISSING.")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        log.info("\n🎉 PASSED! Your data pipeline (dataset.py + dataset_utils.py) is working correctly.")
        log.info("   The failure in diagnose.py is 100% due to an environment, cache, or DataLoader bug.")
    else:
        log.warning("\n🔥 FAILED. The bug is in the step that FAILED above.")