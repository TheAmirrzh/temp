#!/bin/bash

# ==============================================================================
# SOTA Training Pipeline with All Fixes
#
# This script uses the improved loss functions and optimizer configurations
# to fix gradient explosion and improve convergence.
#
# Key changes:
# 1. Uses AdaptiveFocalRankingLoss (most stable)
# 2. Uses AdamW with warmup (prevents early explosion)
# 3. Lower base LR: 5e-5 instead of 1e-4
# 4. Gradient centralization enabled
# 5. Layer-wise LR decay
# ==============================================================================

source "/Users/amirmac/WorkSpace/Codes/LogNet/Phase-2.5.2/.venv2/bin/activate"

# --- Configuration ---
DATA_DIR="./generated_data"
SPECTRAL_DIR="./spectral_cache"
EXP_DIR="./experiments/sota_fixed"
VALIDATION_DIR="./validation_reports"

VENV_PYTHON="/Users/amirmac/WorkSpace/Codes/LogNet/Phase-2.5.2/.venv2/bin/python"

# Data Generation Config
N_EASY=200
N_MEDIUM=200
N_HARD=400
N_VERY_HARD=400

# Spectral Config
SPECTRAL_K=16
NUM_WORKERS_SPECTRAL=$(($(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) - 1))

# Training Config (IMPROVED)
EPOCHS=50
BATCH_SIZE=32  # Reduced from 64 for stability
HIDDEN_DIM=256
NUM_LAYERS=3
GRAD_ACCUM_STEPS=4

# CRITICAL: New optimizer settings to fix gradient explosion
BASE_LR=5e-5          # LOWERED from 1e-4 (critical fix)
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=5       # NEW: prevents early explosion
LOSS_TYPE="focal"     # NEW: most stable loss
OPTIMIZER_TYPE="adamw_warmup"  # NEW: with warmup

# --- Script Logic ---

set -e
echo "🔥 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} +
echo "✅ Cache cleared."

echo "🚀 Starting SOTA Training Pipeline..."
echo "================================================="
date

# 0. Create Directories
echo "📂 [Step 0/6] Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$SPECTRAL_DIR"
mkdir -p "$EXP_DIR"
mkdir -p "$VALIDATION_DIR"
echo "✅ Directories created."
echo "-------------------------------------------------"

# 1. Data Generation
echo "🧬 [Step 1/6] Generating dataset..."
"$VENV_PYTHON" data_generator.py \
    --output-dir "$DATA_DIR" \
    --easy $N_EASY \
    --medium $N_MEDIUM \
    --hard $N_HARD \
    --very-hard $N_VERY_HARD
echo "✅ Dataset generated."
echo "-------------------------------------------------"

# 2. Clear Cache
echo "🔥 [Step 2/6] Clearing stale spectral cache..."
rm -rf "$SPECTRAL_DIR"
mkdir -p "$SPECTRAL_DIR"
echo "✅ Cache cleared."
echo "-------------------------------------------------"

# 3. Spectral Preprocessing
echo "📊 [Step 3/6] Preprocessing spectral features..."
"$VENV_PYTHON" batch_process_spectral.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$SPECTRAL_DIR" \
    --k $SPECTRAL_K \
    --num_workers $NUM_WORKERS_SPECTRAL \
    --no-adaptive-k
echo "✅ Spectral features computed."
echo "-------------------------------------------------"

# 4. SOTA Training (IMPROVED)
echo "🧠 [Step 4/6] Starting SOTA training with all fixes..."
echo ""
echo "CRITICAL IMPROVEMENTS:"
echo "  ✅ Loss: $LOSS_TYPE (numerically stable)"
echo "  ✅ Optimizer: $OPTIMIZER_TYPE (with warmup)"
echo "  ✅ Base LR: $BASE_LR (LOWERED from 1e-4)"
echo "  ✅ Warmup: $WARMUP_EPOCHS epochs (prevents explosion)"
echo "  ✅ Gradient centralization: Enabled"
echo "  ✅ Layer-wise LR decay: Enabled"
echo ""

"$VENV_PYTHON" fixed_train.py \
    --data-dir "$DATA_DIR" \
    --spectral-dir "$SPECTRAL_DIR" \
    --exp-dir "$EXP_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --num-layers $NUM_LAYERS \
    --k-dim $SPECTRAL_K \
    --loss-type $LOSS_TYPE \
    --optimizer-type $OPTIMIZER_TYPE \
    --base-lr $BASE_LR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-epochs $WARMUP_EPOCHS \
    --grad-accum-steps $GRAD_ACCUM_STEPS \
    --use-llrd \
    --llrd-decay 0.9 \
    --device cpu

echo "✅ Training complete."
echo "================================================="

# 5. Validation
echo "🔬 [Step 5/6] Generating validation reports..."
"$VENV_PYTHON" validate_spectral.py \
    --cache-dir "$SPECTRAL_DIR" \
    --output-dir "$VALIDATION_DIR/spectral_report"

"$VENV_PYTHON" validate_temporal.py \
    --output-dir "$VALIDATION_DIR/temporal_report"

echo "✅ Validation complete."
echo "-------------------------------------------------"

# 6. Done
echo "🎉 Full Pipeline Finished Successfully!"
echo ""
echo "Results saved to: $EXP_DIR"
echo "  - Model: $EXP_DIR/best_model.pt"
echo "  - Config: $EXP_DIR/config.json"
echo "  - Results: $EXP_DIR/results.json"
echo ""
date
echo "================================================="

exit 0