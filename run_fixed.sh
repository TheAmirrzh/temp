#!/bin/bash

echo "🚀 Running Fixed Training Pipeline"
echo ""

# Run diagnostics first
echo "Step 1: Running diagnostics..."
python diagnose.py --data-dir generated_data --spectral-dir spectral_cache

if [ $? -ne 0 ]; then
    echo "⚠️  Diagnostics found issues. Check output above."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run training
echo ""
echo "Step 2: Starting training..."
python train_minimal.py \
    --data-dir generated_data \
    --spectral-dir spectral_cache \
    --batch-size 16 \
    --epochs 20 \
    --lr 1e-4 \
    --device cpu

echo ""
echo "✅ Training complete!"
