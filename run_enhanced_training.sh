#!/bin/bash

# Enhanced LogNet Training Runner Script
# This script provides easy access to all the enhanced training parameters

echo "🚀 Enhanced LogNet Training Runner"
echo "=================================="

# Set default values
EASY_SAMPLES=${EASY_SAMPLES:-50}
MEDIUM_SAMPLES=${MEDIUM_SAMPLES:-30}
HARD_SAMPLES=${HARD_SAMPLES:-20}
VERY_HARD_SAMPLES=${VERY_HARD_SAMPLES:-10}
EXTREME_HARD_SAMPLES=${EXTREME_HARD_SAMPLES:-5}

INPUT_DIM=${INPUT_DIM:-10}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_CLASSES=${NUM_CLASSES:-5}
NUM_TACTICS=${NUM_TACTICS:-3}
MAX_STEPS=${MAX_STEPS:-50}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-8}

EPOCHS=${EPOCHS:-20}
LEARNING_RATE=${LEARNING_RATE:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
DROPOUT_RATE=${DROPOUT_RATE:-0.1}
BATCH_SIZE=${BATCH_SIZE:-1}

CURRICULUM_TEMPERATURE=${CURRICULUM_TEMPERATURE:-2.0}
MIN_DIFFICULTY=${MIN_DIFFICULTY:-0.1}
MAX_DIFFICULTY=${MAX_DIFFICULTY:-1.0}

LR_DECAY_FACTOR=${LR_DECAY_FACTOR:-0.5}
LR_PATIENCE=${LR_PATIENCE:-5}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-10}

OUTPUT_DIR=${OUTPUT_DIR:-./enhanced_results}
SEED=${SEED:-42}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Enhanced LogNet Training Parameters:"
    echo ""
    echo "📊 Data Configuration:"
    echo "  --easy-samples N          Number of easy samples (default: $EASY_SAMPLES)"
    echo "  --medium-samples N        Number of medium samples (default: $MEDIUM_SAMPLES)"
    echo "  --hard-samples N          Number of hard samples (default: $HARD_SAMPLES)"
    echo "  --very-hard-samples N     Number of very hard samples (default: $VERY_HARD_SAMPLES)"
    echo "  --extreme-hard-samples N  Number of extreme hard samples (default: $EXTREME_HARD_SAMPLES)"
    echo ""
    echo "🧠 Model Configuration:"
    echo "  --input-dim N             Input dimension (default: $INPUT_DIM)"
    echo "  --hidden-dim N            Hidden dimension (default: $HIDDEN_DIM)"
    echo "  --num-classes N           Number of output classes (default: $NUM_CLASSES)"
    echo "  --num-tactics N           Number of tactics (default: $NUM_TACTICS)"
    echo "  --max-steps N             Maximum proof steps (default: $MAX_STEPS)"
    echo "  --num-attention-heads N   Number of attention heads (default: $NUM_ATTENTION_HEADS)"
    echo ""
    echo "🎓 Training Configuration:"
    echo "  --epochs N                Number of training epochs (default: $EPOCHS)"
    echo "  --learning-rate F         Learning rate (default: $LEARNING_RATE)"
    echo "  --weight-decay F          Weight decay (default: $WEIGHT_DECAY)"
    echo "  --dropout-rate F          Dropout rate (default: $DROPOUT_RATE)"
    echo "  --batch-size N            Batch size (default: $BATCH_SIZE)"
    echo ""
    echo "📚 Curriculum Learning:"
    echo "  --curriculum-temperature F  Curriculum temperature (default: $CURRICULUM_TEMPERATURE)"
    echo "  --min-difficulty F        Minimum difficulty (default: $MIN_DIFFICULTY)"
    echo "  --max-difficulty F        Maximum difficulty (default: $MAX_DIFFICULTY)"
    echo ""
    echo "⚙️  Learning Rate Scheduling:"
    echo "  --lr-decay-factor F       LR decay factor (default: $LR_DECAY_FACTOR)"
    echo "  --lr-patience N           LR patience (default: $LR_PATIENCE)"
    echo ""
    echo "🛑 Early Stopping:"
    echo "  --early-stopping-patience N  Early stopping patience (default: $EARLY_STOPPING_PATIENCE)"
    echo ""
    echo "📁 Output Configuration:"
    echo "  --output-dir DIR          Output directory (default: $OUTPUT_DIR)"
    echo "  --save-model              Save trained model"
    echo "  --plot-curves             Plot learning curves"
    echo ""
    echo "🎲 Random Seed:"
    echo "  --seed N                   Random seed (default: $SEED)"
    echo ""
    echo "Examples:"
    echo "  $0 --epochs 50 --hidden-dim 256 --learning-rate 0.0005"
    echo "  $0 --easy-samples 100 --hard-samples 50 --epochs 30"
    echo "  $0 --save-model --plot-curves --output-dir ./my_results"
    echo ""
    echo "Environment Variables (alternative to command line):"
    echo "  EASY_SAMPLES, MEDIUM_SAMPLES, HARD_SAMPLES, VERY_HARD_SAMPLES, EXTREME_HARD_SAMPLES"
    echo "  INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_TACTICS, MAX_STEPS, NUM_ATTENTION_HEADS"
    echo "  EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, BATCH_SIZE"
    echo "  CURRICULUM_TEMPERATURE, MIN_DIFFICULTY, MAX_DIFFICULTY"
    echo "  LR_DECAY_FACTOR, LR_PATIENCE, EARLY_STOPPING_PATIENCE"
    echo "  OUTPUT_DIR, SEED"
}

# Check for help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Build command
CMD="python main_enhanced_training.py"

# Add all parameters
CMD="$CMD --easy-samples $EASY_SAMPLES"
CMD="$CMD --medium-samples $MEDIUM_SAMPLES"
CMD="$CMD --hard-samples $HARD_SAMPLES"
CMD="$CMD --very-hard-samples $VERY_HARD_SAMPLES"
CMD="$CMD --extreme-hard-samples $EXTREME_HARD_SAMPLES"

CMD="$CMD --input-dim $INPUT_DIM"
CMD="$CMD --hidden-dim $HIDDEN_DIM"
CMD="$CMD --num-classes $NUM_CLASSES"
CMD="$CMD --num-tactics $NUM_TACTICS"
CMD="$CMD --max-steps $MAX_STEPS"
CMD="$CMD --num-attention-heads $NUM_ATTENTION_HEADS"

CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --dropout-rate $DROPOUT_RATE"
CMD="$CMD --batch-size $BATCH_SIZE"

CMD="$CMD --curriculum-temperature $CURRICULUM_TEMPERATURE"
CMD="$CMD --min-difficulty $MIN_DIFFICULTY"
CMD="$CMD --max-difficulty $MAX_DIFFICULTY"

CMD="$CMD --lr-decay-factor $LR_DECAY_FACTOR"
CMD="$CMD --lr-patience $LR_PATIENCE"

CMD="$CMD --early-stopping-patience $EARLY_STOPPING_PATIENCE"

CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --seed $SEED"

# Add optional flags
if [[ "$SAVE_MODEL" == "true" ]]; then
    CMD="$CMD --save-model"
fi

if [[ "$PLOT_CURVES" == "true" ]]; then
    CMD="$CMD --plot-curves"
fi

# Add any additional arguments passed to the script
CMD="$CMD $@"

echo "🔧 Configuration:"
echo "   • Easy samples: $EASY_SAMPLES"
echo "   • Medium samples: $MEDIUM_SAMPLES"
echo "   • Hard samples: $HARD_SAMPLES"
echo "   • Very hard samples: $VERY_HARD_SAMPLES"
echo "   • Extreme hard samples: $EXTREME_HARD_SAMPLES"
echo "   • Hidden dimension: $HIDDEN_DIM"
echo "   • Epochs: $EPOCHS"
echo "   • Learning rate: $LEARNING_RATE"
echo "   • Output directory: $OUTPUT_DIR"
echo ""

echo "🚀 Running enhanced training..."
echo "Command: $CMD"
echo ""

# Run the training
eval $CMD

echo ""
echo "🎉 Enhanced training completed!"
echo "📁 Check results in: $OUTPUT_DIR"
