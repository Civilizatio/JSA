#!/bin/bash


# Parameters (Can be customized)
ENABLE_MONITOR=false # true to enable resource monitor, false to disable
LOG_FILE="train_output.log" # redirect all output to this log file
CUDA_DEVICES=4,5 # CUDA_DEVICES=0,1,2,3 # Example for 4 GPUs
CONFIG_PATH="configs/vq_gan/cifar10.yaml" # Path to your training config file



# Process IDs
# Will be set after starting the training and monitor processes
TRAIN_PID=""
MONITOR_PID=""

############################################
#  Cleanup function 
############################################
cleanup() {
    echo -e "\n[!] Received termination signal, cleaning up..."

    if [ -n "$TRAIN_PID" ]; then
        echo "[+] Terminating training process group (PGID=$TRAIN_PID)..."

        # Kill the entire process group to ensure all child processes are terminated
        kill -TERM -$TRAIN_PID 2>/dev/null

        # Wait a moment for graceful shutdown
        sleep 5

        # Force kill if still alive
        kill -KILL -$TRAIN_PID 2>/dev/null
    fi

    if [ "$ENABLE_MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
        echo "[+] Terminating monitor process..."
        kill -TERM $MONITOR_PID 2>/dev/null
    fi

    echo "[✓] Cleanup completed"
}

# Set trap to catch termination signals and call cleanup
trap cleanup INT TERM EXIT

############################################
# Start training in NEW SESSION
############################################

echo "[+] Starting training..."

setsid env PYTHONPATH=. python scripts/train.py fit \
    --config $CONFIG_PATH \
    --trainer.devices $CUDA_DEVICES \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "[✓] Training started"
echo "    PID : $TRAIN_PID"
echo "    LOG : $LOG_FILE"
echo "    tail -f $LOG_FILE to view logs"

############################################
# 🔍 Start monitor (optional)
############################################

if [ "$ENABLE_MONITOR" = true ]; then
    python scripts/monitor.py \
        --pid $TRAIN_PID \
        --log "$LOG_FILE" \
        --devices $CUDA_DEVICES &

    MONITOR_PID=$!
    echo "[✓] Monitor started (PID=$MONITOR_PID)"
else
    echo "[i] Monitor disabled"
fi

############################################
#  Wait for training
############################################

wait $TRAIN_PID
EXIT_CODE=$?

echo "[+] Training process exited (code=$EXIT_CODE)"

############################################
#  Stop monitor if still alive
############################################

if [ "$ENABLE_MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
fi

############################################
#  Report result
############################################

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "\n[❌ ERROR] Training failed (Exit Code: $EXIT_CODE)"
    echo "Last 20 lines of log:"
    echo "----------------------------------------"
    tail -n 20 "$LOG_FILE"
    echo "----------------------------------------"
else
    echo -e "\n[✅ SUCCESS] Training completed successfully"
fi