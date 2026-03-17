#!/bin/bash

# 0. Monitoring switch (set to true to enable monitoring, set to false to disable monitoring)
ENABLE_MONITOR=true

# 1. Define log file and CUDA devices (you can modify these as needed)
LOG_FILE="train_output.log"
MONITOR_PID="" # Initialize the monitor PID variable, it will be set later if monitoring is enabled
CUDA_DEVICES=0,1,2,3 # Modify this to specify which GPUs to use (e.g., "0,1" for GPU 0 and 1, or "0" for only GPU 0)
CONFIG_PATH="configs/jsa/imagenet.yaml" # Modify this to specify the config file for training

# When the script receives an interrupt signal (e.g., Ctrl+C), it will execute the cleanup function to terminate the training and monitoring processes gracefully.
cleanup() {
    echo -e "\n[!] Received interrupt signal (Ctrl+C), terminating background processes..."
    kill $TRAIN_PID 2>/dev/null
    if [ "$ENABLE_MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
    fi
    echo "Successfully terminated the experiment and related background processes."
    exit 1
}
trap cleanup INT TERM
# ==================================

# 2. Training command (you can modify the config path and other parameters as needed)
PYTHONPATH=. python scripts/train.py fit --config $CONFIG_PATH \
    --trainer.devices $CUDA_DEVICES > "$LOG_FILE" 2>&1 &

# 3. Get the PID of the training process and print it out
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID. Logging to $LOG_FILE"
echo "👉 Tip: You can open a new terminal and run \`tail -f $LOG_FILE\` to view the training output in real time!"

# 4. If monitoring is enabled, start the monitor.py script in the background and print its PID
if [ "$ENABLE_MONITOR" = true ]; then
    python scripts/monitor.py --pid $TRAIN_PID --log "$LOG_FILE" --devices $CUDA_DEVICES &
    MONITOR_PID=$!
    echo "[Monitoring enabled] Monitor process started with PID: $MONITOR_PID"
else
    echo "[Monitoring disabled] monitor.py not started"
fi

# 5. Wait for the training process to finish and print a message when it does
wait $TRAIN_PID
echo "Training process $TRAIN_PID finished."

# 6. If monitoring is enabled, kill the monitor process when training is done and print a message
if [ "$ENABLE_MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
    echo "Monitor process $MONITOR_PID stopped."
fi
