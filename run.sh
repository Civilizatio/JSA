#!/bin/bash
# --config configs/jsa/categorical_prior_continuous_cifar10_conv.yaml \
# --config configs/vq_gan/cifar10.yaml \
# --config configs/cond_transformer/cifar10.yaml \

# 0. 监控开关 (设置为 true 开启监控，设置为 false 关闭监控)
ENABLE_MONITOR=true

# 1. 设置日志文件路径
LOG_FILE="train_output.log"
MONITOR_PID="" # 初始化为空
CUDA_DEVICES=0,1,2,3 # 你可以根据需要修改为你想使用的 GPU 设备，例如 "0,1" 或 "0,1,2,3"

# === 新增：信号捕捉（优雅退出） ===
# 当你按下 Ctrl+C 时，触发 cleanup 函数，把后台进程一起杀掉
cleanup() {
    echo -e "\n[!] 收到中断信号(Ctrl+C)，正在终止后台进程..."
    kill $TRAIN_PID 2>/dev/null
    if [ "$ENABLE_MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
    fi
    echo "已成功终止实验及相关的后台进程。"
    exit 1
}
trap cleanup INT TERM
# ==================================

# 2. 在后台运行训练脚本，并将输出重定向到日志文件
PYTHONPATH=. python scripts/train.py fit --config configs/jsa/imagenet.yaml \
    --trainer.devices $CUDA_DEVICES > "$LOG_FILE" 2>&1 &

# 3. 获取刚刚启动的训练进程的 PID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID. Logging to $LOG_FILE"
echo "👉 提示：你可以打开新的终端运行 \`tail -f $LOG_FILE\` 来实时查看训练输出！"

# 4. 根据开关决定是否启动监控脚本
if [ "$ENABLE_MONITOR" = true ]; then
    python monitor.py --pid $TRAIN_PID --log "$LOG_FILE" --devices $CUDA_DEVICES &
    MONITOR_PID=$!
    echo "[监控已开启] Monitor process started with PID: $MONITOR_PID"
else
    echo "[监控已关闭] 未启动 monitor.py"
fi

# 5. 等待训练进程结束
wait $TRAIN_PID
echo "Training process $TRAIN_PID finished."

# 6. 训练结束后，如果开启了监控则关掉监控程序
if [ "$ENABLE_MONITOR" = true ] && [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
    echo "Monitor process $MONITOR_PID stopped."
fi
