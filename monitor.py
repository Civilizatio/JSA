# monitor.py
# 这个脚本用于监测训练实验的状态，包括：
# 1. 监测训练进程是否还在运行
# 2. 监测 GPU 的状态（显存占用和利用率）
# 3. 监测日志文件中是否有异常输出（例如包含 "Error" 或 "Exception" 的行）
# 如果检测到任何异常情况，都会通过 Server酱发送警报通知用户。

# 需要设置环境变量 SCT_KEY 来使用 Server酱发送通知。

import subprocess
import time
import requests
import re
from datetime import datetime
import os
from typing import List
import dotenv
dotenv.load_dotenv()

MY_KEY = os.getenv("SCT_KEY")

def send_alert(message):
    requests.post(
        f"https://sct.ftqq.com/{MY_KEY}.send",
        data={"text": "实验警报", "desp": message}
    )
    
def check_gpu_status(cuda_devices: List[int], anomaly_counter: dict, max_anomaly=3):
    """ 检查相应的 GPU 状态，如果出现了显存不占用，或者显存占用但是利用率为 0 的情况，说明可能出现了问题 """
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return
        
        for line in result.stdout.strip().split("\n"):
            index, mem_used, gpu_util = map(int, line.split(", "))
            if index in cuda_devices:
                print(f"GPU {index}: Memory Used = {mem_used} MiB, GPU Utilization = {gpu_util}%")
                if mem_used == 0 or (mem_used > 0 and gpu_util == 0):
                    anomaly_counter[index] = anomaly_counter.get(index, 0) + 1
                    if anomaly_counter[index] >= max_anomaly:
                        send_alert(f"GPU {index} 可能出现问题(连续{max_anomaly}次检测异常): Memory Used = {mem_used} MiB, GPU Utilization = {gpu_util}%")
                        print(f"Alert sent for GPU {index}.")
                        anomaly_counter[index] = 0 # 发送警报后重置，避免连续狂轰乱炸
                else:
                    anomaly_counter[index] = 0 # 只要有一次正常，就重置计数器
    except Exception as e:
        print(f"Error checking GPU status: {e}")
    
def is_process_running(pid):
    """ 检查指定 PID 的进程是否还在运行 """
    try:
        # 在 Unix 系统上，发送信号 0 来检查进程是否存在
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def monitor_experiment(pid, log_file_path, cuda_devices: List[int], check_interval=60):
    last_log_position = 0
    gpu_anomaly_counter = {device: 0 for device in cuda_devices} # 新增一个字典来记录每个设备的连续异常次数
    
    # 监测实验进程是否还在运行
    while True:
        if not is_process_running(pid):
            send_alert(f"实验进程 (PID: {pid}) 已停止运行！")
            print(f"Alert sent for experiment process (PID: {pid}).")
            break
        
        # 监测 GPU 状态
        check_gpu_status(cuda_devices, gpu_anomaly_counter, max_anomaly=3)
        
        # 监测日志文件是否有异常输出（例如包含 "Error" 或 "Exception" 的行）
        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as log_file:
                    log_file.seek(last_log_position) # 移动到上次读取的末尾，防止重复发送
                    new_lines = log_file.readlines()
                    last_log_position = log_file.tell() # 更新读取位置
                    
                    for line in new_lines:
                        if re.search(r'Error|Exception', line, re.IGNORECASE):
                            send_alert(f"日志文件中检测到异常: {line.strip()}")
                            print(f"Alert sent for log file anomaly: {line.strip()}")
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        time.sleep(check_interval) # 每隔 check_interval 秒检查一次
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor Training Experiment")
    parser.add_argument("--pid", type=int, required=True, help="PID of the experiment process")
    parser.add_argument("--log", type=str, required=True, help="Path to the log file")
    parser.add_argument("--devices", type=str, required=True, help="Comma-separated list of CUDA device IDs to monitor (e.g., '0,1,2,3')")
    args = parser.parse_args()
    
    # 将逗号分隔的字符串转换为整数列表
    try:
        device_list = [int(d.strip()) for d in args.devices.split(',') if d.strip()]
    except ValueError:
        print("Error: --devices must be a comma-separated list of integers.")
        exit(1)
    
    monitor_experiment(args.pid, args.log, device_list)

