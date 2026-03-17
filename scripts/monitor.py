# scripts/monitor.py
"""
Monitor training experiments and send alerts for anomalies.
This script monitors the status of a training experiment, including:
1. Whether the training process is still running
2. The status of the GPUs (memory usage and utilization)
3. Whether there are any anomalies in the log file (e.g., lines containing "Error" or "Exception")
If any anomalies are detected, an alert will be sent to the user via Server酱.

NOTE: You need to set the environment variable SCT_KEY to use Server酱 for sending notifications.
"""

import subprocess
import time
import requests
import re
import os
from typing import List
import dotenv

dotenv.load_dotenv()

MY_KEY = os.getenv("SCT_KEY")
DEFAULT_CHECK_INTERVAL = 5 * 60  # seconds


def send_alert(message):
    """Send an alert message using Server酱"""

    requests.post(
        f"https://sct.ftqq.com/{MY_KEY}.send",
        data={"text": "实验警报", "desp": message},
    )


def check_gpu_status(cuda_devices: List[int], anomaly_counter: dict, max_anomaly=3):
    """Check the status of the specified GPUs, and send an alert if any anomalies are detected"""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return

        for line in result.stdout.strip().split("\n"):
            index, mem_used, gpu_util = map(int, line.split(", "))
            if index in cuda_devices:
                print(
                    f"GPU {index}: Memory Used = {mem_used} MiB, GPU Utilization = {gpu_util}%"
                )
                if mem_used == 0 or (mem_used > 0 and gpu_util == 0):
                    anomaly_counter[index] = anomaly_counter.get(index, 0) + 1
                    if anomaly_counter[index] >= max_anomaly:
                        send_alert(
                            f"GPU {index} 可能出现问题(连续{max_anomaly}次检测异常): Memory Used = {mem_used} MiB, GPU Utilization = {gpu_util}%"
                        )
                        print(f"Alert sent for GPU {index}.")
                        anomaly_counter[index] = 0  # reset counter after sending alert
                else:
                    anomaly_counter[index] = 0  # reset counter if GPU is normal
    except Exception as e:
        print(f"Error checking GPU status: {e}")


def is_process_running(pid):
    """Check if a process with the given PID is still running"""
    try:
        # `os.kill(pid, 0)` doesn't actually kill the process,
        # but will raise an OSError if the process does not exist
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def monitor_experiment(pid, log_file_path, cuda_devices: List[int], check_interval=60):
    last_log_position = 0
    gpu_anomaly_counter = {
        device: 0 for device in cuda_devices
    }  # record consecutive anomaly counts for each GPU

    while True:
        if not is_process_running(pid):
            send_alert(f"实验进程 (PID: {pid}) 已停止运行！")
            print(f"Alert sent for experiment process (PID: {pid}).")
            break

        check_gpu_status(cuda_devices, gpu_anomaly_counter, max_anomaly=3)

        # Monitor log file for anomalies
        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as log_file:
                    log_file.seek(
                        last_log_position
                    )  # move to the end of the last read, to prevent duplicate sending
                    new_lines = log_file.readlines()
                    last_log_position = log_file.tell()  # update read position

                    for line in new_lines:
                        if re.search(r"Error|Exception", line, re.IGNORECASE):
                            send_alert(f"日志文件中检测到异常: {line.strip()}")
                            print(f"Alert sent for log file anomaly: {line.strip()}")
        except Exception as e:
            print(f"Error reading log file: {e}")

        time.sleep(
            check_interval
        )  # wait for the specified interval before checking again


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Training Experiment")
    parser.add_argument(
        "--pid", type=int, required=True, help="PID of the experiment process"
    )
    parser.add_argument("--log", type=str, required=True, help="Path to the log file")
    parser.add_argument(
        "--devices",
        type=str,
        required=True,
        help="Comma-separated list of CUDA device IDs to monitor (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
        help="Check interval in seconds",
    )
    args = parser.parse_args()

    # Parse the devices argument into a list of integers
    try:
        device_list = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
    except ValueError:
        print("Error: --devices must be a comma-separated list of integers.")
        exit(1)

    monitor_experiment(args.pid, args.log, device_list, check_interval=args.interval)
