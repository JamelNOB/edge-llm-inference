"""
System Resource Monitoring Probe (psutil)
=========================================
Real-time sampling of process RSS memory, system memory utilization, and CPU load.
"""
import os
import time
import psutil
import threading
from typing import Dict, Any, Optional

class ServerResourceMonitor(threading.Thread):
    """
    服务端进程后台高精度资源监测采样线程 (50ms 采样周期)
    用于量化评估推理生成前、中、后的显存/内存峰值抖动。
    """
    def __init__(self, target_pid: Optional[int] = None, interval: float = 0.05):
        super().__init__(daemon=True)
        self.target_pid = target_pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_ram_mb: float = 0.0
        self.peak_cpu_percent: float = 0.0

        try:
            self.process = psutil.Process(target_pid) if target_pid else psutil.Process()
        except Exception:
            self.process = psutil.Process()

    def run(self):
        while not self.stop_event.is_set():
            try:
                ram_mb = self.process.memory_info().rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent(interval=None)
                if ram_mb > self.peak_ram_mb:
                    self.peak_ram_mb = ram_mb
                if cpu_percent > self.peak_cpu_percent:
                    self.peak_cpu_percent = cpu_percent
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


def get_system_metrics() -> Dict[str, Any]:
    """获取当前进程与主机的即时资源开销"""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    system_ram_percent = psutil.virtual_memory().percent
    logical_cores = os.cpu_count() or 1
    cpu_percent = psutil.cpu_percent(interval=None)

    return {
        "memory": {
            "process_ram_rss_mb": round(ram_mb, 2),
            "system_ram_percent": round(system_ram_percent, 1)
        },
        "system": {
            "cpu_cores_logical": logical_cores,
            "cpu_percent": round(cpu_percent, 1)
        }
    }
