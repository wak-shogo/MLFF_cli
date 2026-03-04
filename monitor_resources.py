import psutil
import time
import threading
import os
import sys

try:
    import pynvml
    HAS_NVML = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    HAS_NVML = False

class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_usage = []
        self.mem_usage = []
        self.gpu_usage = []
        self.gpu_mem_usage = []

    def _monitor(self):
        while not self.stop_event.is_set():
            # CPU and RAM
            self.cpu_usage.append(psutil.cpu_percent())
            self.mem_usage.append(psutil.virtual_memory().percent)
            
            # GPU
            if HAS_NVML:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_usage.append(util.gpu)
                    self.gpu_mem_usage.append(100.0 * mem.used / mem.total)
                except:
                    pass
            time.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.report()

    def report(self):
        print("\n" + "="*40)
        print("      RESOURCE USAGE SUMMARY")
        print("="*40)
        if self.cpu_usage:
            print(f"CPU Usage:    Avg: {sum(self.cpu_usage)/len(self.cpu_usage):.1f}%, Max: {max(self.cpu_usage):.1f}%")
        if self.mem_usage:
            print(f"RAM Usage:    Avg: {sum(self.mem_usage)/len(self.mem_usage):.1f}%, Max: {max(self.mem_usage):.1f}%")
        
        if HAS_NVML and self.gpu_usage:
            print(f"GPU Usage:    Avg: {sum(self.gpu_usage)/len(self.gpu_usage):.1f}%, Max: {max(self.gpu_usage):.1f}%")
            print(f"GPU VRAM:     Avg: {sum(self.gpu_mem_usage)/len(self.gpu_mem_usage):.1f}%, Max: {max(self.gpu_mem_usage):.1f}%")
        elif not HAS_NVML:
            print("GPU Monitoring: Not Available (NVIDIA driver/NVML not found)")
        print("="*40 + "\n")

if __name__ == "__main__":
    # If run as a script, it expects to wrap another command
    if len(sys.argv) < 2:
        print("Usage: monitor_resources.py <command> [args...]")
        sys.exit(1)
    
    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        import subprocess
        subprocess.run(sys.argv[1:], check=True)
    finally:
        monitor.stop()
