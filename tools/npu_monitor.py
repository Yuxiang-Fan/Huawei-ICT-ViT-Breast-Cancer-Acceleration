"""
昇腾 NPU 与系统资源实时监控工具 (NPU & System Resource Monitor)

该模块负责:
1. 实时查询昇腾 NPU 的底层状态 (显存占用、温度、AI Core 算力利用率等)。
2. 监控宿主机 (Host) 的 CPU 利用率与内存开销。
3. 针对特定的 Python 进程 (如推理进程) 进行精准的资源采样。
4. 将监控数据持久化为 CSV 文件，便于在实验报告中生成详细的性能波动折线图。
"""

import os
import sys
import time
import csv
import argparse
import subprocess
from datetime import datetime
import re

try:
    import psutil
except ImportError:
    print("⚠️ 缺少 psutil 库。请执行: pip install psutil")
    sys.exit(1)


class NPUMonitor:
    """
    昇腾硬件与系统资源监控器
    """
    def __init__(self, log_dir: str, interval: float = 1.0, target_pid: int = None):
        """
        初始化监控器。
        
        Args:
            log_dir (str): 监控日志 (CSV) 的保存目录。
            interval (float): 采样时间间隔 (秒)。
            target_pid (int, optional): 需要重点监控内存的特定进程 PID。若未指定，则仅监控全局。
        """
        self.log_dir = log_dir
        self.interval = interval
        self.target_pid = target_pid
        
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"npu_resource_monitor_{timestamp}.csv")
        
        # 定义 CSV 表头
        self.headers = [
            "Timestamp", 
            "Host_CPU_Util(%)", 
            "Host_RAM_Util(%)",
            "Target_Process_RAM(GB)",
            "NPU_ID", 
            "NPU_Util(%)", 
            "NPU_Mem_Used(MB)", 
            "NPU_Mem_Total(MB)",
            "NPU_Temp(C)"
        ]
        
        self._init_csv()
        
        # 检查 npu-smi 命令是否可用
        self.has_npu_smi = self._check_npu_smi()
        if not self.has_npu_smi:
            print("⚠️ 警告: 当前环境未检测到 `npu-smi` 命令！工具将仅监控 Host 端 CPU/RAM 资源。")

    def _init_csv(self):
        """初始化 CSV 文件并写入表头"""
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        print(f"📁 监控日志已创建: {self.csv_path}")

    def _check_npu_smi(self) -> bool:
        """检查底层 npu-smi 指令可用性"""
        try:
            subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_system_metrics(self) -> dict:
        """获取 Host 端的 CPU 和 内存指标"""
        metrics = {
            "Host_CPU_Util(%)": psutil.cpu_percent(interval=None),
            "Host_RAM_Util(%)": psutil.virtual_memory().percent,
            "Target_Process_RAM(GB)": 0.0
        }
        
        # 如果指定了具体的 PID (比如推理进程)，则获取其精准内存开销
        if self.target_pid:
            try:
                proc = psutil.Process(self.target_pid)
                # RSS (Resident Set Size): 进程实际占用的物理内存
                metrics["Target_Process_RAM(GB)"] = round(proc.memory_info().rss / (1024 ** 3), 3)
            except psutil.NoSuchProcess:
                metrics["Target_Process_RAM(GB)"] = -1.0 # 进程已结束
        
        return metrics

    def _get_npu_metrics(self) -> dict:
        """
        解析 npu-smi info 的输出，提取 NPU 的利用率、显存和温度信息。
        如果存在多张卡，默认取第一张卡 (NPU_ID = 0) 的数据。
        """
        metrics = {
            "NPU_ID": 0,
            "NPU_Util(%)": 0,
            "NPU_Mem_Used(MB)": 0,
            "NPU_Mem_Total(MB)": 0,
            "NPU_Temp(C)": 0
        }
        
        if not self.has_npu_smi:
            return metrics

        try:
            # 执行 npu-smi info
            result = subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout

            # 解析类似于: | 0    Ascend910B... | 45°C    23W / 300W |   1024 / 32768 MB |   85%      |
            # 这里的正则规则适配大多数昇腾芯片的标准表格输出
            lines = output.split('\n')
            for line in lines:
                # 寻找包含卡号和温度的行
                if re.search(r'\|\s*\d+\s+Ascend', line):
                    # 提取 NPU ID
                    id_match = re.search(r'\|\s*(\d+)\s+Ascend', line)
                    if id_match:
                        metrics["NPU_ID"] = int(id_match.group(1))
                    
                    # 提取温度
                    temp_match = re.search(r'(\d+)°?C', line)
                    if temp_match:
                        metrics["NPU_Temp(C)"] = int(temp_match.group(1))
                        
                    # 提取显存 (例如: 1024 / 32768 MB)
                    mem_match = re.search(r'(\d+)\s*/\s*(\d+)\s*MB', line)
                    if mem_match:
                        metrics["NPU_Mem_Used(MB)"] = int(mem_match.group(1))
                        metrics["NPU_Mem_Total(MB)"] = int(mem_match.group(2))
                        
                    # 提取 NPU (AI Core) 利用率 (例如: 85%)
                    util_match = re.search(r'(\d+)\s*%', line)
                    if util_match:
                        metrics["NPU_Util(%)"] = int(util_match.group(1))
                        
                    # 仅解析第一张卡的通用信息即可退出循环
                    break
                    
        except Exception as e:
            # 静默处理解析异常，防止监控进程崩溃
            pass
            
        return metrics

    def start(self):
        """启动监控守护循环"""
        print(f"🚀 开始持续监控硬件资源 (采样间隔: {self.interval}s)")
        if self.target_pid:
            print(f"🎯 重点关注进程 PID: {self.target_pid}")
        print("💡 提示: 按下 Ctrl+C 可随时停止监控并保存日志。")
        print("-" * 80)
        
        # 预热 psutil 的 CPU 计算
        psutil.cpu_percent(interval=None)
        
        try:
            while True:
                # 获取系统时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 采集指标
                sys_metrics = self._get_system_metrics()
                npu_metrics = self._get_npu_metrics()
                
                # 如果监控目标进程已经结束，则自动停止监控
                if self.target_pid and sys_metrics["Target_Process_RAM(GB)"] == -1.0:
                    print(f"\n⏹️ 目标进程 (PID: {self.target_pid}) 已结束，监控自动终止。")
                    break

                # 拼接 CSV 行
                row = [
                    timestamp,
                    sys_metrics["Host_CPU_Util(%)"],
                    sys_metrics["Host_RAM_Util(%)"],
                    sys_metrics["Target_Process_RAM(GB)"],
                    npu_metrics["NPU_ID"],
                    npu_metrics["NPU_Util(%)"],
                    npu_metrics["NPU_Mem_Used(MB)"],
                    npu_metrics["NPU_Mem_Total(MB)"],
                    npu_metrics["NPU_Temp(C)"]
                ]
                
                # 写入文件并立即刷入磁盘 (防止意外中断导致数据丢失)
                with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    f.flush()
                
                # 控制台轻量级打印
                mem_gb = npu_metrics["NPU_Mem_Used(MB)"] / 1024
                print(f"[{timestamp}] NPU AI-Core: {npu_metrics['NPU_Util(%)']:3d}% | "
                      f"NPU Mem: {mem_gb:5.1f} GB | "
                      f"Host CPU: {sys_metrics['Host_CPU_Util(%)']:4.1f}%", end='\r')
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n" + "-" * 80)
            print("🛑 监控已手动停止。")
            
        finally:
            print(f"💾 完整硬件性能数据已保存至: {os.path.abspath(self.csv_path)}")
            print("📈 你可以使用 Pandas 导入此 CSV 并绘制加速方案前后的显存峰值对比图！")


# =====================================================================
# CLI 命令行入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="昇腾 NPU & 宿主机性能监控守护进程")
    
    parser.add_argument("--log_dir", type=str, default="./logs/hardware_monitor", 
                        help="监控数据的 CSV 保存目录")
    parser.add_argument("--interval", type=float, default=1.0, 
                        help="监控数据采样的时间间隔 (秒)")
    parser.add_argument("--pid", type=int, default=None, 
                        help="(可选) 指定需要精准追踪内存开销的 Python 推理进程 ID")
    
    args = parser.parse_args()
    
    monitor = NPUMonitor(
        log_dir=args.log_dir,
        interval=args.interval,
        target_pid=args.pid
    )
    
    monitor.start()