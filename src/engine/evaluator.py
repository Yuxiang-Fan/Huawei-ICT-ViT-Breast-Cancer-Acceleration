"""
多维评估与 Benchmark 测试引擎

负责:
1. 精度评估: 计算 Accuracy, Macro F1, Precision, Recall 和混淆矩阵。
2. 性能基准测试 (Benchmark): 测量模型在目标硬件上的吞吐量 (FPS) 和端到端推理延迟。
3. 赛题综合打分: 根据自定义逻辑输出最终竞赛得分。
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm


class ModelEvaluator:
    """
    模型评估器，支持精度验证与硬件推理速度压测。
    """
    def __init__(self, model: nn.Module, device: torch.device, use_amp: bool = True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.model.eval()

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        """
        在 DataLoader 上评估模型的分类精度指标。
        """
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        pbar = tqdm(dataloader, desc=desc, leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                output = self.model(images)
                logits = output['logits'] if isinstance(output, dict) else output
                loss = criterion(logits, labels)
                
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader)
        
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
        
        # 假设竞赛评分逻辑以 Macro F1 为主导
        classification_score = macro_f1 * 100 * 0.5
        
        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision": macro_prec,
            "recall": macro_rec,
            "confusion_matrix": conf_matrix,
            "competition_score": classification_score
        }

    @torch.no_grad()
    def benchmark_throughput(self, input_shape: Tuple[int, ...], num_warmup: int = 10, num_runs: int = 50) -> Dict[str, float]:
        """
        对模型进行硬件吞吐量和延迟的 Benchmark 测试。
        """
        print(f"[Benchmark] 开始硬件推理性能压测...")
        print(f"  - 输入形状: {input_shape}")
        print(f"  - 预热次数: {num_warmup} | 测试次数: {num_runs}")
        print(f"  - AMP 混合精度: {'开启' if self.use_amp else '关闭'}")
        
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # 1. 硬件预热 (Warm-up)
        for _ in range(num_warmup):
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                _ = self.model(dummy_input)
                
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            
        # 2. 核心性能测量
        latencies = []
        batch_size = input_shape[0]
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                _ = self.model(dummy_input)
                
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
                
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) 
            
        # 3. 统计学指标计算
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        mean_time_sec = mean_latency / 1000.0
        fps = batch_size / mean_time_sec
        
        print(f"[Benchmark] 压测完成。")
        print(f"  - 吞吐量 (FPS): {fps:.2f} Samples/sec")
        print(f"  - 平均延迟: {mean_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  - P99 尾部延迟: {p99_latency:.2f} ms")
        
        return {
            "fps": fps,
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency
        }
