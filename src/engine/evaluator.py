"""
多维评估与 Benchmark 测试引擎 (Evaluation & Benchmark Engine)

该模块负责:
1. 精度评估 (Accuracy Assessment): 在验证集/测试集上计算 Accuracy, Macro F1, Precision, Recall 和混淆矩阵。
2. 性能基准测试 (Performance Benchmark): 测量模型在目标硬件 (NPU/GPU) 上的吞吐量 (FPS) 和端到端推理延迟。
3. 赛题综合打分 (Competition Scoring): 根据精度和加速比计算综合得分 (如 Macro F1 * 100 * 0.5 等自定义逻辑)。
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm


class ModelEvaluator:
    """
    模型评估器，支持精度验证与硬件推理速度压测。
    """
    def __init__(self, model: nn.Module, device: torch.device, use_amp: bool = True):
        """
        Args:
            model (nn.Module): 待评估的模型 (需要包含 forward 方法输出 logits)。
            device (torch.device): 运行设备 (CPU, CUDA, 或 NPU)。
            use_amp (bool): 是否使用自动混合精度 (FP16) 进行推理加速评估。
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        # 确保模型处于评估模式，关闭 Dropout 和 BatchNorm 的更新
        self.model.eval()

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        """
        在给定的 DataLoader 上评估模型的分类精度指标。
        
        Args:
            dataloader (DataLoader): 测试集或验证集的数据加载器。
            desc (str): 进度条的描述文字。
            
        Returns:
            Dict[str, float]: 包含各类精度指标的字典。
        """
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        pbar = tqdm(dataloader, desc=desc, leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 使用混合精度进行前向推理
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                output = self.model(images)
                # 适配字典输出 (如 EndToEndMILModel) 或直接张量输出
                logits = output['logits'] if isinstance(output, dict) else output
                loss = criterion(logits, labels)
                
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader)
        
        # 计算多维评估指标 (针对医疗影像不均衡数据，优先看 Macro F1)
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # 生成混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
        
        # 自定义赛题打分逻辑 (假设 Macro F1 占主导)
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
        对模型进行严格的硬件吞吐量 (Throughput) 和延迟 (Latency) 测试。
        
        Args:
            input_shape (Tuple): 单个 Batch 的输入形状，例如 (Batch_Size, Num_Patches, Feature_Dim)。
            num_warmup (int): 预热迭代次数，用于克服 GPU/NPU 初始化的开销。
            num_runs (int): 实际记录测量的迭代次数。
            
        Returns:
            Dict[str, float]: 包含 FPS, 每次推理均值、方差等信息的字典。
        """
        print(f"\n🚀 开始硬件推理性能压测 (Benchmark) ...")
        print(f"   -> 输入形状: {input_shape}")
        print(f"   -> 预热次数: {num_warmup} | 测试次数: {num_runs}")
        print(f"   -> 混合精度 (AMP): {'开启 (FP16/BF16)' if self.use_amp else '关闭 (FP32)'}")
        
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # 1. 硬件预热 (Warm-up)
        # NPU/GPU 刚启动时会有 Cudnn 寻优或 Graph 编译的延迟，预热能保证测试的准确性
        for _ in range(num_warmup):
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                _ = self.model(dummy_input)
                
        # 等待计算流同步 (针对异步执行的硬件非常关键)
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
            latencies.append((end_time - start_time) * 1000)  # 转换为毫秒 (ms)
            
        # 3. 统计学指标计算
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # 计算每秒处理的样本数 (Frames Per Second)
        # 这里的样本单位取决于 input_shape[0]。如果是 WSI 级别，就是 WSI/s。
        mean_time_sec = mean_latency / 1000.0
        fps = batch_size / mean_time_sec
        
        print(f"\n✅ 压测完成!")
        print(f"   -> 吞吐量 (FPS): {fps:.2f} Samples/sec")
        print(f"   -> 平均延迟 (Mean Latency): {mean_latency:.2f} ± {std_latency:.2f} ms")
        print(f"   -> 尾部延迟 (P99 Latency): {p99_latency:.2f} ms")
        
        return {
            "fps": fps,
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency
        }


# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    from src.models.mil_aggregator import EndToEndMILModel
    
    print("="*60)
    print("🛠️  正在实例化 Evaluator 测试 ...")
    
    # 1. 构建模拟数据和 DataLoader (假设是特征提取后的形态)
    B, N, D, Num_Classes = 8, 100, 512, 4
    dummy_x = torch.randn(64, N, D)
    dummy_y = torch.randint(0, Num_Classes, (64,))
    
    dataset = TensorDataset(dummy_x, dummy_y)
    test_loader = DataLoader(dataset, batch_size=B, shuffle=False)
    
    # 2. 实例化模型并迁移到硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndMILModel(agg_type='gated', input_dim=D, num_classes=Num_Classes)
    
    # 3. 实例化评估器
    evaluator = ModelEvaluator(model=model, device=device, use_amp=True)
    
    # 4. 测试精度评估逻辑
    print("\n▶️ [测试 1] 运行精度评估 (Accuracy Evaluation) ...")
    metrics = evaluator.evaluate_accuracy(test_loader, desc="Testing Model")
    
    print(f"   ✅ 测试集 Loss: {metrics['loss']:.4f}")
    print(f"   ✅ 测试集 Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ✅ 测试集 Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   ✅ 赛题综合打分: {metrics['competition_score']:.2f}")
    print(f"   ✅ 混淆矩阵:\n{np.array(metrics['confusion_matrix'])}")
    
    # 5. 测试硬件加速压测逻辑
    print("\n▶️ [测试 2] 运行硬件压测 (Performance Benchmark) ...")
    # 输入张量维度: (Batch, Num_Patches, Feature_Dim)
    benchmark_shape = (B, N, D)
    speed_metrics = evaluator.benchmark_throughput(
        input_shape=benchmark_shape, 
        num_warmup=10, 
        num_runs=100
    )
    
    print("="*60)