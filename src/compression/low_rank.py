"""
基于奇异值分解 (SVD) 的低秩压缩算法库 (Low-Rank SVD Compression)

该模块负责:
1. 提取神经网络中的全连接层 (nn.Linear) 的权重矩阵。
2. 对权重矩阵执行 SVD 分解，并根据设定的能量保留阈值 (Energy Threshold) 截断奇异值。
3. 将原始的单层 nn.Linear 替换为由两个小维度 nn.Linear 串联组成的 nn.Sequential。
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class SVDLowRankCompressor:
    """
    SVD 低秩压缩器。
    
    原理: 对于一个输入维度为 N, 输出维度为 M 的权重矩阵 W (M x N)，
    通过 SVD 分解得到 W ≈ U * S * V^T。
    若选取秩 K < min(M, N)，则可将原层替换为两个线性层：
    1. Linear(N, K, bias=False)  -> 权重为 S * V^T
    2. Linear(K, M, bias=True)   -> 权重为 U
    
    参数量从 M*N 下降到 K*(M+N)。
    """
    def __init__(self, energy_threshold: float = 0.95, min_compression_ratio: float = 0.8):
        """
        初始化低秩压缩器。
        
        Args:
            energy_threshold (float): 奇异值能量保留阈值 (通常在 0.90 - 0.99 之间)。
                                      能量定义为奇异值的平方和。
            min_compression_ratio (float): 最小压缩比阈值。如果分解后的参数量 / 原参数量 
                                           大于此值，则认为分解收益太小，放弃对该层的压缩。
        """
        self.energy_threshold = energy_threshold
        self.min_compression_ratio = min_compression_ratio

    @torch.no_grad()
    def _decompose_linear_layer(self, layer: nn.Linear, layer_name: str) -> nn.Module:
        """
        对单个 nn.Linear 执行 SVD 分解和重构。
        """
        # 获取原始权重和偏置
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        out_features, in_features = weight.shape
        
        # 将权重转移到 CPU 进行 SVD，避免部分 NPU/GPU 算子不支持或显存溢出
        weight_cpu = weight.cpu().float()
        
        # 执行奇异值分解 W = U * S * V^T
        # full_matrices=False 表示计算经济型 SVD
        U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
        
        # 计算能量占比以决定保留的秩 K
        squared_s = S ** 2
        total_energy = torch.sum(squared_s)
        cumulative_energy = torch.cumsum(squared_s, dim=0)
        
        # 找到满足能量阈值的最小 K
        k = torch.searchsorted(cumulative_energy, total_energy * self.energy_threshold).item() + 1
        
        # 确保 K 不超过矩阵的满秩
        k = min(k, min(out_features, in_features))
        
        # 计算参数量变化
        original_params = out_features * in_features
        compressed_params = k * (out_features + in_features)
        compression_ratio = compressed_params / original_params
        
        # 如果压缩收益不明显 (比如大于设定阈值)，或者 K 过小导致层失效，则保留原层
        if compression_ratio > self.min_compression_ratio or k < 2:
            return layer
            
        print(f"   [SVD 压缩] 层: {layer_name}")
        print(f"     -> 形状: ({in_features} -> {out_features}) | 选定秩 K: {k}")
        print(f"     -> 参数量: {original_params/1000:.1f}K -> {compressed_params/1000:.1f}K (压缩率: {compression_ratio*100:.1f}%)")
        
        # 截断 U, S, Vh
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        
        # 重构为两个小矩阵 W1 和 W2
        # W1 = diag(S_k) * Vh_k
        # W2 = U_k
        W1 = torch.diag(S_k) @ Vh_k
        W2 = U_k
        
        # 构建新的网络层
        # 第一层: 输入 in_features, 输出 k, 无偏置
        layer1 = nn.Linear(in_features, k, bias=False)
        layer1.weight.data = W1.to(weight.device).to(weight.dtype)
        
        # 第二层: 输入 k, 输出 out_features, 挂载原始偏置
        layer2 = nn.Linear(k, out_features, bias=bias is not None)
        layer2.weight.data = W2.to(weight.device).to(weight.dtype)
        if bias is not None:
            layer2.bias.data = bias.to(weight.device).to(weight.dtype)
            
        # 使用 nn.Sequential 打包
        decomposed_module = nn.Sequential(layer1, layer2)
        return decomposed_module

    def compress_model(self, model: nn.Module) -> nn.Module:
        """
        递归遍历整个模型，将符合条件的 nn.Linear 层替换为低秩分解后的层。
        """
        print(f"\n✂️  [低秩压缩] 开始扫描模型全连接层 (能量保留阈值: {self.energy_threshold*100:.1f}%)...")
        
        # 统计压缩前的参数量
        orig_params = sum(p.numel() for p in model.parameters())
        replaced_count = 0
        
        def recursive_replace(module: nn.Module, prefix: str = ""):
            nonlocal replaced_count
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 如果是 Linear 层，尝试进行 SVD 分解
                if isinstance(child, nn.Linear):
                    new_child = self._decompose_linear_layer(child, full_name)
                    # 如果返回的是不同的模块（即成功压缩了）
                    if new_child is not child:
                        setattr(module, name, new_child)
                        replaced_count += 1
                else:
                    # 递归深入子模块
                    recursive_replace(child, full_name)
                    
        # 执行递归替换
        recursive_replace(model)
        
        # 统计压缩后的参数量
        new_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n✅ [低秩压缩] 完成!")
        print(f"   -> 成功分解了 {replaced_count} 个 Linear 层。")
        print(f"   -> 总参数量: {orig_params / 1e6:.2f} M  ->  {new_params / 1e6:.2f} M")
        print(f"   -> 整体模型压缩率: {(1 - new_params/orig_params)*100:.2f}%")
        
        return model


# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    import copy
    import timm
    
    print("="*60)
    print("🛠️  正在测试 SVD 低秩压缩算法 ...")
    
    # 构造一个极小的虚拟 ViT 模型用于测试
    print("📦 正在实例化测试模型 vit_tiny_patch16_224...")
    dummy_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=4)
    model_for_svd = copy.deepcopy(dummy_model)
    
    # 模拟输入以测试前向传播是否正常
    B, C, H, W = 2, 3, 224, 224
    dummy_input = torch.randn(B, C, H, W)
    
    with torch.no_grad():
        original_output = dummy_model(dummy_input)
    
    # 实例化压缩器
    # 能量阈值 0.95: 意味着放弃 5% 的奇异值能量，换取参数的大幅下降
    compressor = SVDLowRankCompressor(energy_threshold=0.95, min_compression_ratio=0.85)
    
    # 执行全模型压缩
    compressed_model = compressor.compress_model(model_for_svd)
    
    # 测试压缩后的模型能否正常前向传播，且输出维度一致
    with torch.no_grad():
        compressed_output = compressed_model(dummy_input)
        
    print(f"\n✅ 前向传播测试通过!")
    print(f"   原始输出形状: {original_output.shape}")
    print(f"   压缩后输出形状: {compressed_output.shape}")
    
    # 计算均方误差 (MSE) 来直观感受 SVD 截断带来的绝对误差
    mse_error = torch.nn.functional.mse_loss(original_output, compressed_output).item()
    print(f"   SVD 引入的初始截断误差 (MSE): {mse_error:.6f}")
    print("   💡 提示: 截断后输出有少许偏差是正常的，这部分精度损失将由后续的微调 (Fine-tuning) 或蒸馏 (KD) 找回！")
    print("="*60)