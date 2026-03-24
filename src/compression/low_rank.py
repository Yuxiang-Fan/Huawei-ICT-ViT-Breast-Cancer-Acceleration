import torch
import torch.nn as nn
from typing import Tuple, List

class SVDLowRankCompressor:
    """
    通过 SVD 对模型进行低秩压缩
    """
    def __init__(self, energy_threshold: float = 0.95, min_compression_ratio: float = 0.8):
        """
        初始化压缩参数
        energy_threshold: 奇异值能量保留比例，决定截断位置
        min_compression_ratio: 压缩收益阈值，低于该比例才执行替换
        """
        self.energy_threshold = energy_threshold
        self.min_compression_ratio = min_compression_ratio

    @torch.no_grad()
    def _decompose_linear_layer(self, layer: nn.Linear, layer_name: str) -> nn.Module:
        """
        对单个全连接层进行 SVD 分解与重构
        """
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        out_features, in_features = weight.shape
        
        # 转换到 CPU 处理以确保 SVD 算子的兼容性
        weight_cpu = weight.cpu().float()
        
        # 经济型 SVD 分解
        U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
        
        # 根据能量占比确定保留的秩 K
        squared_s = S ** 2
        total_energy = torch.sum(squared_s)
        cumulative_energy = torch.cumsum(squared_s, dim=0)
        
        # 寻找满足能量阈值的最小 K 值
        k = torch.searchsorted(cumulative_energy, total_energy * self.energy_threshold).item() + 1
        k = min(k, min(out_features, in_features))
        
        # 评估压缩收益
        original_params = out_features * in_features
        compressed_params = k * (out_features + in_features)
        compression_ratio = compressed_params / original_params
        
        # 若压缩效果不明显或 K 值过小则跳过
        if compression_ratio > self.min_compression_ratio or k < 2:
            return layer
            
        print(f"层 {layer_name}: 秩 K={k}, 参数量 {original_params} -> {compressed_params}, 压缩率 {compression_ratio:.2%}")
        
        # 截断矩阵并重构
        W1 = torch.diag(S[:k]) @ Vh[:k, :]
        W2 = U[:, :k]
        
        # 构建两个串联的小型线性层
        layer1 = nn.Linear(in_features, k, bias=False)
        layer1.weight.data = W1.to(weight.device).to(weight.dtype)
        
        layer2 = nn.Linear(k, out_features, bias=bias is not None)
        layer2.weight.data = W2.to(weight.device).to(weight.dtype)
        if bias is not None:
            layer2.bias.data = bias.to(weight.device).to(weight.dtype)
            
        return nn.Sequential(layer1, layer2)

    def compress_model(self, model: nn.Module) -> nn.Module:
        """
        递归遍历模型并替换符合条件的全连接层
        """
        orig_params = sum(p.numel() for p in model.parameters())
        replaced_count = 0
        
        def recursive_replace(module: nn.Module, prefix: str = ""):
            nonlocal replaced_count
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear):
                    new_child = self._decompose_linear_layer(child, full_name)
                    if new_child is not child:
                        setattr(module, name, new_child)
                        replaced_count += 1
                else:
                    recursive_replace(child, full_name)
                    
        recursive_replace(model)
        
        new_params = sum(p.numel() for p in model.parameters())
        print(f"压缩完成: 替换层数 {replaced_count}, 总参数量 {orig_params/1e6:.2f}M -> {new_params/1e6:.2f}M")
        
        return model

if __name__ == "__main__":
    import timm
    
    # 针对乳腺癌四分类任务的测试
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=4)
    
    compressor = SVDLowRankCompressor(energy_threshold=0.95, min_compression_ratio=0.85)
    compressed_model = compressor.compress_model(model)
    
    # 验证前向传播
    x = torch.randn(1, 3, 224, 224)
    y = compressed_model(x)
    print(f"输出形状验证: {y.shape}")
