"""
多示例学习注意力聚合器模块 (MIL Aggregator)

该模块负责:
1. 提供多种 MIL 注意力池化策略 (ABMIL, Gated-ABMIL, Multi-Head MIL)。
2. 将切片级别 (Patch-level) 的特征序列聚合成全切片级别 (WSI-level) 的全局特征。
3. 结合分类头 (Classifier Head) 输出最终的乳腺癌四分类 Logits。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ABMILAggregator(nn.Module):
    """
    经典基于注意力的多示例学习聚合器 (Attention-Based MIL)
    参考论文: Attention-based Deep Multiple Instance Learning (Ilse et al., 2018)
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 切片特征序列，形状为 (B, N, input_dim)
        Returns:
            WSI_feature (torch.Tensor): 聚合后的全局特征，形状为 (B, input_dim)
            A (torch.Tensor): 注意力权重，形状为 (B, N, 1)
        """
        # 计算注意力分数: (B, N, input_dim) -> (B, N, 1)
        A = self.attention(x)
        # 沿 N 维度计算 Softmax
        A = F.softmax(A, dim=1) 
        
        # 加权求和: (B, input_dim)
        # x.transpose(1, 2) 形状为 (B, input_dim, N)
        # torch.bmm((B, input_dim, N), (B, N, 1)) -> (B, input_dim, 1) -> squeeze -> (B, input_dim)
        WSI_feature = torch.bmm(x.transpose(1, 2), A).squeeze(2)
        
        return WSI_feature, A


class GatedABMILAggregator(nn.Module):
    """
    门控注意力多示例学习聚合器 (Gated Attention-Based MIL)
    使用 Tanh 和 Sigmoid 门控机制，非线性表达能力更强，能更有效抑制载玻片背景噪声。
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 切片特征序列，形状为 (B, N, input_dim)
        """
        # (B, N, hidden_dim)
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        
        # 门控机制: Tanh 激活与 Sigmoid 激活的 Hadamard 乘积
        A = self.attention_weights(A_V * A_U)  # (B, N, 1)
        A = F.softmax(A, dim=1)                # (B, N, 1)
        
        # 聚合特征
        WSI_feature = torch.bmm(x.transpose(1, 2), A).squeeze(2)  # (B, input_dim)
        
        return WSI_feature, A


class MultiHeadMILAggregator(nn.Module):
    """
    多头注意力多示例学习聚合器 (Multi-Head MIL)
    允许模型从多个不同的特征子空间 (例如: 细胞核形态、腺管结构、间质反应) 并行聚合信息。
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.num_heads = num_heads
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        # 投射到 num_heads 个注意力分支
        self.attention_weights = nn.Linear(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 切片特征序列，形状为 (B, N, input_dim)
        Returns:
            WSI_feature (torch.Tensor): 聚合后的全局多头特征，形状为 (B, input_dim * num_heads)
            A (torch.Tensor): 注意力权重，形状为 (B, num_heads, N)
        """
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        
        # (B, N, num_heads)
        A = self.attention_weights(A_V * A_U)
        A = self.dropout(A)
        
        # 沿 N 维度计算 Softmax，并在计算 bmm 前转置为 (B, num_heads, N)
        A = F.softmax(A, dim=1).transpose(1, 2)
        
        # 聚合多头特征
        # torch.bmm((B, num_heads, N), (B, N, input_dim)) -> (B, num_heads, input_dim)
        multi_head_features = torch.bmm(A, x)
        
        # 展平多头特征: (B, num_heads * input_dim)
        B, num_heads, input_dim = multi_head_features.shape
        WSI_feature = multi_head_features.view(B, num_heads * input_dim)
        
        return WSI_feature, A


class WSIClassifierHead(nn.Module):
    """
    WSI 级别的最终分类头。将聚合后的全局特征映射为目标类别的 Logits。
    """
    def __init__(self, input_dim: int, num_classes: int = 4, dropout: float = 0.25):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, wsi_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wsi_feature (torch.Tensor): (B, input_dim)
        Returns:
            logits (torch.Tensor): (B, num_classes)
        """
        return self.classifier(wsi_feature)


class EndToEndMILModel(nn.Module):
    """
    组装好的端到端 MIL 模型，包含聚合器和分类头。
    此结构设计旨在方便推理时统一调用与导出 ONNX。
    """
    def __init__(self, agg_type: str = 'gated', input_dim: int = 512, num_classes: int = 4, num_heads: int = 4):
        """
        Args:
            agg_type (str): 'abmil', 'gated', 或 'multihead'
            input_dim (int): 输入特征维度 (对应 enhancers.py 的输出)
            num_classes (int): 分类数 (乳腺癌四分类任务为 4)
            num_heads (int): 仅在 agg_type 为 'multihead' 时生效
        """
        super().__init__()
        self.agg_type = agg_type.lower()
        
        # 1. 实例化对应的聚合器
        if self.agg_type == 'abmil':
            self.aggregator = ABMILAggregator(input_dim=input_dim)
            clf_input_dim = input_dim
        elif self.agg_type == 'gated':
            self.aggregator = GatedABMILAggregator(input_dim=input_dim)
            clf_input_dim = input_dim
        elif self.agg_type == 'multihead':
            self.aggregator = MultiHeadMILAggregator(input_dim=input_dim, num_heads=num_heads)
            clf_input_dim = input_dim * num_heads
        else:
            raise ValueError(f"不支持的聚合器类型: {agg_type}")
            
        # 2. 实例化分类头
        self.classifier = WSIClassifierHead(input_dim=clf_input_dim, num_classes=num_classes)

    def forward(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_features (torch.Tensor): (B, N, input_dim)
        Returns:
            Dict: 包含 'logits' (分类结果), 'wsi_feature' (全局特征), 'attention_weights' (注意力热力图权重)
        """
        wsi_feature, attention_weights = self.aggregator(patch_features)
        logits = self.classifier(wsi_feature)
        
        return {
            "logits": logits,
            "wsi_feature": wsi_feature,
            "attention_weights": attention_weights
        }


# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    print("="*60)
    print("🛠️  正在测试多示例学习聚合器 (MIL Aggregators) ...")
    
    # 模拟从 Enhancer 传来的特征张量 (Batch Size = 2, 3000 个 Patches, 512 维特征)
    B, N, Input_Dim = 2, 3000, 512
    dummy_patch_features = torch.randn(B, N, Input_Dim)
    
    print(f"📦 模拟输入特征形状: {dummy_patch_features.shape}")
    
    # 1. 测试门控注意力 MIL (Gated ABMIL) - 赛题最常用
    print("\n✅ [1] 测试 Gated ABMIL 端到端模型...")
    gated_model = EndToEndMILModel(agg_type='gated', input_dim=Input_Dim, num_classes=4)
    out_gated = gated_model(dummy_patch_features)
    
    print(f"   Logits 形状: {out_gated['logits'].shape} (预期: [{B}, 4])")
    print(f"   注意力权重形状: {out_gated['attention_weights'].shape} (预期: [{B}, {N}, 1])")
    
    # 2. 测试多头注意力 MIL (Multi-Head MIL)
    print("\n✅ [2] 测试 Multi-Head MIL 端到端模型 (4个头)...")
    multihead_model = EndToEndMILModel(agg_type='multihead', input_dim=Input_Dim, num_classes=4, num_heads=4)
    out_multihead = multihead_model(dummy_patch_features)
    
    print(f"   Logits 形状: {out_multihead['logits'].shape} (预期: [{B}, 4])")
    print(f"   多头聚合全局特征形状: {out_multihead['wsi_feature'].shape} (预期: [{B}, {Input_Dim * 4}])")
    print(f"   注意力权重形状: {out_multihead['attention_weights'].shape} (预期: [{B}, 4, {N}])")
    
    # 测试对 NPU/GPU 的兼容性 (Batch Matrix Multiplication 必须支持设备迁移)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gated_model = gated_model.to(device)
    dummy_tensor_device = dummy_patch_features.to(device)
    
    with torch.no_grad():
        _ = gated_model(dummy_tensor_device)
        
    print(f"\n✅ 设备迁移与前向传播测试通过! (当前设备: {device})")
    print("="*60)