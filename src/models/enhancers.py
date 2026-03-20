"""
病理特征增强与注意力精粹模块 (Feature Enhancers & Attention Refiners)

该模块位于 UNI Backbone 与 MIL 聚合器之间，负责:
1. 特征域适应与降维：将 1024 维通用特征映射到特定任务的高效子空间 (如 512 维)。
2. 门控信息过滤：通过门控机制抑制背景 (如脂肪、载玻片空白区) 噪声。
3. 先验上下文注入：利用可学习的组织学先验 (Histological Priors) 精炼切片特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedFeatureEnhancer(nn.Module):
    """
    基于门控调制的特征增强模块 (Gated Feature Enhancer)
    
    采用 Gated Linear Unit (GLU) 变体结构。一方面通过线性变换对特征进行降维压缩，
    另一方面通过 Sigmoid 门控分支自动学习不同通道的重要性，实现背景抑制和病灶特征增强。
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, dropout_rate: float = 0.25):
        """
        初始化特征增强模块。
        
        Args:
            input_dim (int): UNI 模型输出的特征维度 (默认 1024)。
            hidden_dim (int): 增强与降维后的特征维度 (默认 512，适配 NPU 计算优化)。
            dropout_rate (float): Dropout 概率，防止过拟合。
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征映射分支
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 门控调制分支
        self.gate_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        # 特征正则化
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            features (torch.Tensor): 从 Backbone 提取的特征，形状为 (B, N, input_dim) 
                                     或 (N, input_dim)，其中 B 为 Batch Size，N 为 Patch 数量。
                                     
        Returns:
            torch.Tensor: 门控调制后的特征，形状为 (B, N, hidden_dim) 或 (N, hidden_dim)。
        """
        # 计算特征表达
        h = self.feature_proj(features)
        
        # 计算门控权重 (0 ~ 1)
        g = self.gate_proj(features)
        
        # 门控调制 (Hadamard 乘积)
        enhanced_features = h * g
        
        return self.dropout(enhanced_features)


class PriorAttentionRefiner(nn.Module):
    """
    基于可学习先验的注意力精粹模块 (Prior-based Attention Refiner)
    
    在切片输入 MIL 聚合器之前，引入 N 个可学习的先验向量 (可以理解为代表良性、原位、浸润等
    组织学形态的 Prototype)。通过 Cross-Attention 机制，让每一个切片特征与这些全局先验进行交互，
    从而获得更丰富的上下文感知能力。
    """
    def __init__(self, feature_dim: int = 512, num_priors: int = 4, num_heads: int = 4):
        """
        初始化注意力精粹模块。
        
        Args:
            feature_dim (int): 输入特征维度 (通常是 GatedFeatureEnhancer 的输出维度 512)。
            num_priors (int): 可学习先验向量的数量 (对应任务的潜在组织形态数量，如四分类任务设为 4 或 8)。
            num_heads (int): 多头注意力的头数。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_priors = num_priors
        
        # 初始化可学习的先验向量 (Prototypes)
        # 形状: (1, num_priors, feature_dim) - 1 用于 Broadcasting 到不同的 Batch
        self.priors = nn.Parameter(torch.randn(1, num_priors, feature_dim))
        nn.init.trunc_normal_(self.priors, std=0.02)
        
        # 交叉注意力层 (Patch 特征作为 Query，先验向量作为 Key 和 Value)
        # 目的：让切片特征向先验形态对齐，提取共有特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            dropout=0.1, 
            batch_first=True
        )
        
        # 前馈网络与残差连接
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            features (torch.Tensor): 增强后的切片特征，形状必须为 (B, N, feature_dim)。
                                     如果是 (N, feature_dim)，需在外部扩充 B 维度。
                                     
        Returns:
            torch.Tensor: 精粹重组后的特征，形状为 (B, N, feature_dim)。
        """
        B, N, D = features.shape
        
        # 扩展 Priors 以匹配 Batch Size
        # priors_expanded 形状: (B, num_priors, feature_dim)
        priors_expanded = self.priors.expand(B, -1, -1)
        
        # Cross Attention: 
        # Query = Patch Features, Key/Value = Learnable Priors
        # 这一步计算每个 Patch 对应多少先验组织学信息，并将其注入回 Patch
        attn_out, _ = self.cross_attn(
            query=features,
            key=priors_expanded,
            value=priors_expanded
        )
        
        # 第一个残差连接 + LayerNorm
        features = self.norm1(features + attn_out)
        
        # FFN + 第二个残差连接 + LayerNorm
        ffn_out = self.ffn(features)
        refined_features = self.norm2(features + ffn_out)
        
        return refined_features


# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    print("="*60)
    print("🛠️  正在测试特征增强与精粹模块 ...")
    
    # 模拟 Backbone 输出的特征 (Batch Size = 2, 1000 个 Patches, 1024 维)
    B, N, Input_Dim = 2, 1000, 1024
    dummy_uni_features = torch.randn(B, N, Input_Dim)
    
    print(f"📦 模拟输入特征形状: {dummy_uni_features.shape}")
    
    # 1. 测试门控特征增强模块
    enhancer = GatedFeatureEnhancer(input_dim=1024, hidden_dim=512)
    enhanced_features = enhancer(dummy_uni_features)
    
    print(f"\n✅ [1] GatedFeatureEnhancer 测试通过!")
    print(f"   输出形状: {enhanced_features.shape} (预期: [{B}, {N}, 512])")
    print(f"   参数量: {sum(p.numel() for p in enhancer.parameters()) / 1e6:.2f} M")
    
    # 2. 测试先验注意力精粹模块
    refiner = PriorAttentionRefiner(feature_dim=512, num_priors=4, num_heads=4)
    refined_features = refiner(enhanced_features)
    
    print(f"\n✅ [2] PriorAttentionRefiner 测试通过!")
    print(f"   输出形状: {refined_features.shape} (预期: [{B}, {N}, 512])")
    print(f"   参数量: {sum(p.numel() for p in refiner.parameters()) / 1e6:.2f} M")
    
    # 测试设备迁移 (确保对 NPU/GPU 友好)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("npu" if hasattr(torch, 'npu') and torch.npu.is_available() else device)
    
    enhancer = enhancer.to(device)
    refiner = refiner.to(device)
    dummy_tensor_device = dummy_uni_features.to(device)
    
    with torch.no_grad():
        out = refiner(enhancer(dummy_tensor_device))
        
    print(f"\n✅ 设备迁移测试通过! (当前设备: {device})")
    print("="*60)