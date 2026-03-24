"""
病理特征增强与注意力精粹模块

本模块主要负责:
1. 特征降维与域适应：将原始 Backbone 输出的高维通用特征映射到 512 维的任务特定空间。
2. 门控噪声过滤：利用门控机制抑制病理切片中的背景噪声，增强病灶区域信号。
3. 组织学先验注入：通过可学习的先验向量精炼 Patch 特征，提升全局上下文感知力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedFeatureEnhancer(nn.Module):
    """
    基于门控调制的特征增强模块
    
    借鉴 GLU 结构，通过线性变换进行维度压缩，并利用 Sigmoid 门控分支自动筛选特征通道，
    有效抑制脂肪或载玻片空白区的干扰，强化病变特征。
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, dropout_rate: float = 0.25):
        """
        参数说明:
        input_dim: 原始特征维度
        hidden_dim: 降维后的隐藏层维度
        dropout_rate: 丢弃率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 主特征映射分支
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 门控信号分支
        self.gate_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        输入形状: Batch Size, Patch 数量, input_dim
        """
        # 提取特征表达
        h = self.feature_proj(features)
        
        # 计算门控权重
        g = self.gate_proj(features)
        
        # 门控特征融合（Hadamard 乘积）
        enhanced_features = h * g
        
        return self.dropout(enhanced_features)


class PriorAttentionRefiner(nn.Module):
    """
    基于组织学先验的注意力精粹模块
    
    在切片输入 MIL 聚合器前，通过 Cross Attention 机制让每个 Patch 特征与一组可学习的
    组织学先验向量进行交互，利用先验知识辅助模型识别典型的病理形态特征。
    """
    def __init__(self, feature_dim: int = 512, num_priors: int = 4, num_heads: int = 4):
        """
        参数说明:
        feature_dim: 输入特征维度
        num_priors: 可学习先验向量的数量（通常对应任务的病理类别数）
        num_heads: 多头注意力的头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_priors = num_priors
        
        # 定义可学习的组织学先验向量
        self.priors = nn.Parameter(torch.randn(1, num_priors, feature_dim))
        nn.init.trunc_normal_(self.priors, std=0.02)
        
        # 交叉注意力层：Patch 特征作为 Query，先验向量作为 Key 和 Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            dropout=0.1, 
            batch_first=True
        )
        
        # 前馈网络与归一化结构
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
        前向传播
        输入形状: Batch Size, Patch 数量, feature_dim
        """
        B, N, D = features.shape
        
        # 扩展先验向量以匹配当前 Batch Size
        priors_expanded = self.priors.expand(B, -1, -1)
        
        # 执行 Cross Attention，将先验知识注入 Patch 特征
        attn_out, _ = self.cross_attn(
            query=features,
            key=priors_expanded,
            value=priors_expanded
        )
        
        # 残差连接与归一化
        features = self.norm1(features + attn_out)
        
        # 前馈网络进一步精粹特征
        refined_features = self.norm2(features + self.ffn(features))
        
        return refined_features


if __name__ == "__main__":
    # 局部测试逻辑
    B, N, Input_Dim = 2, 1000, 1024
    test_features = torch.randn(B, N, Input_Dim)
    
    # 测试特征增强
    enhancer = GatedFeatureEnhancer(input_dim=1024, hidden_dim=512)
    enhanced = enhancer(test_features)
    print(f"增强模块输出形状: {enhanced.shape}")
    
    # 测试注意力精炼
    refiner = PriorAttentionRefiner(feature_dim=512, num_priors=4)
    refined = refiner(enhanced)
    print(f"精炼模块输出形状: {refined.shape}")
