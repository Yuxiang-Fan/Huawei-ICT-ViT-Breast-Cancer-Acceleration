"""
多示例学习注意力聚合器模块

本模块负责:
1. 提供多种多示例学习注意力池化策略，包括 ABMIL、Gated-ABMIL 以及多头 MIL。
2. 将切片级别的特征序列聚合成全切片级别的全局特征。
3. 结合分类头输出最终的乳腺癌四分类结果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ABMILAggregator(nn.Module):
    """
    基于注意力的多示例学习聚合器
    参考自 Ilse 等人于 2018 年提出的经典注意力池化机制。
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
        前向传播逻辑
        输入形状: Batch Size, Patch 数量, input_dim
        输出形状: 聚合特征, 注意力权重
        """
        # 计算每个切片的注意力得分
        A = self.attention(x)
        # 在 Patch 维度执行归一化
        A = F.softmax(A, dim=1) 
        
        # 执行加权求和聚合
        wsi_feature = torch.bmm(x.transpose(1, 2), A).squeeze(2)
        
        return wsi_feature, A


class GatedABMILAggregator(nn.Module):
    """
    门控注意力多示例学习聚合器
    通过双分支门控机制增强非线性表达，能够更精准地识别关键病灶切片并抑制背景噪声。
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
        前向传播逻辑
        """
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        
        # 门控机制：利用 Tanh 分支与 Sigmoid 分支的元素级乘法提取特征
        A = self.attention_weights(A_V * A_U)
        A = F.softmax(A, dim=1)
        
        # 聚合生成全局特征
        wsi_feature = torch.bmm(x.transpose(1, 2), A).squeeze(2)
        
        return wsi_feature, A


class MultiHeadMILAggregator(nn.Module):
    """
    多头注意力多示例学习聚合器
    支持从多个特征子空间并行聚合信息，例如同时关注细胞核异型性与间质反应。
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
        # 映射到多个独立的注意力头
        self.attention_weights = nn.Linear(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播逻辑
        输出聚合特征形状: Batch Size, num_heads 乘以 input_dim
        """
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        
        A = self.attention_weights(A_V * A_U)
        A = self.dropout(A)
        
        # 转置以适配批量矩阵乘法
        A = F.softmax(A, dim=1).transpose(1, 2)
        
        # 计算多头特征聚合
        multi_head_features = torch.bmm(A, x)
        
        # 拼接多个头的输出特征
        B, n_heads, dim = multi_head_features.shape
        wsi_feature = multi_head_features.view(B, n_heads * dim)
        
        return wsi_feature, A


class WSIClassifierHead(nn.Module):
    """
    全切片级别分类头
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
        输出分类结果的 Logits
        """
        return self.classifier(wsi_feature)


class EndToEndMILModel(nn.Module):
    """
    端到端多示例学习模型封装
    整合了特征增强之后的聚合逻辑与分类预测。
    """
    def __init__(self, agg_type: str = 'gated', input_dim: int = 512, num_classes: int = 4, num_heads: int = 4):
        super().__init__()
        self.agg_type = agg_type.lower()
        
        # 选择聚合算法
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
            raise ValueError(f"暂不支持的聚合器类型: {agg_type}")
            
        self.classifier = WSIClassifierHead(input_dim=clf_input_dim, num_classes=num_classes)

    def forward(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回包含预测结果、全局特征及注意力权重的字典
        """
        wsi_feature, attention_weights = self.aggregator(patch_features)
        logits = self.classifier(wsi_feature)
        
        return {
            "logits": logits,
            "wsi_feature": wsi_feature,
            "attention_weights": attention_weights
        }


if __name__ == "__main__":
    # 局部功能测试
    B, N, dim = 2, 3000, 512
    test_input = torch.randn(B, N, dim)
    
    # 测试门控聚合模型
    model = EndToEndMILModel(agg_type='gated', input_dim=512)
    output = model(test_input)
    
    print(f"Logits 形状: {output['logits'].shape}")
    print(f"注意力权重形状: {output['attention_weights'].shape}")
