import os
import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple

class UNIBackbone(nn.Module):
    """
    UNI 特征提取底座
    采用 ViT-Large 架构，将 224x224 的病理 Patch 映射为 1024 维特征向量。
    """
    
    def __init__(
        self, 
        weight_path: str = "weights/uni_pytorch_model.bin",
        freeze_blocks: int = 18, 
        embed_dim: int = 1024
    ):
        """
        初始化特征提取器。
        freeze_blocks: 需要冻结的 Block 数量。默认冻结前 18 层，仅训练后 6 层，
                       这能有效防止在 BACH 等小规模病理数据集上产生过拟合。
        """
        super().__init__()
        self.weight_path = weight_path
        self.freeze_blocks = freeze_blocks
        self.embed_dim = embed_dim
        
        # 构建基础模型结构，去掉最后的分类层，直接输出 CLS 向量
        self.model = timm.create_model(
            "vit_large_patch16_224", 
            pretrained=False, 
            num_classes=0, 
            dynamic_img_size=False
        )
        
        # 加载本地预训练参数
        self._load_pretrained_weights()
        
        # 执行权重冻结策略
        self._apply_freezing_strategy()

    def _load_pretrained_weights(self) -> None:
        """从本地磁盘加载权重"""
        if not os.path.exists(self.weight_path):
            print(f"警告: 未找到权重文件 {self.weight_path}，当前使用随机初始化权重进行调试。")
            return
            
        try:
            state_dict = torch.load(self.weight_path, map_location="cpu")
            
            # 处理不同格式的权重字典映射
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # 加载参数，忽略不匹配的分类头部分
            self.model.load_state_dict(state_dict, strict=False)
            print(f"成功加载预训练权重: {self.weight_path}")
                
        except Exception as e:
            raise RuntimeError(f"模型参数加载失败: {e}")

    def _apply_freezing_strategy(self) -> None:
        """
        实施层级冻结策略，锁定底层特征，仅允许深层特征进行微调。
        """
        if self.freeze_blocks <= 0:
            print("当前模式: 全参数微调")
            return
            
        # 1. 锁定基础特征层
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
            
        # 2. 锁定位置编码
        if hasattr(self.model, 'cls_token') and self.model.cls_token is not None:
            self.model.cls_token.requires_grad = False
        if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
            self.model.pos_embed.requires_grad = False

        # 3. 锁定前 N 个 Block
        for i in range(self.freeze_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = False
                
        print(f"权重冻结完成: 已锁定前 {self.freeze_blocks} 个 Block。")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征。
        输入形状: Batch Size, 3, 224, 224
        输出形状: Batch Size, 1024
        """
        return self.model(x)

if __name__ == "__main__":
    # 针对乳腺癌分类任务的模拟测试
    dummy_input = torch.randn(8, 3, 224, 224)
    backbone = UNIBackbone(weight_path="dummy_path.bin", freeze_blocks=18)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    
    backbone.eval()
    with torch.no_grad():
        features = backbone(dummy_input.to(device))
    
    print(f"特征提取测试成功。输入: {dummy_input.shape}, 输出特征维度: {features.shape}")
