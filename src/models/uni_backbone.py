"""
UNI 大模型特征提取底座 (UNI Foundation Model Backbone)

该模块负责:
1. 加载基于 Vision Transformer (ViT-Large) 架构的 UNI 病理预训练模型。
2. 提供灵活的层级冻结机制 (Layer Freezing)，支持按指定 Block 数量冻结权重。
3. 提取 WSI 局部切片 (Patch) 的高维特征表示 (通常为 1024 维)。
"""

import os
import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple


class UNIBackbone(nn.Module):
    """
    UNI (ViT-Large) 特征提取器。
    该模型将 224x224 的病理切片映射为 1024 维的特征向量。
    """
    
    def __init__(
        self, 
        weight_path: str = "path/to/your/weights/uni_pytorch_model.bin", # 外部预训练权重占位符
        freeze_blocks: int = 18, 
        embed_dim: int = 1024
    ):
        """
        初始化 UNI 特征提取底座。
        
        Args:
            weight_path (str): 本地 UNI 预训练权重文件的路径。
            freeze_blocks (int): 需要冻结的 Transformer Block 数量 (UNI 总共有 24 个 Block)。
                                默认冻结前 18 层，仅微调后 6 层，以防止过拟合病理少样本数据集并大幅节约显存。
            embed_dim (int): ViT-Large 的特征维度，UNI 默认为 1024。
        """
        super().__init__()
        self.weight_path = weight_path
        self.freeze_blocks = freeze_blocks
        self.embed_dim = embed_dim
        
        # 1. 构建 ViT-Large 基础网络结构
        # num_classes=0 表示去掉最后的分类头，直接输出 pool 后的特征 (CLS token)
        self.model = timm.create_model(
            "vit_large_patch16_224", 
            pretrained=False, 
            num_classes=0, 
            dynamic_img_size=False
        )
        
        # 2. 加载本地预训练权重
        self._load_pretrained_weights()
        
        # 3. 执行梯度冻结策略
        self._apply_freezing_strategy()


    def _load_pretrained_weights(self) -> None:
        """从本地路径加载预训练权重字典"""
        if not os.path.exists(self.weight_path):
            print(f"⚠️  警告: UNI 权重文件未找到 [{self.weight_path}]。")
            print("   模型将使用随机初始化权重，这仅供网络结构调试使用！")
            print("   实际训练前，请确保挂载了包含 UNI 权重的 OBS 目录。")
            return
            
        try:
            # 读取权重，适配不同的保存格式 (.pth 或 .bin)
            state_dict = torch.load(self.weight_path, map_location="cpu")
            
            # 兼容有些权重字典被封装在 'model' 或 'state_dict' 键下的情况
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # strict=False 允许忽略分类头部分的缺失
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            print(f"✅ 成功加载 UNI 预训练权重: {self.weight_path}")
            if unexpected_keys:
                print(f"   [提示] 未匹配的参数 (通常是原始分类头): {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"❌ 加载 UNI 权重失败，原因: {e}")


    def _apply_freezing_strategy(self) -> None:
        """
        应用层级冻结策略。
        冻结底层的 Patch Embedding、位置编码以及前 N 个 Transformer Blocks。
        """
        if self.freeze_blocks <= 0:
            print("🔓 所有层已解冻 (全参微调模式 Full Fine-tuning)。")
            return
            
        total_blocks = len(self.model.blocks)
        if self.freeze_blocks > total_blocks:
            raise ValueError(f"要求冻结的 block 数量 ({self.freeze_blocks}) 超过了模型总 block 数 ({total_blocks})")

        # 1. 冻结 Patch Embedding 层
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
            
        # 2. 冻结 CLS Token 和 Position Embedding (如果是参数)
        if hasattr(self.model, 'cls_token') and self.model.cls_token is not None:
            self.model.cls_token.requires_grad = False
        if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
            self.model.pos_embed.requires_grad = False

        # 3. 冻结指定数量的 Transformer Blocks
        for i in range(self.freeze_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = False
                
        print(f"🔒 冻结策略已激活: 已冻结 Embedding 层及前 {self.freeze_blocks}/{total_blocks} 个 Transformer Blocks。")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播提取特征。
        
        Args:
            x (torch.Tensor): 输入的病理切片张量，形状为 (B, 3, 224, 224)。
            
        Returns:
            torch.Tensor: 提取的 1024 维特征向量，形状为 (B, 1024)。
        """
        # timm 的 ViT 设置 num_classes=0 时，forward 会自动返回 CLS token (或全局池化特征)
        features = self.model(x)
        return features



# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    import time
    
    # 模拟外部输入 (Batch Size = 8, 3 Channels, 224x224 Resolution)
    dummy_input = torch.randn(8, 3, 224, 224)
    
    print("="*60)
    print("🛠️  正在实例化 UNI Backbone ...")
    
    # 实例化模型 (使用默认的权重占位符，会触发警告但允许通过)
    backbone = UNIBackbone(
        weight_path="dummy/path/to/uni.bin", # 占位路径
        freeze_blocks=18
    )
    
    # 将模型放置于可用的加速硬件上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 如果环境是昇腾NPU (torch_npu)，可以使用 device="npu"
    # device = torch.device("npu" if hasattr(torch, 'npu') and torch.npu.is_available() else device)
    
    backbone = backbone.to(device)
    dummy_input = dummy_input.to(device)
    
    print("\n🚀 开始前向传播测试 ...")
    backbone.eval()
    
    start_time = time.time()
    with torch.no_grad():
        output_features = backbone(dummy_input)
    end_time = time.time()
    
    print(f"\n✅ 前向传播成功!")
    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output_features.shape} (预期为 [8, 1024])")
    print(f"   推理耗时: {(end_time - start_time) * 1000:.2f} ms")
    
    # 验证梯度冻结状态
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    
    print(f"\n📊 参数统计:")
    print(f"   总参数量: {total_params / 1e6:.2f} M")
    print(f"   已冻结参数: {frozen_params / 1e6:.2f} M ({(frozen_params/total_params)*100:.1f}%)")
    print(f"   可训练参数: {trainable_params / 1e6:.2f} M ({(trainable_params/total_params)*100:.1f}%)")
    print("="*60)