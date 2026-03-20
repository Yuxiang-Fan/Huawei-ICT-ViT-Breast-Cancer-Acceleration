"""
模型剪枝算法库 (Model Pruning Algorithms)

该模块负责:
1. 结构化剪枝 (Structured Pruning): 面向 NPU 极度友好的 Transformer Block 直接截断策略。
2. 非结构化剪枝 (Unstructured Pruning): 基于 L1 幅度 (Magnitude) 或全局 K-th Value 的细粒度权重置零。
3. 提供统一的 PruningManager 接口，方便在消融实验中快速切换策略。
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Dict, Any, Optional
import copy


class StructuredBlockPruner:
    """
    结构化 Transformer Block 截断剪枝器。
    
    原理：由于深度 ViT 模型的深层特征往往存在过度平滑 (Over-smoothing) 或冗余，
    直接删除网络后部的若干 Transformer Block 是硬件加速最直接、最有效的方法。
    这种方法不需要 NPU 的稀疏算子支持，能直接带来线性的 FPS 提升。
    """
    
    def __init__(self, keep_ratio: float = 0.75, total_blocks: int = 24):
        """
        Args:
            keep_ratio (float): 保留 Block 的比例 (例如 0.75 意味着保留 18/24 个 blocks)。
            total_blocks (int): 原始模型总 Block 数。
        """
        self.keep_ratio = keep_ratio
        self.total_blocks = total_blocks
        self.keep_blocks = int(total_blocks * keep_ratio)
        
    def prune(self, model: nn.Module) -> nn.Module:
        """
        对传入的 UNI Backbone (timm ViT 架构) 执行 Block 截断。
        
        Args:
            model (nn.Module): 包含了 `.blocks` 属性的 ViT 模型实例。
            
        Returns:
            nn.Module: 截断后的新模型。
        """
        print(f"\n✂️  [结构化剪枝] 执行 Transformer Block 截断...")
        print(f"   -> 原始 Block 数: {self.total_blocks}, 目标保留数: {self.keep_blocks}")
        
        # 确保我们操作的是内部的 timm 模型结构 (如果外面包了一层的话)
        target_model = model.model if hasattr(model, 'model') else model
        
        if not hasattr(target_model, 'blocks'):
            raise AttributeError("❌ 无法执行结构化剪枝：未在模型中找到 'blocks' 属性，请确认模型是否为标准的 timm ViT 结构。")
            
        current_blocks = len(target_model.blocks)
        if self.keep_blocks >= current_blocks:
            print("   -> ⚠️ 目标保留层数大于等于当前层数，取消截断。")
            return model
            
        # 直接使用切片截断并重新赋值为 nn.Sequential
        # 抛弃最后的 (current_blocks - keep_blocks) 个 Transformer Block
        target_model.blocks = nn.Sequential(*list(target_model.blocks.children())[:self.keep_blocks])
        
        print(f"   -> ✅ 截断完成! 当前模型 Block 数量: {len(target_model.blocks)}")
        return model


class UnstructuredMagnitudePruner:
    """
    非结构化幅度剪枝器 (Unstructured Magnitude Pruning / K-th Value Thresholding)。
    
    原理：遍历模型中所有全连接层 (Linear) 和卷积层 (Conv2d)，统计其权重分布的直方图，
    将绝对值 (L1 norm) 最小的末尾 K% 的权重强制置为 0。
    """
    
    def __init__(self, pruning_ratio: float = 0.3, method: str = 'l1'):
        """
        Args:
            pruning_ratio (float): 需要置 0 的权重比例 (例如 0.3 表示裁剪掉 30% 绝对值最小的权重)。
            method (str): 剪枝策略。目前默认使用 'l1' (基于 L1 norm 的全局或局部阈值)。
        """
        self.pruning_ratio = pruning_ratio
        self.method = method
        self.parameters_to_prune = []
        
    def _collect_target_layers(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """递归收集模型中所有适合进行非结构化剪枝的层 (Linear 和 Conv2d)"""
        params_to_prune = []
        for name, module in model.named_modules():
            # 通常我们只对 Linear 和 Conv 层的 weight 进行剪枝，不剪枝 bias 和 LayerNorm
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                params_to_prune.append((module, 'weight'))
        return params_to_prune

    def prune_global(self, model: nn.Module) -> nn.Module:
        """
        执行全局 K-th Value 非结构化剪枝。
        计算整个模型所有目标权重的阈值，而不是按单层计算，这能更好地保留重要层的特征。
        """
        print(f"\n✂️  [非结构化剪枝] 执行全局 L1 剪枝 (稀疏度 {self.pruning_ratio*100:.1f}%)...")
        
        self.parameters_to_prune = self._collect_target_layers(model)
        
        if not self.parameters_to_prune:
            print("   -> ⚠️ 未找到可剪枝的 Linear/Conv2d 层。")
            return model
            
        # 使用 PyTorch 官方的全局剪枝 API
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_ratio,
        )
        
        self._print_sparsity(model)
        return model
        
    def prune_local(self, model: nn.Module) -> nn.Module:
        """
        执行局部 (Layer-wise) 直方图/幅度剪枝。
        对每个单独的 Linear 层严格裁剪掉指定比例的权重。
        """
        print(f"\n✂️  [非结构化剪枝] 执行局部 L1 剪枝 (每层稀疏度 {self.pruning_ratio*100:.1f}%)...")
        
        self.parameters_to_prune = self._collect_target_layers(model)
        
        for module, name in self.parameters_to_prune:
            prune.l1_unstructured(module, name=name, amount=self.pruning_ratio)
            
        self._print_sparsity(model)
        return model

    def make_permanent(self, model: nn.Module) -> nn.Module:
        """
        使剪枝永久化。
        PyTorch 的剪枝会在模块中添加 `weight_orig` 和 `weight_mask` 缓冲区。
        在导出 ONNX 或进行最终推理前，必须调用此方法将掩码固化到真实的 `weight` 中并移除缓冲区。
        """
        if not self.parameters_to_prune:
            self.parameters_to_prune = self._collect_target_layers(model)
            
        print("\n🔒 [非结构化剪枝] 正在将稀疏掩码 (Masks) 永久固化到模型权重中...")
        for module, name in self.parameters_to_prune:
            try:
                prune.remove(module, name)
            except ValueError:
                # 忽略尚未应用剪枝的层
                pass
        print("   -> ✅ 固化完成，模型准备好进行量化或导出。")
        return model

    def _print_sparsity(self, model: nn.Module):
        """打印模型当前的整体稀疏度统计信息"""
        total_zeros = 0
        total_elements = 0
        for module, name in self.parameters_to_prune:
            weight = getattr(module, name)
            total_zeros += int(torch.sum(weight == 0))
            total_elements += weight.nelement()
            
        sparsity = 100. * total_zeros / total_elements if total_elements > 0 else 0
        print(f"   -> ✅ 剪枝完成! 目标层全局稀疏度: {sparsity:.2f}% ({total_zeros}/{total_elements} 权重已置零)")


class PruningManager:
    """
    模型剪枝流水线管理器。
    提供统一入口，支持单独或组合调用结构化与非结构化剪枝策略。
    """
    def __init__(self, 
                 use_structured: bool = True, struct_keep_ratio: float = 0.75, total_blocks: int = 24,
                 use_unstructured: bool = False, unstruct_ratio: float = 0.3, unstruct_global: bool = True):
        
        self.use_structured = use_structured
        self.use_unstructured = use_unstructured
        
        if self.use_structured:
            self.struct_pruner = StructuredBlockPruner(keep_ratio=struct_keep_ratio, total_blocks=total_blocks)
            
        if self.use_unstructured:
            self.unstruct_pruner = UnstructuredMagnitudePruner(pruning_ratio=unstruct_ratio)
            self.unstruct_global = unstruct_global

    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """执行配置好的流水线剪枝"""
        # 为了安全，通常不对原模型进行原地破坏操作，可以根据需求取消 deepcopy
        # model = copy.deepcopy(model)
        
        if self.use_structured:
            model = self.struct_pruner.prune(model)
            
        if self.use_unstructured:
            if self.unstruct_global:
                model = self.unstruct_pruner.prune_global(model)
            else:
                model = self.unstruct_pruner.prune_local(model)
                
        return model
        
    def finalize_unstructured_masks(self, model: nn.Module) -> nn.Module:
        """在蒸馏或微调结束后，移除非结构化剪枝的钩子，固化权重"""
        if self.use_unstructured:
            model = self.unstruct_pruner.make_permanent(model)
        return model


# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    import timm
    
    print("="*60)
    print("🛠️  正在测试剪枝算法库 ...")
    
    # 构造一个极小的虚拟 ViT 模型用于测试 (避免下载巨大的权重)
    # vit_tiny 有 12 个 block
    print("📦 正在实例化测试模型 vit_tiny_patch16_224...")
    dummy_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    
    # 获取原始参数量
    orig_params = sum(p.numel() for p in dummy_model.parameters())
    print(f"   原始模型总参数量: {orig_params / 1e6:.2f} M")
    
    # ---------------------------------------------------------
    # 测试 1: 结构化剪枝 (保留前 50% 的 Block)
    # ---------------------------------------------------------
    print("\n▶️ 测试分支 1: 结构化剪枝")
    struct_manager = PruningManager(use_structured=True, struct_keep_ratio=0.5, total_blocks=12, use_unstructured=False)
    pruned_model_1 = struct_manager.apply_pruning(copy.deepcopy(dummy_model))
    
    struct_params = sum(p.numel() for p in pruned_model_1.parameters())
    print(f"   结构化剪枝后参数量: {struct_params / 1e6:.2f} M (压缩率: {(1 - struct_params/orig_params)*100:.1f}%)")
    
    # ---------------------------------------------------------
    # 测试 2: 非结构化全局幅度剪枝 (置零 40% 的权重)
    # ---------------------------------------------------------
    print("\n▶️ 测试分支 2: 非结构化全局剪枝")
    unstruct_manager = PruningManager(use_structured=False, use_unstructured=True, unstruct_ratio=0.4, unstruct_global=True)
    pruned_model_2 = unstruct_manager.apply_pruning(copy.deepcopy(dummy_model))
    
    # 固化权重
    pruned_model_2 = unstruct_manager.finalize_unstructured_masks(pruned_model_2)
    print("="*60)