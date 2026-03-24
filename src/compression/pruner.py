"""
模型剪枝模块

包含针对 ViT 架构的 Block 截断（结构化剪枝）
以及基于 L1 范数的权重置零（非结构化剪枝）。
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple
import copy


class StructuredBlockPruner:
    """
    Block 截断剪枝器
    通过直接移除 ViT 模型深层的 Block 来实现硬件友好的物理加速。
    """
    def __init__(self, keep_ratio: float = 0.75, total_blocks: int = 24):
        self.keep_ratio = keep_ratio
        self.total_blocks = total_blocks
        self.keep_blocks = int(total_blocks * keep_ratio)
        
    def prune(self, model: nn.Module) -> nn.Module:
        print(f"[结构化剪枝] 原始 Block 数: {self.total_blocks}, 目标保留数: {self.keep_blocks}")
        
        target_model = model.model if hasattr(model, 'model') else model
        
        if not hasattr(target_model, 'blocks'):
            raise AttributeError("未在模型中找到 blocks 属性，请确认模型是否为标准 timm ViT 结构。")
            
        current_blocks = len(target_model.blocks)
        if self.keep_blocks >= current_blocks:
            print("[结构化剪枝] 目标保留层数大于等于当前层数，跳过截断。")
            return model
            
        target_model.blocks = nn.Sequential(*list(target_model.blocks.children())[:self.keep_blocks])
        
        print(f"[结构化剪枝] 截断完成，当前 Block 数量: {len(target_model.blocks)}")
        return model


class UnstructuredMagnitudePruner:
    """
    非结构化幅度剪枝器
    基于权重的 L1 范数，将绝对值最小的部分权重置零。
    """
    def __init__(self, pruning_ratio: float = 0.3, method: str = 'l1'):
        self.pruning_ratio = pruning_ratio
        self.method = method
        self.parameters_to_prune = []
        
    def _collect_target_layers(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        params_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                params_to_prune.append((module, 'weight'))
        return params_to_prune

    def prune_global(self, model: nn.Module) -> nn.Module:
        print(f"[非结构化剪枝] 执行全局 L1 剪枝，目标稀疏度: {self.pruning_ratio:.2%}")
        
        self.parameters_to_prune = self._collect_target_layers(model)
        
        if not self.parameters_to_prune:
            print("[非结构化剪枝] 未找到可剪枝的 Linear 或 Conv2d 层。")
            return model
            
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_ratio,
        )
        
        self._print_sparsity(model)
        return model
        
    def prune_local(self, model: nn.Module) -> nn.Module:
        print(f"[非结构化剪枝] 执行局部 L1 剪枝，每层稀疏度: {self.pruning_ratio:.2%}")
        
        self.parameters_to_prune = self._collect_target_layers(model)
        
        for module, name in self.parameters_to_prune:
            prune.l1_unstructured(module, name=name, amount=self.pruning_ratio)
            
        self._print_sparsity(model)
        return model

    def make_permanent(self, model: nn.Module) -> nn.Module:
        if not self.parameters_to_prune:
            self.parameters_to_prune = self._collect_target_layers(model)
            
        print("[非结构化剪枝] 正在固化稀疏掩码...")
        for module, name in self.parameters_to_prune:
            try:
                prune.remove(module, name)
            except ValueError:
                pass
        print("[非结构化剪枝] 掩码固化完成。")
        return model

    def _print_sparsity(self, model: nn.Module):
        total_zeros = 0
        total_elements = 0
        for module, name in self.parameters_to_prune:
            weight = getattr(module, name)
            total_zeros += int(torch.sum(weight == 0))
            total_elements += weight.nelement()
            
        sparsity = total_zeros / total_elements if total_elements > 0 else 0
        print(f"[非结构化剪枝] 全局稀疏度: {sparsity:.2%} ({total_zeros}/{total_elements})")


class PruningManager:
    """
    剪枝流水线管理器
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
        if self.use_structured:
            model = self.struct_pruner.prune(model)
            
        if self.use_unstructured:
            if self.unstruct_global:
                model = self.unstruct_pruner.prune_global(model)
            else:
                model = self.unstruct_pruner.prune_local(model)
                
        return model
        
    def finalize_unstructured_masks(self, model: nn.Module) -> nn.Module:
        if self.use_unstructured:
            model = self.unstruct_pruner.make_permanent(model)
        return model


if __name__ == "__main__":
    import timm
    
    # 测试环境初始化
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=4)
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"原始模型参数量: {orig_params / 1e6:.2f}M")
    
    # 场景 1: 结构化剪枝
    struct_manager = PruningManager(use_structured=True, struct_keep_ratio=0.5, total_blocks=12, use_unstructured=False)
    pruned_model_1 = struct_manager.apply_pruning(copy.deepcopy(model))
    struct_params = sum(p.numel() for p in pruned_model_1.parameters())
    print(f"结构化剪枝后参数量: {struct_params / 1e6:.2f}M")
    
    # 场景 2: 非结构化全局剪枝
    unstruct_manager = PruningManager(use_structured=False, use_unstructured=True, unstruct_ratio=0.4, unstruct_global=True)
    pruned_model_2 = unstruct_manager.apply_pruning(copy.deepcopy(model))
    pruned_model_2 = unstruct_manager.finalize_unstructured_masks(pruned_model_2)
