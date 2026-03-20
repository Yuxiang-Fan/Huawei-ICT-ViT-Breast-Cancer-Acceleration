"""
知识蒸馏算法库 (Knowledge Distillation Library)

该模块负责:
1. 传统知识蒸馏 (Traditional KD): 基于软标签的 KL 散度与硬标签的交叉熵混合损失。
2. 解耦知识蒸馏 (Decoupled KD, DKD): 将蒸馏解耦为目标类 (TCKD) 和非目标类 (NCKD)，突破传统 KD 的瓶颈。
3. 蒸馏包装器 (DistillationWrapper): 简化 Teacher 和 Student 模型的联合前向传播与梯度管理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class KDLoss(nn.Module):
    """
    传统知识蒸馏损失函数 (Traditional Knowledge Distillation Loss)
    参考论文: Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
    """
    def __init__(self, T: float = 4.0, alpha: float = 0.5):
        """
        Args:
            T (float): 温度系数 (Temperature)。T 越大，软标签的分布越平滑。
            alpha (float): 软损失 (KL散度) 的权重。硬损失 (CE) 的权重为 1 - alpha。
        """
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: 学生模型输出的 Logits，形状 (B, C)
            teacher_logits: 教师模型输出的 Logits (必须不需要梯度)，形状 (B, C)
            targets: 真实的硬标签 (Ground Truth)，形状 (B,)
            
        Returns:
            torch.Tensor: 混合损失标量
        """
        # 1. 计算硬标签的交叉熵损失 (Hard Loss)
        hard_loss = self.cross_entropy(student_logits, targets)
        
        # 2. 计算软标签的 KL 散度损失 (Soft Loss)
        # 注意: PyTorch 的 kl_div 期望 input 是 log-probabilities，target 是 probabilities
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
        
        # 乘以 T^2 是为了在反向传播时与硬损失的梯度量级保持一致
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.T * self.T)
        
        # 3. 加权组合
        loss = (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss
        return loss


class DKDLoss(nn.Module):
    """
    解耦知识蒸馏损失函数 (Decoupled Knowledge Distillation Loss)
    参考论文: Decoupled Knowledge Distillation (Zhao et al., CVPR 2022)
    
    原理: 传统 KD 压制了非目标类 (Non-target classes) 之间的关系。
    DKD 将蒸馏分为:
    - TCKD (Target Class KD): 目标类与所有非目标类集合的二分类蒸馏。
    - NCKD (Non-Target Class KD): 仅在非目标类内部进行的多分类蒸馏。
    """
    def __init__(self, T: float = 4.0, alpha: float = 1.0, beta: float = 8.0):
        """
        Args:
            T (float): 温度系数。
            alpha (float): TCKD (目标类知识) 的权重。
            beta (float): NCKD (非目标类知识) 的权重。在病理等细粒度分类中，适当调高 beta 有助于学习类间相似度。
        """
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss()

    def _get_gt_mask(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """生成 Ground Truth 的 One-hot 掩码，确保张量设备对齐"""
        mask = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """生成非 Ground Truth 的掩码"""
        mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0).bool()
        return mask

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 DKD 损失。
        """
        # 计算基础硬损失
        hard_loss = self.cross_entropy(student_logits, targets)

        # 获取目标掩码
        gt_mask = self._get_gt_mask(student_logits, targets)
        other_mask = self._get_other_mask(student_logits, targets)

        # ================== 1. 计算 TCKD (Target Class KD) ==================
        # 将预测解耦为二元分布: [目标类概率, 非目标类概率总和]
        student_probs = F.softmax(student_logits / self.T, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1)

        student_pt = student_probs[gt_mask]
        student_pnt = student_probs[other_mask].view(student_probs.size(0), -1).sum(dim=1)
        student_binary_probs = torch.stack([student_pt, student_pnt], dim=1)

        teacher_pt = teacher_probs[gt_mask]
        teacher_pnt = teacher_probs[other_mask].view(teacher_probs.size(0), -1).sum(dim=1)
        teacher_binary_probs = torch.stack([teacher_pt, teacher_pnt], dim=1)

        tckd_loss = F.kl_div(
            torch.log(student_binary_probs + 1e-8), 
            teacher_binary_probs, 
            reduction='batchmean'
        ) * (self.T ** 2)

        # ================== 2. 计算 NCKD (Non-Target Class KD) ==================
        # 仅在非目标类上计算软标签蒸馏
        # 为了防止 softmax 时目标类的干扰，将目标类的 logit 设为一个极小值 (-1000.0)
        student_logits_nontarget = student_logits.masked_fill(gt_mask, -1000.0)
        teacher_logits_nontarget = teacher_logits.masked_fill(gt_mask, -1000.0)

        student_nontarget_log_probs = F.log_softmax(student_logits_nontarget / self.T, dim=1)
        teacher_nontarget_probs = F.softmax(teacher_logits_nontarget / self.T, dim=1)

        nckd_loss = F.kl_div(
            student_nontarget_log_probs, 
            teacher_nontarget_probs, 
            reduction='batchmean'
        ) * (self.T ** 2)

        # 3. 组合损失
        dkd_loss = self.alpha * tckd_loss + self.beta * nckd_loss
        total_loss = hard_loss + dkd_loss
        
        return total_loss


class DistillationWrapper(nn.Module):
    """
    模型蒸馏包装器。
    用于在训练循环中同时管理 Teacher 和 Student 模型，确保 Teacher 模型安全冻结不参与反向传播。
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # 彻底冻结教师模型的梯度
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """重写 train 方法，确保 Teacher 永远处于 eval 模式"""
        self.student.train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时执行学生和教师的前向传播。
        
        Args:
            x (torch.Tensor): 输入数据
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (学生 logits, 教师 logits)
        """
        # 教师前向传播 (强制无梯度)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
            # 适配字典输出形式 (如 MIL 模型)
            if isinstance(teacher_logits, dict):
                teacher_logits = teacher_logits['logits']
                
        # 学生前向传播
        student_logits = self.student(x)
        if isinstance(student_logits, dict):
            student_logits = student_logits['logits']
            
        return student_logits, teacher_logits


# =====================================================================
# 独立测试入口 (Local Validation)
# =====================================================================

if __name__ == "__main__":
    print("="*60)
    print("🛠️  正在测试知识蒸馏算法库 (KD & DKD) ...")
    
    # 模拟输入参数 (Batch Size = 8, 4 分类任务)
    B, Num_Classes = 8, 4
    
    # 模拟学生和教师输出的 Logits 以及真实的标签
    dummy_student_logits = torch.randn(B, Num_Classes, requires_grad=True)
    dummy_teacher_logits = torch.randn(B, Num_Classes) # 教师模型输出通常不需要梯度
    dummy_targets = torch.randint(0, Num_Classes, (B,))
    
    print(f"📦 模拟输入:")
    print(f"   Student Logits 形状: {dummy_student_logits.shape}")
    print(f"   Teacher Logits 形状: {dummy_teacher_logits.shape}")
    print(f"   Targets 形状: {dummy_targets.shape}")
    
    # 1. 测试传统 KD
    print("\n▶️ [1] 测试传统知识蒸馏 (KD Loss) ...")
    kd_criterion = KDLoss(T=4.0, alpha=0.5)
    kd_loss = kd_criterion(dummy_student_logits, dummy_teacher_logits, dummy_targets)
    
    print(f"   ✅ KD Loss 计算成功! 值: {kd_loss.item():.4f}")
    
    # 测试反向传播
    kd_loss.backward(retain_graph=True)
    print(f"   ✅ KD 反向传播梯度已生成。")
    
    # 2. 测试解耦知识蒸馏 DKD
    print("\n▶️ [2] 测试解耦知识蒸馏 (DKD Loss) ...")
    # 清空之前的梯度
    dummy_student_logits.grad = None
    
    dkd_criterion = DKDLoss(T=4.0, alpha=1.0, beta=8.0)
    dkd_loss = dkd_criterion(dummy_student_logits, dummy_teacher_logits, dummy_targets)
    
    print(f"   ✅ DKD Loss 计算成功! 值: {dkd_loss.item():.4f}")
    
    # 测试反向传播
    dkd_loss.backward()
    print(f"   ✅ DKD 反向传播梯度已生成。")
    
    # 测试 NPU/GPU 兼容性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ 设备迁移测试准备完毕 (目标设备: {device})")
    
    dkd_criterion.to(device)
    dummy_s_dev = dummy_student_logits.detach().to(device).requires_grad_(True)
    dummy_t_dev = dummy_teacher_logits.to(device)
    dummy_tg_dev = dummy_targets.to(device)
    
    loss_dev = dkd_criterion(dummy_s_dev, dummy_t_dev, dummy_tg_dev)
    print(f"✅ NPU/GPU 上的 DKD Loss 值: {loss_dev.item():.4f}")
    print("="*60)