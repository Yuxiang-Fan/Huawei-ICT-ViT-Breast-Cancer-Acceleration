"""
知识蒸馏模块
包含标准 KD 与解耦知识蒸馏 。
并提供 DistillationWrapper 用于简化 Teacher-Student 的联合前向传播。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KDLoss(nn.Module):
    """
    标准知识蒸馏损失
    """
    def __init__(self, T: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hard_loss = self.cross_entropy(student_logits, targets)
        
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
        
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.T ** 2)
        
        return (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss


class DKDLoss(nn.Module):
    """
    解耦知识蒸馏损失 
    将蒸馏解耦为目标类和非目标类的监督。
    """
    def __init__(self, T: float = 4.0, alpha: float = 1.0, beta: float = 8.0):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss()

    def _get_masks(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_mask = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1).bool()
        other_mask = ~gt_mask
        return gt_mask, other_mask

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hard_loss = self.cross_entropy(student_logits, targets)

        gt_mask, other_mask = self._get_masks(student_logits, targets)

        # 1. TCKD (Target Class KD)
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

        # 2. NCKD (Non-Target Class KD)
        # 屏蔽目标类的影响，仅在非目标类分布上计算 KL
        student_logits_nontarget = student_logits.masked_fill(gt_mask, -1e4)
        teacher_logits_nontarget = teacher_logits.masked_fill(gt_mask, -1e4)

        student_nontarget_log_probs = F.log_softmax(student_logits_nontarget / self.T, dim=1)
        teacher_nontarget_probs = F.softmax(teacher_logits_nontarget / self.T, dim=1)

        nckd_loss = F.kl_div(
            student_nontarget_log_probs, 
            teacher_nontarget_probs, 
            reduction='batchmean'
        ) * (self.T ** 2)

        return hard_loss + self.alpha * tckd_loss + self.beta * nckd_loss


class DistillationWrapper(nn.Module):
    """
    联合前向传播包装器，自动冻结Teacher模型以避免计算图冗余和意外更新。
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        self.student.train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_logits = self.teacher(x)
            if isinstance(teacher_logits, dict):
                teacher_logits = teacher_logits['logits']
                
        student_logits = self.student(x)
        if isinstance(student_logits, dict):
            student_logits = student_logits['logits']
            
        return student_logits, teacher_logits


if __name__ == "__main__":
    # 局部测试逻辑，配合BACH数据集的四分类任务
    torch.manual_seed(42)
    
    batch_size = 8
    num_classes = 4  
    
    student_out = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_out = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # KD 验证
    kd = KDLoss()
    loss_kd = kd(student_out, teacher_out, labels)
    loss_kd.backward(retain_graph=True)
    assert student_out.grad is not None, "KD backward failed"
    student_out.grad.zero_()
    
    # DKD 验证
    dkd = DKDLoss(alpha=1.0, beta=8.0)
    loss_dkd = dkd(student_out, teacher_out, labels)
    loss_dkd.backward()
    assert student_out.grad is not None, "DKD backward failed"
    
    print("All tests passed.")
