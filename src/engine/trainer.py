"""
模型训练与蒸馏驱动引擎

负责:
1. 封装标准的模型训练与验证循环。
2. 支持 AMP 自动混合精度，优化显存占用并加速训练。
3. 整合早停机制与最佳权重保存逻辑。
4. 支持常规微调与知识蒸馏两种训练模式。
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


class EarlyStopping:
    """
    早停机制与权重保存器
    当验证集指标在指定周期内没有提升时，自动停止训练。
    """
    def __init__(self, save_dir: str, patience: int = 10, mode: str = 'max', min_delta: float = 1e-4):
        self.save_dir = save_dir
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
        self.best_epoch = 0
        
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, current_score: float, epoch: int, model: nn.Module) -> None:
        if self.mode == 'max':
            is_better = current_score > self.best_score + self.min_delta
        else:
            is_better = current_score < self.best_score - self.min_delta

        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, is_best=True)
            print(f"检测到指标提升 (第 {epoch} 轮): 当前值 {current_score:.4f}，权重已保存")
        else:
            self.counter += 1
            print(f"指标未提升计数: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"触发早停，最佳模型位于第 {self.best_epoch} 轮")
                
        if epoch > 0 and epoch % 10 == 0:
            self._save_checkpoint(model, is_best=False, epoch=epoch)

    def _save_checkpoint(self, model: nn.Module, is_best: bool, epoch: int = 0) -> None:
        # 处理模型包装情况，确保提取原始参数
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model = raw_model.student if hasattr(raw_model, 'student') else raw_model
        
        state_dict = raw_model.state_dict()
        
        if is_best:
            path = os.path.join(self.save_dir, "best_model.pth")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
            
        torch.save(state_dict, path)


class MILTrainer:
    """
    MIL 训练与验证驱动模块
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        teacher_model: Optional[nn.Module] = None,
        use_amp: bool = True,
        save_dir: str = "./checkpoints",
        patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        
        self.teacher_model = teacher_model.to(device) if teacher_model else None
        if self.teacher_model:
            self.teacher_model.eval()
            
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
        
        self.early_stopping = EarlyStopping(save_dir=save_dir, patience=patience, mode='max')
        
    def _calculate_metrics(self, y_true: list, y_pred: list) -> Dict[str, float]:
        """计算分类多维评价指标"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='macro', zero_division=0)
        }

    def train_epoch(self, epoch: int) -> float:
        """执行单周期训练"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 自动混合精度训练
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                # 蒸馏模式
                if self.teacher_model is not None:
                    with torch.no_grad():
                        t_out = self.teacher_model(images)
                        teacher_logits = t_out['logits'] if isinstance(t_out, dict) else t_out
                        
                    s_out = self.model(images)
                    student_logits = s_out['logits'] if isinstance(s_out, dict) else s_out
                    
                    # 采用蒸馏损失逻辑 (如 DKD)
                    loss = self.criterion(student_logits, teacher_logits, labels)
                    
                # 常规训练模式
                else:
                    out = self.model(images)
                    logits = out['logits'] if isinstance(out, dict) else out
                    loss = self.criterion(logits, labels)
            
            # 梯度缩放与更新
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """验证集评估"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        eval_criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                out = self.model(images)
                logits = out['logits'] if isinstance(out, dict) else out
                loss = eval_criterion(logits, labels)
                
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        metrics = self._calculate_metrics(all_labels, all_preds)
        return total_loss / len(self.val_loader), metrics

    def fit(self, num_epochs: int) -> None:
        """启动完整训练流水线"""
        print(f"启动训练周期: {num_epochs}, 混合精度: {self.use_amp}, 蒸馏模式: {self.teacher_model is not None}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"轮次 [{epoch}/{num_epochs}] | 学习率: {current_lr:.2e}")
            print(f"训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f}")
            print(f"验证 Acc: {val_metrics['accuracy']:.4f} | 验证 Macro F1: {val_metrics['macro_f1']:.4f}")
            
            # 基于 Macro F1 进行早停判定
            self.early_stopping(current_score=val_metrics['macro_f1'], epoch=epoch, model=self.model)
            
            if self.early_stopping.early_stop:
                break
                
        print(f"训练结束，最佳模型已保存至: {self.early_stopping.save_dir}")
