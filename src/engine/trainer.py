"""
实验驱动引擎 (Model Training & Distillation Engine)

该模块负责:
1. 封装标准的模型训练循环 (Training Loop) 与验证循环 (Validation Loop)。
2. 支持 AMP (自动混合精度) 以降低显存占用并加速 NPU/GPU 训练。
3. 内置 EarlyStopping 早停机制与最佳权重保存逻辑。
4. 支持常规微调 (Fine-tuning) 与知识蒸馏 (Knowledge Distillation) 两种模式。
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

# 如果后续集成在昇腾环境，可以使用 torch_npu 替换 cuda
# import torch_npu


class EarlyStopping:
    """
    早停机制与模型断点保存器。
    当验证集指标 (如 Loss 或 F1 Score) 在指定的 epoch 数内没有提升时，自动停止训练。
    """
    def __init__(self, save_dir: str, patience: int = 10, mode: str = 'max', min_delta: float = 1e-4):
        """
        Args:
            save_dir (str): 模型权重保存路径。
            patience (int): 容忍多少个 epoch 指标没有提升。
            mode (str): 'min' (如用于 Loss) 或 'max' (如用于 F1 Score)。
            min_delta (float): 被认为是“提升”的最小变化量。
        """
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
            print(f"   🌟 发现最佳模型 (Epoch {epoch})! 指标: {current_score:.4f}，已保存至 {self.save_dir}/best_model.pth")
        else:
            self.counter += 1
            print(f"   ⚠️ 模型未提升。早停计数器: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⏹️ Early Stopping 触发！最佳模型停留在 Epoch {self.best_epoch}。")
                
        # 按频率保存常规 Checkpoint (如每 10 个 Epoch)
        if epoch > 0 and epoch % 10 == 0:
            self._save_checkpoint(model, is_best=False, epoch=epoch)

    def _save_checkpoint(self, model: nn.Module, is_best: bool, epoch: int = 0) -> None:
        """安全地保存模型权重"""
        # 如果模型被 DistributedDataParallel 或 DistillationWrapper 包装，提取其内部参数
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save = model_to_save.student if hasattr(model_to_save, 'student') else model_to_save
        
        state_dict = model_to_save.state_dict()
        
        if is_best:
            path = os.path.join(self.save_dir, "best_model.pth")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
            
        torch.save(state_dict, path)


class MILTrainer:
    """
    MIL 训练与验证全流程驱动器。
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
            self.teacher_model.eval()  # 教师模型永远处于评估模式
            
        self.use_amp = use_amp
        # 混合精度 Scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
        
        self.early_stopping = EarlyStopping(save_dir=save_dir, patience=patience, mode='max')
        
    def _calculate_metrics(self, y_true: list, y_pred: list) -> Dict[str, float]:
        """计算四分类任务的多维评价指标"""
        acc = accuracy_score(y_true, y_pred)
        # 医疗场景常用 Macro F1 评估不均衡类别
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision": macro_prec,
            "recall": macro_rec
        }

    def train_epoch(self, epoch: int) -> float:
        """执行单个 Epoch 的训练"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # --------- AMP 混合精度前向传播 ---------
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                
                # 如果是蒸馏模式
                if self.teacher_model is not None:
                    with torch.no_grad():
                        t_out = self.teacher_model(images)
                        teacher_logits = t_out['logits'] if isinstance(t_out, dict) else t_out
                        
                    s_out = self.model(images)
                    student_logits = s_out['logits'] if isinstance(s_out, dict) else s_out
                    
                    # Distillation Loss (e.g., DKDLoss)
                    loss = self.criterion(student_logits, teacher_logits, labels)
                    
                # 正常微调模式
                else:
                    out = self.model(images)
                    logits = out['logits'] if isinstance(out, dict) else out
                    loss = self.criterion(logits, labels)
            
            # --------- 反向传播与梯度更新 ---------
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """执行验证集评估"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # 验证时不使用蒸馏损失，仅计算交叉熵用于观察，但主要根据 F1 Score 进行早停
        eval_criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                out = self.model(images)
                logits = out['logits'] if isinstance(out, dict) else out
                loss = eval_criterion(logits, labels)
                
            total_loss += loss.item()
            
            # 获取预测类别
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics

    def fit(self, num_epochs: int) -> None:
        """启动完整训练/蒸馏流水线"""
        print(f"\n🚀 开始训练流水线 (Total Epochs: {num_epochs}, AMP: {self.use_amp}, Distillation: {self.teacher_model is not None})")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印日志
            print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | LR: {current_lr:.2e}")
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"   Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")
            
            # 根据 Macro F1 触发早停机制
            self.early_stopping(current_score=val_metrics['macro_f1'], epoch=epoch, model=self.model)
            
            if self.early_stopping.early_stop:
                break
                
        total_time = (time.time() - start_time) / 60
        print("="*80)
        print(f"🏁 训练结束！总耗时: {total_time:.2f} 分钟。最佳模型已保存在: {self.early_stopping.save_dir}")


# =====================================================================
# 独立测试与使用范例 (Example Usage)
# =====================================================================

if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    from src.models.mil_aggregator import EndToEndMILModel
    from src.compression.distiller import DKDLoss
    
    print("🛠️  正在实例化 Trainer 测试 ...")
    
    # 1. 模拟数据 (Batch = 4, 100 Patches, 512 Dim) -> 这是假定 Enhancer 后的特征输入
    B, N, D, Num_Classes = 4, 100, 512, 4
    dummy_x = torch.randn(16, N, D)
    dummy_y = torch.randint(0, Num_Classes, (16,))
    
    dataset = TensorDataset(dummy_x, dummy_y)
    train_loader = DataLoader(dataset, batch_size=B, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=B, shuffle=False)
    
    # 2. 实例化模型 (学生与教师)
    student_model = EndToEndMILModel(agg_type='gated', input_dim=D, num_classes=Num_Classes)
    teacher_model = EndToEndMILModel(agg_type='multihead', input_dim=D, num_classes=Num_Classes)
    
    # 3. 优化器、损失函数与设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    
    # 使用蒸馏损失
    criterion = DKDLoss(T=4.0, alpha=1.0, beta=8.0)
    
    # 4. 实例化训练器
    trainer = MILTrainer(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        teacher_model=teacher_model, # 传入教师即开启蒸馏模式
        use_amp=True,
        save_dir="./test_checkpoints",
        patience=3
    )
    
    # 5. 启动训练 (仅跑 2 个 epoch 做测试)
    trainer.fit(num_epochs=2)