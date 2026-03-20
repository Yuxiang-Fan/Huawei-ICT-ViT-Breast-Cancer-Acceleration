"""
病理切片数据加载模块 (WSI Patch DataLoader)

该模块负责:
1. 定义适应多折(Multi-Fold)目录结构的 PyTorch Dataset。
2. 提供针对病理图像 (H&E 染色) 的数据增强策略 (Data Augmentation)。
3. 构建并返回配置了预取(Prefetch)和固定内存(Pinned Memory)的高效 DataLoader，
   适配 ModelArts 和昇腾 NPU/GPU 训练环境。
"""

import os
import glob
from typing import List, Tuple, Dict, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# =====================================================================
# 全局配置与类别映射 (Configuration & Class Mapping)
# =====================================================================

# 赛题的四分类映射
CLASS_MAP = {
    "Normal": 0,
    "Benign": 1,
    "In_Situ": 2,
    "Invasive": 3
}

# ImageNet 标准均值与方差 (适用于 UNI / ViT 等预训练模型)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# 数据增强管道 (Data Transforms)
# =====================================================================

def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    获取指定数据划分的数据增强策略。
    
    Args:
        split (str): 'train' 或 'val' / 'test'
        image_size (int): 模型期望的输入分辨率大小
        
    Returns:
        transforms.Compose: 组合好的图像变换操作
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 病理图像对旋转不敏感，加入随机旋转增强鲁棒性
            transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
            # 颜色抖动，模拟不同医院 H&E 染色的色彩差异
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # 验证集和测试集只做尺寸调整和标准化，保证评估的确定性
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


# =====================================================================
# 自定义 PyTorch Dataset
# =====================================================================

class WSIPatchDataset(Dataset):
    """
    病理切片数据集，用于读取组织 Patch 并返回 Tensor。
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        """
        初始化数据集。
        
        Args:
            data_dir (str): 数据集目录，例如 'path/to/dataset/fold1/train'
            transform (Callable, optional): 图像预处理/增强函数
            
        Raises:
            FileNotFoundError: 如果数据目录不存在或为空
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"❌ 在 {data_dir} 中未找到任何有效的图片文件 (.jpg)")
            
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        扫描目录结构，收集所有图片的路径及其对应的整数标签。
        """
        samples = []
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"❌ 目录不存在: {self.data_dir}")
            
        for class_name, class_idx in CLASS_MAP.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠️  警告: 未找到类别目录 {class_dir}")
                continue
                
            # 获取该类别下所有的 jpg 切片
            # 注意: 这里假设从 builder.py 生成的切片格式为 .jpg
            img_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
            for img_path in img_paths:
                samples.append((img_path, class_idx))
                
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        读取单张图片，应用变换，并返回张量和标签。
        """
        img_path, label = self.samples[idx]
        
        try:
            # 使用 PIL 打开并转换为 RGB，避免灰度图导致通道维度不匹配
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 读取图片失败: {img_path}, 错误: {e}")
            # 如果某张图片损坏，创建一个全零的占位 Tensor (防止训练崩溃)
            # 实际生产中建议在 builder 阶段过滤损坏图片
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label


# =====================================================================
# DataLoader 构建器 (DataLoader Builders)
# =====================================================================

def create_fold_dataloaders(
    dataset_root: str, 
    fold_idx: int, 
    batch_size: int = 64, 
    num_workers: int = 8, 
    image_size: int = 224,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    为指定的 Fold 创建训练集和验证集的 DataLoader。
    
    Args:
        dataset_root (str): 生成的多折数据集根目录
        fold_idx (int): 当前要加载的 Fold 编号 (例如 1, 2, 3, 4)
        batch_size (int): 批处理大小
        num_workers (int): 数据加载的子进程数 (推荐设置为 CPU 核心数)
        image_size (int): 模型输入分辨率
        pin_memory (bool): 是否将数据锁定在页锁定内存中 (开启可加速向 NPU/GPU 的传输)
        
    Returns:
        Dict[str, DataLoader]: 包含 'train' 和 'val' DataLoader 的字典
    """
    fold_dir = os.path.join(dataset_root, f"fold{fold_idx}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")
    
    # 获取数据增强
    train_transform = get_transforms(split='train', image_size=image_size)
    val_transform = get_transforms(split='val', image_size=image_size)
    
    # 实例化 Dataset
    print(f"📦 正在加载 Fold {fold_idx} 数据集...")
    train_dataset = WSIPatchDataset(data_dir=train_dir, transform=train_transform)
    val_dataset = WSIPatchDataset(data_dir=val_dir, transform=val_transform)
    
    print(f"   - 训练集切片数: {len(train_dataset)}")
    print(f"   - 验证集切片数: {len(val_dataset)}")
    
    # 实例化 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True          # 丢弃最后一个不完整的 Batch，有利于 NPU/GPU 内存对齐
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,          # 验证集无需打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return {
        "train": train_loader,
        "val": val_loader
    }


# =====================================================================
# 独立测试入口 (Local Testing)
# =====================================================================

if __name__ == "__main__":
    # 外部数据集占位符
    dummy_dataset_root = "path/to/your/processed/wsi_patches_multi_fold"
    
    try:
        loaders = create_fold_dataloaders(
            dataset_root=dummy_dataset_root,
            fold_idx=1,
            batch_size=32,
            num_workers=4
        )
        
        train_loader = loaders["train"]
        
        # 尝试获取一个 Batch
        for images, labels in train_loader:
            print(f"✅ 成功加载一个 Batch!")
            print(f"   Images shape: {images.shape} # 预期: [32, 3, 224, 224]")
            print(f"   Labels shape: {labels.shape} # 预期: [32]")
            break
            
    except FileNotFoundError as e:
        print(f"💡 提示: 这是一个占位路径。请在实际运行中提供真实的数据目录。")
        print(f"   具体报错: {e}")