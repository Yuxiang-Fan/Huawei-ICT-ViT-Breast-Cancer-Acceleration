"""
病理切片数据加载模块

该模块负责:
1. 定义适应多折目录结构的 PyTorch Dataset。
2. 提供针对 H&E 染色病理图像的数据增强策略。
3. 构建并返回配置了 prefetch 和 pinned memory 的 DataLoader，
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
# 全局配置与类别映射
# =====================================================================

CLASS_MAP = {
    "Normal": 0,
    "Benign": 1,
    "In_Situ": 2,
    "Invasive": 3
}

# ImageNet 标准均值与方差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# 数据增强管道
# =====================================================================

def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    获取指定数据划分的数据增强策略。
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 组织病理学图像无固定方向，随机旋转可大幅增强几何鲁棒性
            transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
            # 模拟不同医院或扫描仪的 H&E 染色差异
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # 验证集仅做缩放和归一化，确保评估的严格一致性
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
    病理 Patch 数据集解析器。
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何有效的 .jpg 切片文件")
            
    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
            
        for class_name, class_idx in CLASS_MAP.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[Warning] 未找到类别子目录: {class_dir}")
                continue
                
            img_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
            for img_path in img_paths:
                samples.append((img_path, class_idx))
                
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Error] 图片损坏已被捕获: {img_path}, Exception: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label


# =====================================================================
# DataLoader 构建器
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
    为指定的 Fold 实例化训练与验证 DataLoader。
    """
    fold_dir = os.path.join(dataset_root, f"fold{fold_idx}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")
    
    train_transform = get_transforms(split='train', image_size=image_size)
    val_transform = get_transforms(split='val', image_size=image_size)
    
    print(f"[DataLoader] 初始化 Fold {fold_idx} 数据管道...")
    train_dataset = WSIPatchDataset(data_dir=train_dir, transform=train_transform)
    val_dataset = WSIPatchDataset(data_dir=val_dir, transform=val_transform)
    
    print(f"  - Train Samples: {len(train_dataset)}")
    print(f"  - Val Samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return {
        "train": train_loader,
        "val": val_loader
    }


if __name__ == "__main__":
    dummy_dataset_root = "data/processed_patches"
    
    try:
        loaders = create_fold_dataloaders(
            dataset_root=dummy_dataset_root,
            fold_idx=1,
            batch_size=32,
            num_workers=4
        )
        train_loader = loaders["train"]
        
        for images, labels in train_loader:
            print(f"[Test] Batch loaded successfully. Image tensor: {images.shape}, Label tensor: {labels.shape}")
            break
            
    except FileNotFoundError as e:
        print(f"[Test Config] {e}")
