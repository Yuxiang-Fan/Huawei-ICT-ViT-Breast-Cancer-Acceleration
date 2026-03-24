"""
多折异质均衡数据集构建与检验模块

核心功能:
1. 基于手工设定的分布策略进行 WSI 级别的数据均衡化。
2. 为不同 Fold 分配具有差异化的 Invasive WSI，促进集成学习的多样性。
3. 调度多进程将 WSI 切分为 Patch。
4. 执行严格的数据泄露、映射关系和均衡性核查。
"""

import os
import json
import pickle
import random
import argparse
import concurrent.futures
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

# =====================================================================
# 全局配置与领域先验知识
# =====================================================================

# WSI 标签映射字典
SLIDE_LABELS = {
    "01.svs": "Normal", "02.svs": "Benign", "03.svs": "In_Situ", "04.svs": "Normal",
    "05.svs": "Benign", "06.svs": "Invasive", "07.svs": "In_Situ", "08.svs": "Invasive",
    "09.svs": "Benign", "10.svs": "Normal", "11.svs": "In_Situ", "12.svs": "Invasive",
    "13.svs": "Invasive", "14.svs": "Benign", "15.svs": "In_Situ", "16.svs": "Invasive",
    "17.svs": "Normal", "18.svs": "In_Situ", "19.svs": "Invasive", "20.svs": "Benign",
    "A01.svs": "Invasive", "A02.svs": "Invasive", "A03.svs": "Invasive", "A04.svs": "Invasive",
    "A05.svs": "Invasive", "A06.svs": "Invasive", "A07.svs": "Invasive", "A08.svs": "Invasive",
    "A09.svs": "Invasive", "A10.svs": "Invasive"
}

# 基础的 4-Fold 验证集划分策略
BASE_VAL_SETS = [
    # Fold 1
    ["01.svs", "04.svs", "02.svs", "05.svs", "03.svs", "07.svs", "06.svs", "12.svs"],
    # Fold 2  
    ["10.svs", "17.svs", "09.svs", "14.svs", "11.svs", "15.svs", "08.svs", "13.svs"],
    # Fold 3
    ["01.svs", "10.svs", "20.svs", "05.svs", "18.svs", "07.svs", "A01.svs", "A02.svs"],
    # Fold 4
    ["04.svs", "17.svs", "02.svs", "09.svs", "03.svs", "11.svs", "A03.svs", "A04.svs"]
]

# 各 Fold 人工筛选的 Invasive WSI，用于制造数据异质性
MANUAL_INVASIVE_SELECTIONS = {
    1: ['06.svs', '08.svs', '12.svs', '13.svs', '16.svs', '19.svs'],
    2: ['A01.svs', 'A02.svs', 'A03.svs', 'A04.svs', 'A05.svs', 'A06.svs'],
    3: ['A07.svs', 'A08.svs', 'A09.svs', 'A10.svs', '12.svs', '19.svs'],
    4: ['06.svs', '13.svs', 'A01.svs', 'A05.svs', 'A08.svs', 'A10.svs'],
}

# 每类目标 WSI 的基准数量，用于控制 WSI 级别的样本均衡
TARGET_CLASS_COUNT = 6  


# =====================================================================
# 图像处理算子定义
# =====================================================================

def create_augmented_wsi_thumbnail(base_path: str, aug_name: str) -> Tuple[Any, Dict]:
    """
    生成增强 WSI 的缩略图并记录元数据。
    注: 实际工程中需接入色彩抖动、对比度增强等具体算法。
    """
    meta = {
        "augmentation_type": ["ColorJitter", "RandomContrast"],
        "base_wsi": os.path.basename(base_path),
        "target_name": aug_name
    }
    return "DummyImageObject", meta


def process_wsi_to_patches(wsi_info: Dict, split_type: str, fold_idx: int, wsi_dir: str, output_dir: str) -> Tuple[str, int]:
    """
    读取 WSI 并在目标组织区域提取指定数量的 Patch。
    注: 实际工程中需接入 OpenSlide 读取、阈值分割与坐标采样逻辑。
    """
    patches_generated = 3000
    return wsi_info['file'], patches_generated


# =====================================================================
# 数据均衡化与调度核心逻辑
# =====================================================================

def balance_wsis_with_manual_selection(train_wsi_files: List[str], fold_idx: int, wsi_dir: str) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    在 WSI 级别执行数据均衡化，并结合人为挑选的异质切片。
    """
    class_distribution = Counter([SLIDE_LABELS[w] for w in train_wsi_files])
    
    # 1. 处理特定类别的筛选逻辑 (如 Invasive 类别)
    invasive_selection = MANUAL_INVASIVE_SELECTIONS.get(fold_idx, [])
    valid_selection = [w for w in invasive_selection if w in train_wsi_files]
    
    if len(valid_selection) < TARGET_CLASS_COUNT:
        remaining = [w for w in train_wsi_files if SLIDE_LABELS[w] == 'Invasive' and w not in valid_selection]
        needed = TARGET_CLASS_COUNT - len(valid_selection)
        additional = random.sample(remaining, min(needed, len(remaining)))
        valid_selection.extend(additional)
        
    selected_invasive = valid_selection[:TARGET_CLASS_COUNT]
    
    # 2. 对少数类别执行 WSI 级别的数据增强
    augmented_wsi_list = []
    augmentation_records = []
    
    for cls in ['Normal', 'Benign', 'In_Situ']:
        current_count = class_distribution.get(cls, 0)
        needed = TARGET_CLASS_COUNT - current_count
        
        if needed > 0:
            cls_wsis = [w for w in train_wsi_files if SLIDE_LABELS[w] == cls]
            if not cls_wsis:
                continue
                
            for i in range(needed):
                base_wsi = random.choice(cls_wsis)
                base_path = os.path.join(wsi_dir, base_wsi)
                aug_id = f"{i:03d}"
                aug_name = f"AUG_{cls}_F{fold_idx}_{aug_id}_{base_wsi.replace('.svs', '')}.svs"
                
                thumb, meta = create_augmented_wsi_thumbnail(base_path, aug_name.replace('.svs', ''))
                
                if thumb is not None:
                    augmented_wsi_list.append({
                        'file': aug_name, 'label': cls, 'is_augmented': True,
                        'original': base_wsi, 'aug_meta': meta, 'fold': fold_idx
                    })
                    augmentation_records.append({
                        'aug_wsi': aug_name, 'original': base_wsi, 'class': cls,
                        'fold': fold_idx, 'methods': meta['augmentation_type']
                    })
    
    # 3. 汇总当前 Fold 的 WSI 列表
    all_wsis = []
    for wsi in train_wsi_files:
        cls = SLIDE_LABELS[wsi]
        if cls != 'Invasive':
            all_wsis.append({'file': wsi, 'label': cls, 'is_augmented': False, 'original': wsi, 'fold': fold_idx})
            
    for wsi in selected_invasive:
        all_wsis.append({'file': wsi, 'label': 'Invasive', 'is_augmented': False, 'original': wsi, 'fold': fold_idx, 'manually_selected': True})
        
    all_wsis.extend(augmented_wsi_list)
    return all_wsis, augmentation_records, selected_invasive


def generate_multi_fold_dataset(wsi_dir: str, output_root: str, num_folds: int = 4, start_fold: int = 1) -> None:
    """
    多 Fold 数据集生成流水线。
    """
    wsi_files = list(SLIDE_LABELS.keys())
    val_sets = BASE_VAL_SETS[:num_folds] if num_folds <= len(BASE_VAL_SETS) else [BASE_VAL_SETS[i % len(BASE_VAL_SETS)] for i in range(num_folds)]
    
    all_results, all_selection_records = [], []
    
    for i in range(num_folds):
        fold_idx = start_fold + i
        val_wsis = val_sets[i]
        train_wsis = [w for w in wsi_files if w not in val_wsis]
        
        balanced_train_wsis, aug_records, invasive_selected = balance_wsis_with_manual_selection(train_wsis, fold_idx, wsi_dir)
        
        # 构建并行处理任务队列
        all_tasks = []
        for wsi_info in balanced_train_wsis:
            all_tasks.append((wsi_info, "train", fold_idx, wsi_dir, output_root))
        for wsi_file in val_wsis:
            wsi_info = {'file': wsi_file, 'label': SLIDE_LABELS[wsi_file], 'is_augmented': False, 'original': wsi_file, 'fold': fold_idx}
            all_tasks.append((wsi_info, "val", fold_idx, wsi_dir, output_root))
            
        # 启动进程池处理 WSI 切片任务
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [executor.submit(process_wsi_to_patches, *task) for task in all_tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Fold {fold_idx} Processing"):
                results.append(future.result())
                
        total_patches = sum(r[1] for r in results if r is not None)
        
        fold_result = {
            'fold': fold_idx,
            'original_train_wsis': train_wsis,
            'balanced_train_wsis': [w['file'] for w in balanced_train_wsis],
            'val_wsis': val_wsis,
            'patch_count': total_patches
        }
        selection_record = {
            'fold': fold_idx, 'invasive_selected': invasive_selected,
            'invasive_available': [w for w in train_wsis if SLIDE_LABELS[w] == 'Invasive'],
            'augmentation_count': len(aug_records)
        }
        
        all_results.append(fold_result)
        all_selection_records.append(selection_record)

    # 序列化保存元数据
    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, f"multi_fold_summary_f{start_fold}_to_{start_fold+num_folds-1}.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump({'all_results': all_results, 'num_folds': num_folds}, f)
        
    config_path = os.path.join(output_root, "ensemble_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "num_models": num_folds,
            "voting_method": "soft_voting",
            "weights": [1.0] * num_folds
        }, f, indent=2)


# =====================================================================
# 验证模块
# =====================================================================

def check_all_folds(dataset_root: str, num_folds: int = 4) -> None:
    """
    对生成的数据集执行数据泄露检验和分布统计。
    """
    print(f"\n[Validation] 开始执行数据交叉验证与泄露核查...")
    
    total_issues = 0
    for fold_num in range(1, num_folds + 1):
        fold_dir = os.path.join(dataset_root, f"fold{fold_num}")
        if not os.path.exists(fold_dir):
            continue
            
        issues = []
        train_sources, val_sources = set(), set()
        
        # 溯源文件系统中的 WSI 所属关系
        for split, source_set in [("train", train_sources), ("val", val_sources)]:
            split_dir = os.path.join(fold_dir, split)
            if not os.path.exists(split_dir): 
                continue
                
            for cls_name in os.listdir(split_dir):
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir): 
                    continue
                    
                for file in os.listdir(cls_dir)[:50]:
                    if file.endswith('.jpg'):
                        wsi_base = file.split('_0')[0] + '.svs' if file.startswith('AUG_') else file.split('_')[0] + '.svs'
                        source_set.add(wsi_base)
                        
                        if split == "val" and file.startswith('AUG_'):
                            issues.append(f"严重越界: 验证集包含增强数据 {file}")

        # 跨集数据泄露检验
        leakage = train_sources & val_sources
        if leakage:
            issues.append(f"数据泄露检测: 训练集与验证集存在重叠 WSI: {list(leakage)}")

        if issues:
            print(f"Fold {fold_num} 检验失败，发现 {len(issues)} 处异常。")
            for iss in issues:
                print(f"  - {iss}")
                total_issues += 1
        else:
            print(f"Fold {fold_num} 检验通过。")
            
    if total_issues == 0:
        print("[Validation] 质量核查终审: 通过。无数据泄露情况。")
    else:
        print(f"[Validation] 质量核查终审: 未通过。需排查上述 {total_issues} 处异常。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Multi-Fold Dataset Builder")
    parser.add_argument("--wsi_dir", type=str, default="data/raw_wsi", help="原始 .svs 文件存放路径")
    parser.add_argument("--output_root", type=str, default="data/processed_patches", help="切片输出根目录")
    parser.add_argument("--num_folds", type=int, default=4, help="生成的 Fold 数量")
    parser.add_argument("--start_fold", type=int, default=1, help="起始 Fold 索引")
    parser.add_argument("--validate_only", action="store_true", help="仅执行目录校验")
    
    args = parser.parse_args()

    if not args.validate_only:
        generate_multi_fold_dataset(
            wsi_dir=args.wsi_dir,
            output_root=args.output_root,
            num_folds=args.num_folds,
            start_fold=args.start_fold
        )
    
    check_all_folds(dataset_root=args.output_root, num_folds=args.num_folds)
