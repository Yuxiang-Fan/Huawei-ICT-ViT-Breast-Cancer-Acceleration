"""
昇腾 NPU 量化与离线编译引擎 (NPU Quantization & ATC Compilation Engine)

该模块负责:
1. 模型导出: 将训练/剪枝后的 PyTorch MIL 模型安全导出为静态 ONNX。
2. AMCT 量化: 调用华为 Ascend Model Compression Toolkit (AMCT) 进行 INT8 训练后量化 (PTQ)。
3. ATC 编译: 自动构建并执行 ATC (Ascend Tensor Compiler) 命令，生成部署所需的 .om 文件。
"""

import os
import subprocess
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.data import DataLoader

# 尝试导入华为昇腾量化工具包 (AMCT)
try:
    import amct_pytorch as amct
    AMCT_AVAILABLE = True
except ImportError:
    AMCT_AVAILABLE = False
    print("⚠️  未检测到 amct_pytorch，INT8 量化功能将被禁用。如需量化，请在昇腾环境中安装 AMCT。")


class NPUDeployPipeline:
    """
    昇腾 NPU 部署流水线：包含 ONNX 导出、INT8 量化以及 OM 编译。
    """
    def __init__(self, model: nn.Module, device: torch.device, work_dir: str = "./npu_deployment"):
        """
        Args:
            model (nn.Module): 准备部署的 PyTorch 端到端模型。
            device (torch.device): 当前运行设备。
            work_dir (str): 输出文件 (ONNX, OM, 量化配置) 的工作目录。
        """
        self.model = model.to(device).eval()
        self.device = device
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def export_to_onnx(self, input_shape: Tuple[int, ...], onnx_name: str = "mil_model_fp16.onnx", dynamic_batch: bool = False) -> str:
        """
        将 PyTorch 模型导出为静态 ONNX 格式。
        
        Args:
            input_shape (Tuple): 输入张量的形状，例如 (BatchSize, NumPatches, FeatureDim)。
            onnx_name (str): 导出的 ONNX 文件名。
            dynamic_batch (bool): 是否支持动态 Batch Size。注意：部分 NPU 算子在静态形状下性能最优。
            
        Returns:
            str: 生成的 ONNX 文件路径。
        """
        onnx_path = os.path.join(self.work_dir, onnx_name)
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        print(f"\n📦 开始导出 ONNX 模型: {onnx_path} ...")
        
        # 动态轴配置 (可选)
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None
        
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=14,            # 昇腾通常对 Opset 11-14 支持最好
                do_constant_folding=True,    # 开启常量折叠优化计算图
                export_params=True
            )
            
        print(f"✅ ONNX 导出成功! 文件大小: {os.path.getsize(onnx_path) / (1024**2):.2f} MB")
        return onnx_path

    def ptq_quantize_int8(self, calib_loader: DataLoader, input_shape: Tuple[int, ...]) -> str:
        """
        使用华为 AMCT 执行 INT8 训练后量化 (Post-Training Quantization, PTQ)。
        
        Args:
            calib_loader (DataLoader): 包含少量数据 (如 16-32 个 Batch) 的校准数据加载器。
            input_shape (Tuple): 输入张量的形状。
            
        Returns:
            str: 包含量化权重的伪量化 ONNX 路径 (可送入 ATC 编译)。
        """
        if not AMCT_AVAILABLE:
            raise RuntimeError("❌ AMCT 工具未安装，无法执行量化。请确保在 ModelArts 或昇腾物理机中运行。")
            
        print("\n🗜️ 开始执行 AMCT INT8 训练后量化 (PTQ) ...")
        
        dummy_input = torch.randn(*input_shape, device=self.device)
        config_path = os.path.join(self.work_dir, "quant.cfg")
        record_path = os.path.join(self.work_dir, "record.txt")
        quant_out_prefix = os.path.join(self.work_dir, "mil_model_int8")
        
        # 1. 生成量化配置文件
        amct.create_quant_config(
            config_file=config_path,
            model=self.model,
            input_data=dummy_input
        )
        print("   -> 量化配置已生成。")
        
        # 2. 修改模型图，插入伪量化节点
        calibration_model = amct.restore_calibration_model(
            model=self.model,
            config_file=config_path
        )
        calibration_model = calibration_model.to(self.device).eval()
        
        # 3. 在校准数据集上进行前向推理，收集激活值的分布统计信息 (Min/Max/KLD)
        print("   -> 开始执行校准前向传播...")
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                images = images.to(self.device)
                _ = calibration_model(images)
                if i >= 10:  # 通常 10-20 个 Batch 的校准数据已足够
                    break
                    
        # 4. 保存最终的量化模型 (内部会导出为 ONNX)
        print("   -> 正在固化量化参数并导出...")
        amct.save_quant_model(
            model=calibration_model,
            record_file=record_path,
            save_path=quant_out_prefix,
            input_data=dummy_input
        )
        
        quant_onnx_path = f"{quant_out_prefix}.onnx"
        print(f"✅ AMCT INT8 量化完成! 量化模型已保存至: {quant_onnx_path}")
        return quant_onnx_path

    def compile_om_via_atc(self, onnx_path: str, soc_version: str = "Ascend310P3", enable_fp16: bool = True) -> str:
        """
        调用系统命令行执行 ATC (Ascend Tensor Compiler)，将 ONNX 编译为 .om 离线模型。
        
        Args:
            onnx_path (str): 源 ONNX 文件路径。
            soc_version (str): 目标芯片版本 (如 Ascend310P3 对应昇腾 310P 系列，Ascend910B 对应 910B)。
            enable_fp16 (bool): 是否强制开启 FP16 精度 (对于非量化的 ONNX 推荐开启)。
            
        Returns:
            str: 编译成功的 .om 文件路径。
        """
        print(f"\n⚙️ 启动 ATC 编译离线模型 (.om) ...")
        om_prefix = onnx_path.rsplit('.', 1)[0]
        om_out_path = f"{om_prefix}.om"
        
        # 构建 ATC 命令
        atc_cmd = [
            "atc",
            f"--framework=5",                       # 5 代表 ONNX 框架
            f"--model={onnx_path}",                 # 输入模型路径
            f"--output={om_prefix}",                # 输出路径前缀 (ATC会自动加.om后缀)
            f"--soc_version={soc_version}",         # 目标芯片架构
            "--log=error"                           # 减少控制台冗余日志
        ]
        
        # 精度控制策略
        if enable_fp16:
            atc_cmd.append("--precision_mode=force_fp16")
        
        print(f"   -> 执行命令: {' '.join(atc_cmd)}")
        
        try:
            # 开启子进程调用系统 ATC 工具
            process = subprocess.run(
                atc_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            print(f"✅ ATC 编译成功! .om 模型就绪: {om_out_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ATC 编译失败! 请检查环境变量 (如 source set_env.sh) 是否配置正确。")
            print(f"   错误详情:\n{e.stderr}")
            raise
            
        return om_out_path


# =====================================================================
# 独立测试与使用范例 (Example Usage)
# =====================================================================

if __name__ == "__main__":
    from src.models.mil_aggregator import EndToEndMILModel
    from torch.utils.data import TensorDataset, DataLoader
    
    print("="*60)
    print("🛠️  正在测试 NPU 部署与量化流水线 ...")
    
    # 1. 模拟网络与输入环境
    B, N, D = 1, 1000, 512 # NPU 推理通常采用 BatchSize = 1 追求最低延迟
    input_shape = (B, N, D)
    
    device = torch.device("cpu") # 导出和编译逻辑在 CPU 上即可完成
    model = EndToEndMILModel(agg_type='gated', input_dim=D, num_classes=4)
    
    # 模拟构建一个极小的校准数据集用于量化 (Calibration Data)
    dummy_calib_data = torch.randn(8, N, D)
    dummy_calib_labels = torch.zeros(8)
    calib_loader = DataLoader(TensorDataset(dummy_calib_data, dummy_calib_labels), batch_size=1)
    
    # 2. 实例化部署流水线
    pipeline = NPUDeployPipeline(model=model, device=device, work_dir="./npu_om_models")
    
    # 3. 路线 A：直接导出 FP16 ONNX 并尝试编译
    print("\n[路线 A: 纯 FP16 加速]")
    fp16_onnx = pipeline.export_to_onnx(input_shape=input_shape, onnx_name="mil_gated_fp16.onnx")
    
    # 注意：此处的 ATC 编译在未安装 CANN 工具链的普通 PC 上会失败并被捕获
    try:
        pipeline.compile_om_via_atc(onnx_path=fp16_onnx, soc_version="Ascend310P3", enable_fp16=True)
    except Exception as e:
        print("   -> (预期的跳过) 由于当前环境无昇腾 ATC 工具，跳过 OM 生成测试。")
        
    # 4. 路线 B：INT8 训练后量化
    print("\n[路线 B: INT8 量化压缩]")
    if AMCT_AVAILABLE:
        int8_onnx = pipeline.ptq_quantize_int8(calib_loader=calib_loader, input_shape=input_shape)
        try:
            pipeline.compile_om_via_atc(onnx_path=int8_onnx, soc_version="Ascend310P3", enable_fp16=False)
        except Exception:
            pass
    else:
        print("   -> (预期的跳过) 未安装 amct_pytorch。")
        
    print("="*60)