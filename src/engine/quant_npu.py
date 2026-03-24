"""
昇腾 NPU 量化与离线编译引擎

负责:
1. 模型导出: 将训练或剪枝后的 PyTorch 模型导出为静态 ONNX 格式。
2. AMCT 量化: 调用华为 AMCT 工具链执行 INT8 PTQ 量化。
3. ATC 编译: 构建并执行 ATC 命令，生成 NPU 部署所需的 .om 离线模型。
"""

import os
import subprocess
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.data import DataLoader

try:
    import amct_pytorch as amct
    AMCT_AVAILABLE = True
except ImportError:
    AMCT_AVAILABLE = False
    print("[Warning] 未检测到 amct_pytorch，INT8 量化功能将被禁用。")


class NPUDeployPipeline:
    """
    昇腾 NPU 部署流水线，包含 ONNX 导出、INT8 量化以及 OM 编译。
    """
    def __init__(self, model: nn.Module, device: torch.device, work_dir: str = "./npu_deployment"):
        self.model = model.to(device).eval()
        self.device = device
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def export_to_onnx(self, input_shape: Tuple[int, ...], onnx_name: str = "mil_model_fp16.onnx", dynamic_batch: bool = False) -> str:
        """
        将 PyTorch 模型导出为静态 ONNX 格式。
        """
        onnx_path = os.path.join(self.work_dir, onnx_name)
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        print(f"[ONNX Export] 开始导出模型至: {onnx_path}")
        
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None
        
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=14,            
                do_constant_folding=True,    
                export_params=True
            )
            
        print(f"[ONNX Export] 导出成功。文件大小: {os.path.getsize(onnx_path) / (1024**2):.2f} MB")
        return onnx_path

    def ptq_quantize_int8(self, calib_loader: DataLoader, input_shape: Tuple[int, ...]) -> str:
        """
        使用华为 AMCT 执行 INT8 PTQ 量化。
        """
        if not AMCT_AVAILABLE:
            raise RuntimeError("AMCT 工具未安装，无法执行量化。请在昇腾环境中运行。")
            
        print("[AMCT Quantization] 开始执行 INT8 PTQ 量化...")
        
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
        print("  - 量化配置已生成")
        
        # 2. 插入伪量化节点
        calibration_model = amct.restore_calibration_model(
            model=self.model,
            config_file=config_path
        )
        calibration_model = calibration_model.to(self.device).eval()
        
        # 3. 执行前向推理收集激活值分布统计信息
        print("  - 正在执行校准前向传播...")
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                images = images.to(self.device)
                _ = calibration_model(images)
                if i >= 10:  
                    break
                    
        # 4. 固化量化参数并导出 ONNX
        print("  - 正在固化量化参数并导出...")
        amct.save_quant_model(
            model=calibration_model,
            record_file=record_path,
            save_path=quant_out_prefix,
            input_data=dummy_input
        )
        
        quant_onnx_path = f"{quant_out_prefix}.onnx"
        print(f"[AMCT Quantization] 量化完成。模型已保存至: {quant_onnx_path}")
        return quant_onnx_path

    def compile_om_via_atc(self, onnx_path: str, soc_version: str = "Ascend310P3", enable_fp16: bool = True) -> str:
        """
        调用系统命令行执行 ATC 工具，将 ONNX 编译为 .om 离线模型。
        """
        print(f"[ATC Compilation] 启动 ATC 编译离线模型 (.om)...")
        om_prefix = onnx_path.rsplit('.', 1)[0]
        om_out_path = f"{om_prefix}.om"
        
        atc_cmd = [
            "atc",
            f"--framework=5",                       
            f"--model={onnx_path}",                 
            f"--output={om_prefix}",                
            f"--soc_version={soc_version}",         
            "--log=error"                           
        ]
        
        if enable_fp16:
            atc_cmd.append("--precision_mode=force_fp16")
        
        print(f"  - 执行命令: {' '.join(atc_cmd)}")
        
        try:
            process = subprocess.run(
                atc_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            print(f"[ATC Compilation] 编译成功。离线模型就绪: {om_out_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"[Error] ATC 编译失败，请检查 CANN 环境变量配置。")
            print(f"详细信息:\n{e.stderr}")
            raise
            
        return om_out_path


if __name__ == "__main__":
    from src.models.mil_aggregator import EndToEndMILModel
    from torch.utils.data import TensorDataset, DataLoader
    
    print("=== NPU 部署与量化流水线测试 ===")
    
    B, N, D = 1, 1000, 512 
    input_shape = (B, N, D)
    
    device = torch.device("cpu") 
    model = EndToEndMILModel(agg_type='gated', input_dim=D, num_classes=4)
    
    dummy_calib_data = torch.randn(8, N, D)
    dummy_calib_labels = torch.zeros(8)
    calib_loader = DataLoader(TensorDataset(dummy_calib_data, dummy_calib_labels), batch_size=1)
    
    pipeline = NPUDeployPipeline(model=model, device=device, work_dir="./npu_om_models")
    
    print("\n[流程 A: 纯 FP16 导出与编译]")
    fp16_onnx = pipeline.export_to_onnx(input_shape=input_shape, onnx_name="mil_gated_fp16.onnx")
    
    try:
        pipeline.compile_om_via_atc(onnx_path=fp16_onnx, soc_version="Ascend310P3", enable_fp16=True)
    except Exception:
        print("  - (预期的跳过) 当前环境无昇腾 ATC 工具，跳过 OM 编译。")
        
    print("\n[流程 B: INT8 训练后量化 (PTQ)]")
    if AMCT_AVAILABLE:
        int8_onnx = pipeline.ptq_quantize_int8(calib_loader=calib_loader, input_shape=input_shape)
        try:
            pipeline.compile_om_via_atc(onnx_path=int8_onnx, soc_version="Ascend310P3", enable_fp16=False)
        except Exception:
            pass
    else:
        print("  - (预期的跳过) 未安装 amct_pytorch 依赖。")
