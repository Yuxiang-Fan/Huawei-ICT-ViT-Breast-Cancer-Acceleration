"""
云端环境文件导出与打包工具 (Cloud Environment Export Utilities)

该模块负责:
1. 将指定的文件夹或多个文件压缩为标准的 ZIP 归档文件，并提供实时的 tqdm 进度条。
2. 针对 Jupyter Notebook / ModelArts 环境，生成基于 Base64 编码的 HTML 下载直链。
   (突破部分云平台无法直接双击下载大文件的限制)
"""

import os
import base64
import zipfile
import argparse
from typing import List, Optional, Union
from tqdm import tqdm

try:
    from IPython.display import HTML, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


def zip_target(target_path: str, output_zip_name: str) -> str:
    """
    将指定的目录或单文件压缩为 ZIP 文件。
    
    Args:
        target_path (str): 需要压缩的目录或文件路径。
        output_zip_name (str): 输出的 ZIP 文件名 (例如: 'dataset_patches.zip')。
        
    Returns:
        str: 生成的 ZIP 文件的绝对路径。
    """
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"❌ 找不到目标路径: {target_path}")

    # 1. 统计需要压缩的文件总数，用于进度条
    total_files = 0
    if os.path.isfile(target_path):
        total_files = 1
    else:
        for root, _, files in os.walk(target_path):
            total_files += len(files)
            
    print(f"📦 开始打包压缩: {target_path}")
    print(f"   -> 共计发现 {total_files} 个文件待压缩")

    # 2. 执行压缩并更新进度条
    # 使用 ZIP_DEFLATED 进行标准压缩，既能减小体积又不会太慢
    with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Zipping", unit="file") as pbar:
            if os.path.isfile(target_path):
                # 单文件压缩
                arcname = os.path.basename(target_path)
                zipf.write(target_path, arcname)
                pbar.update(1)
            else:
                # 目录递归压缩
                for root, _, files in os.walk(target_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 计算相对路径以保持 ZIP 内的目录结构清晰
                        arcname = os.path.relpath(file_path, os.path.dirname(target_path))
                        zipf.write(file_path, arcname)
                        pbar.update(1)

    # 3. 计算压缩包大小
    file_size_gb = os.path.getsize(output_zip_name) / (1024 ** 3)
    file_size_mb = os.path.getsize(output_zip_name) / (1024 ** 2)
    
    print(f"\n✅ 压缩完成！")
    print(f"   输出文件: {os.path.abspath(output_zip_name)}")
    if file_size_gb >= 1.0:
        print(f"   文件大小: {file_size_gb:.2f} GB")
    else:
        print(f"   文件大小: {file_size_mb:.2f} MB")
        
    return os.path.abspath(output_zip_name)


def create_download_link(file_path: str, custom_text: Optional[str] = None) -> None:
    """
    在 Jupyter Notebook 环境中生成文件的 Base64 HTML 下载直链。
    注意：此方法会将文件读入内存进行编码，不建议用于大于 2GB 的超大文件！
    
    Args:
        file_path (str): 需要下载的文件路径 (通常是上面生成的 ZIP 文件)。
        custom_text (str, optional): 链接显示的自定义文本。
    """
    if not IPYTHON_AVAILABLE:
        print("⚠️ 警告: 未检测到 IPython 环境，无法生成 HTML 下载链接。")
        print("   本功能仅在 Jupyter Notebook / ModelArts 交互式界面中有效。")
        return

    if not os.path.exists(file_path):
        print(f"❌ 找不到目标文件: {file_path}")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
    file_size_gb = file_size_mb / 1024
    
    # 安全警告：防止内存溢出 (OOM)
    if file_size_gb > 2.0:
        print(f"⚠️ 警告: 文件大小 ({file_size_gb:.2f} GB) 过大！")
        print("   使用 Base64 编码下载可能会导致 Notebook 内存溢出 (OOM) 崩溃。")
        print("   建议使用 tools/obs_sync.py 将超大文件同步至 OBS，或使用平台自带的文件树下载功能。")
        return

    print(f"⏳ 正在生成 Base64 下载链接，请稍候 (这可能需要花费一些时间)...")
    
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        
        filename = os.path.basename(file_path)
        size_str = f"{file_size_gb:.2f} GB" if file_size_gb >= 1.0 else f"{file_size_mb:.2f} MB"
        link_text = custom_text if custom_text else f"点击下载 {filename} ({size_str})"
        
        # 构建美观的 HTML 标签
        html_code = f"""
        <div style="margin: 20px 0; padding: 15px; border: 2px dashed #1a73e8; border-radius: 8px; text-align: center; background-color: #f8fbff;">
            <h3 style="color: #1a73e8; margin-top: 0;">🚀 导出就绪</h3>
            <a href="data:application/zip;base64,{b64}" download="{filename}" 
               style="display: inline-block; padding: 10px 20px; background-color: #1a73e8; color: white; 
                      text-decoration: none; font-weight: bold; font-size: 16px; border-radius: 5px; 
                      box-shadow: 0 4px 6px rgba(26,115,232,0.2); transition: all 0.3s ease;">
                📥 {link_text}
            </a>
            <p style="color: #666; font-size: 12px; margin-bottom: 0; margin-top: 10px;">
                注意: 下载过程中请勿关闭此页面。如果下载无响应，说明文件大小超出了浏览器 Base64 解析上限。
            </p>
        </div>
        """
        display(HTML(html_code))
        
    except Exception as e:
        print(f"❌ 生成下载链接失败: {e}")


# =====================================================================
# CLI 命令行入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文件打包压缩工具")
    
    parser.add_argument("--target", type=str, required=True, 
                        help="需要打包的目录或文件路径")
    parser.add_argument("--out", type=str, default="export_archive.zip", 
                        help="输出的 ZIP 文件名 (默认: export_archive.zip)")
    
    args = parser.parse_args()
    
    try:
        zip_target(target_path=args.target, output_zip_name=args.out)
    except Exception as err:
        print(f"❌ 程序执行异常中止: {err}")