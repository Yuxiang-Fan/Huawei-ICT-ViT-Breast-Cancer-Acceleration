"""
华为云 OBS 数据同步工具 (OBS Synchronization Tool)

该脚本负责:
1. 提供本地与华为云 OBS (Object Storage Service) 之间的高效数据传输。
2. 支持大文件的分片上传 (Multipart Upload) 与断点续传 (Checkpoint)。
3. 提供目录级别的递归批量上传。
4. 集成 tqdm 进度条与网络异常自动重试机制。
"""

import os
import sys
import argparse
from typing import Optional, List
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 导入华为云 OBS SDK
try:
    from obs import ObsClient
except ImportError:
    print("⚠️ 未检测到 esdk-obs-python 库，请执行: pip install esdk-obs-python")
    sys.exit(1)


class ProgressCallback:
    """
    适配 OBS SDK 的自定义进度回调类，用于在终端渲染 tqdm 进度条。
    """
    def __init__(self, total_size: int, desc: str = "Uploading"):
        self.total_size = total_size
        self.progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        self.transferred = 0

    def progress(self, transferred: int, total: int, *args, **kwargs):
        increment = transferred - self.transferred
        self.progress_bar.update(increment)
        self.transferred = transferred

    def close(self):
        self.progress_bar.close()


class OBSManager:
    """
    华为云 OBS 客户端管理器，封装了文件与目录的上传逻辑。
    """
    def __init__(self, access_key: str, secret_key: str, server: str):
        """
        初始化 OBS 客户端。
        
        Args:
            access_key (str): 华为云 AK (Access Key)
            secret_key (str): 华为云 SK (Secret Key)
            server (str): OBS 终端节点 (例如: obs.cn-southwest-2.myhuaweicloud.com)
        """
        self.server = server
        self.obs_client = ObsClient(
            access_key_id=access_key,
            secret_access_key=secret_key,
            server=server,
            signature='obs',
            is_cname=False
        )

    # 使用 tenacity 装饰器，遇到网络波动时指数退避重试，最多重试 5 次
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _upload_single_file_with_retry(
        self, 
        bucket_name: str, 
        object_key: str, 
        local_path: str, 
        part_size: int = 50 * 1024 * 1024, # 默认 50MB 分片
        task_num: int = 10
    ) -> bool:
        """底层带有重试机制的单文件上传实现"""
        
        file_size = os.path.getsize(local_path)
        progress_cb = ProgressCallback(file_size, desc=f"上传 {os.path.basename(local_path)}")
        
        try:
            # 调用 OBS SDK 的高级上传接口
            resp = self.obs_client.uploadFile(
                bucketName=bucket_name,
                objectKey=object_key,
                uploadFile=local_path,
                partSize=part_size,
                taskNum=task_num,
                enableCheckpoint=True,       # 开启断点续传
                progressCallback=progress_cb.progress
            )
            
            progress_cb.close()
            
            if resp.status < 300:
                print(f"✅ 上传成功: obs://{bucket_name}/{object_key} (ETag: {resp.body.etag})")
                return True
            else:
                print(f"❌ 上传失败: {resp.errorMessage}")
                raise RuntimeError(f"OBS API 返回错误状态码: {resp.status}")
                
        except Exception as e:
            progress_cb.close()
            print(f"⚠️ 上传中断，将触发自动重试... 错误详情: {e}")
            raise e

    def upload_file(self, bucket_name: str, object_key: str, local_path: str) -> None:
        """
        对外暴露的单文件上传接口。
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"❌ 找不到本地文件: {local_path}")
            
        print(f"\n🚀 开始传输单文件: {local_path} -> obs://{bucket_name}/{object_key}")
        self._upload_single_file_with_retry(bucket_name, object_key, local_path)

    def upload_directory(self, bucket_name: str, target_prefix: str, local_dir: str) -> None:
        """
        递归上传整个本地目录到 OBS 指定前缀下。
        
        Args:
            bucket_name (str): 目标桶名称
            target_prefix (str): OBS 上的目标文件夹前缀 (如 'dataset/wsi_patches/')
            local_dir (str): 本地目录路径
        """
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(f"❌ 指定的路径不是一个文件夹: {local_dir}")
            
        print(f"\n🚀 开始递归传输文件夹: {local_dir} -> obs://{bucket_name}/{target_prefix}")
        
        files_to_upload = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                # 计算相对路径，以保持云端的目录结构
                relative_path = os.path.relpath(local_file_path, local_dir)
                # 统一使用 Linux 风格的斜杠作为 OBS Key
                object_key = os.path.join(target_prefix, relative_path).replace("\\", "/")
                files_to_upload.append((local_file_path, object_key))
                
        print(f"📦 共发现 {len(files_to_upload)} 个文件待上传。")
        
        success_count = 0
        for local_file_path, object_key in files_to_upload:
            try:
                self._upload_single_file_with_retry(bucket_name, object_key, local_file_path)
                success_count += 1
            except Exception as e:
                print(f"❌ 文件 {local_file_path} 上传彻底失败，跳过。错误: {e}")
                
        print(f"\n🏁 文件夹同步完成！成功: {success_count}/{len(files_to_upload)}")


# =====================================================================
# CLI 命令行入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="华为云 OBS 数据同步工具")
    
    parser.add_argument("--mode", type=str, choices=['file', 'dir'], required=True, 
                        help="上传模式: 'file' 为单文件上传，'dir' 为整个文件夹递归上传")
    parser.add_argument("--local_path", type=str, required=True, 
                        help="本地待上传的文件或文件夹路径")
    parser.add_argument("--bucket", type=str, required=True, 
                        help="目标 OBS 桶名称")
    parser.add_argument("--target_key", type=str, required=True, 
                        help="目标 OBS 路径 (如果 mode=file，则为对象键；如果 mode=dir，则为目录前缀)")
    parser.add_argument("--endpoint", type=str, default="obs.cn-southwest-2.myhuaweicloud.com", 
                        help="OBS 终端节点 (默认: 贵阳一 obs.cn-southwest-2.myhuaweicloud.com)")
    
    args = parser.parse_args()

    # 安全规范：从环境变量中读取 AK/SK，严禁在开源代码中硬编码！
    ak = os.environ.get("HUAWEICLOUD_SDK_AK")
    sk = os.environ.get("HUAWEICLOUD_SDK_SK")

    if not ak or not sk:
        print("❌ 错误: 找不到华为云凭证！")
        print("💡 提示: 请在运行前设置环境变量 HUAWEICLOUD_SDK_AK 和 HUAWEICLOUD_SDK_SK。")
        print("   Linux/Mac: export HUAWEICLOUD_SDK_AK='your_ak' && export HUAWEICLOUD_SDK_SK='your_sk'")
        print("   Windows:   set HUAWEICLOUD_SDK_AK=your_ak && set HUAWEICLOUD_SDK_SK=your_sk")
        sys.exit(1)

    try:
        manager = OBSManager(access_key=ak, secret_key=sk, server=args.endpoint)
        
        if args.mode == 'file':
            manager.upload_file(
                bucket_name=args.bucket, 
                object_key=args.target_key, 
                local_path=args.local_path
            )
        elif args.mode == 'dir':
            manager.upload_directory(
                bucket_name=args.bucket, 
                target_prefix=args.target_key, 
                local_dir=args.local_path
            )
            
    except Exception as err:
        print(f"\n❌ 程序执行异常中止: {err}")
        sys.exit(1)