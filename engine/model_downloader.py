#!/usr/bin/env python3
"""
模型自动下载模块
首次运行时自动检查并下载缺失的模型文件
"""

import os
from pathlib import Path
from typing import Dict, Any

MODELS_CONFIG = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/insightface/inswapper/resolve/main/inswapper_128.onnx",
        "path": Path.home() / ".insightface/models/buffalo_l/inswapper_128.onnx",
        "description": "InsightFace 换脸核心模型",
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "path": Path.home() / "ComfyUI/models/facerestore_models/GFPGANv1.4.pth",
        "description": "GFPGAN 面部修复模型",
    },
}


def check_and_download_models(verbose: bool = True):
    """
    检查并下载缺失的模型文件
    
    Args:
        verbose: 是否显示详细输出
    """
    import urllib.request
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    missing_models = []
    for name, info in MODELS_CONFIG.items():
        model_path = info["path"]
        if not model_path.exists():
            missing_models.append((name, info))
    
    if not missing_models:
        if verbose:
            print("✓ 所有模型文件已存在")
        return  # 所有模型都已存在
    
    if verbose:
        print("\n" + "=" * 60)
        print("🔧 检测到缺失的模型文件，正在自动下载...")
        print("=" * 60)
    
    downloaded_count = 0
    failed_count = 0
    
    for name, info in missing_models:
        model_path = info["path"]
        
        if verbose:
            print(f"\n⬇️  下载 {name} ...")
            print(f"    说明：{info['description']}")
            print(f"    目标路径：{model_path}")
        
        try:
            # 创建目录
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            request = urllib.request.urlopen(info["url"])
            total_size = int(request.headers.get('Content-Length', 0))
            
            with open(model_path, 'wb') as out_file:
                if has_tqdm and total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                              desc=name, ncols=80) as bar:
                        while chunk := request.read(8192):
                            out_file.write(chunk)
                            bar.update(len(chunk))
                else:
                    # 无进度条模式
                    downloaded = 0
                    while chunk := request.read(8192):
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if verbose and downloaded % (10 * 1024 * 1024) == 0:
                            print(f"    已下载 {downloaded / 1024 / 1024:.1f} MB")
            
            if verbose:
                file_size_mb = model_path.stat().st_size / 1024 / 1024
                print(f"✅ {name} 下载完成 ({file_size_mb:.1f} MB)")
            
            downloaded_count += 1
            
        except Exception as e:
            if verbose:
                print(f"❌ {name} 下载失败：{e}")
                print(f"   请手动下载后放到：{model_path}")
                print(f"   下载地址：{info['url']}")
            failed_count += 1
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"🎉 模型检查完成 - 成功：{downloaded_count}, 失败：{failed_count}")
        print("=" * 60 + "\n")
    
    return downloaded_count, failed_count


if __name__ == "__main__":
    # 直接运行此脚本时执行下载
    check_and_download_models(verbose=True)