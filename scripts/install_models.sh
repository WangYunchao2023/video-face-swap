#!/bin/bash
# 视频换脸工具 - 模型安装脚本
# 首次使用时运行：bash scripts/install_models.sh

set -e

echo "=========================================="
echo "🔧 视频换脸工具 - 模型自动安装"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查并下载 inswapper_128.onnx
INSWAPPER_PATH="$HOME/.insightface/models/buffalo_l/inswapper_128.onnx"
if [ ! -f "$INSWAPPER_PATH" ]; then
    echo -e "${BLUE}⬇️  下载 inswapper_128.onnx...${NC}"
    echo "   目标路径：$INSWAPPER_PATH"
    
    mkdir -p "$(dirname "$INSWAPPER_PATH")"
    
    if command -v wget &> /dev/null; then
        wget --show-progress -q -O "$INSWAPPER_PATH" \
          https://huggingface.co/insightface/inswapper/resolve/main/inswapper_128.onnx
    elif command -v curl &> /dev/null; then
        curl -L -o "$INSWAPPER_PATH" \
          https://huggingface.co/insightface/inswapper/resolve/main/inswapper_128.onnx
    else
        echo -e "${YELLOW}⚠️  未找到 wget 或 curl，请手动下载:${NC}"
        echo "   下载地址：https://huggingface.co/insightface/inswapper/resolve/main/inswapper_128.onnx"
        echo "   目标路径：$INSWAPPER_PATH"
        exit 1
    fi
    
    echo -e "${GREEN}✅ inswapper_128.onnx 下载完成${NC}"
else
    echo -e "${GREEN}✓ inswapper_128.onnx 已存在${NC}"
fi

# 检查并软链到 ComfyUI
COMFY_INSWAPPER="$HOME/ComfyUI/models/insightface/inswapper_128.onnx"
if [ ! -f "$COMFY_INSWAPPER" ] && [ ! -L "$COMFY_INSWAPPER" ]; then
    echo ""
    echo -e "${BLUE}🔗 创建 ComfyUI 软链接...${NC}"
    mkdir -p "$(dirname "$COMFY_INSWAPPER")"
    ln -sf "$INSWAPPER_PATH" "$COMFY_INSWAPPER"
    echo -e "${GREEN}✅ ComfyUI 软链接创建完成${NC}"
else
    echo -e "${GREEN}✓ ComfyUI inswapper 链接已存在${NC}"
fi

# 检查并下载 GFPGANv1.4.pth
GFPGAN_PATH="$HOME/ComfyUI/models/facerestore_models/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN_PATH" ]; then
    echo ""
    echo -e "${BLUE}⬇️  下载 GFPGANv1.4.pth...${NC}"
    echo "   目标路径：$GFPGAN_PATH"
    
    mkdir -p "$(dirname "$GFPGAN_PATH")"
    
    if command -v wget &> /dev/null; then
        wget --show-progress -q -O "$GFPGAN_PATH" \
          https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
    elif command -v curl &> /dev/null; then
        curl -L -o "$GFPGAN_PATH" \
          https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
    else
        echo -e "${YELLOW}⚠️  未找到 wget 或 curl，请手动下载:${NC}"
        echo "   下载地址：https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        echo "   目标路径：$GFPGAN_PATH"
        exit 1
    fi
    
    echo -e "${GREEN}✅ GFPGANv1.4.pth 下载完成${NC}"
else
    echo -e "${GREEN}✓ GFPGANv1.4.pth 已存在${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 所有模型安装完成！${NC}"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 安装 Python 依赖：pip install -r requirements.txt"
echo "  2. 运行换脸：python3 skill/video_face_swap_skill.py -r ref.jpg -v input.mp4"
echo ""