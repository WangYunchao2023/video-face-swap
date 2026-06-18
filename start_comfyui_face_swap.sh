#!/bin/bash
# ComfyUI 启动脚本 - 输出到 ~/Videos/face_swap/comfyui/
# 原 ~/ComfyUI/output/ 仍保持不变（其他任务输出）

FACE_SWAP_DIR="$HOME/Videos/face_swap/comfyui"
mkdir -p "$FACE_SWAP_DIR"

cd ~/ComfyUI
python3 main.py \
  --output-directory "$FACE_SWAP_DIR" \
  --listen 0.0.0.0 \
  --port 8188
