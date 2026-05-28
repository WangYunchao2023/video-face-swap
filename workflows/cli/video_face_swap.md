# video_face_swap — 视频换脸 CLI 工具

## 概述

基于 insightface inswapper_128 的视频换脸命令行工具。
支持单参考图 / 多参考图（自动按人脸嵌入相似度匹配）、跳帧加速、起止帧控制。

## 文件位置

- 脚本：`../../video_face_swap.py`
- 依赖：insightface 0.7.3+, opencv-python, onnxruntime-gpu, numpy<2.0

## 用法

```bash
# 单人换脸
python3 ../../video_face_swap.py swap \
  --source ref.jpg \
  --video input.mp4 \
  -o output.mp4

# 多人脸换脸（自动匹配）
python3 ../../video_face_swap.py swap \
  --sources personA.jpg personB.jpg \
  --video group.mp4 \
  -o result.mp4

# 跳帧加速（每 N+1 帧处理一帧）
python3 ../../video_face_swap.py swap \
  --source ref.jpg \
  --video input.mp4 \
  --skip 1 \
  -o result.mp4

# 带 GFPGAN 增强（需要预装 gfpgan）
python3 ../../video_face_swap.py swap \
  --source ref.jpg \
  --video input.mp4 \
  --enhance \
  -o result.mp4
```

## 完整参数

```
swap      执行视频换脸
  -s, --source PATH      单张参考图（与 --sources 二选一）
  --sources PATH [PATH]  多张参考图
  -v, --video PATH       目标视频
  -o, --output PATH      输出视频（默认: {视频名}_swapped.mp4）
  --start N              起始帧（默认 0）
  --end N                结束帧（默认 -1 = 全部）
  --skip N               跳帧数（默认 0，每 N+1 帧处理一帧）
  --device auto|cpu|cuda 计算设备（默认 auto）
  --enhance              GFPGAN 面部增强
  --threshold FLOAT      相似度阈值 0~1（默认 0.35，越低越宽松）

extract   提取视频帧
  --video PATH           视频路径
  --frames-dir DIR       输出目录（默认 ./frames）
  --fps FLOAT            提取帧率（0=原始）

assemble  组装帧为视频
  --frames-dir DIR       帧目录
  --output PATH          输出视频
  --fps FLOAT            输出帧率（默认 30）
```

## 参数调优指南

| 场景 | 推荐参数 |
|------|---------|
| 正面大头近景 | `--threshold 0.35` 默认即可 |
| 侧脸/低头/遮挡 | `--threshold 0.25` 降低匹配门槛 |
| 多人密集场景 | `--threshold 0.4` 提高精度，减少误配 |
| 长视频（>5min） | `--skip 2` 或 `--skip 3` 大幅加速 |
| 只看某一段 | `--start N --end M` 指定范围 |
| 质量优先 | `--skip 0 --enhance` 全帧+增强 |

## 版本历史

- v1.0.0 (2026-05-28): 基础单图换脸
- v2.0.0 (2026-05-28): 多参考图支持，嵌入相似度自动匹配
