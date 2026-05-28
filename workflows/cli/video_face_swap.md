# video_face_swap — 视频换脸 CLI 工具

## 概述

基于 insightface inswapper_128 的视频换脸工具。支持三种模式：
- **目标锁定模式**：用目标截图锁定视频中特定人脸，只换此人（推荐）
- **强换模式**：所有人脸都换
- **多人脸模式**：多张参考图 + 自动匹配（v2 兼容）

## 文件位置

- 脚本：`../../video_face_swap.py`
- 依赖：insightface 0.7.3+, opencv-python, onnxruntime-gpu, numpy<2.0

## 用法

### 目标锁定模式（推荐，只换特定人）
```bash
python3 ../../video_face_swap.py swap \
  --source ref.jpg \           # 换成这个人的脸
  --target target_face.jpg \   # 换掉这个人（截图）
  --video input.mp4 \
  -o output.mp4 \
  --target-threshold 0.2       # 匹配门槛，越低越宽松
```

### 强换模式（所有人脸都换）
```bash
python3 ../../video_face_swap.py swap \
  --source ref.jpg \
  --video input.mp4 \
  -o output.mp4
```

### 额外参数
```
  --skip N          跳帧加速（每 N+1 帧处理一帧）
  --start N         起始帧
  --end N           结束帧（-1 = 全部）
  --enhance         GFPGAN 面部增强
  --device auto|cpu|cuda
```

## 参数调优指南

### --target-threshold（目标锁定模式关键参数）

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 目标人脸角度一致 | 0.3~0.4 | 严格，几乎不误换 |
| 目标有侧脸/低头 | 0.15~0.2 | 宽容些，不易漏 |
| 多人物且长相差异大 | 0.3+ | 提高精度 |
| 目标时有时无 | 0.1~0.15 | 尽量覆盖 |

### --skip

| 视频时长 | 推荐 | 加速效果 |
|---------|------|---------|
| <30s | 0 | 全帧无跳 |
| 30s~3min | 1 | 约 2x |
| >3min | 2~3 | 约 3~4x |

## 完整子命令

```
swap      视频换脸（三种模式）
enhance   对已有视频做 GFPGAN 面部增强
```

## 版本历史

- v1.0.0 (2026-05-28): 基础单图强换
- v2.0.0 (2026-05-28): 多参考图 + 嵌入匹配
- v3.0.0 (2026-05-28): 目标锁定模式 + 统一接口