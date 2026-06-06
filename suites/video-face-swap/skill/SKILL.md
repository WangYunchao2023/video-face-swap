---
name: video-face-swap
description: >-
  视频换脸 Skill。通过 insightface inswapper_128 模型实现高质量视频换脸，
  支持单人/多人换脸、目标锁定、参考图预处理（对齐+增强）、面部修复。
  触发词：视频换脸、换脸视频、face swap、video face swap
version: 1.2.0
parent_skill: image-video-generation
---

# Video Face Swap Skill v1.2.0

通过本地 AI 模型（insightface inswapper_128）实现高质量视频换脸，支持多种换脸模式和后处理增强。

## 架构

```
参考图 + 目标视频 [+ 目标人脸图]
    ↓
视频换脸引擎 (video_face_swap_skill.py v1.2.0)
    ↓
智能参数分析 (参考图质量 + 视频信息)
    ↓
GPU 加速处理 (insightface + ONNX Runtime)
    ↓
可选面部增强 (GFPGAN/CodeFormer)
    ↓
输出 MP4 → ~/Videos/face_swap/
```

**v1.2.0 新增**：智能参数自动选择 — 分析参考图质量（尺寸、角度、清晰度、光照）和视频信息（分辨率、时长、人脸数），自动推荐最优质量档位、跳帧策略、增强策略和匹配阈值。

---

## 架构
    ↓
视频换脸引擎 (video_face_swap.py)
    ↓
GPU 加速处理 (insightface + ONNX Runtime)
    ↓
可选面部增强 (GFPGAN/CodeFormer)
    ↓
输出 MP4 → ~/Videos/face_swap/
```

## 支持功能

| 功能 | 说明 |
|------|------|
| **单人换脸** | 视频中所有人脸都换成同一个参考人脸 |
| **目标锁定换脸** | 仅换换与目标人脸嵌入相似度超过阈值的人脸 |
| **多人参考图** | 支持多张参考图，按嵌入相似度自动匹配 |
| **参考图预处理** | 自动对齐 (norm_crop) + 可选 GFPGAN 增强，提升换脸质量 |
| **面部修复** | 换脸后可选 GFPGAN/CodeFormer 面部修复增强 |
| **跳帧处理** | 支持跳帧加速长视频处理 |
| **起止帧控制** | 只处理视频的指定帧范围 |
| **自适应检测** | 根据输入图片尺寸自动优化人脸检测参数，解决小图检测失败问题 |

## 前置条件

- 依赖已安装：`insightface==0.7.3`, `opencv-python==4.11.0`, `onnxruntime-gpu`, `numpy==1.26.4`
- 模型文件存在：`~/.insightface/models/buffalo_l/inswapper_128.onnx`
- 可选增强模型：`~/ComfyUI/models/facerestore_models/GFPGANv1.4.pth`
- ComfyUI 环境（用于工作流后备选项，但非必需）
- VRAM：建议 ≥ 6GB（取决于视频分辨率和批处理大小）

## Agent 调用方式（推荐）

```python
# 该 Skill 通过提供的 Python 接口调用
# 实际使用时，请通过 OpenClaw 的 skill 调用机制
# 下面示例展示了底层接口，实际使用请参考 Skill 文档

from video_face_swap_skill import VideoFaceSwap

# 初始化换脸处理器
swapper = VideoFaceSwap(device="auto")

# 基础换脸（所有人脸都换）
result = swapper.process(
    reference_image="/path/to/reference.jpg",
    target_video="/path/to/input.mp4",
    output_path="/path/to/output.mp4",  # 可选，默认 ~/Videos/face_swap/
    enhance=True  # 可选，默认 True
)

# 目标锁定换脸（仅换指定目标人脸）
result = swapper.process(
    reference_image="/path/to/reference.jpg",
    target_video="/path/to/input.mp4",
    target_face_image="/path/to/target.jpg",  # 可选，启用目标锁定
    output_path="/path/to/output.mp4",
    enhance=True,
    target_threshold=0.2  # 可选，默认 0.2
)

# 智能模式：自动选择最优参数（v1.2.0 新增）
result = swapper.process(
    reference_image="/path/to/reference.jpg",
    target_video="/path/to/input.mp4",
    output_path="/path/to/output.mp4",
    auto_params=True  # 自动分析并选择最优参数
)

# 高级选项：指定质量档位
result = swapper.process(
    reference_image="/path/to/reference.jpg",
    target_video="/path/to/input.mp4",
    output_path="/path/to/output.mp4",
    quality="hq",  # draft/normal/hq/best
    enhance=True,
    skip_frames=2  # 每 3 帧处理一帧
)
```

## CLI 调用方式

```bash
# 基础换脸
python3 ~/.openclaw/workspace-cortana-shadow/skills/video-face-swap/video_face_swap_skill.py \
  --reference /path/to/reference.jpg \
  --video /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --enhance

# 目标锁定换脸
python3 video_face_swap_skill.py \
  --reference /path/to/reference.jpg \
  --video /path/to/input.mp4 \
  --target /path/to/target.jpg \
  --output /path/to/output.mp4 \
  --enhance \
  --target-threshold 0.25

# 使用预处理的参考图（推荐用于反复使用同一参考图）
python3 video_face_swap_skill.py \
  --reference /path/to/reference.jpg \
  --video /path/to/input.mp4 \
  --preprocess-dir /path/to/preprocessed_ref/ \
  --output /path/to/output.mp4 \
  --enhance

# 指定质量档位
python3 video_face_swap_skill.py \
  --reference /path/to/reference.jpg \
  --video /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --quality hq \
  --enhance
```

## 高级选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--reference` | 参考人脸图片路径（必需） | - |
| `--video` | 输入视频路径（必需） | - |
| `--output` | 输出视频路径 | `~/Videos/face_swap/<timestamp>_<input_name>_face_swap.mp4` |
| `--target` | 目标人脸图片路径（启用目标锁定模式） | 未指定则换所有人 |
| `--target-threshold` | 目标匹配阈值 (0-1)，越高越严格 | 0.2 |
| `--match-threshold` | 多人脸匹配阈值（用于多参考图） | 0.35 |
| `--preprocess` | 自动预处理参考图（对齐+增强） | False |
| `--preprocess-dir` | 使用已预处理的参考图目录（跳过预处理步骤） | 未指定 |
| `--enhance` | 换脸后做面部修复增强 | False |
| `--restore-model` | 面部修复模型类型 | `gfpgan` |
| `--quality` | 质量档位：draft/normal/hq/best | `normal` |
| `--skip-frames` | 跳帧（每 N+1 帧处理一帧） | 0 |
| `--start-frame` | 起始帧 | 0 |
| `--end-frame` | 结束帧（-1 表示视频末尾） | -1 |
| `--device` | 计算设备 | `auto` |
| `--seed` | 随机种子（用于可复现结果） | 随机 |
| `--verbose` | 显示详细日志 | False |

## 质量档位说明

| 档位 | 检测尺寸 | 处理分辨率 | 面部增强 | 适用场景 |
|------|----------|------------|----------|----------|
| 🏃 草稿 (draft) | 自适应 (min 160) | 原始分辨率的 50% | 可选 | 快速预览、参数调试 |
| 📋 普通 (normal) | 自适应 (min 320) | 原始分辨率 | 开启 | 日常使用，平衡质量与速度 |
| ⭐ 高质量 (hq) | 自适应 (min 480) | 原始分辨率 | 开启 | 正式作品，需要更好的人脸细节 |
| 👑 终极 (best) | 自适应 (min 640) | 原始分辨率 | 开启 + 双重增强 | 最高质量输出，对细节要求极高 |

## 输出

- **路径**：`~/Videos/face_swap/{output_name}.mp4`（或自定义路径）
- **格式**：H.264 MP4
- **帧率**：保持原始视频帧率
- **分辨率**：保持原始视频分辨率（除非在低质量档位下启用了分辨率缩减）

## 工作原理

1. **参考图处理**：
   - 如果启用预处理：自动检测人脸 → 对齐裁剪 (norm_crop) → 可选 GFPGAN 增强 → 提取面部嵌入
   - 如果使用预处理目录：直接加载已有的对齐图、增强图和嵌入向量
   - 否则：直接从原始参考图提取人脸和嵌入

2. **视频处理**：
   - 按帧读取输入视频
   - 对每帧使用 insightface 检测人脸
   - 根据模式选择目标人脸：
     - 全换模式：帧内所有人脸
     - 目标锁定模式：与目标嵌入相似度最高且超过阈值的人脸
     - 多人参考图模式：按参考图顺序匹配
   - 使用 inswapper_128 执行换脸
   - 可选：对换脸结果进行面部修复增强 (GFPGAN/CodeFormer)
   - 写入输出视频

3. **资源管理**：
   - 自动调用 VRAM 调度系统 (vram_switcher) 管理显存
   - 处理完成后自动释放资源
   - 临时文件自动清理

## 依赖与环境要求

- Python 3.8+
- insightface==0.7.3
- opencv-python==4.11.0
- onnxruntime-gpu>=1.15.0
- numpy==1.26.4
- tqdm
- （可选）gfpgan（用于面部增强）

## 性能提示

1. **参考图质量**：对换脸结果影响重大。建议使用：
   - 人脸区域 ≥ 320×320 像素
   - 正脸或轻微偏转 (<30°)
   - 均匀光照，无遮挡
   - 高分辨率、无压缩 artifacts

2. **处理速度**：
   - 首次运行会加载模型到 VRAM（约 1-2GB）
   - 处理速度主要取决于视频分辨率和帧率
   - 720p30视频约 1-2 秒/帧（GTX 2080 Ti 级别）
   - 使用跳帧（如 `--skip-frames 2`）可显著加快处理速度

3. **VRAM 使用**：
   - 基础模型加载：约 1.5GB
   - 处理过程中：根据批处理大小和图像尺寸动态变化
   - 该 Skill 会自动通过 vram_switcher 管理显存，避免与其他进程冲突

## 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 未检测到人脸 | 参考图或视频中人脸太小、模糊或角度极端 | 提高参考图质量；使用预处理功能；检查是否需要调整检测尺寸 |
| 换脸后人脸不像参考图 | 参考图质量差；检测模型不一致；嵌入提取不佳 | 改善参考图质量；启用参考图预处理；确保使用一致的检测模型 |
| 处理速度慢 | 视频分辨率过高；未启用跳帧；VRAM 不足 | 降低质量档位；增加跳帧数；检查 VRAM 使用情况 |
| 输出视频花脸或畸变 | 换脸过程中的姿态极端；遮挡导致跟踪丢失 | 目标锁定模式可能有助于；考虑使用更稳定的源视频 |
| CUDA 错误 | GPU 驱动问题；显存不足；版本不兼容 | 重启以释放显存；检查 GPU 驱动；尝试使用 CPU 模式（较慢） |

## 最佳实践

1. **预处理参考图**：如果计划多次使用同一参考图，先运行预处理步骤：
   ```bash
   python3 video_face_swap_skill.py --reference /path/to/ref.jpg --preprocess
   ```
   然后在后续换脸任务中使用 `--preprocess-dir` 参数。

2. **目标锁定模式**：当只想换特定人脸时使用，可避免误换背景中的其他人脸。

3. **质量与速度平衡**：
   - 快速预览：使用 `draft` 档位 + 适当跳帧
   - 日常使用：`normal` 档位
   - 重要输出：`hq` 或 `best` 档位

4. **结果验证**：建议先处理短片段（使用 `--start-frame` 和 `--end-frame`）验证效果，再处理全片。

## 版本历史

- **v1.2.0** (2026-06-06)：新增智能参数自动选择
  - 新增 `_analyze_reference_quality()`：分析参考图质量（尺寸、角度、清晰度、光照）
  - 新增 `_analyze_video()`：分析视频信息（分辨率、帧率、时长、人脸数、场景变化）
  - 新增 `_auto_select_params()`：基于分析结果自动选择最优参数
  - 新增 `--auto` CLI 参数：一键启用智能参数推荐
  - 自动决策：质量档位、跳帧策略、增强策略、匹配阈值、预处理需求
- **v1.1.0** (2026-06-06)：关键 bug 修复 — 换脸后不像参考图的问题
  - 修复：Skill 层从缩略图（512x512）重新检测人脸，导致嵌入精度低
  - 修复方法：预处理后加载 `face_embedding.npy` 嵌入，覆盖检测到的 face 对象
  - 修复：`final_reference.png` 改为原始分辨率裁剪版（1.5x 外扩），保留检测细节
  - 新增 `_load_source_face_with_embedding()` 确保嵌入来自预处理而非缩略图
- **v1.0.0** (2026-06-05)：初始版本，封装了 `video_face_swap.py` v3.2.0 的全部功能
  - 支持自适应人脸检测
  - 参考图预处理（对齐+增强）
  - 目标锁定和多人参考图模式
  - 面部修复增强（GFPGAN/CodeFormer）
  - VRAM 自动管理通过 vram_switcher
  - 质量档位控制
  - 跳帧和起止帧控制

---
*此 Skill 自动管理显存资源，使用前无需手动调度 VRAM.*