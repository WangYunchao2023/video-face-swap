# 视频换脸工具 - 详细文档

## 目录结构

```
video-face-swap/
├── engine/           # 核心换脸引擎
│   ├── video_face_swap.py      # 底层引擎 v3.3.0
│   └── gfpgan_light.py         # GFPGAN 轻量封装
├── skill/            # OpenClaw Skill 封装
│   ├── SKILL.md                # Skill 说明文档
│   └── video_face_swap_skill.py # Skill 实现 v1.2.0
├── comfyui/          # ComfyUI 工作流
│   ├── workflow_video_face_swap.json
│   └── README.md
├── scripts/          # 辅助脚本
│   └── start_comfyui.sh
├── examples/         # 示例文件和脚本
└── docs/             # 详细文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install insightface==0.7.3 opencv-python==4.11.0 \
            onnxruntime-gpu numpy==1.26.4 tqdm

# 下载模型文件
# inswapper_128.onnx → ~/.insightface/models/buffalo_l/
# GFPGANv1.4.pth → ~/ComfyUI/models/facerestore_models/
```

### 2. 基础使用

```bash
cd ~/.openclaw/workspace-cortana-shadow/video-face-swap

# 智能模式（推荐）
python3 skill/video_face_swap_skill.py \
  --reference examples/ref.jpg \
  --video examples/input.mp4 \
  --auto

# 查看自动选择的参数
python3 skill/video_face_swap_skill.py \
  --reference examples/ref.jpg \
  --video examples/input.mp4 \
  --auto --verbose
```

### 3. 高级功能

#### 参考图预处理
```bash
python3 engine/video_face_swap.py preprocess \
  --source ref.jpg \
  --output optimized_ref/
```

#### 目标锁定换脸
```bash
python3 skill/video_face_swap_skill.py \
  --reference ref.jpg \
  --video input.mp4 \
  --target target.jpg \
  --output output.mp4
```

#### 质量档位控制
```bash
# 草稿模式（快速预览）
python3 skill/video_face_swap_skill.py \
  -r ref.jpg -v input.mp4 --quality draft

# 高质量模式
python3 skill/video_face_swap_skill.py \
  -r ref.jpg -v input.mp4 --quality hq
```

## 智能参数系统

### 参考图质量分析
- **尺寸评分**: 人脸宽度/高度，最佳 400px
- **角度评分**: 偏转角度，0°最佳，>45°扣分
- **亮度评分**: 0.5 最佳，过暗/过亮扣分
- **清晰度评分**: 拉普拉斯方差检测模糊

### 视频信息分析
- 分辨率、帧率、时长
- 场景中最大人脸数
- 场景变化次数（亮度突变）

### 自动决策逻辑

| 条件 | 决策 |
|------|------|
| 参考图评分 < 0.4 | → normal 档位 |
| 视频 ≤ 720p | → normal 档位 |
| 参考图 ≥ 0.7 + 1080p | → hq 档位 |
| 视频 > 5 分钟 | → skip_frames=2 |
| 视频 2-5 分钟 | → skip_frames=1 |
| 人脸 < 150px | → 禁用增强 |
| 角度 > 25° 或模糊 | → 启用增强 |
| ≥3 人脸 | → threshold=0.25 |

## 故障排查

### 未检测到人脸
- 提高参考图质量
- 使用 `--preprocess` 自动优化
- 检查光线和角度

### 换脸后不像
- 确保使用 v1.2.0+ 版本
- 使用 `--auto` 智能参数
- 预处理参考图

### 处理速度慢
- 使用 `--skip-frames 2` 跳帧
- 降低质量档位 `--quality draft`
- 检查 VRAM 使用情况

## 相关文件

- [VERSION.md](../VERSION.md) - 版本对照表
- [README.md](../README.md) - 项目总览
- [skill/SKILL.md](../skill/SKILL.md) - Skill 详细文档
- [comfyui/README.md](../comfyui/README.md) - ComfyUI 使用说明
