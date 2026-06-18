# 🎬 Video Face Swap - 视频换脸完整套件

基于 insightface inswapper_128 模型的高质量视频换脸解决方案，包含核心引擎、OpenClaw Skill 封装和 ComfyUI 工作流。

![Version](https://img.shields.io/badge/version-v1.2.0-blue)
![Engine](https://img.shields.io/badge/engine-v3.3.0-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-v2.1.0-purple)

## ✨ 核心特性

- 🧠 **智能参数自动选择** - 分析参考图和视频，自动推荐最优参数
- 🎭 **多种换脸模式** - 全换/目标锁定/多人匹配
- 📸 **参考图预处理** - 自动对齐 + 增强，提升换脸质量
- 💪 **面部修复增强** - GFPGAN/CodeFormer 可选
- ⚡ **性能优化** - 跳帧处理、VRAM 自动管理、自适应检测
- 🎨 **可视化工作流** - ComfyUI 节点式操作
- 📦 **模型自动下载** - 首次运行自动检测并下载缺失模型

## 📦 项目结构

```
video-face-swap/
├── README.md               # 本文件
├── VERSION.md              # 版本对照表
├── .gitignore              # Git 忽略规则
├── requirements.txt        # Python 依赖
├── engine/                 # 核心引擎
│   ├── video_face_swap.py  # 底层换脸引擎 v3.3.0
│   ├── gfpgan_light.py     # GFPGAN 轻量封装 v1.0.0
│   └── model_downloader.py # 模型自动下载模块
├── skill/                  # OpenClaw Skill 封装
│   ├── SKILL.md            # Skill 文档 v1.2.0
│   └── video_face_swap_skill.py  # Skill 实现
├── comfyui/                # ComfyUI 工作流
│   ├── workflow_video_face_swap.json  # 主工作流 v2.1.0
│   ├── workflow_faceid_sdxl.json      # SDXL 人脸 ID 工作流
│   └── README.md           # ComfyUI 使用说明
├── scripts/                # 辅助脚本
│   ├── start_comfyui.sh    # ComfyUI 启动脚本
│   └── install_models.sh   # 模型手动安装脚本
├── examples/               # 示例文件
│   └── example_usage.sh    # 使用示例脚本
└── docs/                   # 详细文档
    └── README.md           # 详细使用说明
```

## 🚀 快速开始

### 方法一：自动安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/WangYunchao2023/video-face-swap.git
cd video-face-swap

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 首次运行会自动下载模型（无需手动操作）
python3 skill/video_face_swap_skill.py -r ref.jpg -v input.mp4 -o output.mp4
```

**模型自动下载说明**：
- 首次运行时会检测缺失的模型文件并自动下载
- 下载位置：
  - `~/.insightface/models/buffalo_l/inswapper_128.onnx` (~180MB)
  - `~/ComfyUI/models/facerestore_models/GFPGANv1.4.pth` (~340MB)
- 如果自动下载失败，会提示手动运行安装脚本

### 方法二：手动安装模型

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 手动下载模型
bash scripts/install_models.sh
```

### 基础使用

```bash
# 🎯 智能模式（推荐）- 自动选择最优参数
python3 skill/video_face_swap_skill.py \
  --reference examples/ref.jpg \
  --video examples/input.mp4 \
  --output output.mp4 \
  --auto

# 🔒 目标锁定模式 - 仅换指定人脸
python3 skill/video_face_swap_skill.py \
  --reference ref.jpg \
  --video input.mp4 \
  --target target.jpg \
  --output output.mp4 \
  --quality hq

# 📸 预处理参考图（推荐用于重复使用）
python3 engine/video_face_swap.py preprocess \
  --source ref.jpg \
  --output optimized_ref/
```

## 📊 智能参数系统

### 自动分析维度

| 参考图分析 | 视频分析 |
|-----------|----------|
| 人脸尺寸 (最佳 400px) | 分辨率、帧率、时长 |
| 偏转角度 (<30° 佳) | 场景中人脸数量 |
| 亮度/对比度 (0.5 最佳) | 场景变化次数 |
| 清晰度 (拉普拉斯方差) | - |

### 自动决策示例

```
[自动参数] 分析参考图和视频...
[自动参数] 参考图质量评分：0.72
[自动参数] 视频：1920x1080 @ 30fps, 180s, 2 人
[自动参数] → quality=hq (参考图高分 + 1080p)
[自动参数] → skip_frames=1 (2-5 分钟视频)
[自动参数] → enhance=true (默认启用)
[自动参数] → threshold=0.2 (2 人脸场景)
```

## 🎭 换脸模式对比

| 模式 | 说明 | CLI 参数 | 适用场景 |
|------|------|----------|----------|
| **全换模式** | 所有人脸都替换 | 默认 | 单人视频、群像全换 |
| **目标锁定** | 仅换指定人脸 | `--target` | 群像中换特定人 |
| **多人匹配** | 多参考图自动匹配 | 多参考图 | 多人分别换脸 |

## 🏷️ 版本信息

当前版本：**v1.2.0** (2026-06-06)

| 组件 | 版本 | 更新内容 |
|------|------|----------|
| 核心引擎 | v3.3.0 | 智能参数分析 + ID 保真度修复 |
| Skill 封装 | v1.2.0 | 自动参数推荐引擎 + `--auto` 参数 |
| ComfyUI 工作流 | v2.1.0 | 目标锁定支持 + Face Boost 节点 |

详细版本历史请查看 [VERSION.md](VERSION.md)

## 📖 文档指引

- [VERSION.md](VERSION.md) - 版本对照表和升级指南
- [skill/SKILL.md](skill/SKILL.md) - Skill 详细文档和 API 说明
- [comfyui/README.md](comfyui/README.md) - ComfyUI 工作流使用说明
- [docs/README.md](docs/README.md) - 详细使用教程和故障排查
- [examples/example_usage.sh](examples/example_usage.sh) - 可运行的示例脚本

## 🔧 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 未检测到人脸 | 图片太小/角度太极端 | 使用 `--preprocess`；提高参考图质量 |
| 换脸后不像 | 嵌入质量差 | 确保 v1.2.0+；使用 `--auto` |
| 处理速度慢 | 视频太长/分辨率高 | `--skip-frames 2`；降低质量档位 |
| CUDA 错误 | 显存不足 | 重启释放显存；尝试 CPU 模式 |
| 模型下载失败 | 网络问题 | 手动运行 `bash scripts/install_models.sh` |

## 📝 Git 使用

```bash
# 查看项目提交历史
git log --oneline

# 查看特定版本
git show v1.2.0

# 标签列表
git tag -l
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**维护者**: Cortana-shadow  
**最后更新**: 2026-06-06  
**项目位置**: `~/.openclaw/workspaces/video-face-swap/`  
**GitHub**: https://github.com/WangYunchao2023/video-face-swap