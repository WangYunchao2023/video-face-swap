# ComfyUI 视频换脸工作流版本管理

## 当前版本

| 组件 | 版本 | 日期 | 说明 |
|------|------|------|------|
| **workflow_video_face_swap.json** | v2.1.0 | 2026-06-06 | 支持目标锁定模式 + 参数优化 |
| **video_face_swap.py** | v3.3.0 | 2026-06-06 | 智能参数选择 + ID 保真度改进 |
| **video_face_swap_skill.py** | v1.2.0 | 2026-06-06 | 智能参数自动选择 |

## 工作流特性

### v2.1.0 更新 (2026-06-06)
- ✅ 支持目标锁定换脸模式（仅换指定人脸）
- ✅ 集成参考图预处理节点（对齐 + 增强）
- ✅ 可切换 GFPGAN/CodeFormer 面部修复
- ✅ 支持跳帧处理长视频
- ✅ 质量档位控制（draft/normal/hq/best）

### 节点组成
1. **VHS_LoadVideoPath** - 加载输入视频
2. **LoadImage** - 参考人脸图片
3. **LoadImage** - 目标人脸图片（可选，用于目标锁定）
4. **ReActorFaceSwap** - 核心换脸节点
5. **FaceBoost** - 面部增强（可选）
6. **VHS_VideoCombine** - 输出视频合成

## 与 Python 版本的对应关系

| 功能 | Python CLI | ComfyUI 节点 |
|------|-----------|-------------|
| 参考图预处理 | `--preprocess` | 预处理子工作流 |
| 目标锁定 | `--target` | 目标人脸输入 + ReActor 参数 |
| 质量档位 | `--quality` | ReActor detection 尺寸 |
| 跳帧 | `--skip-frames` | Video Combine 帧选择 |
| 面部修复 | `--enhance` | FaceBoost 节点 |

## 使用示例

### 基础换脸（所有人）
```bash
# Python
python3 video_face_swap_skill.py -r ref.jpg -v input.mp4 -o out.mp4 --auto

# ComfyUI: 加载 workflow_video_face_swap.json
# - 输入视频：input.mp4
# - 参考人脸：ref.jpg
# - 目标人脸：留空（全换模式）
```

### 目标锁定换脸
```bash
# Python
python3 video_face_swap_skill.py -r ref.jpg -v input.mp4 -t target.jpg -o out.mp4 --auto

# ComfyUI:
# - 输入视频：input.mp4
# - 参考人脸：ref.jpg
# - 目标人脸：target.jpg
# - ReActor: 启用性别/年龄过滤（可选）
```

## 依赖项

### ComfyUI 管理器插件
- **ComfyUI-VideoHelperSuite** (VHS) - 视频加载/输出
- **ComfyUI_Reacto r** - 换脸核心节点
- **ComfyUI-FaceAnalysis** - 人脸检测/分析

### 模型文件
- `~/.insightface/models/buffalo_l/inswapper_128.onnx`
- `~/ComfyUI/models/facerestore_models/GFPGANv1.4.pth`

## 性能参考

| 视频规格 | Python (GTX 2080 Ti) | ComfyUI |
|----------|---------------------|---------|
| 720p 30fps, 1 分钟 | ~2-3 分钟 | ~3-4 分钟 |
| 1080p 30fps, 1 分钟 | ~4-5 分钟 | ~5-6 分钟 |
| 1080p 30fps, 5 分钟 (skip=1) | ~8-10 分钟 | ~10-12 分钟 |

---

*最后更新：2026-06-06*
