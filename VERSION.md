# Video Face Swap - 版本对照表

## 当前版本 (2026-06-06)

| 组件 | 版本 | 路径 | 发布日期 |
|------|------|------|----------|
| **核心引擎** | v3.3.0 | `engine/video_face_swap.py` | 2026-06-06 |
| **Skill 封装** | v1.2.0 | `skill/video_face_swap_skill.py` | 2026-06-06 |
| **ComfyUI 工作流** | v2.1.0 | `comfyui/workflow_video_face_swap.json` | 2026-06-06 |
| **GFPGAN 封装** | v1.0.0 | `engine/gfpgan_light.py` | 2026-06-06 |

## 版本兼容性

| Skill 版本 | 引擎版本 | ComfyUI 版本 | 主要特性 |
|-----------|---------|-------------|----------|
| v1.2.0 | ≥ v3.3.0 | ≥ v2.1.0 | 智能参数自动选择 |
| v1.1.0 | ≥ v3.2.0 | ≥ v2.0.0 | 目标锁定模式 |
| v1.0.0 | ≥ v3.0.0 | ≥ v1.0.0 | 初始版本 |

## 各版本详细说明

### v3.3.0 / v1.2.0 / v2.1.0 (2026-06-06)

#### 核心引擎 v3.3.0
- 🎯 **智能参数分析**：参考图质量评分 + 视频信息分析
- 🐛 **ID 保真度修复**：
  - norm_crop 从 256px → 512px
  - 从 `face_embedding.npy` 加载高质量嵌入
  - `final_reference.png` 改为原始分辨率裁剪（1.5x 外扩）
- 🔧 **自适应检测**：小图自动放大，多 det_size 重试
- 🔄 **模型回退**：antelopev2 失败 → buffalo_l

#### Skill v1.2.0
- 🤖 `_analyze_reference_quality()`: 参考图 5 维分析
- 📊 `_analyze_video()`: 视频 4 维分析
- 🧠 `_auto_select_params()`: 智能参数推荐引擎
- 💻 `--auto` CLI 参数：一键启用最优参数

#### ComfyUI v2.1.0
- 👤 目标人脸输入节点（目标锁定模式）
- 💪 Face Boost 节点（增强身份保留）
- 📝 更新的使用说明文本

---

### v3.2.0 / v1.1.0 / v2.0.0 (2026-06-05)

#### 核心引擎 v3.2.0
- 🔒 **目标锁定模式**：仅换指定人脸
- 📐 **自适应人脸检测**：根据图片尺寸选择 det_size
- 🔄 **模型回退机制**

#### Skill v1.1.0
- 🐛 **关键 Bug 修复**：嵌入质量优化

#### ComfyUI v2.0.0
- 🎭 目标锁定模式支持
- 🎨 优化的节点布局

---

### v3.0.0 / v1.0.0 / v1.0.0 (2026-05-28)

- 🎉 **初始版本**
- ✅ 基础换脸功能
- 🎨 ComfyUI 工作流集成

---

## Git 标签

```bash
# 查看所有标签
git tag -l 'video-face-swap/*'

# 查看特定版本
git show video-face-swap/v1.2.0
```

## 升级指南

### v1.1.0 → v1.2.0

1. **备份现有文件**
2. **更新 Skill 文件**
   ```bash
   cp video-face-swap/skill/video_face_swap_skill.py \
      ~/.openclaw/skills/video-face-swap/
   ```
3. **更新引擎文件**
   ```bash
   cp video-face-swap/engine/video_face_swap.py \
      ~/.openclaw/workspace-cortana-shadow/
   ```
4. **测试智能参数功能**
   ```bash
   python3 video_face_swap_skill.py -r ref.jpg -v input.mp4 --auto
   ```

---

**维护者**: Cortana-shadow  
**最后更新**: 2026-06-06
