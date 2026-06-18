---
name: image-generation
description: >-
  AI 图片生成 Skill v2.0。支持自然语言触发本地 ComfyUI 生图（SD3/Flux），
  自动分析场景选择最佳模型和参数，支持反馈调参。
  触发词：生图、生成图片、AI图片、图片生成、画图、SD3、Flux、ComfyUI、
  老照片、复古、风景、人像、肖像
version: 2.0.0
---

# ImageGen Skill v2.0.0

## 架构

```
用户自然语言描述
    ↓
analyze_request()  ← 场景分析器（主体/风格/质量分类）
    ↓
模型选择（SD3 / Flux Schnell / Flux Dev）
    ↓
参数优化（步数/CFG/分辨率/负面词）
    ↓
提示词增强
    ↓
┌─ ComfyUI 运行中？── 是 ─→ 本地 ComfyUI 执行（自动 VRAM 调度）
└─ 否 ─────────────────→ 降级到 image_generate 云工具
    ↓
返回图片路径
    ↓
效果不好？ ──→ retry_with_feedback() ──→ 调参/换模型重试
```

## 自然语言触发

最推荐的使用方式：直接用中文/英文描述你要的图片。

### 示例

```python
from image_gen import generate

# 最简单的方式：一句话描述
result = generate(description="一位80岁老奶奶坐在院子里晒太阳，复古老照片风格")
# → 自动选择 SD3 Medium + 胶片刻画 + 高步数

result = generate(description="长城日出，云雾缭绕，风景照片")
# → 自动选择 Flux Schnell + 风景参数

result = generate(description="一只橘猫趴在窗台上，写实风格，4k")
# → 自动选择 Flux Schnell + 高质量
```

### 在 OpenClaw 对话中触发

用户说：
> "帮我生成一张80年代农村老奶奶在院子里的照片"

→ Skill 自动匹配 → `analyze_request()` 分析：
- 主体类型: portrait（人物肖像）
- 风格: vintage/retro（老照片）
- 质量: medium
- 选模型: SD3 Medium（人物+复古场景最优）
- 参数: steps=40, cfg=5.0, 768×1024
- 增强提示词: 加入 Kodak胶片、自然光、皱纹细节等
→ ComfyUI 执行 → 返回图片

用户说：
> "手画得太奇怪了"

→ `retry_with_feedback()` 分析到"手"关键词：
- 换用 SD3 Medium（人物细节更好）
- 增加手部负面词
- 重新生成 `_retry` 版本

## 模型选择逻辑

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 人物肖像、特写 | **SD3 Medium** (30-50步) | 人物细节、CLIP编码好 |
| 复古老照片 | **SD3 Medium** (35-40步) | 胶片质感最佳 |
| 多人、全身照 | **Flux Schnell FP8** (4-8步) | 构图好、速度快 |
| 风景、建筑 | **Flux Schnell FP8** (4-8步) | 场景渲染好 |
| 精细渲染/高质量 | **Flux Dev FP8** (25-40步) | 最大质量 |
| 快速草稿 | **Flux Schnell FP8** (4步) | 最快 |
| 动物、静物 | **Flux Schnell FP8** (4-8步) | 通用场景 |

### 模型能力矩阵

```python
MODEL_CAPABILITIES = {
    "sd3":           # 人物近景/复古 → 10GB VRAM
    "flux_schnell":  # 风景/多人/通用 → 9GB VRAM
    "flux_dev":      # 高质量精细 → 12GB VRAM
}
```

### 风格自动增强

| 检测到的风格 | 自动添加的关键词 |
|-------------|----------------|
| 复古/老照片 | vintage, film grain, Kodak, faded colors |
| 赛博朋克 | neon lights, dark, futuristic |
| 写实 | photorealistic, detailed, DSLR |
| 动漫 | anime style, cel shaded |
| 水彩 | watercolor, soft colors |
| 电影感 | cinematic lighting, film still |
| 极简 | minimalist, clean |

### 参数自适应

- **步数**: 低质量→快 (4-20步) / 中等→默认 / 高质量→满 (40-50步)
- **CFG**: SD3 4-7 / Flux 1-3.5
- **分辨率**: 人像 768×1024 / 风景 1024×768 / 方图 1024×1024
- **负面词**: 自动根据主体补充（人像加手部/皮肤负面词）

## CLI 使用

```bash
# 一句话描述
python3 image_gen.py --description "80年代中国农村老奶奶，自然光线，胶片质感"

# 只分析不生成（查看模型选择）
python3 image_gen.py --description "长城风景照" --dry-run

# 强制指定模型
python3 image_gen.py --description "夕阳下的海滩" --model flux_schnell

# 覆盖参数
python3 image_gen.py --description "猫咪特写" --steps 50 --seed 42

# 旧式拆分参数（兼容）
python3 image_gen.py --subject "卡通机器人" --style "Pixar风格" --scene "未来城市"

# 仅本地，不降级到云
python3 image_gen.py --description "测试" --no-cloud
```

## Python API

### generate()

```python
from image_gen import generate, analyze_request, retry_with_feedback

# 一、智能生成
result = generate(
    description="自然语言描述",     # 最推荐的方式
    output_name="my_image",        # 可选
    output_dir="/path/to/dir",     # 可选
    prefer_local=True,             # 优先本地 ComfyUI
    fallback_to_cloud=True,        # 本地不可用时降级到云
)

# 返回:
# {"success": True, "output_file": "/path/to/img.png",
#  "analysis": {...}, "model_used": "sd3"}

# 二、分析请求（不生成）
analysis = analyze_request("一位老奶奶，复古老照片")
# → {"model_key": "sd3", "params": {...}, "reason": "...", ...}

# 三、反馈调参重试
result2 = retry_with_feedback(prev_result, "手画得太奇怪了")
# → 自动换 SD3 + 手部修复 重新生成
```

### analyze_request()

```python
analysis = analyze_request(description)
# → {
#     "model_key": "sd3" | "flux_schnell" | "flux_dev",
#     "params": {"steps": 40, "cfg": 5.0, "width": 768, "height": 1024, ...},
#     "reason": "老照片/复古人像 → SD3 Medium",
#     "subject_type": "portrait",
#     "style_hints": ["vintage"],
#     "quality_tier": "medium",
#     "enhanced_prompt": "original + 增强词",
# }
```

### retry_with_feedback()

| 反馈关键词 | 自动处理 |
|-----------|---------|
| "模糊"/"blurry" | 增加步数 +10 |
| "手"/"hand" | 换 SD3 + 手部负面词 |
| "人脸"/"face" | 换 SD3 Medium |
| "构图"/"composition" | Flux Schnell 8步 |
| "颜色太淡"/"太艳" | 调 CFG ±1.0 |

## VRAM 管理

- 自动通过 `VMgr` 调度：生图前 evict LLM，生图后恢复
- 2080Ti 22GB：SD3 ~10G / Flux Schnell ~9G / Flux Dev ~12G
- 查看状态：`python3 ~/.openclaw/scripts/vram_manager.py status`
- 强制抢占：`python3 ~/.openclaw/scripts/vram_manager.py acquire`

## 输出

- 默认保存到: `~/Pictures/image-gen/`（可被 PICTURE_DIR 或 output_dir 覆盖）
- 文件名: `{model_key}_{timestamp}.png` 或自定义
- 重试文件: `{original}_retry.png`

## 依赖

- Python 3.10+（`requests`）
- ComfyUI 运行在 `http://127.0.0.1:8188`
- `comfy_client.py`（ComfyUI API 封装，位于 `~/.openclaw/scripts/`）
- `vram_manager.py`（VRAM 调度，位于 `~/.openclaw/scripts/`）
- 模型文件位于 `~/ComfyUI/models/`
- Ollama `qwen2.5:14b` 常驻显存

## 状态

- ✅ 自然语言场景分析
- ✅ 模型自动选择（SD3/Flux Schnell/Flux Dev）
- ✅ 参数自适应（步数/CFG/分辨率/负面词）
- ✅ 提示词增强
- ✅ ComfyUI 本地执行 + VRAM 调度
- ✅ 云模型降级方案
- ✅ 反馈循环（调参/换模型重试）
- ✅ CLI + API 双模式
- ✅ 兼容旧版 subject/style/scene/extra 参数

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 2.0.0 | 2026-05-19 | 自然语言分析、模型自动选择、ComfyUI 本地执行、反馈循环、VRAM 调度 |
| 1.0.0 | - | 初始版本（仅 image_generate 云工具） |
