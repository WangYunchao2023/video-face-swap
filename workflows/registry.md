# Workflow 注册表

统一管理 CLI 脚本和 ComfyUI 工作流，方便重复调用和扩展。

## 索引

| # | 名称 | 类型 | 路径 | 简介 | 状态 |
|---|------|------|------|------|------|
| 1 | 视频换脸 | CLI/ComfyUI | [`cli/video_face_swap.md`](./cli/video_face_swap.md) / [`comfyui/video_face_swap.json`](./comfyui/video_face_swap.json) | insightface inswapper 视频换脸，支持多人脸 | ✅ v2.0.0 |

（后续新增工作流从此处 append，格式保持统一）

## 新增工作流规范

1. CLI 脚本 → 在 `cli/` 下建 `.md` 文档（用法+参数+示例）
2. ComfyUI 工作流 → 在 `comfyui/` 下放 `.json`，同时复制到 `~/ComfyUI/user/default/workflows/`
3. 在此表追加一行索引
4. Git 提交 + tag

## 备注

- ComfyUI 工作流文件内置 `_meta` 字段，加载时显示说明
- CLI 脚本统一放在 workspace 根目录，注册表只维护用法文档
