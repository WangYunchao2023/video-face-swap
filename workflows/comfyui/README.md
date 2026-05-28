# video_face_swap — ComfyUI 视频换脸工作流

## 节点结构

```
VHS_LoadVideoPath ──IMAGE──→ ReActorFaceSwap ──IMAGE──→ VHS_VideoCombine
       │                              ↑                     ↑
       │                     LoadImage │            audio────┘
       │                      (参考人脸) │                      
       └──VHS_VIDEOINFO──→ VHS_VideoInfoLoaded (查看帧率)
```

## 模型依赖

| 模型 | 路径 | 大小 | 来源 |
|------|------|------|------|
| inswapper_128.onnx | `~/ComfyUI/models/insightface/inswapper_128.onnx` | 528MB | insightface buffalo_l → 软链到此 |

模型已通过软链就绪：`~/.insightface/models/buffalo_l/inswapper_128.onnx` → `~/ComfyUI/models/insightface/`

## 如何运行

1. 启动 ComfyUI
2. 加载此工作流（或从 File→Load 选择 `workflow_video_face_swap`）
3. 设置参数：

   | 节点 | 必须设置 | 说明 |
   |------|---------|------|
   | VHS_LoadVideoPath | `video` | 填写目标视频的完整路径 |
   | LoadImage | `image` | 上传参考人脸图 |
   | VHS_VideoCombine | `frame_rate` | 改为视频源帧率（查看 VideoInfoLoaded 节点输出） |

4. 点击"添加到队列"运行

## 已知限制

- **单参考图**：ReActor 节点一次只接受一张参考图
- 如需换多人脸，在工作流中串联多个 ReActorFaceSwap 节点（第一个换完 → 第二个换另一个人）
- 或者使用 CLI 工具（原生支持 `--sources` 多人脸自动匹配）

## 参数调优

| 节点参数 | 可选值 | 推荐 |
|---------|--------|------|
| swap_model | inswapper_128.onnx / 其他 | 默认即可 |
| facedetection | retinaface_resnet50 / yolov8_l | retinaface 更稳 |
| face_restore_model | none / codeformer / gfpgan | 由其他节点处理 |
| input_faces_index | "0" = 第一张脸; "0,1" = 多张 | 视频中的人脸序号 |
| source_faces_index | "0" | 参考图中默认取最大脸 |

## 版本历史

- v1.0.0 (2026-05-28): 基础视频换脸工作流
