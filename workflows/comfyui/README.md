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

| 模型 | 路径 | 大小 |
|------|------|------|
| inswapper_128.onnx | `~/ComfyUI/models/insightface/inswapper_128.onnx` | 528MB |

模型已通过软链就绪：`~/.insightface/models/buffalo_l/inswapper_128.onnx` → `~/ComfyUI/models/insightface/`

## 如何运行

1. 启动 ComfyUI
2. 加载此工作流
3. 设置参数：

   | 节点 | 必须设置 | 说明 |
   |------|---------|------|
   | VHS_LoadVideoPath | `video` | 填写目标视频的完整路径 |
   | LoadImage | `image` | 上传参考人脸图（换成这个人）|
   | VHS_VideoCombine | `frame_rate` | 改为视频源帧率 |

4. 添加到队列运行

## 已知限制

- **单参考图**：ReActor 节点一次只接受一张参考图
- 多人脸需串联多个 ReActorFaceSwap 节点
- 如需目标锁定（只换特定人的脸），建议使用 CLI 脚本：
  ```bash
  python3 video_face_swap.py swap \
    --source ref.jpg \
    --target target_face.jpg \
    --video input.mp4 \
    -o output.mp4
  ```

## 参数参考

| 节点参数 | 可选值 |
|---------|--------|
| swap_model | inswapper_128.onnx |
| facedetection | retinaface_resnet50 / yolov8_l |
| face_restore_model | none / codeformer / gfpgan |
| input_faces_index | "0" = 第一人脸; "0,1" = 多张 |

## CLI 对应工具

CLI 脚本位置：`~/openclaw/workspace-cortana-shadow/video_face_swap.py`

### 目标锁定模式（推荐，只换特定人）
```bash
python3 video_face_swap.py swap \
  -s reference_face.jpeg \     # 换成这张脸
  -t target_face.png \         # 换掉这个人（截图）
  -v input_video.mp4 \
  -o output.mp4 \
  --target-threshold 0.2       # 匹配严格度
```

### 强换模式（所有人脸都换）
```bash
python3 video_face_swap.py swap \
  -s reference_face.jpeg \
  -v input_video.mp4 \
  -o output.mp4
```

### 额外参数
- `--skip N`：跳帧加速 (每 N+1 帧处理一帧)
- `--start N --end N`：指定帧范围
- `--enhance`：GFPGAN 面部增强
- `--device cuda`：GPU 加速

## 版本历史

- v1.0.0 (2026-05-28): 基础视频换脸
- v2.0.0 (2026-05-28): 增加 --target 目标锁定模式