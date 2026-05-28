#!/usr/bin/env python3
"""
视频换脸工具 v2.0.0 — Video Face Swap Pipeline

基于 insightface inswapper_128 + OpenCV 的高效视频换脸方案。
支持：多人脸同时换脸（多参考图自动匹配）、批量帧处理、跳帧加速、GFPGAN 增强。

v2.0.0 新增：
  - 多参考图换脸：传入多张参考图，自动按人脸嵌入相似度匹配
  - 可同时替换视频中多个不同人物的脸
  - --sources 参数接收多个参考图路径
  - 嵌入相似度匹配 + 阈值过滤

依赖：insightface==0.7.3, opencv-python, onnxruntime-gpu, numpy, tqdm
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# ─── 核心换脸引擎 ─────────────────────────────────────────────

class VideoFaceSwapper:
    """视频换脸引擎（支持多人脸）"""

    def __init__(
        self,
        det_name: str = "antelopev2",
        det_size: Tuple[int, int] = (640, 640),
        device: str = "auto",   # "auto" | "cpu" | "cuda"
        enable_restoration: bool = False,
        match_threshold: float = 0.35,
    ):
        self.device = device
        self.enable_restoration = enable_restoration
        self.match_threshold = match_threshold
        self._providers = self._resolve_providers()
        self._load_models(det_name, det_size)

    def _resolve_providers(self) -> List:
        """选择 ONNX Runtime provider"""
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cpu":
            return ["CPUExecutionProvider"]
        else:  # auto
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except:
                pass
            return ["CPUExecutionProvider"]

    def _load_models(self, det_name: str, det_size: Tuple[int, int]):
        """加载 insightface 模型"""
        from insightface.app import FaceAnalysis
        from insightface.model_zoo.inswapper import INSwapper
        import onnxruntime

        log_info("[加载] 初始化 FaceAnalysis (%s)..." % det_name)
        self.app = FaceAnalysis(name=det_name, providers=self._providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        log_info("[加载] FaceAnalysis 就绪")

        # 查找 inswapper 模型
        model_path = os.path.expanduser("~/.insightface/models/buffalo_l/inswapper_128.onnx")
        alt_paths = [
            model_path,
            os.path.expanduser("~/.insightface/models/inswapper_128.onnx"),
            os.path.join(os.path.dirname(__file__), "models", "inswapper_128.onnx"),
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                found = True
                break

        if not found:
            raise FileNotFoundError(
                "找不到 inswapper_128.onnx！\n"
                "请下载后放到 ~/.insightface/models/buffalo_l/inswapper_128.onnx"
            )

        log_info("[加载] 加载 inswapper (%s, %.1f MB)..." % (
            model_path, os.path.getsize(model_path) / 1024 / 1024))

        # 创建 ONNX 会话 (避免 INSwapper 默认构造)
        sess = onnxruntime.InferenceSession(model_path, providers=self._providers)
        self.swapper = INSwapper(model_file=model_path, session=sess)
        log_info("[加载] inswapper 就绪 (输入尺寸 %s)" % str(self.swapper.input_size))

    # ── 参考人脸提取 ──────────────────────────────────────────

    def get_source_face(self, img: np.ndarray, label: str = ""):
        """从单张参考图像提取面积最大的人脸"""
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("参考图像%s未检测到人脸" % (" " + label if label else ""))
        largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3] - f.bbox[0] * f.bbox[1])
        log_info("[检测] 参考人脸%s: bbox=%s, det_score=%.3f" %
                 (" " + label if label else "",
                  largest.bbox.astype(int).tolist(), largest.det_score))
        return largest

    def load_source_faces(self, source_paths: List[str]) -> List:
        """
        加载多张参考图，提取每张图的人脸

        Args:
            source_paths: 参考图片路径列表

        Returns:
            source_faces: 人脸列表，顺序与输入一致
        """
        source_faces = []
        for i, path in enumerate(source_paths):
            img = cv2.imread(path)
            if img is None:
                raise IOError("无法读取参考图: %s" % path)
            label = "[%d] %s" % (i, os.path.basename(path))
            face = self.get_source_face(img, label=label)
            source_faces.append(face)
        log_info("[检测] 共加载 %d 个参考人脸" % len(source_faces))
        return source_faces

    # ── 人脸匹配 ──────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算归一化嵌入的余弦相似度"""
        return float(np.dot(emb1, emb2))

    def match_faces(
        self,
        target_faces: List,
        source_faces: List,
    ) -> List[Tuple]:
        """
        将检测到的人脸与参考人脸按嵌入相似度匹配

        Args:
            target_faces: 视频帧中检测到的人脸列表
            source_faces: 参考人脸列表

        Returns:
            matches: [(target_face, source_face), ...] 已匹配的对
                     未匹配到的 target_faces 将被跳过
        """
        matches = []

        # 为每个 source_face 找最佳匹配的 target
        used_targets = set()
        for si, src in enumerate(source_faces):
            best_score = -1.0
            best_ti = -1
            for ti, tgt in enumerate(target_faces):
                if ti in used_targets:
                    continue
                score = self._cosine_similarity(src.normed_embedding, tgt.normed_embedding)
                if score > best_score:
                    best_score = score
                    best_ti = ti

            if best_ti >= 0 and best_score >= self.match_threshold:
                used_targets.add(best_ti)
                matches.append((target_faces[best_ti], src, best_score))
                log_info("[匹配] 参考[%d] → 目标[%d], 相似度=%.3f" %
                         (si, best_ti, best_score))
            elif best_ti >= 0:
                log_info("[匹配] 参考[%d] → 目标[%d] 相似度=%.3f (低于阈值 %.2f, 跳过)" %
                         (si, best_ti, best_score, self.match_threshold))

        # 输出未匹配的参考图信息
        matched_srcs = set(id(m[1]) for m in matches)
        for si, src in enumerate(source_faces):
            if id(src) not in matched_srcs:
                log_info("[匹配] 参考[%d] 未匹配到视频中的人脸" % si)

        return matches

    # ── 单帧换脸 ──────────────────────────────────────────────

    def swap_face(self, frame: np.ndarray, source_face, target_face=None) -> np.ndarray:
        """
        对一帧图像执行单人脸换脸

        Args:
            frame: BGR 图像 (H, W, 3)
            source_face: 源人脸 (参考)
            target_face: 目标人脸 (若 None 则自动检测面积最大的)

        Returns:
            换脸后的 BGR 图像
        """
        if target_face is None:
            faces = self.app.get(frame)
            if len(faces) == 0:
                return frame
            target_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3] - f.bbox[0] * f.bbox[1])

        result = self.swapper.get(frame, target_face, source_face, paste_back=True)

        if self.enable_restoration:
            result = self._enhance_face(result, target_face)

        return result

    def swap_face_multi(self, frame: np.ndarray, source_faces: List) -> np.ndarray:
        """
        对一帧图像执行多人脸换脸

        检测帧中所有人脸 → 按嵌入相似度匹配参考人脸 → 逐对执行换脸

        Args:
            frame: BGR 图像 (H, W, 3)
            source_faces: 参考人脸列表（来自 load_source_faces）

        Returns:
            换脸后的 BGR 图像
        """
        # 检测帧中人脸
        faces = self.app.get(frame)
        if len(faces) == 0:
            return frame  # 无脸，跳过

        # 匹配
        matches = self.match_faces(faces, source_faces)
        if not matches:
            return frame

        # 按面积倒序换脸（先换大脸，减少重叠干扰）
        matches.sort(key=lambda m: (
            m[0].bbox[2] - m[0].bbox[0]) * (m[0].bbox[3] - m[0].bbox[1]),
            reverse=True)

        result = frame.copy()
        for tgt, src, score in matches:
            result = self.swapper.get(result, tgt, src, paste_back=True)

        # 单帧仅做一次增强（增强整个结果，而非逐脸）
        if self.enable_restoration:
            try:
                from gfpgan import GFPGANer
                if not hasattr(self, '_gfpgan'):
                    log_info("[增强] 初始化 GFPGAN...")
                    self._gfpgan = GFPGANer(
                        model_path=os.path.expanduser(
                            "~/ComfyUI/models/face_restoration/GFPGANv1.4.pth"),
                        upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
                _, _, result = self._gfpgan.enhance(
                    result, has_aligned=False, only_center_face=False, paste_back=True)
            except Exception as e:
                log_info("[增强] 跳过: %s" % e)

        return result

    def _enhance_face(self, img: np.ndarray, face) -> np.ndarray:
        """GFPGAN 面部增强（单脸版，保留兼容）"""
        try:
            from gfpgan import GFPGANer
            if not hasattr(self, '_gfpgan'):
                log_info("[增强] 初始化 GFPGAN...")
                self._gfpgan = GFPGANer(
                    model_path=os.path.expanduser(
                        "~/ComfyUI/models/face_restoration/GFPGANv1.4.pth"),
                    upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            _, _, result = self._gfpgan.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True)
            return result
        except Exception as e:
            log_info("[增强] 跳过: %s" % e)
            return img

    # ── 视频处理 ──────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        source_faces: List,
        output_path: str,
        max_faces: int = 0,
        skip_frames: int = 0,
        start_frame: int = 0,
        end_frame: int = -1,
        progress_callback=None,
    ) -> str:
        """
        处理完整视频（支持多人脸）

        Args:
            video_path: 输入视频路径
            source_faces: 参考人脸列表（来自 load_source_faces）
            output_path: 输出视频路径
            max_faces: 每帧最多处理的人脸数 (0=不限制)
            skip_frames: 跳帧数 (每 N 帧处理一帧)
            start_frame: 起始帧
            end_frame: 结束帧 (-1=结尾)
            progress_callback: 进度回调 func(current, total)

        Returns:
            输出视频路径
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("无法打开视频: %s" % video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        end = end_frame if end_frame > 0 else total_frames

        # 准备输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed = 0
        skipped_count = 0
        no_face_frames = 0
        total_swaps = 0

        log_info("[处理] 开始逐帧多人脸换脸...")
        log_info("  视频: %s | %d×%d | %.2f fps | %d 帧 | %d 个参考人脸" % (
            os.path.basename(video_path), width, height, fps, total_frames,
            len(source_faces)))

        frame_idx = 0
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧
            if frame_idx < start_frame:
                out.write(frame)
                frame_idx += 1
                continue

            if skip_frames > 0 and (frame_idx - start_frame) % (skip_frames + 1) != 0:
                out.write(frame)
                skipped_count += 1
                frame_idx += 1
                continue

            # 执行多人脸换脸
            try:
                swapped = self.swap_face_multi(frame, source_faces)
                if not np.array_equal(swapped, frame):
                    total_swaps += 1
            except Exception as e:
                log_info("[警告] 帧 %d 处理失败: %s" % (frame_idx, e))
                swapped = frame
                no_face_frames += 1

            out.write(swapped)
            processed += 1

            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)

            frame_idx += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        log_info("[完成] 处理了 %d 帧 (%d 跳过, %d 无脸, %d 帧有换脸)" %
                 (processed, skipped_count, no_face_frames, total_swaps))
        log_info("  输出: %s" % output_path)

        return output_path


# ─── 工具函数 ─────────────────────────────────────────────────

def log_info(msg: str):
    print("[视频换脸] %s" % msg)
    sys.stdout.flush()


def extract_frames(video_path: str, output_dir: str, fps: float = 0):
    """
    提取视频帧到图片序列（用于 ComfyUI 逐帧处理）

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        fps: 提取帧率 (0=使用视频原始帧率)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("无法打开视频: %s" % video_path)

    os.makedirs(output_dir, exist_ok=True)
    target_fps = fps if fps > 0 else cap.get(cv2.CAP_PROP_FPS)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / target_fps))

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, "frame_%06d.png" % saved)
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    log_info("[提取] 从 %s 提取了 %d 帧到 %s" % (
        os.path.basename(video_path), saved, output_dir))
    return output_dir


def assemble_video(frames_dir: str, output_path: str, fps: float = 30.0):
    """
    从图片序列组装视频

    Args:
        frames_dir: 帧目录 (frame_000000.png 格式)
        output_path: 输出视频路径
        fps: 输出帧率
    """
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not files:
        raise ValueError("目录中没有图片: %s" % frames_dir)

    first = cv2.imread(os.path.join(frames_dir, files[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        if img is not None:
            out.write(img)

    out.release()
    log_info("[组装] 已生成 %s (%d 帧, %.2f fps)" % (output_path, len(files), fps))
    return output_path


# ─── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="视频换脸工具 v2.0 — Video Face Swap (多人脸)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 单人换脸 (兼容 v1)
  %(prog)s swap --source ref1.jpg --video target.mp4 -o result.mp4

  # 多人换脸 (自动按相似度匹配)
  %(prog)s swap --sources ref_A.jpg ref_B.jpg --video group.mp4 -o result.mp4

  # 跳帧加速 + 调低匹配阈值
  %(prog)s swap --sources ref.jpg --video target.mp4 -o result.mp4 --skip 1 --threshold 0.3

  # 提取帧 → ComfyUI 处理 → 组装
  %(prog)s extract --video target.mp4 --frames-dir ./frames --fps 30
  %(prog)s assemble --frames-dir ./frames_out --output result.mp4 --fps 30
        """)

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── 主命令: 换脸 ──
    swap_parser = subparsers.add_parser("swap", help="执行视频换脸")
    # 兼容 v1 的 --source 单图模式
    swap_parser.add_argument("-s", "--source", default=None, help="源参考人脸图片路径（单图模式，与--sources二选一）")
    # v2 多图模式
    swap_parser.add_argument("--sources", nargs="+", default=None,
                             help="多张参考人脸图片路径（与--source二选一）")
    swap_parser.add_argument("-v", "--video", required=True, help="目标视频路径")
    swap_parser.add_argument("-o", "--output", default=None, help="输出视频路径 (默认: 自动命名)")
    swap_parser.add_argument("--start", type=int, default=0, help="起始帧")
    swap_parser.add_argument("--end", type=int, default=-1, help="结束帧 (-1=全部)")
    swap_parser.add_argument("--skip", type=int, default=0, help="跳帧数 (每 N+1 帧处理一帧)")
    swap_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                             help="计算设备")
    swap_parser.add_argument("--enhance", action="store_true", help="启用 GFPGAN 面部增强")
    swap_parser.add_argument("--threshold", type=float, default=0.35,
                             help="人脸匹配相似度阈值 (默认0.35, 范围0~1, 越低越宽松)")

    # ── 提取帧 ──
    extract_parser = subparsers.add_parser("extract", help="提取视频帧")
    extract_parser.add_argument("--video", required=True, help="视频路径")
    extract_parser.add_argument("--frames-dir", default="./frames", help="输出帧目录")
    extract_parser.add_argument("--fps", type=float, default=0, help="提取帧率 (0=原始)")

    # ── 组装视频 ──
    assemble_parser = subparsers.add_parser("assemble", help="组装帧为视频")
    assemble_parser.add_argument("--frames-dir", required=True, help="帧目录")
    assemble_parser.add_argument("--output", required=True, help="输出视频路径")
    assemble_parser.add_argument("--fps", type=float, default=30.0, help="输出帧率")

    args = parser.parse_args()

    if args.command == "swap":
        # 确定参考图列表
        source_paths = []
        if args.source and args.sources:
            log_info("[错误] --source 和 --sources 不能同时使用，请选择其一")
            sys.exit(1)
        elif args.source:
            source_paths = [args.source]
        elif args.sources:
            source_paths = args.sources
        else:
            log_info("[错误] 必须提供 --source 或 --sources 指定参考人脸图")
            sys.exit(1)

        # 自动输出路径
        if args.output is None:
            base = os.path.splitext(os.path.basename(args.video))[0]
            args.output = "%s_swapped.mp4" % base

        # 初始化引擎
        swapper = VideoFaceSwapper(
            device=args.device,
            enable_restoration=args.enhance,
            match_threshold=args.threshold,
        )

        # 加载参考人脸
        log_info("[加载] 加载 %d 张参考人脸图..." % len(source_paths))
        source_faces = swapper.load_source_faces(source_paths)

        # 执行换脸
        t0 = time.time()
        swapper.process_video(
            video_path=args.video,
            source_faces=source_faces,
            output_path=args.output,
            start_frame=args.start,
            end_frame=args.end,
            skip_frames=args.skip,
        )
        elapsed = time.time() - t0
        log_info("[耗时] %.1f 秒" % elapsed)

    elif args.command == "extract":
        extract_frames(args.video, args.frames_dir, args.fps)

    elif args.command == "assemble":
        assemble_video(args.frames_dir, args.output, args.fps)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
