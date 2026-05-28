#!/usr/bin/env python3
"""
视频换脸工具 v3.0.0 — Video Face Swap Pipeline

基于 insightface inswapper_128 + OpenCV 的高效视频换脸方案。

v3.0.0 核心改动：
  - 引入"目标人脸"概念：用目标人脸锁定视频中"换谁"，用参考人脸确定"换成什么样"
  - 新增 --target 参数指定目标人脸（视频中需要被换掉的人）
  - 新增 --target-threshold 控制目标匹配门槛
  - 简化单图模式：--source 为"换成谁"，--target 为"换谁"
  - 保留 --sources 多参考图（参考图列表）
  - 多人脸场景仍可用 --sources / --match-threshold 自动匹配

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
    """视频换脸引擎"""

    def __init__(
        self,
        det_name: str = "antelopev2",
        det_size: Tuple[int, int] = (640, 640),
        device: str = "auto",
        enable_restoration: bool = False,
        target_threshold: float = 0.2,
    ):
        self.device = device
        self.enable_restoration = enable_restoration
        self.target_threshold = target_threshold
        self._providers = self._resolve_providers()
        self._load_models(det_name, det_size)

    def _resolve_providers(self) -> List:
        """选择 ONNX Runtime provider"""
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cpu":
            return ["CPUExecutionProvider"]
        else:
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

        log("[加载] 初始化 FaceAnalysis (%s)..." % det_name)
        self.app = FaceAnalysis(name=det_name, providers=self._providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        log("[加载] FaceAnalysis 就绪")

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
        log("[加载] 加载 inswapper (%s, %.1f MB)..." %
            (model_path, os.path.getsize(model_path) / 1024 / 1024))
        sess = onnxruntime.InferenceSession(model_path, providers=self._providers)
        self.swapper = INSwapper(model_file=model_path, session=sess)
        log("[加载] inswapper 就绪 (输入尺寸 %s)" % str(self.swapper.input_size))

    # ── 人脸提取 ──────────────────────────────────────────────

    def extract_face(self, img: np.ndarray, label: str = "") -> object:
        """从图像中提取面积最大的人脸"""
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("图像%s未检测到人脸" % (" " + label if label else ""))
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        log("[检测] 人脸%s: bbox=%s, score=%.3f" %
            (" " + label if label else "", largest.bbox.astype(int).tolist(), largest.det_score))
        return largest

    # ── 帧内换脸（统一入口） ──────────────────────────────────

    def swap_frame(
        self,
        frame: np.ndarray,
        source_face: object,
        target_face: object = None,
    ) -> np.ndarray:
        """
        单帧执行一张人脸的换脸

        Args:
            frame: BGR 图像
            source_face: 参考人脸（换成这个人）
            target_face: 目标人脸（换这个人），None 则取面积最大的

        Returns:
            换脸后的图像
        """
        result = self.swapper.get(frame, target_face, source_face, paste_back=True)
        return result

    def swap_frame_by_target_embedding(
        self,
        frame: np.ndarray,
        source_face: object,
        target_embedding: np.ndarray = None,
        target_faces_in_frame: List[object] = None,
    ) -> np.ndarray:
        """
        单帧：通过嵌入对比找到目标人脸并换脸

        Args:
            frame: BGR 图像
            source_face: 参考人脸（换成这个人）
            target_embedding: 目标人脸的 normed_embedding
            target_faces_in_frame: 本帧已检测的人脸（可选，省一次检测）

        Returns:
            换脸后的图像
        """
        if target_embedding is None:
            return frame

        faces = target_faces_in_frame
        if faces is None:
            faces = self.app.get(frame)
        if not faces:
            return frame

        # 找到与目标嵌入最匹配的人脸
        best_face = None
        best_score = -1.0
        for face in faces:
            score = float(np.dot(target_embedding, face.normed_embedding))
            if score > best_score:
                best_score = score
                best_face = face

        if best_face is None or best_score < self.target_threshold:
            return frame  # 没找到目标，不动

        # 执行换脸
        result = self.swapper.get(frame, best_face, source_face, paste_back=True)
        return result

    def swap_frame_all_faces(
        self,
        frame: np.ndarray,
        source_face: object,
        faces_in_frame: List[object] = None,
    ) -> np.ndarray:
        """
        单帧：帧内所有人脸都换成参考脸（不匹配，强换）

        Args:
            frame: BGR 图像
            source_face: 参考人脸
            faces_in_frame: 本帧已检测的人脸

        Returns:
            换脸后的图像
        """
        faces = faces_in_frame
        if faces is None:
            faces = self.app.get(frame)
        if not faces:
            return frame

        # 按面积倒序换（先大后小，减少重叠干扰）
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

        result = frame.copy()
        for face in faces:
            result = self.swapper.get(result, face, source_face, paste_back=True)

        return result

    def swap_frame_multi_sources(
        self,
        frame: np.ndarray,
        source_faces: List[object],
        faces_in_frame: List[object] = None,
        target_embedding: np.ndarray = None,
        match_threshold: float = 0.35,
    ) -> np.ndarray:
        """
        单帧：多人脸多参考图模式（v2 兼容）
        source_faces 有多个时，各自匹配各自的人脸
        """
        faces = faces_in_frame
        if faces is None:
            faces = self.app.get(frame)
        if not faces or not source_faces:
            return frame

        if target_embedding is not None:
            # 优先用目标嵌入锁定一个 target
            best_face = None
            best_score = -1.0
            for face in faces:
                score = float(np.dot(target_embedding, face.normed_embedding))
                if score > best_score:
                    best_score = score
                    best_face = face
            if best_face is not None and best_score >= self.target_threshold:
                return self.swapper.get(frame, best_face, source_faces[0], paste_back=True)
            return frame

        # 回退：嵌入相似度匹配（v2 逻辑）
        matches = []
        used_targets = set()
        for si, src in enumerate(source_faces):
            best_score = -1.0
            best_ti = -1
            for ti, tgt in enumerate(faces):
                if ti in used_targets:
                    continue
                score = float(np.dot(src.normed_embedding, tgt.normed_embedding))
                if score > best_score:
                    best_score = score
                    best_ti = ti
            if best_ti >= 0 and best_score >= match_threshold:
                used_targets.add(best_ti)
                matches.append((faces[best_ti], src, best_score))
            elif best_ti >= 0:
                log("[匹配] 参考[%d] → 目标[%d] = %.3f (低于阈值 %.2f, 跳过)" %
                    (si, best_ti, best_score, match_threshold))

        if not matches:
            return frame

        matches.sort(key=lambda m: (m[0].bbox[2]-m[0].bbox[0])*(m[0].bbox[3]-m[0].bbox[1]), reverse=True)
        result = frame.copy()
        for tgt, src, _ in matches:
            result = self.swapper.get(result, tgt, src, paste_back=True)
        return result

    # ── 视频处理（三种模式） ──────────────────────────────────

    def process_video_all_faces(
        self,
        video_path: str,
        source_face: object,
        output_path: str,
        skip_frames: int = 0,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> str:
        """
        模式1：强换——视频中所有人脸都换

        Args:
            video_path: 输入视频
            source_face: 参考人脸
            output_path: 输出路径
            skip_frames: 跳帧
            start_frame: 起始
            end_frame: 结束
        Returns:
            输出路径
        """
        return self._process(
            video_path=video_path,
            source_face=source_face,
            output_path=output_path,
            mode="all",
            skip_frames=skip_frames,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def process_video_target(
        self,
        video_path: str,
        source_face: object,
        target_embedding: np.ndarray,
        output_path: str,
        skip_frames: int = 0,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> str:
        """
        模式2：目标锁定——只换与 target_embedding 匹配的人

        Args:
            video_path: 输入视频
            source_face: 参考人脸（换成谁）
            target_embedding: 目标人脸嵌入（换谁）
            output_path: 输出路径
        Returns:
            输出路径
        """
        self._target_embedding = target_embedding
        return self._process(
            video_path=video_path,
            source_face=source_face,
            output_path=output_path,
            mode="target",
            skip_frames=skip_frames,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def _process(self, video_path, source_face, output_path, mode="all",
                 skip_frames=0, start_frame=0, end_frame=-1):
        """统一的逐帧处理循环"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("无法打开视频: %s" % video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        end = end_frame if end_frame > 0 else total_frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed = 0
        skipped = 0
        no_face = 0
        face_swapped = 0

        label = "所有人强换" if mode == "all" else "仅匹配目标"
        log("[处理] 模式=%s | %s | %d×%d | %.2f fps | %d帧" %
            (label, os.path.basename(video_path), width, height, fps, total_frames))

        t0 = time.time()
        frame_idx = 0
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧 + start_frame 处理
            if frame_idx < start_frame:
                out.write(frame)
                frame_idx += 1
                continue
            if skip_frames > 0 and (frame_idx - start_frame) % (skip_frames + 1) != 0:
                out.write(frame)
                skipped += 1
                frame_idx += 1
                continue

            # 检测人脸
            faces = self.app.get(frame)

            if not faces:
                no_face += 1
                out.write(frame)
                frame_idx += 1
                continue

            if mode == "all":
                # 所有人脸强换
                faces_sorted = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                result = frame.copy()
                for face in faces_sorted:
                    result = self.swapper.get(result, face, source_face, paste_back=True)
                face_swapped += 1
            elif mode == "target":
                # 找最匹配目标的人脸
                target_emb = self._target_embedding
                best_face = None
                best_score = -1.0
                for face in faces:
                    score = float(np.dot(target_emb, face.normed_embedding))
                    if score > best_score:
                        best_score = score
                        best_face = face
                if best_face is not None and best_score >= self.target_threshold:
                    result = self.swapper.get(frame, best_face, source_face, paste_back=True)
                    face_swapped += 1
                else:
                    result = frame.copy()
            else:
                result = frame.copy()

            out.write(result)
            processed += 1
            frame_idx += 1

            if frame_idx % 500 == 0:
                elapsed = time.time() - t0
                log("[进度] %d/%d (%.0f%%) | %.0fs" %
                    (frame_idx, total_frames, 100*frame_idx/total_frames, elapsed))

        cap.release()
        out.release()
        elapsed = time.time() - t0

        log("[完成] %d帧 | %d换脸 | %d无脸 | %d跳过 | %.0fs" %
            (processed, face_swapped, no_face, skipped, elapsed))
        log("  输出: %s" % output_path)
        return output_path

    # ── GFPGAN 增强（可选） ───────────────────────────────────

    def enhance_video(self, video_path: str, output_path: str) -> str:
        """对视频逐帧做 GFPGAN 增强（保持原尺寸）"""
        try:
            from gfpgan import GFPGANer
        except ImportError:
            log("[增强] gfpgan 未安装，跳过")
            return video_path

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        gfpgan = GFPGANer(
            model_path=os.path.expanduser("~/ComfyUI/models/face_restoration/GFPGANv1.4.pth"),
            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        log("[增强] GFPGAN 逐帧增强 (%d 帧)..." % total)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, _, result = gfpgan.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h))
            out.write(result)
            frame_idx += 1
            if frame_idx % 200 == 0:
                log("[增强] %d/%d" % (frame_idx, total))

        cap.release()
        out.release()
        log("[增强] 完成: %s" % output_path)
        return output_path


# ─── 工具函数 ─────────────────────────────────────────────────

def log(msg: str):
    print("[视频换脸] %s" % msg)
    sys.stdout.flush()


# ─── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="视频换脸工具 v3.0 — Video Face Swap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 模式1: 强换 — 所有脸都换
  %(prog)s swap --source ref.jpg --video input.mp4 -o out.mp4

  # 模式2: 目标锁定 — 只换 target 指向的人 (推荐!)
  %(prog)s swap --source ref.jpg --target target_face.jpg --video input.mp4 -o out.mp4

  # 全帧 + GFPGAN 增强
  %(prog)s swap --source ref.jpg --video input.mp4 -o out.mp4 --enhance

  # 仅增强已有视频
  %(prog)s enhance --video result.mp4 -o result_enhanced.mp4
        """)

    subparsers = parser.add_subparsers(dest="command")

    # ── 换脸 ──
    sp = subparsers.add_parser("swap", help="执行视频换脸")
    sp.add_argument("-s", "--source", required=True, help="参考人脸（换成这个人的脸）")
    sp.add_argument("-t", "--target", default=None, help="目标人脸（视频中被换的那个人，截图）")
    sp.add_argument("-v", "--video", required=True, help="目标视频")
    sp.add_argument("-o", "--output", default=None, help="输出视频")
    sp.add_argument("--start", type=int, default=0, help="起始帧")
    sp.add_argument("--end", type=int, default=-1, help="结束帧")
    sp.add_argument("--skip", type=int, default=0, help="跳帧 (每N+1帧处理一帧)")
    sp.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    sp.add_argument("--enhance", action="store_true", help="GFPGAN 增强")
    sp.add_argument("--target-threshold", type=float, default=0.2,
                    help="目标匹配阈值 0~1 (默认0.2, 越高越严格)")

    # ── 增强 ──
    sp2 = subparsers.add_parser("enhance", help="GFPGAN 面部增强")
    sp2.add_argument("-v", "--video", required=True)
    sp2.add_argument("-o", "--output", required=True)
    sp2.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()

    if args.command == "swap":
        if args.output is None:
            base = os.path.splitext(os.path.basename(args.video))[0]
            args.output = "%s_swapped.mp4" % base

        # 初始化引擎
        swapper = VideoFaceSwapper(
            device=args.device,
            enable_restoration=args.enhance,
            target_threshold=args.target_threshold,
        )

        # 加载参考人脸（换成谁）
        ref_img = cv2.imread(args.source)
        if ref_img is None:
            log("[错误] 无法读取参考图: %s" % args.source); sys.exit(1)
        source_face = swapper.extract_face(ref_img, label="reference(换成)")

        # 模式选择
        if args.target:
            # 目标锁定模式
            tgt_img = cv2.imread(args.target)
            if tgt_img is None:
                log("[错误] 无法读取目标图: %s" % args.target); sys.exit(1)
            target_face = swapper.extract_face(tgt_img, label="target(换掉)")
            target_embedding = target_face.normed_embedding
            log("[匹配] 将用目标嵌入锁定视频中此人 (阈值=%.2f)" % args.target_threshold)
            log("[匹配] 参考(%s) ↔ 目标(%s) 嵌入相似度: %.4f" %
                (os.path.basename(args.source), os.path.basename(args.target),
                 float(np.dot(source_face.normed_embedding, target_embedding))))

            t0 = time.time()
            swapper.process_video_target(
                video_path=args.video,
                source_face=source_face,
                target_embedding=target_embedding,
                output_path=args.output,
                skip_frames=args.skip,
                start_frame=args.start,
                end_frame=args.end,
            )
            log("[耗时] %.1f 秒" % (time.time() - t0))
        else:
            # 强换所有人模式
            log("[模式] 无 --target, 将对视频中所有人脸强换")
            t0 = time.time()
            swapper.process_video_all_faces(
                video_path=args.video,
                source_face=source_face,
                output_path=args.output,
                skip_frames=args.skip,
                start_frame=args.start,
                end_frame=args.end,
            )
            log("[耗时] %.1f 秒" % (time.time() - t0))

    elif args.command == "enhance":
        swapper = VideoFaceSwapper(device=args.device)
        swapper.enhance_video(args.video, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()