#!/usr/bin/env python3
"""
视频换脸工具 v3.3.0 — Video Face Swap Pipeline
================================================

基于 insightface inswapper_128 + 面部预处理 + GFPGAN 增强的完整视频换脸方案。

v3.3.0 (2026-06-06) 核心修复：ID 保真度改进
  - P0-1: norm_crop2 输出从 256 提升至 512，保证对齐图有足够像素细节
  - P0-2: GFPGAN 增强后不再用于重新提取嵌入（GFPGAN 改变 ID），嵌入始终来自对齐图
  - P0-3: 新增 load_source_face_from_preprocess() — 从 final_reference.png（原始分辨率
    裁剪版）检测人脸，然后覆盖加载 face_embedding.npy 中的高质量嵌入
  - P0-4: Skill 层同理修复，不再从缩略图重新检测 source_face
  - P1: final_reference.png 改为保存原始分辨率的面部裁剪版（1.5x 外扩），而非 512x512 对齐缩略图
  - P2: _det_size_cache 检测失败时清除缓存后重试，不再永久锁定失败的 det_size
  - P2: _load_models 增加 det_name 回退机制（antelopev2 失败时回退 buffalo_l）
  - P2: 新增 det_name_inswapper 参数，显示指定 inswapper 所属模型包路径

v3.2.0 (2026-05-31) 核心修复：
  - 修复关键 bug：antelopev2 默认 det_size=(640,640) 在小参考图(如 262×368)上检测不
    到人脸，导致 source_face 实际为 None/空值，换出的脸与参考图完全无关
  - 新增自适应检测 `_detect_adaptive()`：根据图片实际尺寸自动选择最优 det_size
  - 修复 extract_face()：首次检测失败时，自动缩小 det_size 重试
  - 修复 preprocess_face()：小图自动放大后再检测
  - 修复 swap --preprocess 流程：现在优先从预处理（对齐+增强）后的图像提取 source_face，
    而非继续使用原始小图
  - 新增 --det-size CLI 参数：允许用户手动指定检测尺寸

v3.1.0 核心改动：
  - 新增 preprocess 子命令：自动对齐、增强参考人脸，修正偏转角
  - swap 命令支持 --preprocess 旗帜：自动预处理参考图后执行换脸
  - 新增 --face-restore-model 参数：指定面部修复模型（gfpgan / codeformer / none）
  - ComfyUI 工作流同步升级：增加参考图预处理 + face_boost + 面部修复

依赖：insightface==0.7.3, opencv-python, onnxruntime-gpu, numpy, tqdm, gfpgan
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import json
import pickle
import os
import sys
from pathlib import Path
import subprocess

# 解决 529MB ONNX 模型的 protobuf arena 分配失败（RTX 5080 Blackwell 环境）
os.environ.setdefault("PROTOBUF_PYTHON_IMPLEMENTATION", "python")

# onnxruntime-gpu 需要 cudart/cudnn/cublas DLL。
# 优先 pip 官方 nvidia-cudnn-cu12 包（新版 cudnn，对新架构支持更好），
# 其次用 torch 捆绑的 cudnn 作为兜底。
# 注意：若进程同时 import 了 torch，DLL 按模块名先到先得，
# torch/lib 里捆绑的旧版 cudnn 会抢先载入并覆盖 PATH 注入，导致 ORT CUDA 推理失败。
# 因此使用本引擎时请勿在同一进程内 import torch（GFPGAN 增强依赖 torch，
# GPU 模式下建议关闭 --enhance）。
_sp = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
_dll_dirs = [
    os.path.join(_sp, "nvidia", "cudnn", "bin"),
    os.path.join(_sp, "torch", "lib"),
]
_dll_dirs = [d for d in _dll_dirs if os.path.isdir(d)]
if _dll_dirs:
    os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict


# ─── 数据类 ──────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """预处理结果"""
    aligned_face_path: str          # 对齐后的参考图路径
    enhanced_path: str = ""         # 增强后的路径（如有）
    embedding_path: str = ""        # 嵌入向量路径 (.npy)
    original_angle: float = 0.0     # 原始偏转角度
    quality_score: float = 0.0      # 质量评分 0-1
    enhancement_applied: bool = False


# ─── 核心换脸引擎 ─────────────────────────────────────────────

class VideoFaceSwapper:
    """视频换脸引擎"""

    def __init__(
        self,
        det_name: str = "buffalo_l",
        det_name_inswapper: str = "buffalo_l",  # 显式指定 inswapper 对应的模型包
        det_size: Tuple[int, int] = (640, 640),
        device: str = "auto",
        enable_restoration: bool = False,
        target_threshold: float = 0.2,
        face_restore_model: str = "none",
    ):
        self.device = device
        self.enable_restoration = enable_restoration
        self.target_threshold = target_threshold
        self.face_restore_model = face_restore_model
        self._providers = self._resolve_providers()
        self._inswapper_providers, self._inswapper_provider_options = self._resolve_inswapper_providers()
        self._load_models(det_name, det_name_inswapper, det_size)
        self._gfpgan = None  # lazy load
        # 缓存自适应检测的 det_size（避免重复重试）
        self._det_size_cache: Dict[str, Tuple[int, int]] = {}

    def _detect_adaptive(self, img: np.ndarray) -> List:
        """
        自适应检测人脸：根据图片实际尺寸选择最优 det_size。

        重要：insightface 的 FaceAnalysis 内部存在状态问题：
        - 同一个 app 实例，如果第一次 get() 返回 0 张人脸，
          即使之后调用 prepare() 更改 det_size，再次 get() 仍会返回 0。
        - 每次更改 det_size 需要创建全新的 FaceAnalysis 实例。

        策略：
          1. 根据图片尺寸自动选择最佳 det_size
          2. 如果当前 det_size 合适，直接用 self.app.get()
          3. 如果不合适，创建新的 FaceAnalysis 实例
          4. 如果图片太小，先放大再检测
          5. 结果缓存在 _det_size_cache 中

        Returns:
            list of face objects
        """
        h, w = img.shape[:2]
        cache_key = f"{h}x{w}"

        # 计算最佳 det_size
        # antelopev2 的推荐输入：图片最短边 >= det_size × 0.8 左右
        best_det = self._initial_det_size
        img_min_side = min(w, h)

        if img_min_side < best_det[0]:
            # 图片小于默认检测尺寸 → 缩小 det_size
            for sz in [(320, 320), (160, 160), (100, 100)]:
                if img_min_side >= sz[0] * 0.6:
                    best_det = sz
                    break
            else:
                # 图片太小，放大后检测
                scale = best_det[0] / max(img_min_side, 1)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                h, w = new_h, new_w  # update for cache key
                log("[检测] 原图太小 (%dx%d)，放大到 %dx%d" % (w, h, new_w, new_h))

        # 如果已经有缓存的检测尺寸，用缓存
        if cache_key in self._det_size_cache:
            best_det = self._det_size_cache[cache_key]

        # 用最佳 det_size
        if best_det == self._current_det_size:
            # 当前 app 实例已经用这个尺寸，直接检测
            faces = self.app.get(img)
            if faces:
                self._det_size_cache[cache_key] = best_det
                return faces
            # 检测失败 → 清除缓存 + 重新创建 app
            if cache_key in self._det_size_cache:
                del self._det_size_cache[cache_key]
                log("[检测] 缓存 det_size=%s 失效，移除后重试" % str(self._current_det_size))
            best_det = self._initial_det_size

        # 需要创建新的 app 实例
        if best_det != self._initial_det_size:
            log("[检测] 更换 det_size=%s (原图 %dx%d)" % (str(best_det), w, h))
            try:
                from insightface.app import FaceAnalysis
                new_app = FaceAnalysis(name=self._det_name, providers=self._providers, provider_options=self._provider_options)
                new_app.prepare(ctx_id=0, det_size=best_det)
                faces = new_app.get(img)
                if faces:
                    self._det_size_cache[cache_key] = best_det
                    self.app = new_app
                    self._current_det_size = best_det
                    log("[检测] [OK] det_size=%s 成功，%d 张人脸" % (str(best_det), len(faces)))
                    return faces
            except Exception as e:
                log("[检测] 更换 det_size 失败: %s" % str(e))
                if cache_key in self._det_size_cache:
                    del self._det_size_cache[cache_key]

        # 所有尝试都失败
        self._current_det_size = self._initial_det_size
        return []

    def _load_models(self, det_name: str, det_name_inswapper: str, det_size: Tuple[int, int]):
        """加载 insightface 检测模型 + inswapper 换脸模型"""
        from insightface.app import FaceAnalysis
        from insightface.model_zoo.inswapper import INSwapper
        import onnxruntime

        log("[加载] 初始化 FaceAnalysis (%s)..." % det_name)
        try:
            self.app = FaceAnalysis(name=det_name, providers=self._providers, provider_options=self._provider_options)
            self.app.prepare(ctx_id=0, det_size=det_size)
            log("[加载] FaceAnalysis (%s) 就绪" % det_name)
        except Exception as e:
            log("[警告] FaceAnalysis '%s' 加载失败: %s" % (det_name, e))
            if det_name != "buffalo_l":
                log("[加载] 回退到 buffalo_l...")
                det_name = "buffalo_l"
                self.app = FaceAnalysis(name=det_name, providers=self._providers, provider_options=self._provider_options)
                self.app.prepare(ctx_id=0, det_size=det_size)
                log("[加载] FaceAnalysis (buffalo_l) 就绪")
            else:
                raise
        self._det_name = det_name
        self._initial_det_size = det_size
        self._current_det_size = det_size
        log("[加载] FaceAnalysis 就绪 (检测模型=%s, 初始 det_size=%s)" % (det_name, str(det_size)))

        # 查找 inswapper 模型 — 优先使用与检测模型同包路径
        inswapper_model_name = "inswapper_128.onnx"
        model_search_paths = [
            os.path.expanduser("~/.insightface/models/%s/%s" % (det_name_inswapper, inswapper_model_name)),
            os.path.expanduser("~/.insightface/models/buffalo_l/%s" % inswapper_model_name),
            os.path.expanduser("~/.insightface/models/%s" % inswapper_model_name),
            os.path.join(os.path.dirname(__file__), "models", inswapper_model_name),
        ]
        found = False
        for p in model_search_paths:
            if os.path.exists(p):
                model_path = p
                found = True
                break
        if not found:
            raise FileNotFoundError(
                "找不到 inswapper_128.onnx！\n"
                "请下载后放到 ~/.insightface/models/buffalo_l/inswapper_128.onnx"
            )
        log("[加载] 加载 inswapper (%s, %.1f MB, providers=%s)..." %
            (model_path, os.path.getsize(model_path) / 1024 / 1024, self._inswapper_providers))
        sess = onnxruntime.InferenceSession(model_path, providers=self._inswapper_providers, provider_options=self._inswapper_provider_options)
        self.swapper = INSwapper(model_file=model_path, session=sess)
        log("[加载] inswapper 就绪 (输入尺寸 %s, 模型包=%s)" % (str(self.swapper.input_size), det_name_inswapper))

    def _resolve_providers(self) -> List:
        """选择 ONNX Runtime provider（检测/识别模型），CUDA 优先（NVIDIA 显卡），回退 DML/CPU"""
        import onnxruntime as ort
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                self._provider_options = [{}, {}]
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in avail:
                self._provider_options = [{}, {}]
                return ["DmlExecutionProvider", "CPUExecutionProvider"]
        except:
            pass
        self._provider_options = [{}]
        return ["CPUExecutionProvider"]

    def _resolve_inswapper_providers(self) -> Tuple[List, List]:
        """选择 inswapper 的 provider，CUDA 优先（NVIDIA 显卡），回退 DirectML/CPU"""
        import onnxruntime as ort
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                return (["CUDAExecutionProvider", "CPUExecutionProvider"], [{"cudnn_conv_algo_search": "EXHAUSTIVE"}, {}])
            if "DmlExecutionProvider" in avail:
                return (["DmlExecutionProvider", "CPUExecutionProvider"], [{}, {}])
        except:
            pass
        return (["CPUExecutionProvider"], [{}])

    def _load_gfpgan(self):
        """懒加载 GFPGAN（轻量封装，无需 facexlib 检测模型）"""
        if self._gfpgan is not None:
            return
        try:
            from gfpgan_light import GFPGANLight
            self._gfpgan = GFPGANLight(
                model_path=os.path.expanduser(
                    "~/ComfyUI/models/facerestore_models/GFPGANv1.4.pth"))
            log("[加载] GFPGANLight 修复模型就绪")
        except Exception as e:
            log("[警告] GFPGAN 加载失败: %s，跳过" % e)
            self._gfpgan = None

    # ── 参考图预处理 ──────────────────────────────────────────

    def preprocess_face(
        self,
        image_path: str,
        output_dir: str = None,
        enhance: bool = True,
        align: bool = True,
    ) -> PreprocessResult:
        """
        对参考照片进行自动预处理：
        1. 检测人脸 → 提取对齐裁剪（纠正偏转角）
        2. 可选 GFPGAN 增强
        3. 提取增强后的面部嵌入
        4. 保存对齐图 + 增强图 + 嵌入向量

        Args:
            image_path: 参考照片路径
            output_dir: 输出目录（默认 ref_face_optimized/）
            enhance: 是否做 GFPGAN 增强
            align: 是否做对齐裁剪

        Returns:
            PreprocessResult
        """
        import shutil

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像: %s" % image_path)

        log("[预处理] 读取: %s (%dx%d)" % (image_path, img.shape[1], img.shape[0]))

        # 自适应检测人脸
        faces = self._detect_adaptive(img)
        if len(faces) == 0:
            raise ValueError("图像中未检测到人脸，无法预处理")

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        bbox = face.bbox.astype(int)
        log("[预处理] 检测到人脸: bbox=%s, score=%.4f" %
            (bbox.tolist(), face.det_score))

        # 估算偏转角度
        kps = face.kps
        eye_dist = np.linalg.norm(kps[0] - kps[1])
        nose_offset = abs(kps[2][0] - (kps[0][0] + kps[1][0]) / 2)
        yaw_deg = np.degrees(np.arctan(nose_offset / max(eye_dist, 1)))
        log("[预处理] 头部偏转: %.1f°" % yaw_deg)

        # 质量评分
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean() / 255.0
        contrast = gray.std() / 255.0
        quality = 1.0 - (yaw_deg / 45.0) * 0.5  # 角度扣分
        quality = quality * (0.5 + 0.5 * brightness)  # 亮度加分
        quality = min(max(quality, 0.2), 1.0)
        log("[预处理] 照度=%.2f, 对比度=%.2f, 质量评分=%.2f" %
            (brightness, contrast, quality))

        # 构建输出目录
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(image_path),
                "ref_face_optimized"
            )
        os.makedirs(output_dir, exist_ok=True)

        # 1. 对齐裁剪 — 用 insightface 的 norm_crop 修正角度
        aligned_face = None
        aligned_path = None
        if align:
            from insightface.utils import face_align
            # norm_crop 用 5 个关键点做仿射变化把脸摆正
            # v3.3.0: 从 256 提升到 512，提供更多人脸细节给检测模型，确保嵌入质量
            aimg, M = face_align.norm_crop2(img, face.kps, 512)
            aligned_face = aimg
            aligned_path = os.path.join(output_dir, "aligned_face.png")
            cv2.imwrite(aligned_path, aimg)
            log("[预处理] 对齐保存: %s (512x512)" % aligned_path)

            # 同时也保存对齐前裁剪（对比用）
            before_path = os.path.join(output_dir, "original_crop.png")
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), \
                              min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
            cv2.imwrite(before_path, img[y1:y2, x1:x2])

        # 2. GFPGAN 增强
        enhanced_path = None
        enhancement_applied = False
        if enhance and align and aligned_face is not None:
            self._load_gfpgan()
            if self._gfpgan is not None:
                try:
                    enhanced = self._gfpgan.enhance_aligned(aligned_face, weight=0.5)
                    enhanced_path = os.path.join(output_dir, "enhanced_face.png")
                    cv2.imwrite(enhanced_path, enhanced)
                    log("[预处理] 增强保存: %s (仅供后处理视觉对比，不用于嵌入提取)" % enhanced_path)
                    enhancement_applied = True
                    # v3.3.0 修复：GFPGAN 增强会改变面部 ID，不应用它重新提取嵌入。
                    # 嵌入从原始对齐图提取，确保身份一致。
                    log("[预处理] 嵌入继续从对齐图提取，以保持身份一致性")
                except Exception as e:
                    log("[警告] 增强失败: %s" % e)

        # 3. 提取最优嵌入向量
        # 策略：优先从对齐图检测人脸提取嵌入（512x512 已足够高分辨率）
        # 如果对齐图检测失败，回退到原始图检测的人脸
        if aligned_face is not None:
            # 对齐图 512x512：直接检测（antelopev2 在 512x512 上效果良好）
            aligned_faces = self._detect_adaptive(aligned_face)
            if aligned_faces:
                best_af = max(aligned_faces,
                              key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                face = best_af
                log("[预处理] 从对齐图重新提取嵌入，score=%.4f" % best_af.det_score)
            else:
                log("[预处理] 对齐图检测失败，保留原始图检测的人脸")

        embedding = face.normed_embedding
        emb_path = os.path.join(output_dir, "face_embedding.npy")
        np.save(emb_path, embedding)
        log("[预处理] 嵌入向量保存: %s (shape=%s)" % (emb_path, embedding.shape))

        # 4. 保存元数据
        meta = {
            "source": os.path.abspath(image_path),
            "aligned": bool(align),
            "enhanced": enhancement_applied,
            "yaw_degrees": float(yaw_deg),
            "quality_score": float(quality),
            "embedding_file": "face_embedding.npy",
            "aligned_file": "aligned_face.png" if align else None,
            "enhanced_file": "enhanced_face.png" if enhancement_applied else None,
        }
        meta_path = os.path.join(output_dir, "preprocess_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log("[预处理] 元数据保存: %s" % meta_path)

        result = PreprocessResult(
            aligned_face_path=aligned_path or "",
            enhanced_path=enhanced_path or "",
            embedding_path=emb_path,
            original_angle=float(yaw_deg),
            quality_score=float(quality),
            enhancement_applied=enhancement_applied,
        )
        log("[预处理] [OK] 完成! (角度: %.1f° → 对齐, 增强: %s, 评分: %.2f)" %
            (yaw_deg, "是" if enhancement_applied else "否", quality))

        # 5. 保存最终参考图 — 原始分辨率裁剪版（供换脸时直接检测使用）
        # v3.3.0 修复：不再复制 512x512 的对齐缩略图作为最终参考图
        # 改为保存原始图片的裁剪版，保留更多像素细节给检测模型
        final_ref_path = os.path.join(output_dir, "final_reference.png")
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), \
                          min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
        # 向外扩展 50% 确保包含完整面部周围环境，帮助检测
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w, half_h = (x2 - x1) // 2, (y2 - y1) // 2
        expand = 1.5
        x1 = max(0, int(cx - half_w * expand))
        y1 = max(0, int(cy - half_h * expand))
        x2 = min(img.shape[1], int(cx + half_w * expand))
        y2 = min(img.shape[0], int(cy + half_h * expand))
        original_crop = img[y1:y2, x1:x2]
        cv2.imwrite(final_ref_path, original_crop)
        log("[预处理] 最终参考图 (原始分辨率裁剪, %dx%d): %s" %
            (original_crop.shape[1], original_crop.shape[0], final_ref_path))
        result.final_reference_path = final_ref_path

        return result

    def load_preprocessed_face(
        self,
        preprocess_dir: str,
    ) -> Tuple[np.ndarray, str]:
        """
        从预处理目录加载优化后的嵌入和参考图路径
        Returns:
            (embedding, ref_image_path)
        """
        emb_path = os.path.join(preprocess_dir, "face_embedding.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError("未找到嵌入向量: %s" % emb_path)

        embedding = np.load(emb_path)

        # 优先使用增强图，其次对齐图
        ref_candidates = [
            os.path.join(preprocess_dir, "final_reference.png"),
            os.path.join(preprocess_dir, "enhanced_face.png"),
            os.path.join(preprocess_dir, "aligned_face.png"),
        ]
        ref_path = None
        for c in ref_candidates:
            if os.path.exists(c):
                ref_path = c
                break

        return embedding, ref_path

    # ── 人脸提取 ──────────────────────────────────────────────

    def extract_face(self, img: np.ndarray, label: str = "") -> object:
        """从图像中提取面积最大的人脸（自适应检测）"""
        h, w = img.shape[:2]
        # 如果图片太小，先放大到至少 256x256
        if max(h, w) < 256:
            scale = 512.0 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            log("[检测] %s: 原图太小 (%dx%d)，放大到 %dx%d" %
                (label or "参考图", h, w, new_h, new_w))

        faces = self._detect_adaptive(img)
        if len(faces) == 0:
            raise ValueError("图像%s未检测到人脸" % (" " + label if label else ""))
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        log("[检测] 人脸%s: bbox=%s, score=%.3f" %
            (" " + label if label else "", largest.bbox.astype(int).tolist(), largest.det_score))
        return largest

    def load_source_face_from_preprocess(
        self,
        reference_image: str,
        preprocess_dir: str,
    ) -> object:
        """
        从预处理目录加载 source_face：检测人脸 + 用保存的嵌入覆盖。

        v3.3.0 新增：
        - insightface 的 Face.normed_embedding 是 @property（从 self.embedding 实时计算）
        - 因此直接给 face.embedding 赋值即可替换 normed_embedding
        - 优先从 final_reference.png（原始分辨率裁剪版）检测人脸
        - 加载 face_embedding.npy 覆盖 face.embedding
        - 确保嵌入质量不受缩略图检测影响

        Args:
            reference_image: 原始参考图路径（回退用）
            preprocess_dir: 预处理输出目录

        Returns:
            face 对象（embedding 已替换为保存的高质量嵌入）
        """
        emb_path = os.path.join(preprocess_dir, "face_embedding.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError("预处理目录中未找到 face_embedding.npy")

        saved_embedding = np.load(emb_path)  # 已经归一化的向量 norm≈1.0
        log("[加载] [OK] 加载预处理嵌入: %s (norm=%.4f)" %
            (emb_path, np.linalg.norm(saved_embedding)))

        def _detect_and_override(img, label):
            """检测人脸，用保存的嵌入覆盖 face.embedding"""
            face = self.extract_face(img, label=label)
            # normed_embedding 是 property：self.embedding / l2norm(self.embedding)
            # 所以给 self.embedding 赋归一化后的值即可
            original_normed = face.normed_embedding.copy()
            cos_sim = float(np.dot(original_normed, saved_embedding))
            face.embedding = saved_embedding.copy()
            # 验证 normed_embedding 已更新
            log("[加载] [OK] %s: 检测 score=%.3f, 覆盖后 normed_embedding norm=%.4f" %
                (label, face.det_score, np.linalg.norm(face.normed_embedding)))
            log("[加载]    与原检测嵌入 cosine=%.4f" % cos_sim)
            return face

        # 1. 优先从 final_reference.png（原始分辨率裁剪版）检测人脸
        final_ref = os.path.join(preprocess_dir, "final_reference.png")
        if os.path.exists(final_ref):
            img = cv2.imread(final_ref)
            if img is not None:
                try:
                    return _detect_and_override(img, "最终参考图")
                except ValueError:
                    log("[加载] final_reference.png 未检测到人脸")

        # 2. 回退：从原始参考图检测
        img = cv2.imread(reference_image)
        if img is None:
            raise ValueError("无法读取原始参考图: %s" % reference_image)
        return _detect_and_override(img, "原始参考图")

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
        """
        if target_embedding is None:
            return frame

        faces = target_faces_in_frame
        if faces is None:
            faces = self.app.get(frame)
        if not faces:
            return frame

        best_face = None
        best_score = -1.0
        for face in faces:
            score = float(np.dot(target_embedding, face.normed_embedding))
            if score > best_score:
                best_score = score
                best_face = face

        if best_face is None or best_score < self.target_threshold:
            return frame

        result = self.swapper.get(frame, best_face, source_face, paste_back=True)
        return result

    def swap_frame_all_faces(
        self,
        frame: np.ndarray,
        source_face: object,
        faces_in_frame: List[object] = None,
    ) -> np.ndarray:
        """
        单帧：帧内所有人脸都换成参考脸（不匹配）
        """
        faces = faces_in_frame
        if faces is None:
            faces = self.app.get(frame)
        if not faces:
            return frame

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
        单帧：多人脸多参考图模式
        """
        faces = faces_in_frame
        if faces is None:
            faces = self.app.get(frame)
        if not faces or not source_faces:
            return frame

        if target_embedding is not None:
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

    # ── 视频处理 ──────────────────────────────────────────────

    def process_video_all_faces(
        self,
        video_path: str,
        source_face: object,
        output_path: str,
        skip_frames: int = 0,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> str:
        '''模式1：强换——视频中所有人脸都换'''
        return self._process(
            video_path=video_path, source_face=source_face,
            output_path=output_path, mode="all",
            skip_frames=skip_frames, start_frame=start_frame, end_frame=end_frame)

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
        '''模式2：目标锁定——只换与 target_embedding 匹配的人'''
        self._target_embedding = target_embedding
        return self._process(
            video_path=video_path, source_face=source_face,
            output_path=output_path, mode="target",
            skip_frames=skip_frames, start_frame=start_frame, end_frame=end_frame)

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
        log("[处理] 模式=%s | %s | %dx%d | %.2f fps | %d帧" %
            (label, os.path.basename(video_path), width, height, fps, total_frames))

        t0 = time.time()
        frame_idx = 0
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < start_frame:
                out.write(frame)
                frame_idx += 1
                continue
            if skip_frames > 0 and (frame_idx - start_frame) % (skip_frames + 1) != 0:
                out.write(frame)
                skipped += 1
                frame_idx += 1
                continue

            faces = self.app.get(frame)

            if not faces:
                no_face += 1
                out.write(frame)
                frame_idx += 1
                continue

            if mode == "all":
                faces_sorted = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                result = frame.copy()
                for face in faces_sorted:
                    result = self.swapper.get(result, face, source_face, paste_back=True)
                face_swapped += 1
            elif mode == "target":
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

        # 复制音频轨道
        _copy_audio_to_output(video_path, output_path)

        elapsed = time.time() - t0

        log("[完成] %d帧 | %d换脸 | %d无脸 | %d跳过 | %.0fs" %
            (processed, face_swapped, no_face, skipped, elapsed))
        log("  输出: %s" % output_path)
        return output_path

    # ── 帧级别面部修复 ───────────────────────────────────────

    def _apply_face_restoration(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧做面部修复
        """
        if self.face_restore_model == "none":
            return frame
        if self.face_restore_model == "gfpgan":
            self._load_gfpgan()
            if self._gfpgan is not None:
                try:
                    # 在全帧中检测人脸，对齐后增强
                    faces = self.app.get(frame)
                    if not faces:
                        return frame
                    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    from insightface.utils import face_align
                    aimg, _ = face_align.norm_crop2(frame, largest.kps, 512)
                    enhanced = self._gfpgan.enhance_aligned(aimg, weight=0.5)
                    # 贴回原图
                    return self._paste_face_back(frame, largest, enhanced)
                except Exception as e:
                    log("[增强警告] %s" % e)
                    return frame
        return frame

    def _paste_face_back(self, img: np.ndarray, face, enhanced_face: np.ndarray) -> np.ndarray:
        """将增强后的脸部贴回原图位置"""
        from insightface.utils import face_align
        _, M = face_align.norm_crop2(img, face.kps, 512)
        M_inv = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        warped = cv2.warpAffine(enhanced_face, M_inv, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        # 用脸部区域做 alpha 融合
        bbox = face.bbox.astype(int)
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, ((x1+x2)//2, (y1+y2)//2), ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 15)
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (img * (1 - mask_3ch) + warped * mask_3ch).astype(np.uint8)
        return result

    def enhance_video(self, video_path: str, output_path: str, restore_model: str = "gfpgan") -> str:
        """
        对视频逐帧做面部修复增强

        Args:
            video_path: 输入视频
            output_path: 输出视频
            restore_model: 修复模型类型 ("gfpgan" / "codeformer" / "none")
        """
        if restore_model == "none":
            log("[增强] restore_model=none，跳过")
            return video_path

        self.face_restore_model = restore_model

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        log("[增强] %s 逐帧面部修复 (%d 帧)..." % (restore_model.upper(), total))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self._apply_face_restoration(frame)
            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h))
            out.write(result)
            frame_idx += 1
            if frame_idx % 200 == 0:
                log("[增强] %d/%d" % (frame_idx, total))

        cap.release()
        out.release()

        # 复制音频轨道
        _copy_audio_to_output(video_path, output_path)

        log("[增强] 完成: %s" % output_path)
        return output_path


# ─── 音频复制 ─────────────────────────────────────────────────

def _copy_audio_to_output(input_video: str, output_video: str):
    """用 ffmpeg 把源视频的音频轨道复制到输出视频（源无音频则跳过）"""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()

        # 快速探测输入是否有音频
        probe = subprocess.run(
            [ffmpeg_path, "-i", input_video],
            capture_output=True, text=True
        )
        if "Audio:" not in probe.stderr:
            return

        tmp = output_video + ".audio_tmp.mp4"
        subprocess.run(
            [ffmpeg_path, "-y",
             "-i", output_video,
             "-i", input_video,
             "-c:v", "copy",
             "-c:a", "aac",
             "-map", "0:v:0",
             "-map", "1:a:0?",
             "-shortest",
             tmp],
            capture_output=True, check=True
        )
        os.replace(tmp, output_video)
        log("[音频] [OK] 已添加音频轨道")
    except Exception as e:
        log("[音频] ⚠️ 添加音频失败: %s" % e)


def _reencode_h264(input_video: str, crf: int = 18):
    """把视频重编码为 H.264 高画质（crf 越小越清晰，文件越大）。
    OpenCV 的 mp4v 编码器压缩效率低、容易出块状噪点，
    ffmpeg libx264 + 合理 crf 是画质与体积的最佳平衡。"""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
        tmp = input_video + ".h264_tmp.mp4"
        subprocess.run(
            [ffmpeg_path, "-y",
             "-i", input_video,
             "-c:v", "libx264",
             "-crf", str(crf),
             "-preset", "medium",
             "-pix_fmt", "yuv420p",
             "-c:a", "copy",
             "-movflags", "+faststart",
             tmp],
            capture_output=True, check=True
        )
        os.replace(tmp, input_video)
        log("[编码] [OK] 已重编码为 H.264 (crf=%d)" % crf)
    except Exception as e:
        log("[编码] ⚠️ H.264 重编码失败，保留原编码: %s" % e)

# ─── 工具函数 ─────────────────────────────────────────────────

def log(msg: str):
    # Windows GBK 控制台打不出 ⚠️ 等字符，编码兜底避免整个进程崩溃
    try:
        print("[视频换脸] %s" % msg)
    except UnicodeEncodeError:
        safe = msg.encode("gbk", "replace").decode("gbk")
        print("[视频换脸] %s" % safe)
    sys.stdout.flush()


def load_face_from_image(swapper: VideoFaceSwapper, image_path: str, label: str = "") -> object:
    """从图片文件加载人脸对象"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取: %s" % image_path)
    return swapper.extract_face(img, label=label)


# ─── CLI ──────────────────────────────────────────────────────

# 默认输出根目录
DEFAULT_VIDEOS_DIR = os.path.join(os.path.expanduser("~"), "Videos", "face_swap")


def main():
    parser = argparse.ArgumentParser(
        description="视频换脸工具 v3.3.0 — Video Face Swap (ID 保真度改进)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 0) 先预处理参考图（自动对齐+增强）
  %(prog)s preprocess --source ref.jpg -o optimized_ref/

  # 1) 模式: 强换 — 用优化后的参考图
  %(prog)s swap --source optimized_ref/final_reference.png \\
        --preprocess-dir optimized_ref --video input.mp4 -o out.mp4

  # 2) 模式: 目标锁定 — 只换 target 指向的人
  %(prog)s swap --source optimized_ref/final_reference.png \\
        --preprocess-dir optimized_ref \\
        --target target_face.jpg --video input.mp4 -o out.mp4

  # 3) 一键模式: 自动预处理 + 换脸 + 增强
  %(prog)s swap --source ref.jpg --preprocess --enhance \\
        --video input.mp4 -o out.mp4

  # 4) 仅面部修复增强已有视频
  %(prog)s enhance --video result.mp4 -o enhanced.mp4 --restore gfpgan

  # 5) 流程: 预处理 → 强换 → 增强
  %(prog)s preprocess --source ref.jpg -o optimized_ref/
  %(prog)s swap --source optimized_ref/final_reference.png \\
        --preprocess-dir optimized_ref --video input.mp4 -o raw.mp4
  %(prog)s enhance --video raw.mp4 -o final.mp4 --restore gfpgan
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── preprocess ────────────────────────────────────────────
    pp = subparsers.add_parser("preprocess", help="预处理参考图：对齐+增强+提取嵌入")
    pp.add_argument("--source", "-s", required=True, help="参考照片路径")
    pp.add_argument("--output", "-o", default=None, help="输出目录")
    pp.add_argument("--no-enhance", action="store_true", help="跳过 GFPGAN 增强")
    pp.add_argument("--no-align", action="store_true", help="跳过对齐裁剪")
    pp.add_argument("--det-size", type=int, default=0,
                    help="手动指定检测尺寸 (如 320)，0=自适应 (default: 0)")

    # ── swap ──────────────────────────────────────────────────
    sp = subparsers.add_parser("swap", help="执行视频换脸")
    sp.add_argument("--source", "-s", required=True, help="参考人脸图片路径")
    sp.add_argument("--video", "-v", required=True, help="输入视频路径")
    sp.add_argument("--output", "-o", default=None, help="输出视频路径")
    sp.add_argument("--target", "-t", default=None,
                    help="目标人脸图片（视频中要替换的人），不指定则换所有人")
    sp.add_argument("--preprocess", action="store_true",
                    help="自动预处理参考图（对齐+增强）后再换脸")
    sp.add_argument("--preprocess-dir", default=None,
                    help="使用已预处理的结果目录（跳过预处理步骤）")
    sp.add_argument("--enhance", action="store_true",
                    help="换脸后做 GFPGAN 面部修复增强")
    sp.add_argument("--restore-model", default="gfpgan",
                    choices=["gfpgan", "codeformer", "none"],
                    help="面部修复模型 (default: gfpgan)")
    sp.add_argument("--target-threshold", type=float, default=0.2,
                    help="目标匹配阈值 (default: 0.2)")
    sp.add_argument("--match-threshold", type=float, default=0.35,
                    help="多人脸匹配阈值 (default: 0.35)")
    sp.add_argument("--skip-frames", type=int, default=0,
                    help="跳帧（每 N+1 帧处理一帧）")
    sp.add_argument("--start-frame", type=int, default=0, help="起始帧")
    sp.add_argument("--end-frame", type=int, default=-1, help="结束帧")
    sp.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="计算设备 (default: auto)")
    sp.add_argument("--det-size", type=int, default=0,
                    help="手动指定检测尺寸 (如 320)，0=自适应 (default: 0)")
    sp.add_argument("--crf", type=int, default=16,
                    help="H.264 编码质量 0-51 (默认 16，越小越清晰、文件越大; 设 0 跳过重编码)")

    # ── enhance ───────────────────────────────────────────────
    eh = subparsers.add_parser("enhance", help="对视频做面部修复增强")
    eh.add_argument("--video", "-v", required=True, help="输入视频路径")
    eh.add_argument("--output", "-o", required=True, help="输出视频路径")
    eh.add_argument("--restore", default="gfpgan",
                    choices=["gfpgan", "codeformer", "none"],
                    help="面部修复模型")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # ─── 执行预处理 ────────────────────────────────────────────
    # ─── 执行预处理 ────────────────────────────────────────────
    if args.command == "preprocess":
        det_size = (args.det_size, args.det_size) if args.det_size > 0 else (640, 640)
        swapper = VideoFaceSwapper(device="auto", det_size=det_size, enable_restoration=(not args.no_enhance))
        # 默认输出到 ~/Videos/face_swap/preprocess/<日期>/
        if args.output is None:
            src_base = os.path.splitext(os.path.basename(args.source))[0]
            date_subdir = time.strftime("%Y-%m-%d")
            args.output = os.path.join(DEFAULT_VIDEOS_DIR, "preprocess", date_subdir, src_base)
            os.makedirs(args.output, exist_ok=True)
        result = swapper.preprocess_face(
            image_path=args.source,
            output_dir=args.output,
            enhance=(not args.no_enhance),
            align=(not args.no_align),
        )
        print("\n[OK] 预处理完成!")
        print("   对齐图: ", result.aligned_face_path)
        print("   增强图: ", result.enhanced_path or "无")
        print("   嵌入:   ", result.embedding_path)
        print("   原始角度: %.1f°" % result.original_angle)
        print("   质量评分: %.2f" % result.quality_score)
        print("   最终参考图:", getattr(result, 'final_reference_path', 'N/A'))
        if hasattr(result, 'final_reference_path') and os.path.exists(result.final_reference_path):
            print("   ---> 换脸时使用 --source '%s' --preprocess-dir '%s'" %
                  (result.final_reference_path, os.path.dirname(result.embedding_path)))
        return

    # ─── 执行换脸 ──────────────────────────────────────────────
    if args.command == "swap":
        # 自动处理输出路径 → 默认 ~/Videos/face_swap/<日期>/
        if args.output is None:
            base = os.path.splitext(os.path.basename(args.video))[0]
            date_subdir = time.strftime("%Y-%m-%d")
            out_dir = os.path.join(DEFAULT_VIDEOS_DIR, date_subdir)
            os.makedirs(out_dir, exist_ok=True)
            args.output = os.path.join(out_dir, "%s_face_swapped.mp4" % base)

        det_size = (args.det_size, args.det_size) if args.det_size > 0 else (640, 640)
        swapper = VideoFaceSwapper(
            device=args.device,
            det_size=det_size,
            target_threshold=args.target_threshold,
            face_restore_model=args.restore_model if args.enhance else "none",
        )

        # 加载参考图 — v3.3.0: 嵌入优先策略
        source_face = None
        preprocess_dir = args.preprocess_dir

        if args.preprocess:
            # 自动预处理
            log("[流程] 自动预处理参考图...")
            source_dir = os.path.join(
                os.path.dirname(args.source),
                "ref_face_optimized"
            )
            result = swapper.preprocess_face(
                image_path=args.source,
                output_dir=source_dir,
                enhance=(args.restore_model != "none"),
                align=True,
            )
            preprocess_dir = os.path.dirname(result.embedding_path)
            # v3.3.0: 使用新方法加载（检测人脸 + 覆盖嵌入）
            source_face = swapper.load_source_face_from_preprocess(
                reference_image=args.source,
                preprocess_dir=preprocess_dir,
            )
            log("[流程] 参考人脸加载完成 [OK] (嵌入来自预处理)")

        elif preprocess_dir and os.path.isdir(preprocess_dir):
            # 使用已预处理的目录
            log("[流程] 从预处理目录加载参考人脸: %s" % preprocess_dir)
            source_face = swapper.load_source_face_from_preprocess(
                reference_image=args.source,
                preprocess_dir=preprocess_dir,
            )
            log("[流程] 参考人脸加载完成 [OK] (嵌入来自预处理)")

        else:
            # 直接从原始参考图加载
            source_face = load_face_from_image(swapper, args.source, label="原始参考图")

        # 加载目标人脸
        target_embedding = None
        if args.target:
            target_img = cv2.imread(args.target)
            target_face = swapper.extract_face(target_img, label="目标人脸")
            target_embedding = target_face.normed_embedding
            log("[流程] 目标人脸加载完成，锁定模式")
        else:
            log("[流程] 未指定目标 → 视频中所有人脸都会替换")

        # 执行换脸
        final_output = args.output
        if target_embedding is not None:
            raw_output = args.output
            swapper.process_video_target(
                video_path=args.video,
                source_face=source_face,
                target_embedding=target_embedding,
                output_path=raw_output,
                skip_frames=args.skip_frames,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
        else:
            raw_output = args.output
            swapper.process_video_all_faces(
                video_path=args.video,
                source_face=source_face,
                output_path=raw_output,
                skip_frames=args.skip_frames,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )

        # 面部修复增强
        if args.enhance:
            enhanced_output = os.path.splitext(args.output)[0] + "_enhanced.mp4"
            swapper.enhance_video(
                video_path=raw_output,
                output_path=enhanced_output,
                restore_model=args.restore_model,
            )
            final_output = enhanced_output

        # H.264 高画质重编码（--crf 0 可跳过）
        if args.crf > 0:
            _reencode_h264(final_output, crf=args.crf)

        print("\n[OK] 换脸完成!")
        print("   输出: %s" % final_output)
        return

    # ─── 执行增强 ──────────────────────────────────────────────
    if args.command == "enhance":
        swapper = VideoFaceSwapper(device="auto")
        swapper.enhance_video(
            video_path=args.video,
            output_path=args.output,
            restore_model=args.restore,
        )
        return


if __name__ == "__main__":
    main()
