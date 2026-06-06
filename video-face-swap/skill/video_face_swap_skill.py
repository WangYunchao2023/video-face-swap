#!/usr/bin/env python3
"""
视频换脸 Skill 实现
包装现有的 video_face_swap.py 功能，提供简洁的接口供 Skill 系统使用
版本: 1.2.0

v1.2.0 (2026-06-06) 新增智能参数自动选择：
  - 新增 _analyze_reference_quality()：分析参考图质量（尺寸、角度、清晰度、光照）
  - 新增 _analyze_video()：分析视频信息（分辨率、帧率、时长、人脸数）
  - 新增 _auto_select_params()：根据分析结果自动选择最优参数
  - 新增 --auto 参数：一键启用智能参数推荐
  - 质量评分现在用于自动决策（质量档位、增强策略、阈值）

v1.1.0 (2026-06-06) 核心修复：
  - 修复重大 bug：预处理模式下从 512x512 缩略图重新检测提取 source_face，
    导致嵌入质量差、换脸结果不像参考图
  - 修复：预处理后优先加载 face_embedding.npy 的嵌入，再结合 final_reference.png
    （原始分辨率裁剪版）检测人脸，然后用保存的嵌入覆盖 source_face.normed_embedding
  - 修复：不使用 GFPGAN 增强后的图提取嵌入（避免 ID 漂移）
  - 同步升级至底层 v3.3.0 的 norm_crop=512 和 final_reference 原始分辨率输出
"""

import os
import sys
import argparse
import time
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# 将项目根目录添加到 Python 路径以便导入现有模块
# 将 engine 目录添加到路径
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.join(current_dir, '..', 'engine')
sys.path.insert(0, engine_dir)

# 导入现有的换脸工具
try:
    # 从 engine 目录导入底层引擎
from video_face_swap import VideoFaceSwapper, log
except ImportError as e:
    log(f"[错误] 无法导入 video_face_swap 模块: {e}")
    sys.exit(1)


class VideoFaceSwapSkill:
    """视频换脸 Skill 实现类"""

    def __init__(self, device: str = "auto"):
        """
        初始化换脸 Skill

        Args:
            device: 计算设备 ("auto", "cuda", "cpu")
        """
        self.device = device
        self.swapper = None
        self._initialize_swapper()

    def _initialize_swapper(self):
        """初始化底层换脸引擎"""
        try:
            self.swapper = VideoFaceSwapper(device=self.device)
            log("[Skill] 视频换脸引擎初始化成功")
        except Exception as e:
            log(f"[错误] 初始化换脸引擎失败: {e}")
            raise

    def _load_source_face_with_embedding(self, img: np.ndarray, label: str,
                                         embedding: np.ndarray = None) -> object:
        """
        从图像检测人脸，并可选覆盖其嵌入向量。

        核心修复(v1.1.0)：
        - 检测人脸后，如果提供了 embedding，用其覆盖 face.embedding
        - 注意：Face.normed_embedding 是 @property（从 self.embedding 实时计算）
        - 因此给 face.embedding 赋值即可
        - 确保换脸时使用的嵌入来自高质量的预处理流程而非缩略图检测

        Args:
            img: BGR 图像
            label: 日志标签
            embedding: 可选，来自预处理的高质量嵌入向量（已归一化 norm≈1.0）

        Returns:
            face 对象（embedding 可能已被替换）
        """
        face = self.swapper.extract_face(img, label=label)
        if embedding is not None:
            original_normed = face.normed_embedding.copy()
            face.embedding = embedding.copy()
            cos_sim = float(np.dot(original_normed, face.normed_embedding))
            log(f"[Skill] ✅ {label}: 嵌入替换完成 "
                f"(替换后 normed_embedding norm={np.linalg.norm(face.normed_embedding):.4f}, "
                f"与原检测嵌入 cosine={cos_sim:.4f})")
        return face

    def process(self,
                reference_image: str,
                target_video: str,
                output_path: Optional[str] = None,
                target_face_image: Optional[str] = None,
                enhance: bool = True,
                restore_model: str = "gfpgan",
                target_threshold: float = 0.2,
                match_threshold: float = 0.35,
                skip_frames: int = 0,
                start_frame: int = 0,
                end_frame: int = -1,
                quality: str = "normal",
                preprocess: bool = False,
                preprocess_dir: Optional[str] = None,
                auto_params: bool = False,
                verbose: bool = False) -> str:
        """
        执行视频换脸处理

        Args:
            reference_image: 参考人脸图片路径
            target_video: 输入视频路径
            output_path: 输出视频路径（可选）
            target_face_image: 目标人脸图片路径（可选，启用目标锁定）
            enhance: 是否进行面部修复增强
            restore_model: 面部修复模型类型 ("gfpgan", "codeformer", "none")
            target_threshold: 目标匹配阈值 (0-1)
            match_threshold: 多人脸匹配阈值
            skip_frames: 跳帧数（每 N+1 帧处理一帧）
            start_frame: 起始帧
            end_frame: 结束帧（-1 表示视频末尾）
            quality: 质量档位 ("draft", "normal", "hq", "best")
            preprocess: 是否自动预处理参考图
            preprocess_dir: 使用已预处理的参考图目录
            auto_params: 是否自动选择参数（根据参考图和视频质量）
            verbose: 是否显示详细日志

        Returns:
            输出视频路径
        """
        if not self.swapper:
            raise RuntimeError("换脸引擎未初始化")

        # 根据质量档位调整参数
        quality_settings = self._get_quality_settings(quality)
        if verbose:
            log(f"[Skill] 使用质量档位: {quality}, 设置: {quality_settings}")

        # 确定输出路径
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = Path(target_video).stem
            output_dir = os.path.join(os.path.expanduser("~"), "Videos", "face_swap")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"{video_name}_{timestamp}_face_swap.mp4"
            )

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # =======================================================
            # 自动参数选择（v1.2.0 新增）
            # =======================================================
            if auto_params and not preprocess_dir:
                log("[Skill] 启用智能参数自动选择...")
                auto_params_dict = self._auto_select_params(
                    reference_image, target_video,
                    target_mode=(target_face_image is not None)
                )
                
                # 应用自动选择的参数（用户显式指定的参数优先）
                if quality == "normal":  # 用户未指定
                    quality = auto_params_dict['quality']
                if skip_frames == 0:  # 用户未指定
                    skip_frames = auto_params_dict['skip_frames']
                if not preprocess:  # 用户未指定
                    preprocess = auto_params_dict['preprocess']
                
                # 增强相关参数
                if restore_model == "gfpgan" and not enhance:
                    restore_model = auto_params_dict['restore_model']
                
                if target_face_image and target_threshold == 0.2:
                    target_threshold = auto_params_dict['target_threshold']
                
                log(f"[Skill] 自动参数：quality={quality}, skip_frames={skip_frames}, "
                    f"preprocess={preprocess}, enhance={auto_params_dict['enhance']}")
            
            # =======================================================
            # 加载参考人脸 — 核心修复 (v1.1.0)
            # =======================================================
            source_face = None
            saved_embedding = None  # 如有预处理，从此文件加载
            final_ref_path = None   # 最终参考图路径（原始分辨率裁剪版）

            if preprocess:
                # ---- 模式 A: 自动预处理 ----
                log("[Skill] 自动预处理参考图...")
                preprocess_result = self.swapper.preprocess_face(
                    image_path=reference_image,
                    output_dir=None,
                    enhance=(restore_model != "none"),
                    align=True
                )
                preprocess_dir_used = os.path.dirname(preprocess_result.embedding_path)

                # 加载保存的高质量嵌入
                emb_path = os.path.join(preprocess_dir_used, "face_embedding.npy")
                if os.path.exists(emb_path):
                    saved_embedding = np.load(emb_path)
                    log(f"[Skill] ✅ 加载保存的嵌入: {emb_path} (shape={saved_embedding.shape})")

                # v1.1.0 修复：从 final_reference.png（原始分辨率裁剪版）检测人脸
                # 不再从 512x512 的对齐/增强缩略图检测
                final_ref_path = os.path.join(preprocess_dir_used, "final_reference.png")
                if os.path.exists(final_ref_path):
                    ref_img = cv2.imread(final_ref_path)
                    if ref_img is not None:
                        try:
                            source_face = self._load_source_face_with_embedding(
                                ref_img, "最终参考图", saved_embedding)
                        except ValueError:
                            log("[Skill] final_reference.png 未检测到人脸，尝试从原始图检测")
                            source_face = None

                # 回退：从原始图检测 + 用保存的嵌入覆盖
                if source_face is None:
                    log("[Skill] 从原始参考图提取人脸（用已保存嵌入覆盖）...")
                    source_img = cv2.imread(reference_image)
                    source_face = self._load_source_face_with_embedding(
                        source_img, "原始参考图", saved_embedding)

            elif preprocess_dir and os.path.isdir(preprocess_dir):
                # ---- 模式 B: 使用已预处理的目录 ----
                log(f"[Skill] 从预处理目录加载: {preprocess_dir}")

                # 加载保存的嵌入
                emb_path = os.path.join(preprocess_dir, "face_embedding.npy")
                if os.path.exists(emb_path):
                    saved_embedding = np.load(emb_path)
                    log(f"[Skill] ✅ 加载保存的嵌入: {emb_path}")

                # 从 final_reference.png（原始分辨率）检测人脸 + 覆盖嵌入
                final_ref_path = os.path.join(preprocess_dir, "final_reference.png")
                if os.path.exists(final_ref_path):
                    ref_img = cv2.imread(final_ref_path)
                    if ref_img is not None:
                        try:
                            source_face = self._load_source_face_with_embedding(
                                ref_img, "最终参考图", saved_embedding)
                        except ValueError:
                            log("[Skill] final_reference.png 未检测到人脸，回退到原始图")

                # 回退：meta.json 中的原始路径
                if source_face is None:
                    meta_path = os.path.join(preprocess_dir, "preprocess_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        original_src = meta.get("source", "")
                        if original_src and os.path.exists(original_src):
                            source_img = cv2.imread(original_src)
                            source_face = self._load_source_face_with_embedding(
                                source_img, "原始参考图(来自meta)", saved_embedding)

                if source_face is None:
                    raise RuntimeError("无法从预处理目录加载参考人脸")

            else:
                # ---- 模式 C: 直接从原始参考图加载（无预处理） ----
                source_img = cv2.imread(reference_image)
                if source_img is None:
                    raise ValueError(f"无法读取参考图像: {reference_image}")
                source_face = self.swapper.extract_face(source_img, label="原始参考图")
                log("[Skill] 参考人脸加载完成（无预处理）")

            # 嵌入已就绪，验证
            if source_face is None:
                raise RuntimeError("无法提取参考人脸")
            log(f"[Skill] 最终 source_face 嵌入 norm={np.linalg.norm(source_face.normed_embedding):.4f}")

            # 加载目标人脸（如果指定）
            target_embedding = None
            if target_face_image:
                target_img = cv2.imread(target_face_image)
                if target_img is None:
                    raise ValueError(f"无法读取目标人脸图像: {target_face_image}")
                target_face = self.swapper.extract_face(target_img, label="目标人脸")
                target_embedding = target_face.normed_embedding
                log("[Skill] 目标人脸加载完成，启用目标锁定模式")
            else:
                log("[Skill] 未指定目标人脸 → 视频中所有人脸都会替换")

            # =======================================================
            # 执行换脸
            # =======================================================
            if target_embedding is not None:
                # 目标锁定模式
                raw_output = output_path
                if enhance and restore_model != "none":
                    raw_output = output_path.replace(".mp4", "_raw.mp4")

                self.swapper.process_video_target(
                    video_path=target_video,
                    source_face=source_face,
                    target_embedding=target_embedding,
                    output_path=raw_output,
                    skip_frames=skip_frames,
                    start_frame=start_frame,
                    end_frame=end_frame
                )

                if enhance and restore_model != "none":
                    enhanced_output = output_path
                    self.swapper.enhance_video(
                        video_path=raw_output,
                        output_path=enhanced_output,
                        restore_model=restore_model
                    )
                    if os.path.exists(raw_output) and raw_output != output_path:
                        os.remove(raw_output)
                elif enhance and restore_model == "none":
                    log("[Skill] 警告: 增强已启用但修复模型设置为'none'，跳过增强步骤")

            else:
                # 全换模式
                raw_output = output_path
                if enhance and restore_model != "none":
                    raw_output = output_path.replace(".mp4", "_raw.mp4")

                self.swapper.process_video_all_faces(
                    video_path=target_video,
                    source_face=source_face,
                    output_path=raw_output,
                    skip_frames=skip_frames,
                    start_frame=start_frame,
                    end_frame=end_frame
                )

                if enhance and restore_model != "none":
                    enhanced_output = output_path
                    self.swapper.enhance_video(
                        video_path=raw_output,
                        output_path=enhanced_output,
                        restore_model=restore_model
                    )
                    if os.path.exists(raw_output) and raw_output != output_path:
                        os.remove(raw_output)
                elif enhance and restore_model == "none":
                    log("[Skill] 警告: 增强已启用但修复模型设置为'none'，跳过增强步骤")

            log(f"[Skill] ✅ 换脸处理完成! 输出: {output_path}")
            return output_path

        except Exception as e:
            log(f"[错误] 换脸处理失败: {e}")
            raise

    def _analyze_reference_quality(self, img: np.ndarray) -> Dict[str, Any]:
        """
        分析参考人脸图片质量
        
        Args:
            img: BGR 图像
            
        Returns:
            质量分析结果字典
        """
        h, w = img.shape[:2]
        
        # 检测人脸
        faces = self.swapper.app.get(img)
        if not faces:
            # 尝试自适应检测
            faces = self.swapper._detect_adaptive(img)
        
        if not faces:
            return {
                'score': 0.0,
                'face_size': 0,
                'yaw': 90.0,
                'brightness': 0.5,
                'sharpness': 0.0,
                'issues': ['未检测到人脸']
            }
        
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        bbox = face.bbox.astype(int)
        face_w, face_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        face_size = min(face_w, face_h)
        
        # 计算偏转角度
        kps = face.kps
        eye_dist = np.linalg.norm(kps[0] - kps[1])
        nose_offset = abs(kps[2][0] - (kps[0][0] + kps[1][0]) / 2)
        yaw = np.degrees(np.arctan(nose_offset / max(eye_dist, 1)))
        
        # 计算亮度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean() / 255.0
        
        # 计算对比度
        contrast = gray.std() / 255.0
        
        # 计算清晰度（拉普拉斯方差）
        face_crop = gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face_crop.size > 0:
            sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var()
            sharpness_norm = min(sharpness / 500.0, 1.0)  # 归一化到 0-1
        else:
            sharpness_norm = 0.0
        
        # 综合质量评分
        angle_score = max(0, 1.0 - yaw / 45.0)
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 0.5 最佳
        size_score = min(face_size / 400.0, 1.0)  # 400px 最佳
        
        overall_score = (
            angle_score * 0.3 +
            brightness_score * 0.2 +
            size_score * 0.3 +
            sharpness_norm * 0.2
        )
        
        issues = []
        if face_size < 200:
            issues.append('人脸尺寸偏小')
        if yaw > 30:
            issues.append('偏转角度较大')
        if brightness < 0.3 or brightness > 0.7:
            issues.append('光照不均')
        if sharpness_norm < 0.3:
            issues.append('图像模糊')
        
        return {
            'score': round(overall_score, 3),
            'face_size': face_size,
            'yaw': round(yaw, 2),
            'brightness': round(brightness, 3),
            'contrast': round(contrast, 3),
            'sharpness': round(sharpness_norm, 3),
            'issues': issues
        }
    
    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        分析视频基本信息
        
        Args:
            video_path: 视频路径
            
        Returns:
            视频分析结果字典
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'error': '无法打开视频',
                'resolution': (0, 0),
                'fps': 0,
                'duration_sec': 0,
                'total_frames': 0,
                'face_count': 0
            }
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        # 采样检测人脸（每 30 帧检测一次）
        face_count = 0
        detection_frames = 0
        sample_interval = max(1, total_frames // 30)  # 最多检测 30 次
        
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.swapper.app.get(frame)
            if faces:
                face_count = max(face_count, len(faces))
            detection_frames += 1
        
        cap.release()
        
        # 估算场景变化数（简化：基于亮度变化）
        scene_changes = 0
        cap = cv2.VideoCapture(video_path)
        prev_brightness = None
        for i in range(0, min(total_frames, 300), 10):  # 每 10 帧采样，最多 300 帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            if prev_brightness is not None:
                if abs(brightness - prev_brightness) > 40:  # 亮度突变
                    scene_changes += 1
            prev_brightness = brightness
        cap.release()
        
        return {
            'resolution': (width, height),
            'fps': round(fps, 2),
            'duration_sec': round(duration_sec, 1),
            'total_frames': total_frames,
            'face_count': face_count,
            'scene_changes': scene_changes,
            'resolution_str': f'{width}x{height}'
        }
    
    def _auto_select_params(self, reference_image: str, target_video: str,
                            target_mode: bool = False) -> Dict[str, Any]:
        """
        根据参考图和视频的实际情况自动选择最优参数
        
        Args:
            reference_image: 参考图路径
            target_video: 目标视频路径
            target_mode: 是否目标锁定模式
            
        Returns:
            推荐的参数字典
        """
        log("[自动参数] 分析参考图和视频...")
        
        # 分析参考图
        ref_img = cv2.imread(reference_image)
        if ref_img is None:
            log("[自动参数] 警告：无法读取参考图，使用默认参数")
            return self._get_default_params()
        
        ref_info = self._analyze_reference_quality(ref_img)
        log(f"[自动参数] 参考图质量评分：{ref_info['score']:.2f}")
        if ref_info['issues']:
            log(f"[自动参数] 参考图问题：{', '.join(ref_info['issues'])}")
        
        # 分析视频
        video_info = self._analyze_video(target_video)
        if 'error' in video_info:
            log(f"[自动参数] 警告：{video_info['error']}，使用默认参数")
            return self._get_default_params()
        
        log(f"[自动参数] 视频：{video_info['resolution_str']} @ {video_info['fps']}fps, "
            f"{video_info['duration_sec']:.1f}s, {video_info['face_count']}人")
        
        # 1. 自动选择质量档位
        if ref_info['score'] < 0.4:
            quality = 'normal'
            log("[自动参数] 参考图质量较低 → 使用 normal 档位（避免浪费资源）")
        elif video_info['resolution'][0] <= 1280:
            quality = 'normal'
            log("[自动参数] 视频分辨率 ≤ 720p → 使用 normal 档位")
        elif ref_info['score'] >= 0.7 and video_info['resolution'][0] >= 1920:
            quality = 'hq'
            log("[自动参数] 参考图质量高 + 视频 1080p+ → 使用 hq 档位")
        else:
            quality = 'normal'
        
        # 2. 自动选择跳帧策略
        if video_info['duration_sec'] > 300:  # > 5 分钟
            skip_frames = 2
            log("[自动参数] 长视频 (>5 分钟) → 跳帧=2（每 3 帧处理 1 帧）")
        elif video_info['duration_sec'] > 120:  # 2-5 分钟
            skip_frames = 1
            log("[自动参数] 中等时长视频 (2-5 分钟) → 跳帧=1（每 2 帧处理 1 帧）")
        elif video_info['scene_changes'] > 15:
            skip_frames = 1
            log("[自动参数] 场景变化频繁 → 跳帧=1 加速")
        else:
            skip_frames = 0
            log("[自动参数] 短视频/场景稳定 → 全帧处理")
        
        # 3. 自动选择增强策略
        if ref_info['face_size'] < 150:
            enhance = False
            log("[自动参数] 人脸尺寸 < 150px → 跳过增强（效果有限）")
        elif ref_info['yaw'] > 25:
            enhance = True
            log("[自动参数] 偏转角度 > 25° → 启用增强（修正姿态）")
        elif ref_info['sharpness'] < 0.4:
            enhance = True
            log("[自动参数] 图像模糊 → 启用增强（提升清晰度）")
        else:
            enhance = True
            log("[自动参数] 默认启用增强")
        
        # 4. 自动选择修复模型
        if enhance:
            if ref_info['face_size'] >= 300 and ref_info['score'] >= 0.6:
                restore_model = 'gfpgan'
                log("[自动参数] 高质量参考图 → 使用 GFPGAN")
            else:
                restore_model = 'gfpgan'
                log("[自动参数] 使用 GFPGAN（通用场景）")
        else:
            restore_model = 'none'
        
        # 5. 自动调整匹配阈值
        if target_mode:
            if video_info['face_count'] >= 3:
                target_threshold = 0.25
                log("[自动参数] 多人脸场景 → 提高目标阈值至 0.25（更严格）")
            else:
                target_threshold = 0.2
                log("[自动参数] 使用默认目标阈值 0.2")
        else:
            target_threshold = 0.2
        
        # 6. 自动决定是否预处理
        if ref_info['score'] < 0.5 or ref_info['yaw'] > 20:
            preprocess = True
            log("[自动参数] 参考图质量一般或角度偏大 → 启用预处理（对齐 + 增强）")
        else:
            preprocess = False
            log("[自动参数] 参考图质量良好 → 跳过预处理")
        
        return {
            'quality': quality,
            'skip_frames': skip_frames,
            'enhance': enhance,
            'restore_model': restore_model,
            'target_threshold': target_threshold,
            'preprocess': preprocess,
            'ref_score': ref_info['score'],
            'video_info': video_info
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        """返回默认参数"""
        return {
            'quality': 'normal',
            'skip_frames': 0,
            'enhance': True,
            'restore_model': 'gfpgan',
            'target_threshold': 0.2,
            'preprocess': False,
            'ref_score': 0.5
        }


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="视频换脸 Skill v1.2.0 — 智能参数自动选择 — 高质量视频换脸工具（嵌入优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础换脸（所有人脸都换）
  %(prog)s --reference ref.jpg --video input.mp4 --output output.mp4

  # 目标锁定换脸（仅换指定目标人脸）
  %(prog)s --reference ref.jpg --video input.mp4 --target target.jpg --output output.mp4

  # 使用预处理的参考图（推荐用于反复使用同一参考图）
  %(prog)s --reference ref.jpg --video input.mp4 --preprocess-dir ./preprocessed_ref/ --output output.mp4

  # 高质量模式
  %(prog)s --reference ref.jpg --video input.mp4 --output output.mp4 --quality hq --enhance

  # 跳帧加速处理长视频
  %(prog)s --reference ref.jpg --video input.mp4 --output output.mp4 --skip-frames 2
        """
    )

    # 必要参数
    parser.add_argument("--reference", "-r", required=True, help="参考人脸图片路径")
    parser.add_argument("--video", "-v", required=True, help="输入视频路径")

    # 输出参数
    parser.add_argument("--output", "-o", help="输出视频路径（可选，默认自动生成）")

    # 目标锁定参数
    parser.add_argument("--target", "-t", help="目标人脸图片路径（启用目标锁定模式）")
    parser.add_argument("--target-threshold", type=float, default=0.2,
                        help="目标匹配阈值 (0-1)，默认: 0.2")
    parser.add_argument("--match-threshold", type=float, default=0.35,
                        help="多人脸匹配阈值，默认: 0.35")

    # 处理选项
    parser.add_argument("--enhance", action="store_true", default=True,
                        help="换脸后做面部修复增强（默认启用）")
    parser.add_argument("--no-enhance", action="store_false", dest="enhance",
                        help="禁用面部修复增强")
    parser.add_argument("--restore-model", default="gfpgan",
                        choices=["gfpgan", "codeformer", "none"],
                        help="面部修复模型类型 (默认: gfpgan)")

    # 质量和性能选项
    parser.add_argument("--quality", choices=["draft", "normal", "hq", "best"],
                        default="normal", help="质量档位 (默认: normal)")
    parser.add_argument("--preprocess", action="store_true",
                        help="自动预处理参考图（对齐+增强）")
    parser.add_argument("--preprocess-dir", help="使用已预处理的参考图目录（跳过预处理步骤）")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="跳帧（每 N+1 帧处理一帧，默认: 0）")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧（默认: 0）")
    parser.add_argument("--end-frame", type=int, default=-1, help="结束帧（-1 表示视频末尾，默认: -1）")

    # 智能参数选项（v1.2.0 新增）
    parser.add_argument("--auto", action="store_true", dest="auto_params",
                        help="自动分析参考图和视频质量，智能选择最优参数（推荐）")

    # 系统选项
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="计算设备 (默认: auto)")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")

    args = parser.parse_args()

    # 参数验证
    if args.preprocess and args.preprocess_dir:
        print("[错误] --preprocess 和 --preprocess-dir 不能同时使用")
        sys.exit(1)

    try:
        # 初始化 Skill
        skill = VideoFaceSwapSkill(device=args.device)

        # 执行换脸
        result = skill.process(
            reference_image=args.reference,
            target_video=args.video,
            output_path=args.output,
            target_face_image=args.target,
            enhance=args.enhance,
            restore_model=args.restore_model,
            target_threshold=args.target_threshold,
            match_threshold=args.match_threshold,
            skip_frames=args.skip_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            quality=args.quality,
            preprocess=args.preprocess,
            preprocess_dir=args.preprocess_dir,
            auto_params=args.auto_params,
            verbose=args.verbose
        )

        print(f"\n✅ 换脸完成!")
        print(f"   输出文件: {result}")

    except Exception as e:
        print(f"[错误] 处理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()