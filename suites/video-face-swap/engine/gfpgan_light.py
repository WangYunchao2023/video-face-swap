#!/usr/bin/env python3
"""
GFPGAN 轻量封装 — 绕过 facexlib 检测模型依赖
=============================================
直接加载 GFPGAN 生成器（跳过 FaceRestoreHelper 的面部检测），
配合 insightface 完成面部增强。

用法:
  from gfpgan_light import GFPGANLight
  enhancer = GFPGANLight()
  enhanced = enhancer.enhance_aligned(aligned_face_bgr)
"""

import os
import cv2
import numpy as np
import torch


class GFPGANLight:
    """轻量级 GFPGAN 封装，无需 facexlib 检测模型"""

    def __init__(self, model_path: str = None, device: str = "auto"):
        if model_path is None:
            model_path = os.path.expanduser(
                "~/ComfyUI/models/facerestore_models/GFPGANv1.4.pth"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError("GFPGAN 模型不存在: %s" % model_path)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        self.gfpgan = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )

        loadnet = torch.load(model_path, map_location="cpu", weights_only=False)
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        ).to(self.device)

        print("[GFPGANLight] 模型加载完成 (%s, %s)" % (
            os.path.basename(model_path), self.device))

    def enhance_aligned(self, img_bgr: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """
        增强已对齐的人脸（norm_crop 输出）

        Args:
            img_bgr: BGR 图像（已对齐，如 norm_crop 输出）
            weight: 原始图像保留权重 (0~1), 0.5 是默认值

        Returns:
            BGR 增强图像
        """
        h, w = img_bgr.shape[:2]

        # GFPGAN 需要 512x512 输入
        if h != 512 or w != 512:
            img_bgr = cv2.resize(img_bgr, (512, 512))

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.gfpgan(img_tensor, weight=weight)
            # GFPGAN returns tuple (output, _restored, weight_param)
            if isinstance(output, (list, tuple)):
                output = output[0]
            output = output.clamp(0, 1)
            output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
            output = (output * 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # 如果输出尺寸不同，缩放到原尺寸
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h))

        return output

    def enhance_unrestored(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        调用 GFPGAN 输出原始质量（不 mix）
        """
        return self.enhance_aligned(img_bgr, weight=0.0)

    def enhance_full_restore(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        完全恢复（weight=1 表示全部替换为生成结果，去掉了原始退化）
        """
        return self.enhance_aligned(img_bgr, weight=1.0)


if __name__ == "__main__":
    # 简单测试
    print("=== GFPGANLight 测试 ===")
    enhancer = GFPGANLight()
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = enhancer.enhance_aligned(test_img)
    print("输入: %s → 输出: %s" % (str(test_img.shape), str(result.shape)))
    print("测试通过 ✅")
