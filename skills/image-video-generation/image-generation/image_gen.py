#!/usr/bin/env python3
"""
image_gen.py — ImageGen Skill v2.0.0
智能图片生成编排器

功能：
  1. 自然语言分析 → 自动选择最佳模型 + 参数
  2. 本地 ComfyUI 执行（SD3 / Flux Schnell / Flux Dev）
  3. 反馈循环：效果不好时自动调参或换模型重试
  4. 降级到 image_generate 工具（云模型）
"""

import os, sys, json, re, time, shutil, hashlib
import argparse
from pathlib import Path
from datetime import datetime

# ── 路径常量 ─────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).parent.resolve()
COMFY_SCRIPTS = Path.home() / ".openclaw" / "skills" / "comfy-workflow" / "scripts"
SCRIPTS_DIR = Path.home() / ".openclaw" / "scripts"
OUTPUT_DIR = Path(os.environ.get("PICTURE_DIR", Path.home() / "Pictures" / "image-gen"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(COMFY_SCRIPTS))
sys.path.insert(0, str(SCRIPTS_DIR))

# ── 模型知识库 ───────────────────────────────────────────────────────

# 模型能力矩阵：决定哪种场景用哪个模型
MODEL_CAPABILITIES = {
    "sd3": {
        "name": "SD3 Medium",
        "file": "sd3-medium-incl_clips.safetensors",
        "vram_gb": 10,
        "default_steps": 30,
        "best_for": ["portrait", "human", "person", "face", "retro", "vintage", "film"],
        "strength": "人物近景、人像、老照片、胶片质感 — 细节最丰富",
        "weakness": "多人复杂场景不如 Flux",
        "cfg_range": (4.0, 7.0),
        "default_cfg": 5.0,
        "max_steps": 50,
    },
    "flux_schnell": {
        "name": "Flux.1 Schnell FP8",
        "file": "flux1-schnell/flux1-schnell-fp8.safetensors",
        "vram_gb": 9,
        "default_steps": 4,
        "best_for": ["landscape", "scenery", "architecture", "animal", "multi_person", "full_body", "indoor", "scene"],
        "strength": "风景、建筑、多人场景、全身 — 构图好速度快",
        "weakness": "低步数时手部细节不如 SD3",
        "cfg_range": (1.0, 2.0),
        "default_cfg": 1.0,
        "max_steps": 8,
    },
    "flux_dev": {
        "name": "Flux.1 Dev FP8",
        "file": "flux1-dev-fp8/flux1-dev-fp8-e4m3fn.safetensors",
        "vram_gb": 12,
        "default_steps": 25,
        "best_for": ["high_quality", "detailed", "masterpiece", "professional", "fine_art"],
        "strength": "最高质量精细渲染",
        "weakness": "速度较慢，需要更多步数",
        "cfg_range": (1.0, 3.5),
        "default_cfg": 2.0,
        "max_steps": 40,
    },
}

# 分辨率指南
RESOLUTION_PRESETS = {
    "portrait": (768, 1024),
    "landscape": (1024, 768),
    "square": (1024, 1024),
    "wide": (1280, 720),
    "small_portrait": (384, 512),  # 小图 → 放大路径
    "small_square": (512, 512),
}

# 风格关键词映射
STYLE_KEYWORDS = {
    "retro": ["vintage", "film grain", "faded colors", "Kodak", "analog", "1970s", "1980s"],
    "cyberpunk": ["neon lights", "dark", "rain", "futuristic", "high tech"],
    "photorealistic": ["photorealistic", "detailed", "DSLR", "professional photography"],
    "anime": ["anime style", "cel shaded", "vibrant"],
    "watercolor": ["watercolor", "soft colors", "hand painted"],
    "oil_painting": ["oil painting", "impasto", "brush strokes"],
    "sketch": ["pencil sketch", "black and white", "line art"],
    "minimalist": ["minimalist", "clean", "simple background"],
    "cinematic": ["cinematic lighting", "film still", "dramatic lighting", "shallow depth of field"],
}

# 负面提示词模板
NEGATIVE_COMMON = "deformed, blurry, bad anatomy, bad hands, extra fingers, watermark, text, logo, cropped, worst quality, low quality, signature, username"

NEGATIVE_PORTRAIT = "smooth skin, perfect teeth, digital skin retouching, beauty filter, overexposed, washed out, flat lighting, perfect symmetric face"

# ── 场景分析器 ────────────────────────────────────────────────────────

def analyze_request(description: str) -> dict:
    """
    分析自然语言请求，返回最合适的模型和参数。
    
    返回:
        {
            "model_key": "sd3" | "flux_schnell" | "flux_dev",
            "params": {
                "steps": int,
                "cfg": float,
                "width": int,
                "height": int,
                "negative": str,
            },
            "reason": str,   # 选择理由
            "subject_type": str,  # portrait/landscape/detailed...
            "style_hints": list,
        }
    """
    desc_lower = description.lower()
    
    # === Step 1: 识别主体类型 ===
    subject_type = _classify_subject(desc_lower)
    
    # === Step 2: 识别风格 ===
    style_hints = _extract_style(desc_lower)
    
    # === Step 3: 识别质量要求 ===
    quality_tier = _detect_quality(desc_lower)
    
    # === Step 4: 选模型 ===
    model_key, reason = _select_model(subject_type, quality_tier, style_hints, desc_lower)
    
    # === Step 5: 定参数 ===
    params = _build_params(model_key, subject_type, style_hints, quality_tier, description)
    
    # === Step 6: 增强提示词 ===
    enhanced_prompt = _enhance_prompt(description, style_hints, model_key, subject_type, quality_tier)
    
    return {
        "model_key": model_key,
        "params": params,
        "reason": reason,
        "subject_type": subject_type,
        "style_hints": style_hints,
        "quality_tier": quality_tier,
        "enhanced_prompt": enhanced_prompt,
    }


def _classify_subject(text: str) -> str:
    """分类主体类型"""
    portrait_kw = [
        "portrait", "人像", "肖像", "人物", "face", "人脸", "headshot", "特写",
        "老奶奶", "老人", "child", "woman", "man", "person",
        "她", "他", "character", "角色", "selfie",
    ]
    landscape_kw = [
        "landscape", "风景", "scenery", "mountain", "山", "river", "河",
        "ocean", "sea", "海", "forest", "森林", "cityscape", "城市",
        "architecture", "建筑", "building", "building",
        "indoor", "室内", "room", "房间", "interior", "interior",
    ]
    animal_kw = ["animal", "动物", "cat", "猫", "dog", "狗", "bird", "bird"]
    food_kw = ["food", "食物", "dish", "菜", "fruit", "水果"]
    still_life_kw = ["still life", "静物", "flower", "花", "vase"]
    
    # 计算分数
    scores = {}
    for cat, kws in [("portrait", portrait_kw), ("landscape", landscape_kw),
                     ("animal", animal_kw), ("food", food_kw), ("still_life", still_life_kw)]:
        score = sum(1 for kw in kws if kw in text)
        if score:
            scores[cat] = score
    
    if not scores:
        return "general"
    
    return max(scores, key=scores.get)


def _extract_style(text: str) -> list:
    """提取风格关键词"""
    found = []
    for style_name, keywords in STYLE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text or style_name in text:
                found.append(style_name)
                break
    # 中文风格检测
    cn_styles = {
        "vintage": ["复古", "怀旧", "年代", "旧", "老照片", "胶片", "1980", "1990"],
        "anime": ["动漫", "卡通", "动画", "二次元"],
        "watercolor": ["水彩"],
        "sketch": ["素描", "速写", "线稿"],
        "cyberpunk": ["赛博", "朋克", "科幻"],
        "minimalist": ["极简", "简约"],
        "cinematic": ["电影感", "电影级", "戏剧"],
    }
    for style_name, cn_kws in cn_styles.items():
        if any(kw in text for kw in cn_kws):
            if style_name not in found:
                found.append(style_name)
    
    return found


def _detect_quality(text: str) -> str:
    """检测质量要求 (low/medium/high)"""
    high_kw = ["high quality", "高质量", "精细", "detailed", "masterpiece",
               "专业", "professional", "极致", "8k", "ultra", "best"]
    low_kw = ["fast", "quick", "快速", "quick", "草稿", "draft", "thumbnail"]
    
    high_score = sum(1 for kw in high_kw if kw in text)
    low_score = sum(1 for kw in low_kw if kw in text)
    
    if high_score > low_score:
        return "high"
    elif low_score > high_score:
        return "low"
    return "medium"


def _select_model(subject_type: str, quality: str, styles: list, text: str) -> tuple:
    """根据分析结果选择最佳模型"""
    # 预先匹配特殊场景
    is_portrait = subject_type in ("portrait",)
    is_landscape = subject_type in ("landscape", "animal")
    is_high_quality = quality == "high"
    is_vintage = any(s in ("vintage", "retro") for s in styles)
    
    # 老照片/复古 → SD3（已验证人物质感最好）
    if is_vintage and is_portrait:
        return "sd3", "老照片/复古人像 → SD3 Medium（胶片质感最佳）"
    
    # 人物近景 → SD3 Medium（人物细节最好）
    if is_portrait:
        if is_high_quality:
            return "sd3", "高质量人像 → SD3 Medium（人物细节最丰富）"
        return "sd3", "人物肖像 → SD3 Medium（CLIP细节好）"
    
    # 高质量精细渲染 → Flux Dev
    if is_high_quality:
        return "flux_dev", f"高质量{subject_type} → Flux.1 Dev FP8（精细渲染）"
    
    # 风景/动物 → Flux Schnell（快且构图好）
    if is_landscape:
        return "flux_schnell", f"{subject_type} → Flux.1 Schnell FP8（快速+构图好）"
    
    # 默认为 Flux Schnell
    return "flux_schnell", f"通用场景 → Flux.1 Schnell FP8（平衡速度质量）"


def _build_params(model_key: str, subject_type: str, styles: list,
                  quality: str, original_text: str) -> dict:
    """构建生成参数"""
    model = MODEL_CAPABILITIES[model_key]
    
    # 步数
    if quality == "high":
        steps = model["max_steps"]
    elif quality == "low":
        steps = model["default_steps"] // 2 if model_key != "flux_schnell" else 4
    else:
        steps = model["default_steps"]
    steps = max(steps, model["default_steps"] if model_key == "flux_schnell" else 20)
    
    # CFG
    cfg_low, cfg_high = model["cfg_range"]
    if quality == "high":
        cfg = cfg_high
    elif quality == "low":
        cfg = cfg_low
    else:
        cfg = model["default_cfg"]
    
    # 分辨率
    if subject_type == "portrait":
        if "全身" in original_text or "full body" in original_text.lower():
            width, height = RESOLUTION_PRESETS["portrait"]
        else:
            width, height = RESOLUTION_PRESETS["portrait"]
    elif subject_type == "landscape":
        width, height = RESOLUTION_PRESETS["landscape"]
    else:
        width, height = RESOLUTION_PRESETS["square"]
    
    # 16:9 检测
    if any(kw in original_text.lower() for kw in ["16:9", "宽屏", "wide screen", "wide"]):
        width, height = RESOLUTION_PRESETS["wide"]
    
    # 负面提示词
    neg_parts = [NEGATIVE_COMMON]
    if subject_type == "portrait":
        neg_parts.append(NEGATIVE_PORTRAIT)
    negative = ", ".join(neg_parts)
    
    return {
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "negative": negative,
        "seed": int(time.time()) % 2**31,  # 随机种子
    }


def _enhance_prompt(original: str, styles: list, model_key: str, subject_type: str, quality: str = "medium") -> str:
    """增强提示词：添加风格词、质量词"""
    enhancements = []
    
    # 风格词
    for s in styles:
        if s in STYLE_KEYWORDS:
            # 选1-2个关键词加入
            enhancements.extend(STYLE_KEYWORDS[s][:2])
    
    # 质量词
    if model_key == "flux_dev":
        enhancements.extend(["masterpiece", "best quality", "highly detailed"])
    elif model_key == "sd3":
        enhancements.extend(["detailed", "high quality"])
    elif quality == "medium":
        enhancements.extend(["high quality"])
    elif quality == "low":
        pass  # 快速模式不加多余词
    
    # 构图提示
    if subject_type == "portrait":
        enhancements.append("sharp focus")
    
    if enhancements:
        return original + ", " + ", ".join(set(enhancements))
    return original


# ── ComfyUI 执行器 ────────────────────────────────────────────────────

def _check_comfyui() -> bool:
    """检查 ComfyUI 是否运行"""
    try:
        import requests
        r = requests.get("http://127.0.0.1:8188/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _generate_local(analysis: dict, output_path: str, overrides: dict = None) -> dict:
    """在本地 ComfyUI 上执行生成"""
    model_key = overrides.get("model", analysis["model_key"]) if overrides else analysis["model_key"]
    params = analysis["params"]
    prompt = analysis["enhanced_prompt"]
    
    # 合并 override 参数
    if overrides:
        for k in ["steps", "cfg", "width", "height", "seed", "negative"]:
            if k in overrides:
                params[k] = overrides[k]
        if "prompt" in overrides:
            prompt = overrides["prompt"]
    
    from vram_manager import VMgr
    
    vm = VMgr()
    print(f"[ImageGen v2.0.0] 选择模型: {model_key} — {analysis['reason']}")
    print(f"  提示词: {prompt[:80]}...")
    print(f"  参数: steps={params['steps']}, cfg={params['cfg']}, "
          f"{params['width']}×{params['height']}, seed={params['seed']}")
    
    print(f"[ImageGen] 获取 VRAM...")
    ok = vm.acquire_for_comfy(reason=f'imggen_{model_key}')
    if not ok:
        return {"success": False, "error": "无法获取 VRAM", "output_file": ""}
    
    try:
        from comfy_client import ComfyUIClient
        client = ComfyUIClient()
        
        # 构建工作流
        if model_key == "sd3":
            workflow = _build_sd3_workflow(prompt, params)
        elif model_key == "flux_schnell":
            workflow = _build_flux_workflow(prompt, params, use_fp8=True)
        elif model_key == "flux_dev":
            workflow = _build_flux_workflow(prompt, params, use_fp8=True, dev_mode=True)
        else:
            # 默认 fallback
            workflow = _build_flux_workflow(prompt, params, use_fp8=True)
        
        print(f"[ImageGen] 提交任务...")
        pid = client.post_prompt(workflow)
        print(f"[ImageGen] prompt_id: {pid}")
        
        # 等待并收集输出
        result = client.wait_for_prompt(pid, timeout_sec=1200)
        
        # 解析输出文件名
        outputs = result.get("outputs", {})
        saved_file = None
        for node_id, node_out in outputs.items():
            for img in node_out.get("images", []):
                saved_file = img.get("filename", "")
                break
            if saved_file:
                break
        
        if not saved_file:
            return {"success": False, "error": "未找到输出文件", "output_file": ""}
        
        # 下载到目标目录
        local_path = client.download_output(node_id, saved_file, str(OUTPUT_DIR))
        # 重命名为指定输出名
        desired_name = Path(output_path).name
        if local_path != output_path:
            shutil.copy2(local_path, output_path)
        
        return {"success": True, "output_file": str(output_path), "prompt_id": pid}
    
    except Exception as e:
        return {"success": False, "error": str(e), "output_file": ""}
    finally:
        print(f"[ImageGen] 释放 VRAM...")
        vm.release_and_restore()


def _build_sd3_workflow(prompt: str, params: dict) -> dict:
    """构建 SD3 Medium 工作流"""
    return {
        "3": {"inputs": {"ckpt_name": "sd3-medium-incl_clips.safetensors"}, "class_type": "CheckpointLoaderSimple"},
        "4": {"inputs": {"text": prompt, "clip": ["3", 1]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"text": params["negative"], "clip": ["3", 1]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"width": params["width"], "height": params["height"], "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "7": {
            "inputs": {
                "seed": params["seed"], "steps": params["steps"], "cfg": params["cfg"],
                "sampler_name": "euler", "scheduler": "normal",
                "positive": ["4", 0], "negative": ["5", 0],
                "latent_image": ["6", 0], "model": ["3", 0], "denoise": 1.0,
            },
            "class_type": "KSampler",
        },
        "8": {"inputs": {"samples": ["7", 0], "vae": ["3", 2]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"images": ["8", 0], "filename_prefix": "SD3-Medium"}, "class_type": "SaveImage"},
    }


def _build_flux_workflow(prompt: str, params: dict, use_fp8: bool = True,
                          dev_mode: bool = False) -> dict:
    """构建 Flux 工作流"""
    if dev_mode:
        model_file = "flux1-dev-fp8/flux1-dev-fp8-e4m3fn.safetensors"
    else:
        model_file = "flux1-schnell/flux1-schnell-fp8.safetensors"
    
    return {
        "3": {"inputs": {"ckpt_name": model_file}, "class_type": "CheckpointLoaderSimple"},
        "4": {
            "inputs": {"clip": ["3", 1], "clip_l": prompt, "t5xxl": prompt, "guidance": 1.0},
            "class_type": "CLIPTextEncodeFlux",
        },
        "5": {
            "inputs": {"clip": ["3", 1], "clip_l": params["negative"], "t5xxl": params["negative"], "guidance": 1.0},
            "class_type": "CLIPTextEncodeFlux",
        },
        "6": {"inputs": {"width": params["width"], "height": params["height"], "batch_size": 1}, "class_type": "EmptyFlux2LatentImage"},
        "7": {"inputs": {"steps": params["steps"], "width": params["width"], "height": params["height"]}, "class_type": "Flux2Scheduler"},
        "8": {"inputs": {"sampler_name": "euler"}, "class_type": "KSamplerSelect"},
        "9": {
            "inputs": {
                "model": ["3", 0], "add_noise": "enable", "noise_seed": params["seed"],
                "cfg": params["cfg"], "positive": ["4", 0], "negative": ["5", 0],
                "sampler": ["8", 0], "sigmas": ["7", 0], "latent_image": ["6", 0],
            },
            "class_type": "SamplerCustom",
        },
        "10": {"inputs": {"samples": ["9", 0], "vae": ["3", 2]}, "class_type": "VAEDecode"},
        "11": {"inputs": {"images": ["10", 0], "filename_prefix": "FLUX"}, "class_type": "SaveImage"},
    }


# ── 云模型降级 ───────────────────────────────────────────────────────

def _generate_cloud(prompt: str, output_path: str, size: str = "1k",
                     aspect_ratio: str = "1:1") -> dict:
    """通过 image_generate 工具生成（降级方案）"""
    try:
        from openclaw_workspacecortana_shadow_skills_image_generation import image_generate as tool_image_generate
    except ImportError:
        return {"success": False, "error": "image_generate 工具不可用（需在 OpenClaw 上下文）", "output_file": ""}
    
    try:
        # 智能选择 aspect_ratio
        result = tool_image_generate(
            prompt=prompt,
            size=size,
            aspect_ratio=aspect_ratio,
            quality="high",
            output_format="png",
        )
        return {"success": True, "output_file": str(result) if result else output_path}
    except Exception as e:
        return {"success": False, "error": str(e), "output_file": ""}


# ── 反馈循环 ──────────────────────────────────────────────────────────

def retry_with_feedback(prev_result: dict, feedback: str) -> dict:
    """
    根据反馈调整参数重新生成。
    
    反馈关键词检测:
    - "模糊"/"blurry" → 增加步数
    - "手"/"hand"/"手指" → 增加手部负面词, 换 SD3
    - "人脸"/"face" → 换 SD3 Medium
    - "构图"/"composition" → 换 Flux, 改 prompt
    """
    if not prev_result.get("analysis"):
        return {"success": False, "error": "需要之前的 analysis 数据", "output_file": ""}
    
    analysis = prev_result["analysis"]
    feedback_lower = feedback.lower()
    overrides = {}
    new_prompt = None
    
    # 模糊 → 增加步数
    if any(kw in feedback_lower for kw in ["模糊", "blurry", "不清晰", "blur"]):
        overrides["steps"] = min(analysis["params"]["steps"] + 10, 50)
        analysis["reason"] += f" | 反馈调参：步数增至 {overrides['steps']}"
    
    # 手部问题 → 换 SD3 + 手部负面词
    if any(kw in feedback_lower for kw in ["手", "hand", "手指", "finger"]):
        overrides["model"] = "sd3"
        hand_neg = "bad hands, malformed hands, extra fingers, missing fingers, deformed hands"
        current_neg = analysis["params"]["negative"]
        if hand_neg not in current_neg:
            overrides["negative"] = current_neg + ", " + hand_neg
        analysis["reason"] += " | 反馈：换 SD3 + 手部修复"
    
    # 人脸问题 → 换 SD3（人脸好）
    if any(kw in feedback_lower for kw in ["脸", "face", "表情", "facial", "丑"]):
        overrides["model"] = "sd3"
        analysis["reason"] += " | 反馈：换 SD3（人脸优势）"
    
    # 构图问题 → 保持 Flux 但改提示词
    if any(kw in feedback_lower for kw in ["构图", "composition", "布局", "layout"]):
        # 保持 flux_schnell，只增加步数
        overrides["model"] = "flux_schnell"
        overrides["steps"] = 8
        analysis["reason"] += " | 反馈：Flux Schnell 8步改善构图"
    
    # 颜色/风格 → 调整 CFG
    if any(kw in feedback_lower for kw in ["颜色", "color", "色彩", "太淡", "太艳"]):
        if "太淡" in feedback_lower or "pale" in feedback_lower or "saturated" not in feedback_lower:
            overrides["cfg"] = min(analysis["params"]["cfg"] + 1.0, 7.0)
        else:
            overrides["cfg"] = max(analysis["params"]["cfg"] - 1.0, 1.0)
        analysis["reason"] += f" | 反馈：调 CFG 至 {overrides['cfg']}"
    
    # 重新生成
    current_output = prev_result.get("output_file", "")
    if current_output:
        p = Path(current_output)
        retry_name = p.stem + "_retry" + p.suffix
        retry_path = str(p.parent / retry_name)
    else:
        retry_name = f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        retry_path = str(OUTPUT_DIR / retry_name)
    
    # 更新 prompt
    if new_prompt:
        analysis["enhanced_prompt"] = new_prompt
    
    result = generate(
        description=analysis.get("original_description", feedback),
        override_analysis=analysis,
        overrides=overrides,
        output_path=retry_path,
        prefer_local=True,
    )
    result["retry_feedback"] = feedback
    return result


# ── 主入口 ────────────────────────────────────────────────────────────

def generate(
    description: str = "",
    subject: str = "",
    style: str = "",
    scene: str = "",
    extra: str = "",
    output_name: str = "",
    output_dir: str | Path | None = None,
    override_analysis: dict = None,
    overrides: dict = None,
    prefer_local: bool = True,
    fallback_to_cloud: bool = True,
) -> dict:
    """
    智能图片生成。

    参数:
        description: 自然语言描述（最推荐的方式）
        subject/style/scene/extra: 拆分描述（兼容旧版）
        output_name: 输出文件名（不含扩展名）
        output_dir: 输出目录
        override_analysis: 预分析结果（反馈循环用）
        overrides: 参数覆盖
        prefer_local: 优先尝试本地 ComfyUI
        fallback_to_cloud: 本地不可用时降级到云工具

    返回:
        {"success": bool, "output_file": str, "error": str,
         "analysis": dict, "model_used": str}
    """
    # Step 1: 合并描述
    if not description and subject:
        parts = [subject]
        for label_val in [style, scene, extra]:
            if label_val:
                parts.append(label_val)
        description = ", ".join(parts)
    
    if not description:
        return {"success": False, "error": "请提供图片描述", "output_file": ""}
    
    # Step 2: 分析请求
    analysis = override_analysis or analyze_request(description)
    analysis["original_description"] = description
    
    # Step 3: 生成输出路径
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if not output_name:
        model_short = analysis["model_key"]
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"{model_short}_{ts}"
    output_path = str(out_dir / f"{output_name}.png")
    
    # Step 4: 执行生成
    comfy_available = _check_comfyui() if prefer_local else False
    model_used = analysis["model_key"]
    
    if comfy_available:
        print(f"[ImageGen v2.0.0] 🔧 本地 ComfyUI 模式")
        result = _generate_local(analysis, output_path, overrides)
        result["analysis"] = analysis
        result["model_used"] = model_used
        if result["success"]:
            return result
    
    if fallback_to_cloud:
        print(f"[ImageGen v2.0.0] ☁️ 云模型模式")
        # 智能选择比例
        if analysis["subject_type"] == "portrait":
            ar = "2:3"
        elif analysis["subject_type"] == "landscape":
            ar = "3:2"
        else:
            ar = "1:1"
        
        # 构建完整提示词
        prompt = analysis["enhanced_prompt"]
        result = _generate_cloud(prompt, output_path, aspect_ratio=ar)
        result["analysis"] = analysis
        result["model_used"] = "cloud"
        return result
    
    if not comfy_available:
        return {
            "success": False,
            "error": "ComfyUI 未运行，且 fallback_to_cloud=False",
            "output_file": "",
            "model_used": "none",
        }
    
    return {
        "success": False,
        "error": "未知错误",
        "output_file": "",
        "model_used": "none",
    }


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ImageGen v2.0.0 — 智能图片生成")
    p.add_argument("--description", "-d", default="", help="自然语言描述（推荐方式）")
    p.add_argument("--subject", default="", help="主体描述（旧版方式）")
    p.add_argument("--style", default="", help="风格描述")
    p.add_argument("--scene", default="", help="场景/背景")
    p.add_argument("--extra", default="", help="额外描述")
    p.add_argument("--output-name", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--model", default=None, choices=["sd3", "flux_schnell", "flux_dev"],
                    help="强制指定模型（覆盖自动选择）")
    p.add_argument("--steps", type=int, default=None, help="覆盖步数")
    p.add_argument("--seed", type=int, default=None, help="覆盖种子")
    p.add_argument("--cfg", type=float, default=None, help="覆盖 CFG")
    p.add_argument("--width", type=int, default=None, help="覆盖宽度")
    p.add_argument("--height", type=int, default=None, help="覆盖高度")
    p.add_argument("--no-cloud", action="store_true", help="不使用云降级")
    p.add_argument("--dry-run", action="store_true", help="只分析不生成")
    args = p.parse_args()
    
    # 构建 overrides
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.steps:
        overrides["steps"] = args.steps
    if args.seed:
        overrides["seed"] = args.seed
    if args.cfg:
        overrides["cfg"] = args.cfg
    if args.width:
        overrides["width"] = args.width
    if args.height:
        overrides["height"] = args.height
    
    # 分析
    desc = args.description or args.subject
    if not desc:
        print("请提供 --description 或 --subject")
        sys.exit(1)
    
    analysis = analyze_request(desc)
    print(f"\n📋 分析结果:")
    print(f"  模型: {analysis['model_key']}")
    print(f"  理由: {analysis['reason']}")
    print(f"  主体: {analysis['subject_type']}")
    print(f"  风格: {analysis['style_hints'] or '无'}")
    print(f"  质量: {analysis['quality_tier']}")
    print(f"  增强提示词: {analysis['enhanced_prompt'][:100]}...")
    print(f"  参数: steps={analysis['params']['steps']}, cfg={analysis['params']['cfg']}, "
          f"{analysis['params']['width']}×{analysis['params']['height']}")
    
    if args.dry_run:
        print("\n--- 仅分析，未生成 ---")
        return
    
    print()
    r = generate(
        description=desc,
        overrides=overrides if overrides else None,
        output_name=args.output_name,
        output_dir=args.output_dir or None,
        prefer_local=True,
        fallback_to_cloud=not args.no_cloud,
    )
    
    if r["success"]:
        print(f"\n✅ {r['output_file']}")
    else:
        print(f"\n❌ {r.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
