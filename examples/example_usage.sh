#!/bin/bash
# 视频换脸工具 - 使用示例

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PROJECT_DIR/engine"
SKILL_DIR="$PROJECT_DIR/skill"

# 将 engine 和 skill 目录添加到 Python 路径
export PYTHONPATH="$ENGINE_DIR:$SKILL_DIR:$PYTHONPATH"

echo "🎬 视频换脸工具 - 使用示例"
echo "================================"
echo ""

# 示例 1: 智能模式（推荐）
echo "📌 示例 1: 智能参数自动选择"
echo "python3 \$SKILL_DIR/video_face_swap_skill.py \\"
echo "  --reference example_reference.jpg \\"
echo "  --video example_input.mp4 \\"
echo "  --output output_smart.mp4 \\"
echo "  --auto"
echo ""

# 示例 2: 目标锁定
echo "📌 示例 2: 目标锁定模式"
echo "python3 \$SKILL_DIR/video_face_swap_skill.py \\"
echo "  --reference example_reference.jpg \\"
echo "  --video example_input.mp4 \\"
echo "  --target example_target.jpg \\"
echo "  --output output_target.mp4 \\"
echo "  --quality hq"
echo ""

# 示例 3: 预处理参考图
echo "📌 示例 3: 预处理参考图（推荐用于重复使用）"
echo "python3 \$ENGINE_DIR/video_face_swap.py preprocess \\"
echo "  --source example_reference.jpg \\"
echo "  --output optimized_ref/"
echo ""
echo "# 然后使用预处理后的参考图"
echo "python3 \$SKILL_DIR/video_face_swap_skill.py \\"
echo "  --reference optimized_ref/final_reference.png \\"
echo "  --preprocess-dir optimized_ref/ \\"
echo "  --video example_input.mp4 \\"
echo "  --output output_preprocessed.mp4"
echo ""

echo "💡 提示：将示例文件放入 examples/ 目录后运行"
