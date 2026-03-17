#!/bin/bash
# ============================================================
# OccluVLA: Push-Then-Grasp Data Generation
# 用法:
#   bash run_generate.sh          # 正式 (100条 + 视频)
#   bash run_generate.sh --debug  # 调试 (5条)
# ============================================================
set -e

echo "============================================"
echo "  OccluVLA: Push-Then-Grasp Data Gen"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate occluvla-sim

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCCLUVLA_DIR="$HOME/workspace/occluvla/project/occluvla"
DATA_GEN="$OCCLUVLA_DIR/data_gen"
DATA_DIR="$OCCLUVLA_DIR/data/push_grasp_demos"
VIS_DIR="$OCCLUVLA_DIR/data/visualizations"

mkdir -p "$DATA_GEN" "$DATA_DIR" "$VIS_DIR"

# 复制脚本
# echo "[1/4] Copying scripts..."
# cp "$SCRIPT_DIR/push_grasp_env.py"  "$DATA_GEN/"
# cp "$SCRIPT_DIR/generate_demos.py"  "$DATA_GEN/"
# cp "$SCRIPT_DIR/inspect_data.py"    "$DATA_GEN/"
# echo "  Done"

# 依赖
echo "[2/4] Dependencies..."
pip install imageio Pillow -q 2>/dev/null || true

# 生成
echo "[3/4] Generating demos..."
cd "$DATA_GEN"

if [[ "$1" == "--debug" ]]; then
    python generate_demos.py --num 5 --save_video --render_size 512 --debug --output_dir "$DATA_DIR"
else
    python generate_demos.py --num 100 --save_video --render_size 512 --output_dir "$DATA_DIR"
fi

# 检查
echo "[4/4] Quality check..."
python inspect_data.py --data_dir "$DATA_DIR" --save_samples

echo ""
echo "============================================"
echo "  Done!"
echo "  Data:   $DATA_DIR"
echo "  Video:  $VIS_DIR"
echo "  Next:   check videos, then run visibility test"
echo "============================================"
