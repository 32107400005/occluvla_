#!/bin/bash
# ============================================================
# OccluVLA: Push-Then-Grasp Data Generation (OccluBench)
#
# 用法:
#   bash run_generate.sh                 # Level 1, 100条 + 视频
#   bash run_generate.sh --debug         # Level 1, 5条 debug
#   bash run_generate.sh --level 2       # Level 2, 100条 + 视频
# ============================================================
set -e

LEVEL=1
MODE="full"

# 解析参数
for arg in "$@"; do
    case $arg in
        --debug)    MODE="debug" ;;
        --level)    shift; LEVEL=$1 ;;
        1|2)        LEVEL=$arg ;;
    esac
done

echo "============================================"
echo "  OccluVLA: Push-Then-Grasp Data Gen"
echo "  Env: OccluBench-Level${LEVEL}-v0"
echo "  Mode: $MODE"
echo "  Cameras: shoulder_camera + hand_camera"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate occluvla-sim

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCCLUVLA_DIR="$HOME/workspace/occluvla/project/occluvla"
DATA_GEN="$OCCLUVLA_DIR/data_gen"
DATA_DIR="$OCCLUVLA_DIR/data/push_grasp_level${LEVEL}"
VIS_DIR="$OCCLUVLA_DIR/data/visualizations"

mkdir -p "$DATA_GEN" "$DATA_DIR" "$VIS_DIR"

# 复制脚本 (含 occlubench_env.py)
# echo "[1/4] Copying scripts..."
# cp "$SCRIPT_DIR/occlubench_env.py"  "$DATA_GEN/"
# cp "$SCRIPT_DIR/push_grasp_env.py"  "$DATA_GEN/"
# cp "$SCRIPT_DIR/generate_demos.py"  "$DATA_GEN/"
# cp "$SCRIPT_DIR/inspect_data.py"    "$DATA_GEN/"
# echo "  Done"

# 依赖
echo "[2/4] Dependencies..."
pip install imageio imageio-ffmpeg Pillow -q 2>/dev/null || true

# 生成
echo "[3/4] Generating demos..."
cd "$DATA_GEN"

if [[ "$MODE" == "debug" ]]; then
    python generate_demos.py \
        --num 5 \
        --level $LEVEL \
        --save_video \
        --render_size 256 \
        --debug \
        --output_dir "$DATA_DIR"
else
    python generate_demos.py \
        --num 100 \
        --level $LEVEL \
        --save_video \
        --render_size 256 \
        --output_dir "$DATA_DIR"
fi

# 质量检查
echo "[4/4] Quality check..."
python inspect_data.py --data_dir "$DATA_DIR" --save_samples

echo ""
echo "============================================"
echo "  Done! Level $LEVEL"
echo "  Data:   $DATA_DIR"
echo "  Video:  $VIS_DIR"
echo "    ├── epXXXX_L${LEVEL}_shoulder_camera.mp4"
echo "    ├── epXXXX_L${LEVEL}_hand_camera.mp4"
echo "    ├── epXXXX_L${LEVEL}_render.mp4"
echo "    └── epXXXX_L${LEVEL}_sidebyside.mp4"
echo "============================================"
