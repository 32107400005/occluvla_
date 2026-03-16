#!/bin/bash
# ============================================================
# OccluVLA Week 2 Task 2: Push-Then-Grasp Scene (修正版)
#
# 用法:
#   bash 5_create_push_grasp_scene.sh          # 完整运行 (100条)
#   bash 5_create_push_grasp_scene.sh --debug   # 调试模式 (5条)
# ============================================================
set -e

echo "============================================"
echo "  OccluVLA: Push-Then-Grasp Scene (修正版)"
echo "============================================"
echo ""

# ── 参数 ──────────────────────────────────────────────────────────
DEBUG_MODE=false
if [[ "$1" == "--debug" ]]; then
    DEBUG_MODE=true
    echo "  🔍 Debug mode: ON (只生成 5 条, 打印详细信息)"
    echo ""
fi

# ── 激活 conda 环境 ──────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate occluvla-sim

# ── 路径设置 ──────────────────────────────────────────────────────
OCCLUVLA_DIR="$HOME/workspace/occluvla/project/occluvla"
DATA_GEN_DIR="$OCCLUVLA_DIR/data_gen"
DATA_DIR="$OCCLUVLA_DIR/data/push_grasp_demos"
VIS_DIR="$OCCLUVLA_DIR/data/visualizations"

mkdir -p "$DATA_GEN_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$VIS_DIR"

# ── 复制 Python 文件 ─────────────────────────────────────────────
# 假设三个 .py 文件放在与此脚本同目录下
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# echo "[1/5] 复制脚本到 $DATA_GEN_DIR ..."

# cp "$SCRIPT_DIR/push_grasp_env.py"  "$DATA_GEN_DIR/push_grasp_env.py"
# cp "$SCRIPT_DIR/generate_demos.py"  "$DATA_GEN_DIR/generate_demos.py"
# cp "$SCRIPT_DIR/inspect_data.py"    "$DATA_GEN_DIR/inspect_data.py"

# echo "  ✓ push_grasp_env.py"
# echo "  ✓ generate_demos.py"
# echo "  ✓ inspect_data.py"
# echo ""

# ── 安装依赖 ─────────────────────────────────────────────────────
echo "[2/5] 检查依赖..."
pip install imageio[ffmpeg] Pillow -q 2>/dev/null || true
# 确保 imageio-ffmpeg 和 pyav 可用 (高质量 MP4 输出)
pip install imageio-ffmpeg pyav -q 2>/dev/null || true
echo "  ✓ 依赖就绪"
echo ""

# ── 先运行一次快速检查 ────────────────────────────────────────────
echo "[3/5] 快速环境检查 (1 条轨迹)..."
cd "$DATA_GEN_DIR"

python -c "
from push_grasp_env import PushGraspSceneManager
import numpy as np

print('  Creating environment...')
mgr = PushGraspSceneManager(num_envs=1, render_size=512)

print('  Reset...')
obs, info = mgr.reset(seed=0)

print('  Reading observations...')
ee = mgr.get_ee_pos(obs)
tgt = mgr.get_target_pos(obs)
imgs = mgr.get_images(obs)

print(f'  EE position:     {ee}')
print(f'  Target position: {tgt}')
for k, v in imgs.items():
    print(f'  Image \"{k}\": shape={v.shape}')

print('  ✓ Environment check passed')
mgr.close()
"

if [ $? -ne 0 ]; then
    echo ""
    echo "  ✗ 环境检查失败! 请检查:"
    echo "    1. ManiSkill3 是否正确安装"
    echo "    2. conda 环境是否正确: conda activate occluvla-sim"
    echo "    3. GPU 是否可用: python -c 'import torch; print(torch.cuda.is_available())'"
    echo ""
    echo "  常见修复:"
    echo "    pip install mani-skill"
    echo "    python -m mani_skill.utils.download_asset PickSingleYCB-v1"
    echo ""
    exit 1
fi
echo ""

# ── 生成轨迹 ─────────────────────────────────────────────────────
echo "[4/5] 生成 push-then-grasp 轨迹..."
echo ""

if [ "$DEBUG_MODE" = true ]; then
    python generate_demos.py \
        --num_episodes 5 \
        --save_video \
        --render_size 512 \
        --debug \
        --output_dir "$DATA_DIR"
else
    python generate_demos.py \
        --num_episodes 100 \
        --save_video \
        --render_size 512 \
        --output_dir "$DATA_DIR"
fi

echo ""

# ── 数据质量检查 ──────────────────────────────────────────────────
echo "[5/5] 数据质量检查..."
echo ""

python inspect_data.py \
    --data_dir "$DATA_DIR" \
    --save_samples

echo ""
echo "============================================"
echo "  ✅ Push-Grasp Scene & Data Generation 完成!"
echo ""
echo "  数据:     $DATA_DIR"
echo "  视频:     $VIS_DIR"
echo "  采样帧:   $DATA_DIR/sample_frames/"
echo "  元数据:   $DATA_DIR/metadata.json"
echo ""
echo "  下一步:"
echo "    1. 检查视频: ls $VIS_DIR/*.mp4"
echo "    2. 检查采样帧: ls $DATA_DIR/sample_frames/"
echo "    3. 如果行为正确, 运行: bash 6_test_visibility.sh"
echo ""
echo "  如果行为不正确, 常见问题:"
echo "    - 机器人不动 → 检查 control_mode 是否匹配"
echo "    - 推不到物体 → 需要自定义环境加入多个物体"
echo "    - 视频太小   → 确认 --render_size 512 生效"
echo "============================================"
