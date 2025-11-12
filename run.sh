#!/usr/bin/env bash
set -euo pipefail

# ---- experiment config (edit these) ----
EXPNR=000
EXPNAME="nvsa_local"
SAVEPATH="$HOME/nvsa_logs"               
DATAPATH="$HOME/Thesis/Datasets/RAVEN-10000"
DATAPATHIRAVEN=""       
SEED=1234
RUN=0 
EXPPATH="${SAVEPATH}/${EXPNAME}/"
CHKPOINTPATH="$HOME/Thesis/neuro-vector-symbolic-architectures-raven/Checkpoint_saved/ckpt"

# ---- conda activate in non-interactive shell ----
# if command -v conda >/dev/null 2>&1; then
#   eval "$(conda shell.bash hook)"
#   conda activate myNVSAenv
# else
#   echo "conda not found on PATH"; exit 1
# fi

mkdir -p "$EXPPATH"

# run_job () {
#   local NAME="$1"; shift
#   echo "=== Running: $NAME ==="
#   python -m cProfile -s cumtime raven/main_nvsa_marg.py "$@" \
#     --exp_dir "$EXPPATH" \
#     --dataset "$DATAPATH" \
#     --dataset-i-raven "$DATAPATHIRAVEN" \
#     --seed "$SEED" \
#     --run "$RUN" \
#     --resume "$CHKPOINTPATH" >> nvsa_profile_gpu.txt
# }

run_job () {
  local NAME="$1"; shift
  echo "=== Running: $NAME ==="
  python raven/main_nvsa_marg.py "$@" \
    --exp_dir "$EXPPATH" \
    --dataset "$DATAPATH" \
    --dataset-i-raven "$DATAPATHIRAVEN" \
    --seed "$SEED" \
    --run "$RUN" \
    --resume "$CHKPOINTPATH"
}

# run_job () {
#   local NAME="$1"; shift
#   echo "=== Running under Nsight Systems: $NAME ==="

#   # Create an output directory for Nsight traces if it doesn’t exist
#   mkdir -p "$HOME/Thesis/neuro-vector-symbolic-architectures-raven/nsys_traces"

#   # Choose output trace file name
#   local TRACE_OUT="$HOME/Thesis/neuro-vector-symbolic-architectures-raven/nsys_traces/${NAME}_trace"

#   # Run under Nsight Systems
#   nsys profile \
#     -w true \
#     --trace=cuda,nvtx,osrt,cudnn,cublas \
#     --sample=process-tree \
#     --cuda-memory-usage=true \
#     --force-overwrite=true \
#     --cudabacktrace=true --osrt-threshold=10000 -x true \
#     --output="$TRACE_OUT" \
#     python raven/main_nvsa_marg.py "$@" \
#       --exp_dir "$EXPPATH" \
#       --dataset "$DATAPATH" \
#       --dataset-i-raven "$DATAPATHIRAVEN" \
#       --seed "$SEED" \
#       --run "$RUN" \
#       --resume "$CHKPOINTPATH"
      
#   echo "Trace saved to ${TRACE_OUT}.qdrep"
# }

# Center
run_job center_single \
  --mode test --config center_single --epochs 50 --s 7 --trainable-s

# # 2x2
# run_job distribute_four \
#   --mode train --config distribute_four --epochs 150 --s 6 --trainable-s

# # 3x3 (uses smaller batch in the original script)
# run_job distribute_nine \
#   --mode train --config distribute_nine --epochs 150 --s 2 --trainable-s --batch-size 8

# # Left-right
# run_job lr \
#   --mode train --config left_center_single_right_center_single --epochs 100 --s 5 --trainable-s

# # Up-down
# run_job ud \
#   --mode train --config up_center_single_down_center_single --epochs 100 --s 5 --trainable-s

# # Out-in center
# run_job inout_center \
#   --mode train --config in_center_single_out_center_single --epochs 100 --s 5 --trainable-s

# # Out-in grid (same config as above per repo script)
# run_job inout_grid \
#   --mode train --config in_center_single_out_center_single --epochs 100 --s 5 --trainable-s