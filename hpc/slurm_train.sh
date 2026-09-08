#!/bin/bash
#SBATCH -J octave
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o runs/slurm_train_%j.log
# Resumable training under a wall-time limit. Submit once; it re-submits itself until
# train_state.pt reports the target step count (results.json exists).
#   sbatch hpc/slurm_train.sh runs/A_flow_phys --nc 64 --nf 128 --dis ... --ic ... --growth ...
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=$1; shift
export PYTHONUNBUFFERED=1
python octave_flow_toy.py "$@" --device cuda --out "$OUT" --resume \
  --max-seconds $(( 4*3600 - 600 ))          # 10 min under the 4 h limit
if [[ ! -f "$OUT/results.json" ]]; then
  sbatch --dependency=afterany:$SLURM_JOB_ID "$0" "$OUT" "$@"
fi
