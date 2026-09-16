#!/bin/bash
# Phase B on the PSC series (HANDOFF §2 B), strictly one training at a time on the held A100:
#   B0  R_reg_psc      regression base, NATIVE 64 label            (control, same data as B1)
#   B1  R_reg_psc_y64  regression base, label R_{128->64} Psi_128  (the experiment)
#   B2  RF_resflow_psc      residual flow on B0 (target = native 64)  (control)
#   B3  RF_resflow_psc_y64  residual flow on B1 (target = native 64)  (the experiment)
# then runs/eval_vs_ref.py on B2/B3 against the native 128 of the test set.
# Usage (from the repo root, inside tmux):  JOBID=<hold job> bash hpc/run_phaseB.sh [B0 B1 B2 B3]
set -euo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?set JOBID to the GPU hold job (hpc/slurm_hold_gpu.sh)}
PY=${PY:-/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python}
STEPS=${STEPS:-3000}
TRAIN=${TRAIN:-"0 1 2 3 4 5 6 7"}
TEST=${TEST:-8}
DIS="data/psc/dmo-{N}/set{seed}/PART_009/disp.npy"
IC="data/psc/dmo-{N}/set{seed}/IC/disp.npy"
COMMON="--nc 32 --nf 64 --box 100000 --offset 0.5 --growth 76.7439 --dis $DIS --ic $IC --train-seeds $TRAIN --test-seeds $TEST --device cuda"
run() {  # run <out> <python args...>
  local out=$1; shift
  mkdir -p "$out"
  echo "=== $(date -Is) $out: $*" | tee -a runs/phaseB_launcher.log
  srun --jobid="$JOBID" --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=06:00:00 \
    --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1 \
    --output="$out/train.log" --error="$out/train.err" \
    "$PY" -u "$@" --out "$out"
  echo "=== $(date -Is) $out finished (exit $?)" | tee -a runs/phaseB_launcher.log
}
for stage in "${@:-B0 B1 B2 B3}"; do
  case $stage in
    B0) run runs/R_reg_psc     octave_flow_toy.py $COMMON --steps $STEPS --base 24 --regression --seed 0 ;;
    B1) run runs/R_reg_psc_y64 octave_flow_toy.py $COMMON --steps $STEPS --base 24 --regression --seed 0 --label-from 128 ;;
    B2) run runs/RF_resflow_psc     runs/resflow_train.py --data psc --train-seeds $TRAIN --test-seeds $TEST --steps $STEPS --device cuda --base-ckpt runs/R_reg_psc/model_ema.pt ;;
    B3) run runs/RF_resflow_psc_y64 runs/resflow_train.py --data psc --train-seeds $TRAIN --test-seeds $TEST --steps $STEPS --device cuda --base-ckpt runs/R_reg_psc_y64/model_ema.pt ;;
    EVAL) $PY runs/eval_vs_ref.py runs/RF_resflow_psc runs/RF_resflow_psc_y64 --set $TEST | tee -a runs/phaseB_launcher.log ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
