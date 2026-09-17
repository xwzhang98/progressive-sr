#!/bin/bash
# Phase C1 on the PSC series (HANDOFF §2 C1; owner 2026-09-16 "三个都做"): production split
# train sets 0-13, test set 14 (set 15 = second box, evaluated afterwards), one training at a time
# on the held A100.
#   C1a  F_reg_128_psc      64->128 regression base (batch 1, base 24, 3000 steps)   laptop: F_reg_128_3k
#   C1b  RF_resflow_128_psc residual flow on C1a (target = native 128)               laptop: RF_resflow_128
#   S32a R_reg_psc14        32->64 regression base on the SAME split (control for C2)
#   S32b RF_resflow_psc14   residual flow on S32a
#   EVAL eval_vs_ref.py: C1b vs native 256 (band k < k_Ny,128), S32b vs native 128 (k < k_Ny,64)
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_phaseC.sh [C1a C1b S32a S32b EVAL] > runs/phaseC_launcher.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?set JOBID to the GPU hold job (hpc/slurm_hold_gpu.sh)}
PY=${PY:-/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python}
STEPS=${STEPS:-3000}
TRAIN=${TRAIN:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13"}
TEST=${TEST:-14}
DIS="data/psc/dmo-{N}/set{seed}/PART_009/disp.npy"
IC="data/psc/dmo-{N}/set{seed}/IC/disp.npy"
COMMON="--box 100000 --offset 0.5 --growth 76.7439 --dis $DIS --ic $IC --train-seeds $TRAIN --test-seeds $TEST --device cuda --octave-sampler full"
LOG=runs/phaseC_launcher.log
run() {  # run <out> <python args...>
  local out=$1; shift
  mkdir -p "$out"
  echo "=== $(date -Is) $out: $*" | tee -a $LOG
  srun --jobid="$JOBID" --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=12:00:00 \
    --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1 \
    --output="$out/train.log" --error="$out/train.err" \
    "$PY" -u "$@" --out "$out"
  echo "=== $(date -Is) $out finished (exit $?)" | tee -a $LOG
}
for stage in "${@:-C1a C1b S32a S32b EVAL}"; do
  case $stage in
    C1a)  run runs/F_reg_128_psc      octave_flow_toy.py --nc 64 --nf 128 $COMMON --steps $STEPS --base 24 --batch 1 --regression --seed 0 ;;
    C1b)  run runs/RF_resflow_128_psc runs/resflow_train.py --data psc --nc 64 --nf 128 --batch 1 --train-seeds $TRAIN --test-seeds $TEST --steps $STEPS --device cuda --base-ckpt runs/F_reg_128_psc/model_ema.pt ;;
    S32a) run runs/R_reg_psc14        octave_flow_toy.py --nc 32 --nf 64 $COMMON --steps $STEPS --base 24 --batch 2 --regression --seed 0 ;;
    S32b) run runs/RF_resflow_psc14   runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN --test-seeds $TEST --steps $STEPS --device cuda --base-ckpt runs/R_reg_psc14/model_ema.pt ;;
    EVAL) $PY runs/eval_vs_ref.py runs/RF_resflow_128_psc --nc 128 --set $TEST | tee -a $LOG
          $PY runs/eval_vs_ref.py runs/RF_resflow_psc14 --nc 64 --set $TEST | tee -a $LOG ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
