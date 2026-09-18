#!/bin/bash
# 2026-09-17 evening: complete the capacity line at 32->64 (split 0-13/14):
#   RFJ48_resflow_psc14  residual flow (48) on the base-48 base + jac-weight 1.2 (no velocity)  = production candidate
#   R_reg_psc14_b96      regression base, width 96 (capacity dose-response 24 -> 48 -> 96)
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"; LOG=runs/cap2_probe.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=04:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1"
out=runs/RFJ48_resflow_psc14; mkdir -p $out
echo "=== $(date -Is) $out" | tee -a $LOG
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN --test-seeds 14 \
  --steps 3000 --device cuda --flow-base 48 --base-ckpt runs/R_reg_psc14_b48/model_ema.pt --jac-weight 1.2 --out $out
grep -E "^base|^emulator|^generative" $out/train.log | tee -a $LOG
$PY runs/eval_vs_ref.py $out --nc 64 --set 14 2>&1 | grep "vs native" | grep -v truth | tee -a $LOG
out=runs/R_reg_psc14_b96; mkdir -p $out
echo "=== $(date -Is) $out (base 96 regression)" | tee -a $LOG
$SRUN --output=$out/train.log --error=$out/train.err $PY -u octave_flow_toy.py --nc 32 --nf 64 --box 100000 --offset 0.5 --growth 76.7439 \
  --dis "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy" --ic "data/psc/dmo-{N}/set{seed}/IC/disp.npy" --train-seeds $TRAIN --test-seeds 14 \
  --device cuda --octave-sampler full --steps 3000 --base 96 --batch 2 --regression --seed 0 --out $out
grep -E "1 Euler|model params|s/step" $out/train.log | tail -3 | cut -c1-200 | tee -a $LOG
echo "=== $(date -Is) cap2 probe done" | tee -a $LOG
