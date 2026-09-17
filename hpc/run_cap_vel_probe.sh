#!/bin/bash
# 2026-09-17 (owner: "可以继续"): complete the capacity / velocity table at 32->64 (split 0-13/14), one run at a time:
#   A  RF48_resflow_psc14        residual flow (width 48) on the base-48 base runs/R_reg_psc14_b48
#   B  V48_reg_psc14 -> VF48_resflow_psc14   base 48 + velocity inputs, then residual flow (48, velocity)
#   C  VFJ48_resflow_psc14       B's base + residual flow (48, velocity) + --jac-weight 1.2  (all positive levers stacked)
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_cap_vel_probe.sh > runs/cap_vel_probe_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"; LOG=runs/cap_vel_probe.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=04:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1"
RF="runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN --test-seeds 14 --steps 3000 --device cuda --flow-base 48"
flow() {  # flow <out> <extra args...>
  local out=$1; shift; mkdir -p $out
  echo "=== $(date -Is) $out: $*" | tee -a $LOG
  $SRUN --output=$out/train.log --error=$out/train.err $PY -u $RF "$@" --out $out
  grep -E "^base|^emulator|^generative" $out/train.log | tee -a $LOG
  $PY runs/eval_vs_ref.py $out --nc 64 --set 14 2>&1 | grep "vs native" | grep -v truth | tee -a $LOG
}
flow runs/RF48_resflow_psc14 --base-ckpt runs/R_reg_psc14_b48/model_ema.pt
out=runs/V48_reg_psc14; mkdir -p $out
echo "=== $(date -Is) $out (base 48 + velocity, regression)" | tee -a $LOG
$SRUN --output=$out/train.log --error=$out/train.err $PY -u octave_flow_toy.py --nc 32 --nf 64 --box 100000 --offset 0.5 --growth 76.7439 \
  --dis "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy" --ic "data/psc/dmo-{N}/set{seed}/IC/disp.npy" --vel "data/psc/dmo-{N}/set{seed}/PART_009/vel.npy" \
  --velocity-inputs --train-seeds $TRAIN --test-seeds 14 --device cuda --octave-sampler full --steps 3000 --base 48 --batch 2 --regression --seed 0 --out $out
grep -E "1 Euler" $out/train.log | cut -c1-200 | tee -a $LOG
flow runs/VF48_resflow_psc14 --velocity-inputs --base-ckpt runs/V48_reg_psc14/model_ema.pt
flow runs/VFJ48_resflow_psc14 --velocity-inputs --base-ckpt runs/V48_reg_psc14/model_ema.pt --jac-weight 1.2
echo "=== $(date -Is) cap/vel probe done" | tee -a $LOG
