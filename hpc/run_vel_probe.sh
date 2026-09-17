#!/bin/bash
# 2026-09-17 (owner: "1和2一起做"): after the capacity probe (runs/R_reg_psc14_b48) finishes, the velocity-input pair at
# 32->64 on the production split: V base (regression) -> V residual flow -> density vs native 128.
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_vel_probe.sh > runs/vel_probe_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"; LOG=runs/vel_probe.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=03:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1"
until [ -f runs/R_reg_psc14_b48/results.json ] || ! squeue --steps -j $JOBID -h | grep -q python; do sleep 30; done
echo "=== $(date -Is) capacity probe (base 48):" | tee -a $LOG; grep -E "1 Euler|model params" runs/R_reg_psc14_b48/train.log | cut -c1-200 | tee -a $LOG
out=runs/V_reg_psc14; mkdir -p $out
echo "=== $(date -Is) $out (velocity inputs, regression base)" | tee -a $LOG
$SRUN --output=$out/train.log --error=$out/train.err $PY -u octave_flow_toy.py --nc 32 --nf 64 --box 100000 --offset 0.5 --growth 76.7439 \
  --dis "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy" --ic "data/psc/dmo-{N}/set{seed}/IC/disp.npy" --vel "data/psc/dmo-{N}/set{seed}/PART_009/vel.npy" \
  --velocity-inputs --train-seeds $TRAIN --test-seeds 14 --device cuda --octave-sampler full --steps 3000 --base 24 --batch 2 --regression --seed 0 --out $out
grep -E "1 Euler|model params" $out/train.log | cut -c1-200 | tee -a $LOG
out2=runs/VF_resflow_psc14; mkdir -p $out2
echo "=== $(date -Is) $out2 (velocity inputs, residual flow)" | tee -a $LOG
$SRUN --output=$out2/train.log --error=$out2/train.err $PY -u runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN \
  --test-seeds 14 --steps 3000 --device cuda --velocity-inputs --base-ckpt $out/model_ema.pt --out $out2
grep -E "^base|^emulator|^generative" $out2/train.log | tee -a $LOG
$PY runs/eval_vs_ref.py $out2 --nc 64 --set 14 2>&1 | grep "vs native" | tee -a $LOG
echo "=== $(date -Is) velocity probe done" | tee -a $LOG
