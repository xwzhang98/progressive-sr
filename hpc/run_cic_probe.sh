#!/bin/bash
# 2026-09-17: (1) correction-stage headroom test (toy --loss-band low), (2) --cic-weight dose-response on the residual
# flow at 32->64 (split 0-13/14, frozen base runs/R_reg_psc14), one training at a time on the held A100.
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_cic_probe.sh > runs/cic_probe_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"; LOG=runs/cic_probe.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=03:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1"
out=runs/G_reg_psc14_lowband; mkdir -p $out
echo "=== $(date -Is) $out (loss on the coarse band only)" | tee -a $LOG
$SRUN --output=$out/train.log --error=$out/train.err $PY -u octave_flow_toy.py --nc 32 --nf 64 --box 100000 --offset 0.5 --growth 76.7439 \
  --dis "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy" --ic "data/psc/dmo-{N}/set{seed}/IC/disp.npy" --train-seeds $TRAIN --test-seeds 14 \
  --device cuda --octave-sampler full --steps 3000 --base 24 --batch 2 --regression --seed 0 --loss-band low --out $out
grep -E "1 Euler" $out/train.log | cut -c1-200 | tee -a $LOG
for spec in log1p:1:30 sqrt:1:30 lin:1:0.3 log1p:2:10 log1p:1:3 sqrt:1:3 lin:1:0.03 log1p:2:1; do
  c=${spec%%:*}; r=${spec#*:}; f=${r%%:*}; w=${r##*:}
  out=runs/RFC_resflow_psc14_${c}_f${f}_w$(echo $w | tr -d .); mkdir -p $out
  echo "=== $(date -Is) $out (cic $c factor $f weight $w)" | tee -a $LOG
  $SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN \
    --test-seeds 14 --steps 3000 --device cuda --base-ckpt runs/R_reg_psc14/model_ema.pt --cic-weight $w --cic-compress $c --cic-factor $f --out $out
  grep -E "\[cic\]" $out/train.log | tail -1 | tee -a $LOG
  grep -E "^emulator|^generative" $out/train.log | tee -a $LOG
  $PY runs/eval_vs_ref.py $out --nc 64 --set 14 --fields emulator generative 2>&1 | grep "vs native" | tee -a $LOG
done
echo "=== $(date -Is) cic probe done" | tee -a $LOG
