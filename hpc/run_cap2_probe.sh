#!/bin/bash
# 2026-09-17 (owner: "可以，按照你的做"): capacity dose-response one step further + second box.
#   W96   runs/R_reg_psc14_b96 -> runs/RF96_resflow_psc14        32->64, width 96 both stages
#   S48   runs/M48_reg_psc -> runs/M48_resflow_psc               shared two-level operator at width 48 (+ EVAL2-style density)
#   SET15 RF48_resflow_psc14 and VFJ48_resflow_psc14 evaluated on set 15 (eval-only)
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_cap2_probe.sh > runs/cap2_probe_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"; LOG=runs/cap2_probe.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=08:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1"
COMMON="--nc 32 --nf 64 --box 100000 --offset 0.5 --growth 76.7439 --dis data/psc/dmo-{N}/set{seed}/PART_009/disp.npy --ic data/psc/dmo-{N}/set{seed}/IC/disp.npy --train-seeds $TRAIN --test-seeds 14 --device cuda --octave-sampler full --steps 3000 --batch 2 --regression --seed 0"
say() { echo "=== $(date -Is) $*" | tee -a $LOG; }
# --- W96
out=runs/R_reg_psc14_b96; mkdir -p $out; say "$out (width 96 base)"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u octave_flow_toy.py $COMMON --base 96 --out $out
grep -E "1 Euler|model params" $out/train.log | cut -c1-200 | tee -a $LOG; grep -E "Error|error" $out/train.err | grep -v Warning | head -2 | tee -a $LOG
out=runs/RF96_resflow_psc14; mkdir -p $out; say "$out (width 96 residual flow)"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN --test-seeds 14 --steps 3000 --device cuda --flow-base 96 --base-ckpt runs/R_reg_psc14_b96/model_ema.pt --out $out
grep -E "^base|^emulator|^generative" $out/train.log | tee -a $LOG; grep -E "Error|error" $out/train.err | grep -v Warning | head -2 | tee -a $LOG
$PY runs/eval_vs_ref.py $out --nc 64 --set 14 2>&1 | grep "vs native" | grep -v truth | tee -a $LOG
# --- S48 (shared operator, both levels, width 48)
out=runs/M48_reg_psc; mkdir -p $out; say "$out (shared base, width 48)"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/multi_resflow_train.py --mode base --base 48 --data psc --train-seeds $TRAIN --test-seeds 14 --steps 3000 --device cuda --out $out
grep -E "^\[.*to.*\]" $out/train.log | tee -a $LOG; grep -E "Error|error" $out/train.err | grep -v Warning | head -2 | tee -a $LOG
out=runs/M48_resflow_psc; mkdir -p $out; say "$out (shared residual flow, width 48)"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/multi_resflow_train.py --mode resflow --base 48 --data psc --train-seeds $TRAIN --test-seeds 14 --steps 3000 --device cuda --base-ckpt runs/M48_reg_psc/model_ema.pt --out $out
grep -E "^\[.*to.*\]" $out/train.log | tee -a $LOG; grep -E "Error|error" $out/train.err | grep -v Warning | head -2 | tee -a $LOG
for lv in 32to64:64 64to128:128; do tag=${lv%%:*}; nc=${lv##*:}; mkdir -p $out/$tag; for f in truth base emulator generative; do ln -sf ../field_${f}_${tag}.npy $out/$tag/field_${f}.npy; done
  $PY runs/eval_vs_ref.py $out/$tag --nc $nc --set 14 2>&1 | grep "vs native" | grep -v truth | tee -a $LOG; done
# --- SET15 for the width-48 pairs
for spec in RF48_resflow_psc14:R_reg_psc14_b48: VFJ48_resflow_psc14:V48_reg_psc14:--velocity-inputs; do
  m=${spec%%:*}; r=${spec#*:}; b=${r%%:*}; extra=${r##*:}
  out=runs/${m}_set15; mkdir -p $out; say "$out (eval-only on set 15)"
  $SRUN --output=$out/eval.log --error=$out/eval.err $PY -u runs/resflow_train.py --data psc --nc 32 --nf 64 --batch 2 --train-seeds $TRAIN --test-seeds 15 --steps 3000 --device cuda --flow-base 48 $extra --base-ckpt runs/$b/model_ema.pt --eval-only runs/$m/model_ema.pt --out $out
  grep -E "^base|^emulator|^generative" $out/eval.log | tee -a $LOG
  $PY runs/eval_vs_ref.py $out --nc 64 --set 15 2>&1 | grep "vs native" | tee -a $LOG
done
say "cap2 probe done"
