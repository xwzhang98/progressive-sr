#!/bin/bash
# 2026-09-18 (owner approved patch cropping): ONE shared width-48 two-stage operator for THREE levels,
# 32->64 and 64->128 on full boxes, 128->256 on 128^3 crops (halo 16), tiled inference (core 64) at that level.
#   C3a runs/M48c_reg_psc       shared regression base
#   C3b runs/M48c_resflow_psc   shared residual flow (Q_J; crop-tapered Q_J at the crop level)
#   C3c runs/chain_M48c_set{14,15}   32->64->128->256 chain, emulator + generative, then RM density job
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_c3.sh > runs/c3_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python; LOG=runs/c3.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=12:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
GEOM="--base 48 --levels 32 64 128 --batches 2 1 1 --crops 0 0 128 --halo 16 --tile-core 64 --data psc --train-seeds $TRAIN --test-seeds 14 --steps 3000 --device cuda"
say() { echo "=== $(date -Is) $*" | tee -a $LOG; }
out=runs/M48c_reg_psc; mkdir -p $out; say "$out"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/multi_resflow_train.py --mode base $GEOM --out $out
grep -E "^level|^\[.*to.*\]" $out/train.log | tee -a $LOG; grep -iE "Traceback|out of memory" $out/train.err | head -2 | tee -a $LOG
[ -f $out/model_ema.pt ] || { say "base failed, stopping"; exit 1; }
out=runs/M48c_resflow_psc; mkdir -p $out; say "$out"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/multi_resflow_train.py --mode resflow $GEOM --base-ckpt runs/M48c_reg_psc/model_ema.pt --out $out
grep -E "^lambda|^\[.*to.*\]" $out/train.log | tee -a $LOG; grep -iE "Traceback|out of memory" $out/train.err | head -2 | tee -a $LOG
[ -f $out/model_ema.pt ] || { say "flow failed, stopping"; exit 1; }
RUNS=""
for s in 14 15; do
  out=runs/chain_M48c_set$s; mkdir -p $out; say "$out"
  $SRUN --output=$out/chain.log --error=$out/chain.err $PY -u runs/chain_shared.py --base-ckpt runs/M48c_reg_psc/model_ema.pt \
     --flow-ckpt runs/M48c_resflow_psc/model_ema.pt --set $s --levels 32 64 128 --out $out
  grep -E "^level|octave:" $out/chain.log | tee -a $LOG
  RUNS="$RUNS $out:$s"
done
J=$(sbatch --parsable --export=ALL,RUNS="$RUNS" hpc/slurm_eval_chain.sh)
say "C3 GPU part done; density evaluation submitted as RM job $J"
