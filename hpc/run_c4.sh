#!/bin/bash
# 2026-09-19 (owner: "先1后2，内存映射读取"): ONE shared width-48 two-stage operator for FOUR levels, 32->64 and 64->128 full-box,
# 128->256 on crops, 256->512 on STREAMED crops (tiling.CropSource: coarse in RAM, target and precomputed IC octave memory-mapped).
#   C4a runs/M48d_reg_psc -> C4b runs/M48d_resflow_psc -> chains 64->128->256->512 on sets 14, 15 -> density (hold allocation CPUs)
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_c4.sh > runs/c4_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python; LOG=runs/c4.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=16:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
TRAIN="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
GEOM="--base 48 --levels 32 64 128 256 --batches 2 1 1 1 --crops 0 0 128 128 --stream-levels 256 --halo 16 --tile-core 64 --data psc --train-seeds $TRAIN --test-seeds 14 --steps 3000 --device cuda --eval-skip-levels 128 256"
say() { echo "=== $(date -Is) $*" | tee -a $LOG; }
out=runs/M48d_reg_psc; mkdir -p $out; say "$out"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/multi_resflow_train.py --mode base $GEOM --out $out
grep -E "^level|^\[.*to.*\]" $out/train.log | tee -a $LOG; grep -iE "Traceback|out of memory" $out/train.err | head -2 | tee -a $LOG
[ -f $out/model_ema.pt ] || { say "base failed"; exit 1; }
out=runs/M48d_resflow_psc; mkdir -p $out; say "$out"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/multi_resflow_train.py --mode resflow $GEOM --base-ckpt runs/M48d_reg_psc/model_ema.pt --out $out
grep -E "^lambda|^\[.*to.*\]" $out/train.log | tee -a $LOG; grep -iE "Traceback|out of memory" $out/train.err | head -2 | tee -a $LOG
[ -f $out/model_ema.pt ] || { say "flow failed"; exit 1; }
RUNS=""
for s in 14 15; do
  c=runs/chain_M48d_set$s; mkdir -p $c; say "$c"
  $SRUN --output=$c/chain.log --error=$c/chain.err $PY -u runs/chain_shared.py --base-ckpt runs/M48d_reg_psc/model_ema.pt \
     --flow-ckpt runs/M48d_resflow_psc/model_ema.pt --set $s --levels 64 128 256 --starts 64 128 256 --gen-starts 64 256 --out $c
  grep -E "^level|octave:" $c/chain.log | sed -E 's/ \(multi.*//' | tee -a $LOG
  RUNS="$RUNS $c:$s"
done
say "chains done; density evaluation on the allocation CPUs"
srun --jobid=$JOBID --overlap --ntasks=1 --cpus-per-task=8 --time=06:00:00 --export=ALL,RUNS="$RUNS",SLURM_CPUS_ON_NODE=8 \
     --output=runs/evalchain_c4.log --error=runs/evalchain_c4.err bash hpc/slurm_eval_chain4.sh
say "density evaluation done (runs/evalchain_c4.log)"
