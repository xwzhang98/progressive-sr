#!/bin/bash
# 2026-09-19 (owner: "先1后2"): rollout-aware fine-tuning of the shared three-level flow (RFT2 protocol), one arm per GPU:
#   train runs/RS_<tag> (mix MIX) -> chain_shared on sets 14, 15 -> density evaluation on the allocation's CPUs.
# Usage: JOBID=<hold job> MIX=0.5 TAG=mix50 [WAIT_CACHE=1] setsid nohup bash hpc/run_rollout_shared.sh > runs/rs_<tag>_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; MIX=${MIX:?}; TAG=${TAG:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
LOG=runs/rs_$TAG.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=08:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
say() { echo "=== $(date -Is) $*" | tee -a $LOG; }
if [ "${WAIT_CACHE:-0}" = 1 ]; then
  until [ "$(ls runs/rollout_cache_M48c/*.npy 2>/dev/null | wc -l)" -ge 42 ] && ! pgrep -f "rollout_shared.py --mix 0.5" >/dev/null || [ -f runs/RS_mix50/model_ema.pt ]; do
    [ "$(ls runs/rollout_cache_M48c/*.npy 2>/dev/null | wc -l)" -ge 42 ] && grep -q "gstep" runs/RS_mix50/train.log 2>/dev/null && break
    sleep 30; done
fi
out=runs/RS_$TAG; mkdir -p $out; say "$out (mix $MIX)"
$SRUN --output=$out/train.log --error=$out/train.err $PY -u runs/rollout_shared.py --mix $MIX --out $out
grep -E "warm start|done:" $out/train.log | cut -c1-200 | tee -a $LOG; grep -iE "Traceback|out of memory" $out/train.err | head -2 | tee -a $LOG
[ -f $out/model_ema.pt ] || { say "training failed"; exit 1; }
RUNS=""
for s in 14 15; do
  c=runs/chain_RS_${TAG}_set$s; mkdir -p $c; say "$c"
  $SRUN --output=$c/chain.log --error=$c/chain.err $PY -u runs/chain_shared.py --base-ckpt runs/M48c_reg_psc/model_ema.pt \
     --flow-ckpt $out/model_ema.pt --set $s --levels 32 64 128 --out $c
  grep -E "emulator.*octave:" $c/chain.log | sed -E 's/ \(multi.*//' | tee -a $LOG
  RUNS="$RUNS $c:$s"
done
say "chains done; density evaluation on the allocation CPUs"
srun --jobid=$JOBID --overlap --ntasks=1 --cpus-per-task=8 --time=03:00:00 --export=ALL,RUNS="$RUNS",SLURM_CPUS_ON_NODE=8 \
     --output=runs/evalchain_rs_$TAG.log --error=runs/evalchain_rs_$TAG.err bash hpc/slurm_eval_chain.sh
say "density evaluation done (runs/evalchain_rs_$TAG.log)"
