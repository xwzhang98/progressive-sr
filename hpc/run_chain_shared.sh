#!/bin/bash
# 2026-09-18: autoregressive chain 32->64->128->256 with the shared width-48 operator (M48, M48J), sets 14 and 15,
# emulator + generative, 128->256 ZERO-SHOT; then the density evaluation as an RM job.
# Usage: JOBID=<hold job> setsid nohup bash hpc/run_chain_shared.sh > runs/chain_shared_launcher.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
JOBID=${JOBID:?}; PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python; LOG=runs/chain_shared.log
SRUN="srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=04:00:00 --export=ALL,PYTHONNOUSERSITE=1,OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
RUNS=""
for m in M48 M48J; do for s in 14 15; do
  out=runs/chain_${m}_set$s; mkdir -p $out
  echo "=== $(date -Is) $out" | tee -a $LOG
  $SRUN --output=$out/chain.log --error=$out/chain.err $PY -u runs/chain_shared.py --base-ckpt runs/M48_reg_psc/model_ema.pt \
     --flow-ckpt runs/${m}_resflow_psc/model_ema.pt --set $s --levels 32 64 128 --out $out
  grep -E "^level|octave:" $out/chain.log | tee -a $LOG
  grep -iE "out of memory|Traceback" $out/chain.err | head -2 | tee -a $LOG
  RUNS="$RUNS $out:$s"
done; done
J=$(sbatch --parsable --export=ALL,RUNS="$RUNS" hpc/slurm_eval_chain.sh)
echo "=== $(date -Is) GPU part done; density evaluation submitted as RM job $J" | tee -a $LOG
