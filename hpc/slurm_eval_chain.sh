#!/bin/bash
#SBATCH -J evalchain
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=110G
#SBATCH -t 04:00:00
#SBATCH -o runs/slurm_evalchain_%j.log
# Density of every chain_shared.py output against the converged native 2c reference (runs/eval_vs_ref.py):
# 64-level fields vs native 128 and 128-level vs native 256 on the 256^3 mesh, 256-level vs native 512 on a 512^3 mesh.
#   sbatch --export=ALL,RUNS="runs/chain_M48_set14:14 runs/chain_M48J_set14:14" hpc/slurm_eval_chain.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE PYTHONUNBUFFERED=1
for spec in $RUNS; do
  d=${spec%%:*}; s=${spec##*:}
  echo "===== $d (set $s)"
  $PY runs/eval_vs_ref.py $d --set $s --nc 64 --mesh 256 --tag 32to64 \
      --fields truth_32to64 base_from32_32to64 emulator_from32_32to64 generative_from32_32to64
  $PY runs/eval_vs_ref.py $d --set $s --nc 128 --mesh 256 --tag 64to128 \
      --fields truth_64to128 base_from64_64to128 emulator_from64_64to128 emulator_from32_64to128 generative_from64_64to128 generative_from32_64to128
  $PY runs/eval_vs_ref.py $d --set $s --nc 256 --mesh 512 --tag 128to256 \
      --fields truth_128to256 base_from128_128to256 emulator_from128_128to256 emulator_from64_128to256 emulator_from32_128to256 \
               generative_from128_128to256 generative_from64_128to256 generative_from32_128to256
done
echo "===== eval chain done"
