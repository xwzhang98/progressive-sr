#!/bin/bash
# Density of chain_shared.py outputs of the FOUR-level operator vs the native runs (runs/eval_vs_ref.py): 128-level vs native 256
# (mesh 256), 256-level vs native 512 (mesh 512), 512-level vs the native 512 run itself (no 1024 run exists; mesh 1024).
#   RUNS="runs/chain_M48d_set14:14 ..." bash hpc/slurm_eval_chain4.sh   (inside an allocation, or sbatch on RM)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8} PYTHONUNBUFFERED=1
for spec in $RUNS; do
  d=${spec%%:*}; s=${spec##*:}
  echo "===== $d (set $s)"
  $PY runs/eval_vs_ref.py $d --set $s --nc 128 --mesh 256 --tag 64to128 --fields truth_64to128 base_from64_64to128 emulator_from64_64to128 generative_from64_64to128
  $PY runs/eval_vs_ref.py $d --set $s --nc 256 --mesh 512 --tag 128to256 \
      --fields truth_128to256 base_from128_128to256 emulator_from128_128to256 emulator_from64_128to256 generative_from64_128to256
  $PY runs/eval_vs_ref.py $d --set $s --nc 512 --ref-level 512 --mesh 1024 --tag 256to512 \
      --fields base_from256_256to512 emulator_from256_256to512 emulator_from128_256to512 emulator_from64_256to512 \
               generative_from256_256to512 generative_from64_256to512
done
echo "===== eval chain4 done"
