#!/bin/bash
#SBATCH -J phase0
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o runs/slurm_phase0_%j.log
# Phase 0 on the real nested series. Edit the data patterns, offset and growth factor.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
DATA=/path/to/series            # expects $DATA/{N}/dis.npy and $DATA/{N}/ic_dis.npy
OFFSET=0.5                      # from the nestedness check on the 64/128 pair
GROWTH=1.0                      # D(z_out)/D(z_ic)
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
python phase0_octaves.py --levels 64 128 256 512 --box 100 \
  --dis "$DATA/{N}/dis.npy" --ic "$DATA/{N}/ic_dis.npy" --vel "$DATA/{N}/vel.npy" \
  --offset $OFFSET --growth $GROWTH --window cube --out runs/phase0_real_$SLURM_JOB_ID
