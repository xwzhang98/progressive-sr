#!/bin/bash
#SBATCH -J ylabel
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=110G
#SBATCH -t 06:00:00
#SBATCH -o runs/slurm_ylabel_%j.log
# Phase A.4 on the PSC series: (a) 128/256/512 convergence in k < k_Ny,64 and (b) Y64/Y128 label
# tests, runs/eval_ylabel.py. The 512^3 deposit on the 256^3 analysis mesh needs ~60 GB.
#   sbatch hpc/slurm_ylabel.sh                 # set 0, nc 64 128, with convergence
#   sbatch --export=ALL,SETS="1 2" hpc/slurm_ylabel.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=${PY:-/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python}
SETS=${SETS:-0}
NC=${NC:-"64 128"}
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE PYTHONUNBUFFERED=1
$PY runs/eval_ylabel.py --sets $SETS --nc $NC --convergence --out runs/ylabel_psc
