#!/bin/bash
#SBATCH -J phase0
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o runs/slurm_phase0_%j.log
# Phase 0 on the real nested series (one set = one seed, all four levels).
# Override DATA/SET/OFFSET/GROWTH/BOX/PY/LEVELS via sbatch --export=ALL,VAR=... ; defaults = the
# PSC series of 2026-09-16 (map2map .npy in kpc/h, IC at z=99 only for the 64 and 512 levels, so
# the 512 IC is passed as the single top-level file and lower-level ICs are cube-truncated from it).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # sbatch spools the script; submit from the repo root
PY=${PY:-/hildafs/projects/phy200018p/xzhangn/source/anaconda3/envs/torch206/bin/python}
DATA=${DATA:-/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology}
SET=${SET:-0}
LEVELS=${LEVELS:-"64 128 256 512"}
TOP=${LEVELS##* }
OFFSET=${OFFSET:-0.5}           # cross-phase audit, RUNLOG 2026-09-16 (PSC series = 0.5)
GROWTH=${GROWTH:-76.7439}       # D(z=0)/D(z=99), same cosmology as sims/ (Omega0 0.2814, h 0.697)
BOX=${BOX:-100000}              # fields on disk are kpc/h
OUT=${OUT:-runs/phase0_real_set${SET}_$SLURM_JOB_ID}
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE PYTHONUNBUFFERED=1
$PY phase0_octaves.py --levels $LEVELS --box $BOX \
  --dis "$DATA/dmo-{N}/set$SET/PART_009/disp.npy" --ic "$DATA/dmo-$TOP/set$SET/IC/disp.npy" \
  --vel "$DATA/dmo-{N}/set$SET/PART_009/vel.npy" \
  --offset $OFFSET --growth $GROWTH --window cube --out "$OUT"
