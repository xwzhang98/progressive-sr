#!/usr/bin/env bash
#SBATCH --job-name=octave_twig_pool
#SBATCH --partition=TWIG-GPU
#SBATCH --qos=twig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm_hold_%x-%j.log
# Hold ONE A100 on TWIG-GPU (owner allowed a second A100 there, 2026-09-16 23:30) and launch experiments into it with
#   srun --jobid=<this job> --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:1 ... python -u ...
# (owner's pattern). State/heartbeat under runs/.claims/ (gitignored). Never run two trainings in it.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
state_dir="runs/.claims/twig_1xa100_${SLURM_JOB_ID}"
mkdir -p "$state_dir"
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "partition=${SLURM_JOB_PARTITION}"; echo "started=$(date -Is)"; } > "$state_dir/owner"
trap 'date -Is > "$state_dir/ended_at"' EXIT TERM INT
while true; do date -Is > "$state_dir/heartbeat"; sleep 60; done
