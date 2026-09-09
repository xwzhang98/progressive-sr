#!/bin/zsh
# Add the 128^3 level to the existing self-run set, giving a 64->128 pair per seed.
#
# Why: the 32->64 pair of sims/run_selfsim.sh is a very coarse transition, and every Stage 5
# conclusion rests on it. One level up answers three things at once —
#   * the self-similarity premise of weight sharing (Sec. 6), on N-body rather than 2LPT;
#   * whether the flow/regression trade-off of REPORT_5 survives a finer transition;
#   * the Haar-vs-spectral ordering flip, which needs the SAME transition on the self-run and
#     production data (production only has 64->128 and up).
# It also matches the production transition, so Phase 0 becomes apples-to-apples.
#
#   ./sims/run_selfsim_128.sh [NSEEDS]      # default 10, i.e. the seeds run_selfsim.sh made
#
# ~28 min per seed measured (MP-GenIC + MP-Gadget to z=0 + two conversions), so 10 seeds is
# about 4.7 h. Writes into the existing data/selfsim/s<seed>/ next to the 32 and 64 cubes, so
# `--dis data/selfsim/s{seed}/dis_{N}.npy` works for --nc 64 --nf 128 with no other change.
# Resumable: a seed whose dis_128.npy exists is skipped.
#
# Conventions identical to run_selfsim.sh: MP-GenIC particles on cell corners (--offset 0),
# lengths in kpc/h, growth D(z=0)/D(z=99) = 76.7439. Seed s is 181170 + 1000*s, the same value
# run_selfsim.sh used, so the 64^3 already on disk belongs to the same realisation.

set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
MPG=$HOME/src/MP-Gadget
PY=$ROOT/.venv/bin/python
NSEEDS=${1:-10}
export OMP_NUM_THREADS=6

for s in $(seq 0 $((NSEEDS-1))); do
  OUT=$ROOT/data/selfsim/s$s
  if [[ -f $OUT/dis_128.npy ]]; then echo "### seed $s SKIP (done)"; continue; fi
  if [[ ! -f $OUT/dis_64.npy ]]; then echo "### seed $s SKIP (no 64^3 yet - run run_selfsim.sh first)"; continue; fi
  echo "### seed $s  128^3 start @$(date +%H:%M:%S)"
  SEED=$((181170 + 1000*s))
  mkdir -p $OUT

  sed -e "s|^OutputDir = .*|OutputDir = $OUT/N128|" \
      -e "s|^Ngrid = .*|Ngrid = 128|" \
      -e "s|^Seed = .*|Seed = $SEED|" \
      $ROOT/sims/genic_64.param > $OUT/genic_128.param
  mpirun -np 2 $MPG/genic/MP-GenIC $OUT/genic_128.param > $OUT/genic_128.log 2>&1

  sed -e "s|^InitCondFile = .*|InitCondFile = $OUT/N128/IC|" \
      -e "s|^OutputDir = .*|OutputDir = $OUT/N128|" \
      $ROOT/sims/gadget_64.param > $OUT/gadget_128.param
  mpirun -np 2 $MPG/gadget/MP-Gadget $OUT/gadget_128.param > $OUT/gadget_128.log 2>&1

  $PY $ROOT/hpc/convert_snapshot.py $OUT/N128/IC $OUT/ic_dis_128.npy \
      --box 100000 --pos-unit 1 --ptype 1 --id-offset 1 --id-order C --offset 0 > /dev/null
  $PY $ROOT/hpc/convert_snapshot.py $OUT/N128/PART_000 $OUT/dis_128.npy --vel $OUT/vel_128.npy \
      --box 100000 --pos-unit 1 --ptype 1 --id-offset 1 --id-order C --offset 0 --force > /dev/null
  rm -rf $OUT/N128/IC $OUT/N128/PART_000     # bigfiles no longer needed; keep the .npy cubes
  echo "### seed $s  128^3 done @$(date +%H:%M:%S)"
done
echo "### ALL 128 SEEDS DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
