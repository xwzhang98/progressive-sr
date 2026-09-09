#!/bin/zsh
# Generate a multi-seed nested 32/64 N-body set with MP-Gadget, on this laptop.
#
# Why: the production snapshots on Box ship without initial conditions, and the physical
# coupling needs the FINE-level IC octave, so no real-data training is possible with them.
# This produces N-body data (with shell crossing, which the 2LPT toy cannot make) together
# with its ICs.
#
#   ./sims/run_selfsim.sh [NSEEDS]        # default 10: seeds 0..9
#
# ~2.2 min per seed (GenIC 32+64, Gadget 32 = 21 s, Gadget 64 = 104 s), so 10 seeds ~22 min.
# Everything lands in data/selfsim/s<seed>/ (gitignored). Resumable: a seed whose
# dis_64.npy already exists is skipped.
#
# CONVENTIONS (do not mix with the production data):
#   * MP-GenIC puts particles on cell CORNERS -> --offset 0 everywhere below, verified on the
#     z=99 snapshot: rms|x-q|/h = 0.033 / 0.066 (the 0.5 alternative gives 0.87).
#     The Box production disp.npy files use cell CENTRES and need --offset 0.5 instead.
#   * Lengths stay in kpc/h (--pos-unit 1, --box 100000), like the production data.
#   * Growth D(z=0)/D(z=99) = 76.7439 for Omega0=0.2814, OmegaLambda=0.7186.
#   * MP-GenIC ICs at one seed are NOT bitwise nested across Ngrid (|R IC_64 - IC_32|/|IC_32|
#     = 1.4e-01); the white noise is shared (r = 1.00000 at low k) and they diverge toward
#     the coarse Nyquist. Only the fine IC is used by the training source, and that is exact.

set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
MPG=$HOME/src/MP-Gadget
PY=$ROOT/.venv/bin/python
NSEEDS=${1:-10}
export OMP_NUM_THREADS=6

for s in $(seq 0 $((NSEEDS-1))); do
  OUT=$ROOT/data/selfsim/s$s
  if [[ -f $OUT/dis_64.npy ]]; then echo "### seed $s SKIP (done)"; continue; fi
  echo "### seed $s  @$(date +%H:%M:%S)"
  mkdir -p $OUT
  SEED=$((181170 + 1000*s))

  for N in 32 64; do
    sed -e "s|^OutputDir = .*|OutputDir = $OUT/N$N|" \
        -e "s|^Ngrid = .*|Ngrid = $N|" \
        -e "s|^Seed = .*|Seed = $SEED|" \
        $ROOT/sims/genic_64.param > $OUT/genic_$N.param
    mpirun -np 2 $MPG/genic/MP-GenIC $OUT/genic_$N.param > $OUT/genic_$N.log 2>&1

    sed -e "s|^InitCondFile = .*|InitCondFile = $OUT/N$N/IC|" \
        -e "s|^OutputDir = .*|OutputDir = $OUT/N$N|" \
        $ROOT/sims/gadget_64.param > $OUT/gadget_$N.param
    mpirun -np 2 $MPG/gadget/MP-Gadget $OUT/gadget_$N.param > $OUT/gadget_$N.log 2>&1

    $PY $ROOT/hpc/convert_snapshot.py $OUT/N$N/IC $OUT/ic_dis_$N.npy \
        --box 100000 --pos-unit 1 --ptype 1 --id-offset 1 --id-order C --offset 0 > /dev/null
    $PY $ROOT/hpc/convert_snapshot.py $OUT/N$N/PART_000 $OUT/dis_$N.npy --vel $OUT/vel_$N.npy \
        --box 100000 --pos-unit 1 --ptype 1 --id-offset 1 --id-order C --offset 0 --force > /dev/null
    rm -rf $OUT/N$N/IC $OUT/N$N/PART_000     # bigfiles no longer needed; keep the .npy cubes
  done
  echo "### seed $s done @$(date +%H:%M:%S)"
done
echo "### ALL SEEDS DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
