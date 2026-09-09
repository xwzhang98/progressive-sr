#!/bin/zsh
# Train on the self-run MP-Gadget 32/64 set (sims/run_selfsim.sh), sequentially.
#
# The question this exists to answer: REPORT_3b found that direct regression with the
# physical coupling BEATS the flow on the 2LPT toy (r 0.9976 vs 0.9963), and argued that the
# case for the flow is the reduced-information regime -- multi-stream regions, real N-body
# chaos -- which 2LPT cannot produce (its multistream fraction is 0 by construction). This
# data has a multistream fraction of 0.29, so it can discriminate.
#
#   PYTHONUNBUFFERED=1 nohup runs/selfsim_train.sh >> runs/selfsim_train.log 2>&1 &
#
# STRICTLY SEQUENTIAL: two concurrent MPS jobs corrupt each other (2026-09-08, loss -> 321
# and a Metal command buffer failure; the identical single job was stable).
# Resumable: a run whose results.json exists is skipped.
#
# Conventions for THIS data (differ from the Box production snapshots):
#   --offset 0      MP-GenIC puts particles on cell corners (production data uses 0.5)
#   --box 100000    lengths in kpc/h
#   --growth 76.7439  D(z=0)/D(z=99) for Omega0=0.2814, OmegaLambda=0.7186

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 1500 --base 24 --batch 2 --device mps"

run () {
  local name=$1; shift
  local out=$1; shift
  if [[ -f "$out/results.json" ]]; then echo "########## $name SKIP (done)"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S) -> $out"
  $PY octave_flow_toy.py ${=COMMON} ${=DATA} "$@" --out "$out" || echo "########## $name FAILED"
  echo "########## $name end @$(date +%H:%M:%S)"
}

run R1_flow_phys runs/R_flow_phys
run R2_reg_phys  runs/R_reg_phys  --regression

echo "########## SELFSIM TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
