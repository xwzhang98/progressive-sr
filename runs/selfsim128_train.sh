#!/bin/zsh
# flow vs regression one level up: 64->128 on the self-run MP-Gadget set.
#
# Purpose: every Stage 5 conclusion rests on the 32->64 pair, which is a very coarse
# transition. This repeats R1/R2 at 64->128 to see whether the accuracy/power trade-off
# survives, and gives the first N-body test of the self-similarity premise behind weight
# sharing (Sec. 6): the same dimensionless quantities, one octave finer.
#
#   PYTHONUNBUFFERED=1 nohup runs/selfsim128_train.sh >> runs/selfsim128_train.log 2>&1 &
#
# BATCH 1, not 2: at 128^3 with --batch 2 the machine thrashes (measured 127 s/step against
# 7.55 s/step at batch 1 -- unified memory, MPS activations do not show up in RSS). 1500 steps
# is then ~3.1 h per run. This is the one setting that differs from the 32->64 runs, and it
# must be quoted with any cross-level comparison.
#
# --octave-sampler full: the training coupling is physical, so the sampled branch is never
# used in training; this only makes the in-run GENERATIVE evaluation use the corrected octave
# (with its transverse component). The 32->64 numbers exist both ways for comparison.
#
# One GPU job at a time. Resumable: a run whose results.json exists is skipped.

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 64 --nf 128 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 1500 --base 24 --batch 1 --device mps --octave-sampler full"

while pgrep -f 'octave_flow_toy.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S)"

run () {
  local name=$1; shift
  local out=$1; shift
  if [[ -f "$out/results.json" ]]; then echo "########## $name SKIP (done)"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S) -> $out"
  $PY octave_flow_toy.py ${=COMMON} ${=DATA} "$@" --out "$out" || echo "########## $name FAILED"
  echo "########## $name end @$(date +%H:%M:%S)"
}

run F1_flow_128 runs/F_flow_128
run F2_reg_128  runs/F_reg_128  --regression

echo "########## SELFSIM128 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
