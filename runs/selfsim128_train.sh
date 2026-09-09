#!/bin/zsh
# flow vs regression one level up: 64->128 on the self-run MP-Gadget set.
# PAUSABLE AND RESUMABLE -- see "Pausing" below.
#
# Purpose: every Stage 5 conclusion rests on the 32->64 pair, which is a very coarse
# transition. This repeats R1/R2 at 64->128 to see whether the accuracy/power trade-off
# survives, and gives the first N-body test of the self-similarity premise behind weight
# sharing (Sec. 6): the same dimensionless quantities, one octave finer.
#
#   PYTHONUNBUFFERED=1 nohup runs/selfsim128_train.sh >> runs/selfsim128_train.log 2>&1 &
#
# Pausing
# -------
#   touch runs/PAUSE          # finishes the current chunk, saves, then exits cleanly
#   rm runs/PAUSE             # then relaunch the same command to carry on
# A hard kill is also safe: at most CHUNK seconds of training are lost, because each chunk
# ends with `--max-seconds` writing <out>/train_state.pt (model, EMA, optimizer, step, loss
# log) and the next chunk picks it up with `--resume`.
#
# BATCH 1, not 2: at 128^3 with --batch 2 the machine thrashes (measured 127 s/step against
# 7.55 s/step at batch 1 -- unified memory, so MPS activations never show up in RSS). 1500
# steps is then ~3.1 h per run. This is the one setting that differs from the 32->64 runs and
# must be quoted with any cross-level comparison.
#
# --octave-sampler full: the training coupling is physical, so the sampled branch is never used
# in training; this only makes the in-run GENERATIVE evaluation use the corrected octave (with
# its transverse component). The 32->64 numbers exist both ways for comparison.
#
# One GPU job at a time. A run whose results.json exists is skipped entirely.

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}          # seconds of training per invocation; a kill costs at most this
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 64 --nf 128 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 1500 --base 24 --batch 1 --device mps --octave-sampler full"

paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S) (rm runs/PAUSE and relaunch to resume)"; exit 0; }; }

while pgrep -f 'octave_flow_toy.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), chunk = ${CHUNK}s"

run () {
  local name=$1; shift
  local out=$1; shift
  if [[ -f "$out/results.json" ]]; then echo "########## $name SKIP (done)"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S) -> $out"
  while [[ ! -f "$out/results.json" ]]; do
    paused
    $PY octave_flow_toy.py ${=COMMON} ${=DATA} "$@" --out "$out" \
        --max-seconds $CHUNK --resume || { echo "########## $name FAILED"; return 1; }
  done
  echo "########## $name end @$(date +%H:%M:%S)"
}

run F1_flow_128 runs/F_flow_128
paused
run F2_reg_128  runs/F_reg_128  --regression

echo "########## SELFSIM128 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
