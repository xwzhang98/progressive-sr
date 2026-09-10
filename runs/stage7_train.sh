#!/bin/zsh
# Stage 7 Part 3 queue: the admissible Q_E / Q_J metrics, 32->64, 3000 steps each.
#
#   E_flow_qe     flow + Q_E on e = v - (x1-x0), positions from the coarse field
#   J_flow_qj     flow + Q_J (p=2, eps=0.1), state from the coarse field
#   E_flow_qe_in  qe + the log(1+delta_c) Eulerian input channel (cin=13)
#   E_reg_qe      regression + Q_E        (if the night allows)
#   J_reg_qj      regression + Q_J        (if the night allows)
#
# lambda-e is auto-equalised at step 1 and recorded in results.json (measured on this box:
# ~0.039 for qe, ~0.010 for qj). Expectations to TEST, from KICKOFF_STAGE6 C: (a) qe/qj move
# emulator P_delta up and the Q_E kernel fraction below 1 with generative numbers unmoved;
# (b) qj matches qe in single-stream, does less in multi-stream and on the kernel fraction;
# (c) the regression's Eulerian excess does NOT go away; (d) eulerian-inputs helps r in
# multi-stream patches more than single-stream. Part 1 found kernel fractions of 14-30 (not
# ~1), so (a)'s "below 1" is a long fall; and Part 2 found the biased jac term already fixes
# the spectrum without fixing the halo-mislocation residual -- the qe run is the one that can.
#
#   PYTHONUNBUFFERED=1 nohup runs/stage7_train.sh >> runs/stage7_train.log 2>&1 &
#   touch runs/PAUSE    # same protocol as always
#
# ~2.0-2.3 s/step measured => ~1.8-2 h per run, ~10 h for all five.

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 3000 --base 24 --batch 2 --device mps --octave-sampler full"

paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S)"; exit 0; }; }

while pgrep -f 'octave_flow_toy.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), commit $(git rev-parse --short HEAD)"

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

run E_flow_qe    runs/E_flow_qe    --loss-metric qe
paused
run J_flow_qj    runs/J_flow_qj    --loss-metric qj
paused
run E_flow_qe_in runs/E_flow_qe_in --loss-metric qe --eulerian-inputs
paused
run E_reg_qe     runs/E_reg_qe     --loss-metric qe --regression
paused
run J_reg_qj     runs/J_reg_qj     --loss-metric qj --regression

echo "########## STAGE7 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
