#!/bin/zsh
# Does an Eulerian term in the loss fix the Eulerian failure?  32->64, flow, from scratch.
#
# Why: the Lagrangian octave-band P/P and the Eulerian density power disagree about which model
# is better, and at k_Ny,c neither model is near 1 (32->64: flow 0.963, regression 1.435, and
# it gets worse one level down -- flow 0.714 at 64->128). The flow-matching loss is blind to
# this: it sees the displacement variance, not whether caustics form in the right place.
# `--cic-weight` adds MSE on log(1+delta) of the CIC density built from the predicted x1.
#
# lambda is calibrated against the two terms at 32->64 convergence (flow term ~8.5e-2, CIC term
# ~8.3e-2 at the baseline), so lambda = 1.0 makes them comparable and 0.3 makes the CIC term
# ~30% of the flow term. Both are run to see a trend rather than to pick a winner.
#
#   PYTHONUNBUFFERED=1 nohup runs/cic_train.sh >> runs/cic_train.log 2>&1 &
#   touch runs/PAUSE   # same protocol as the other drivers
#
# ~2.11 s/step with the CIC term (1.83 without, so +15%), 3000 steps => ~1.8 h per run.

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
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S)"

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

run C1_flow_cic1p0 runs/C_flow_cic1p0 --cic-weight 1.0
paused
run C2_flow_cic0p3 runs/C_flow_cic0p3 --cic-weight 0.3

echo "########## CIC TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
