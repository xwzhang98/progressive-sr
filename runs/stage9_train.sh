#!/bin/zsh
# Stage 9: Q_J ablations at 32->64 (reviewer's item 3), 3000 steps each, ~7.6 h total.
#   AB_qj_p0   caustic weight off      (p=0: absolute-J error)
#   AB_qj_p3   stronger caustic weight (p=3: per-Eulerian-volume)
#   AB_div     divergence-only form    (drops the adj(A) state coupling)
#   AB_grad    full-gradient form      (no null space at all)
# Reference: J_flow_qj (adj, p=2). Same recipe otherwise; lambda auto-equalised per form.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 3000 --base 24 --batch 2 --device mps --octave-sampler full --loss-metric qj"
paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S)"; exit 0; }; }
while pgrep -f 'octave_flow_toy.py|multi_train.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), commit $(git rev-parse --short HEAD)"
run () {
  local name=$1; shift; local out=$1; shift
  if [[ -f "$out/results.json" ]]; then echo "########## $name SKIP"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S)"
  while [[ ! -f "$out/results.json" ]]; do
    paused
    $PY octave_flow_toy.py ${=COMMON} ${=DATA} "$@" --out "$out" \
        --max-seconds $CHUNK --resume || { echo "########## $name FAILED"; return 1; }
  done
  echo "########## $name end @$(date +%H:%M:%S)"
}
run AB_qj_p0 runs/AB_qj_p0 --jac-p 0
paused
run AB_qj_p3 runs/AB_qj_p3 --jac-p 3
paused
run AB_div   runs/AB_div   --jac-form div
paused
run AB_grad  runs/AB_grad  --jac-form grad
echo "########## STAGE9 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
