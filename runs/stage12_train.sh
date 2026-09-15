#!/bin/zsh
# Stage 12: rollout pilot round 2 on the resflow pair (owner's spec).
#   RFT2_mix50 / RFT2_mix0 (control), 800 steps at LR 5e-5, downstream flow only,
#   paired augmentation, X - B target recomputed for the chosen coarse, lambda reused.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S)"; exit 0; }; }
while pgrep -f 'octave_flow_toy.py|resflow_train.py|rollout_ft' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), commit $(git rev-parse --short HEAD)"
loop () {
  local name=$1 out=$2 mix=$3
  if [[ -f "$out/model_ema.pt" ]]; then echo "########## $name SKIP"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S)"
  while [[ ! -f "$out/model_ema.pt" ]]; do
    paused
    $PY runs/rollout_ft2.py --steps 800 --mix $mix --out "$out" --max-seconds $CHUNK --resume \
        || { echo "########## $name FAILED"; return 1; }
  done
  echo "########## $name end @$(date +%H:%M:%S)"
}
loop RFT2_mix50 runs/RFT2_mix50 0.5
paused
loop RFT2_mix0  runs/RFT2_mix0  0.0
echo "########## STAGE12 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
