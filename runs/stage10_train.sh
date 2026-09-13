#!/bin/zsh
# Stage 10: rollout fine-tuning (owner's protocol) + the residual-flow comparison (approved).
#   RFT_mix50  downstream 64->128 fine-tuned from M_qj_multi, coarse = predicted with P=0.5
#   RFT_mix0   CONTROL: same fine-tuning, true coarse only (separates "mixing helps" from
#              "1500 extra low-LR steps help")
#   RF_resflow frozen regression base + residual flow, 32->64, 3000 steps
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S)"; exit 0; }; }
while pgrep -f 'octave_flow_toy.py|multi_train.py|rollout_ft.py|resflow_train.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), commit $(git rev-parse --short HEAD)"
loop () {
  local name=$1 done_file=$2; shift 2
  if [[ -f "$done_file" ]]; then echo "########## $name SKIP"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S)"
  while [[ ! -f "$done_file" ]]; do
    paused
    "$@" --max-seconds $CHUNK --resume || { echo "########## $name FAILED"; return 1; }
  done
  echo "########## $name end @$(date +%H:%M:%S)"
}
loop RFT_mix50  runs/RFT_mix50/model_ema.pt  $PY runs/rollout_ft.py --steps 1500 --mix 0.5 --out runs/RFT_mix50
paused
loop RFT_mix0   runs/RFT_mix0/model_ema.pt   $PY runs/rollout_ft.py --steps 1500 --mix 0.0 --out runs/RFT_mix0
paused
loop RF_resflow runs/RF_resflow/model_ema.pt $PY runs/resflow_train.py --steps 3000 --out runs/RF_resflow
echo "########## STAGE10 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
