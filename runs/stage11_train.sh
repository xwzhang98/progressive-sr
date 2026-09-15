#!/bin/zsh
# Stage 11 (owner-approved 1+2): residual flow at the hard level.
#   RF_resflow_128  base = frozen F_reg_128_3k, residual flow 64->128, 3000 steps, batch 1.
#   Parameter note for all comparisons: base+flow = 3.11M vs 1.56M single-stage.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S)"; exit 0; }; }
while pgrep -f 'octave_flow_toy.py|resflow_train.py|rollout_ft.py|multi_train.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), commit $(git rev-parse --short HEAD)"
echo "########## RF_resflow_128 start @$(date +%Y-%m-%d\ %H:%M:%S)"
while [[ ! -f runs/RF_resflow_128/model_ema.pt ]]; do
  paused
  $PY runs/resflow_train.py --nc 64 --nf 128 --batch 1 --base-ckpt runs/F_reg_128_3k/model_ema.pt \
      --steps 3000 --out runs/RF_resflow_128 --max-seconds $CHUNK --resume \
      || { echo "########## FAILED"; break; }
done
echo "########## STAGE11 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
