#!/bin/zsh
# Stage 8: multi-transition weight sharing (owner-approved), then the single-level reference.
#
#   M_qj_multi   ONE operator for 32->64 AND 64->128, s_l = ln(sigma_c), qj loss,
#                3000 steps per level alternating (runs/multi_train.py). ~9 h.
#   J_flow_128   single-level qj at 64->128, 3000 steps, batch 1 -- (i) does the Stage-7
#                Eulerian gain survive the harder level, (ii) the apples-to-apples
#                reference for the multi model's 128 side. ~7 h.
#
#   PYTHONUNBUFFERED=1 nohup runs/stage8_train.sh >> runs/stage8_train.log 2>&1 &
#   touch runs/PAUSE   # same protocol
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S)"; exit 0; }; }
while pgrep -f 'octave_flow_toy.py|multi_train.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), commit $(git rev-parse --short HEAD)"

echo "########## M_qj_multi start @$(date +%Y-%m-%d\ %H:%M:%S)"
while [[ ! -f runs/M_qj_multi/results_64to128.json ]]; do
  paused
  $PY runs/multi_train.py --steps 3000 --device mps --max-seconds $CHUNK --resume \
      --out runs/M_qj_multi || { echo "########## M_qj_multi FAILED"; break; }
done
echo "########## M_qj_multi end @$(date +%H:%M:%S)"
paused

DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
C128="--nc 64 --nf 128 --box 100000 --offset 0 --growth 76.7439 \
      --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
      --steps 3000 --base 24 --batch 1 --device mps --octave-sampler full --loss-metric qj"
echo "########## J_flow_128 start @$(date +%Y-%m-%d\ %H:%M:%S)"
while [[ ! -f runs/J_flow_128/results.json ]]; do
  paused
  $PY octave_flow_toy.py ${=C128} ${=DATA} --out runs/J_flow_128 \
      --max-seconds $CHUNK --resume || { echo "########## J_flow_128 FAILED"; break; }
done
echo "########## J_flow_128 end @$(date +%H:%M:%S)"
echo "########## STAGE8 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
