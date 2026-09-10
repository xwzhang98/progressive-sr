#!/bin/zsh
# Extend the four flow/regression runs from 1500 to 3000 steps, to settle whether
# "the network gains less one level down" is real or an undertraining artefact.
#
# At 1500 steps neither level is converged: over the last third the loss still falls by 10.4%
# (32->64) and 12.8% (64->128), and the finer level -- the one that looked worse -- is the one
# with further to go. The cross-level comparison is therefore not quotable as it stands.
#
# Uses --resume, so only the extra 1500 steps are paid for. The 1500-step results are preserved
# by copying each run directory to <name>_3k first and deleting only results.json there, so
# both step counts remain on disk.
#
# 32->64 goes first (1.5 h for the pair) so the cheap half is guaranteed even if the night runs
# short; 64->128 follows (6.4 h).
#
#   PYTHONUNBUFFERED=1 nohup runs/extend3k.sh >> runs/extend3k.log 2>&1 &
#   touch runs/PAUSE   # same pause protocol as selfsim128_train.sh
#
# --octave-sampler full on every evaluation: training uses the physical coupling, so the sampler
# never enters it, and this makes the generative lines comparable across all four runs.

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CHUNK=${CHUNK:-900}
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
SEEDS="--train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8"
BASE="--box 100000 --offset 0 --growth 76.7439 $SEEDS --steps 3000 --base 24 --device mps --octave-sampler full"

paused () { [[ -f runs/PAUSE ]] && { echo "########## PAUSED @$(date +%H:%M:%S) (rm runs/PAUSE and relaunch)"; exit 0; }; }

while pgrep -f 'octave_flow_toy.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), chunk = ${CHUNK}s"

extend () {
  local name=$1 src=$2 dst=$3 res=$4; shift 4
  if [[ -f "$dst/results.json" ]]; then echo "########## $name SKIP (done)"; return; fi
  if [[ ! -d "$dst" ]]; then
    if [[ ! -f "$src/train_state.pt" ]]; then echo "########## $name SKIP (no $src/train_state.pt)"; return; fi
    cp -R "$src" "$dst" && rm -f "$dst/results.json" "$dst/summary.png"
    echo "########## $name seeded from $src at step $($PY -c "import torch;print(torch.load('$dst/train_state.pt',map_location='cpu')['step'])")"
  fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S) -> $dst"
  while [[ ! -f "$dst/results.json" ]]; do
    paused
    $PY octave_flow_toy.py ${=BASE} ${=res} ${=DATA} "$@" --out "$dst" \
        --max-seconds $CHUNK --resume || { echo "########## $name FAILED"; return 1; }
  done
  echo "########## $name end @$(date +%H:%M:%S)"
}

# --- 32->64 first: cheap, and needed for the cross-level comparison to be fair ---
extend E1_flow_64_3k  runs/R_flow_phys  runs/R_flow_phys_3k  "--nc 32 --nf 64 --batch 2"
paused
extend E2_reg_64_3k   runs/R_reg_phys   runs/R_reg_phys_3k   "--nc 32 --nf 64 --batch 2" --regression
paused
# --- 64->128: batch 1, see selfsim128_train.sh for why ---
extend E3_flow_128_3k runs/F_flow_128   runs/F_flow_128_3k   "--nc 64 --nf 128 --batch 1"
paused
extend E4_reg_128_3k  runs/F_reg_128    runs/F_reg_128_3k    "--nc 64 --nf 128 --batch 1" --regression

echo "########## EXTEND3K DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
