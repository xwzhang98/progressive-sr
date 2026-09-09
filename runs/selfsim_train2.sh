#!/bin/zsh
# Overnight continuation of runs/selfsim_train.sh on the self-run MP-Gadget 32/64 set.
# Waits for the first chain to finish, then runs, in this priority order:
#
#   1. R1b/R2b -- flow and regression repeated with a different --seed. REPORT_3b found ~3x
#      run-to-run scatter in eps_rel and flagged repeat seeds as the next round's first item;
#      without them a difference of 0.001 in r means nothing.
#   2. R3/R4  -- the independent-coupling arms, completing the experiment-A table on N-body
#      data (the 2LPT version is in REPORT_3b).
#   3. C sweep -- sampler steps on the real-data flow model (eval-only, ~2 min).
#
# STRICTLY SEQUENTIAL (two concurrent MPS jobs corrupt each other). Resumable: any run whose
# results.json exists is skipped, so this can be stopped and restarted.
#
#   PYTHONUNBUFFERED=1 nohup runs/selfsim_train2.sh >> runs/selfsim_train2.log 2>&1 &

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 1500 --base 24 --batch 2 --device mps"

# wait for the first chain, and in any case for the GPU to be free
while pgrep -f 'octave_flow_toy.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S), starting continuation"

run () {
  local name=$1; shift
  local out=$1; shift
  if [[ -f "$out/results.json" ]]; then echo "########## $name SKIP (done)"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S) -> $out"
  $PY octave_flow_toy.py ${=COMMON} ${=DATA} "$@" --out "$out" || echo "########## $name FAILED"
  echo "########## $name end @$(date +%H:%M:%S)"
}

run R1b_flow_phys_s1 runs/R_flow_phys_s1 --seed 1
run R2b_reg_phys_s1  runs/R_reg_phys_s1  --seed 1 --regression
run R3_flow_indep    runs/R_flow_indep   --coupling independent
run R4_reg_indep     runs/R_reg_indep    --coupling independent --regression

for N in 1 2 4 8 16; do
  [[ -f runs/RC_eval_n$N/results.json ]] && { echo "########## RC n=$N SKIP"; continue; }
  echo "=== RC nsteps=$N ==="
  $PY octave_flow_toy.py --nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 \
      ${=DATA} --test-seeds 8 --train-seeds 0 1 2 3 4 5 6 7 \
      --eval-only runs/R_flow_phys/model_ema.pt --nsteps-sample $N \
      --device mps --out runs/RC_eval_n$N
done

echo "########## SELFSIM TRAIN2 DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
