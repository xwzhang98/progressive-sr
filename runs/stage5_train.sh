#!/bin/zsh
# Stage 5 (KICKOFF_STAGE5.md) on the self-run MP-Gadget 32/64 set, strictly sequential.
#
#   R3/R4  the data-driven source, --source-filter wiener (the Stage 5 ask)
#   R1b/R2b the seed repeats of R1/R2, so the R1-R4 table has an error bar; REPORT_3b found
#           ~3x run-to-run scatter in eps_rel and flagged repeats as the next round's first item
#   re-eval of R1/R2 on the CPU, because both were launched before the growth double-count
#           in the sampled-octave branch was fixed, so their generative lines are invalid
#
# One GPU job at a time (two concurrent MPS jobs corrupt each other). Resumable: a run whose
# results.json exists is skipped.
#
#   PYTHONUNBUFFERED=1 nohup runs/stage5_train.sh >> runs/stage5_train.log 2>&1 &

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA='--dis data/selfsim/s{seed}/dis_{N}.npy --ic data/selfsim/s{seed}/ic_dis_{N}.npy'
COMMON="--nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 \
        --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
        --steps 1500 --base 24 --batch 2 --device mps"

while pgrep -f 'octave_flow_toy.py' > /dev/null; do sleep 60; done
echo "########## GPU free @$(date +%Y-%m-%d\ %H:%M:%S)"

run () {
  local name=$1; shift
  local out=$1; shift
  if [[ -f "$out/results.json" ]]; then echo "########## $name SKIP (done)"; return; fi
  echo "########## $name start @$(date +%Y-%m-%d\ %H:%M:%S) -> $out"
  $PY octave_flow_toy.py ${=COMMON} ${=DATA} "$@" --out "$out" || echo "########## $name FAILED"
  echo "########## $name end @$(date +%H:%M:%S)"
}

# --- Stage 5's R3/R4: the Wiener/propagator source -------------------------------
run R3_flow_wiener runs/R_flow_wiener --source-filter wiener
run R4_reg_wiener  runs/R_reg_wiener  --source-filter wiener --regression

# --- seed repeats of R1/R2, for the error bar ------------------------------------
run R1b_flow_s1 runs/R_flow_phys_s1 --seed 1
run R2b_reg_s1  runs/R_reg_phys_s1  --seed 1 --regression

# --- CPU re-evaluation of R1/R2 (they ran with the growth bug; emulator lines were
#     unaffected, generative lines were not). CPU so it never contends for the GPU.
for m in R_flow_phys R_reg_phys; do
  [[ -f runs/${m}_eval/results.json ]] && { echo "########## re-eval $m SKIP"; continue; }
  echo "########## re-eval $m @$(date +%H:%M:%S)"
  $PY octave_flow_toy.py --nc 32 --nf 64 --box 100000 --offset 0 --growth 76.7439 ${=DATA} \
      --train-seeds 0 1 2 3 4 5 6 7 --test-seeds 8 \
      --eval-only runs/$m/model_ema.pt --device cpu --out runs/${m}_eval
done

echo "########## STAGE5 TRAIN DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
