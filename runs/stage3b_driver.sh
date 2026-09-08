#!/bin/zsh
# Stage 3b driver (KICKOFF_STAGE3B.md): experiments A, B, C, run STRICTLY SEQUENTIALLY.
#
# Two MPS jobs on the same GPU corrupt each other: on 2026-09-08 a concurrent pair at
# 32->64 diverged (loss 3.5e-2 -> 321) and hit "command buffer exited with error status"
# on the M2 Max, while the identical single-job run was stable.  Never run two of these
# at once.
#
# Resumable: a run whose <out>/results.json already exists is skipped, so this can be
# stopped (Ctrl-C / shutdown) and restarted; it picks up at the first unfinished run.
#
#   cd /Users/zhangxiaowen/AntigravityProjects/progressive-sr
#   PYTHONUNBUFFERED=1 nohup runs/stage3b_driver.sh >> runs/stage3b.log 2>&1 &
#   tail -f runs/stage3b.log
#
# ~47 min per training run at 1.9 s/step, 6 runs + a fast eval sweep = ~4h45m total.

cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
COMMON="--nc 32 --nf 64 --steps 1500 --base 24 --batch 2 --rms-delta 2.0 --device mps"

run () {
  local name=$1; shift
  local out=$1; shift
  if [[ -f "$out/results.json" ]]; then
    echo "########## $name  SKIP (already done: $out/results.json)"
    return
  fi
  echo "########## $name  start @$(date +%Y-%m-%d\ %H:%M:%S)  -> $out"
  $PY octave_flow_toy.py ${=COMMON} "$@" --out "$out" || echo "########## $name FAILED"
  echo "########## $name  end   @$(date +%H:%M:%S)"
}

# --- A: objective x coupling, cube window (the default) ----------------------
run A1_flow_phys   runs/A_flow_phys
run A2_reg_phys    runs/A_reg_phys    --regression
run A3_flow_indep  runs/A_flow_indep  --coupling independent
run A4_reg_indep   runs/A_reg_indep   --coupling independent --regression

# --- B: the same objectives with the sphere window (ablation) ----------------
run B1_flow_sphere runs/B_flow_sphere --window sphere --alpha 1.0
run B2_reg_sphere  runs/B_reg_sphere  --window sphere --alpha 1.0 --regression

# --- C: sampler steps on the cube flow model, no retraining ------------------
for N in 1 2 4 8 16; do
  if [[ -f "runs/C_eval_n$N/results.json" ]]; then
    echo "########## C n=$N SKIP"; continue
  fi
  echo "=== C nsteps=$N ==="
  $PY octave_flow_toy.py --nc 32 --nf 64 --eval-only runs/A_flow_phys/model_ema.pt \
      --rms-delta 2.0 --nsteps-sample $N --device mps --out runs/C_eval_n$N
done

echo "########## STAGE3B DONE @$(date +%Y-%m-%d\ %H:%M:%S)"
