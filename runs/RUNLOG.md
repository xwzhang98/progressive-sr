# RUNLOG

## 2026-09-08 — run from the Cowork session, inside the desktop-app Linux VM (aarch64, 4 CPU, 3 GB, torch 2.3.1 CPU)

- `python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128 --offset 0.5 --out runs/st_dealias`
  -> operator check 4.0e-7 / 3.8e-7; nestedness 2.6e-7 / 3.1e-7; slopes P_eps 2.19 / 2.18, P_div 4.14 / 4.18 (sphere & cube); Haar 0.85 / 0.64. Matches reference.
- `python phase0_octaves.py --selftest --levels 32 64 128 --offset 0.5 --out runs/st_alias`
  -> aliased coarse levels: sphere slope 0.19 / -0.71 (white floor), as expected.
- `python octave_flow_toy.py --nc 16 --nf 32 --steps 300 --base 16 --rms-delta 2.0 --device cpu --max-seconds 135 [--resume] --out runs/toy16_vm`
  (4 chunks of ~135 s because each VM call is capped at 3 min; 1.43 s/step) -> see results below / results.json
  -> 300 steps: baseline octave r=0.917 P/P=0.834 | emulator (8 Heun) r=0.989 P/P=0.965, coarse eps_rel 5.5e-3 | 1-step r=0.989 P/P=0.937 | generative P/P=0.990, r~0 vs truth; sample-vs-sample coarse r=0.996, gen coarse eps_rel 1.5e-2 > emulator 5.5e-3 (as theory expects). Reference (600 steps, cloud CPU): emulator r=0.994 P/P=0.993.
- `runs/toy16_probe/` is a 3-step probe run; safe to delete (the VM cannot delete files in this folder).

Next: Stage 3 of KICKOFF_PROMPT.md (32->64, 1500 steps) — run natively with `--device mps` (VM has no GPU and a 3-min cap per command).

## 2026-09-08 — Stage 0, native macOS (M-series arm64, macOS 26.1), fresh `.venv`

Env: `python3 -m venv .venv; .venv/bin/pip install -r requirements.txt`
-> torch 2.14.0, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9;
`torch.backends.mps.is_available()` = True, `is_built()` = True.
No MPS fallback needed: conv3d, avg_pool3d, circular pad and GroupNorm all run natively.

- `python octave_flow_toy.py --nc 16 --nf 32 --steps 20 --base 16 --rms-delta 2.0 --device mps --out runs/mps_probe`
  -> spectral ops on cpu / network on mps, as designed. Baseline octave r=0.917 P/P=0.834,
  coarse r=0.9942 P_eps/P=1.16e-02, rms err/h_f=0.186 — identical to the VM CPU run.
  Timing: 2.00 s first step (warm-up), 0.20 s/step steady (cumulative 0.30 s/step over 20).
  20 steps is far too few to train, so emulator == baseline here; this run is a smoke test only.
- `python octave_flow_toy.py --nc 16 --nf 32 --steps 20 --base 16 --rms-delta 2.0 --device cpu --out runs/cpu_probe`
  -> same baseline and same per-step losses to all printed digits as the MPS run (MPS numerics OK).
  Timing: 0.57 s/step.
- Speed summary at 16->32, base=16: MPS 0.20 s/step vs native CPU 0.57 s/step vs 1.43 s/step
  on the aarch64 VM (7.2x faster than the VM, 2.9x faster than native CPU).
- Deleted `runs/toy16_probe/` and the empty `runs/toy16_vm.log` as instructed.
- Note (harness, not a code change): the script's stdout is block-buffered when redirected to a
  file, so background runs are launched with `PYTHONUNBUFFERED=1` to make the log pollable.
  Progress lines are printed every `steps//20` steps.

## 2026-09-08 — Stage 3.1, main 32->64 run (native MPS)

- `PYTHONUNBUFFERED=1 python octave_flow_toy.py --nc 32 --nf 64 --steps 1500 --base 24 --batch 2 --rms-delta 2.0 --device mps --out runs/toy32_main`
  1.87 s/step, 1500 steps = 46.7 min training + eval; model 1.56M params (base=24).
  loss (50-step mean): 3.54e-2 (1) -> 8.33e-3 (150) -> 6.66e-3 (300) -> 5.01e-3 (675) -> 4.18e-3 (1500).
  -> baseline (x0, no network)      octave r=0.918  P/P=0.846 | coarse r=0.9948  P_eps/P=1.05e-02 | rms/h_f=0.188
  -> emulator, 8 Heun steps         octave r=0.997  P/P=1.004 | coarse r=0.9987  P_eps/P=2.55e-03 | rms/h_f=0.059
  -> emulator, 1 Euler step         octave r=0.997  P/P=1.000 | coarse r=0.9987  P_eps/P=2.64e-03 | rms/h_f=0.058
  -> generative, sampled octave A   octave r=0.015  P/P=1.009 | coarse r=0.9913  P_eps/P=1.78e-02 | rms/h_f=0.603
  -> sample A vs sample B           octave r=0.008  P/P=1.000 | coarse r=0.9922  P_eps/P=1.56e-02
  Better than the 16->32 reference (600 steps: r=0.994, P/P=0.993). Generative coarse-band
  eps_rel (1.78e-2) again exceeds the emulator's (2.55e-3) by ~7x, as §5 predicts for the
  stochastic eta-eta backreaction.

## 2026-09-08 — Stage 3.2, sampler-step sweep on runs/toy32_main (no retraining)

`python octave_flow_toy.py --nc 32 --nf 64 --eval-only runs/toy32_main/model_ema.pt --rms-delta 2.0 --nsteps-sample N --device mps --out runs/toy32_eval_nN`, N = 1 2 4 8 16.

| N | emu r | emu P/P | emu eps_rel | gen P/P | gen eps_rel | sample-vs-sample r_low |
|---|-------|---------|-------------|---------|-------------|------------------------|
| 1 | 0.9971 | 1.0001 | 2.64e-03 | 1.0047 | 1.47e-02 | 0.9950 |
| 2 | 0.9974 | 1.0022 | 2.54e-03 | 1.0066 | 1.63e-02 | 0.9935 |
| 4 | 0.9967 | 1.0069 | 2.55e-03 | 1.0113 | 1.73e-02 | 0.9927 |
| 8 | 0.9966 | 1.0044 | 2.55e-03 | 1.0088 | 1.78e-02 | 0.9922 |
| 16| 0.9966 | 1.0049 | 2.55e-03 | 1.0093 | 1.80e-02 | 0.9920 |

Saturated at N=1 for this model: the spread over N is <0.7% and non-monotonic. Very different
from 16->32 (1-step P/P=0.937 vs 0.993 at 8 steps) — see Stage 3.3 for why (nonlinearity).

## 2026-09-08 — Stage 3.3, nonlinearity sweep (800 steps each, 1.89-1.90 s/step)

- `python octave_flow_toy.py --nc 32 --nf 64 --steps 800 --base 24 --batch 2 --rms-delta 1.0 --device mps --out runs/toy32_rms1`
- `python octave_flow_toy.py --nc 32 --nf 64 --steps 800 --base 24 --batch 2 --rms-delta 3.0 --device mps --out runs/toy32_rms3`

| run | rms_delta | steps | base r | base P/P | emu r | emu P/P | 1-step P/P | gen P/P | svs r_low | emu eps_rel | gen eps_rel | rms/h_f base->emu |
|-----|-----------|-------|--------|----------|-------|---------|------------|---------|-----------|-------------|-------------|-------------------|
| toy32_rms1 | 1.0 | 800  | 0.9773 | 0.9569 | 0.9991 | 1.0011 | 0.9985 | 1.0076 | 0.9982 | 9.00e-04 | 4.36e-03 | 0.047 -> 0.015 |
| toy32_main | 2.0 | 1500 | 0.9181 | 0.8465 | 0.9966 | 1.0044 | 1.0001 | 1.0088 | 0.9922 | 2.55e-03 | 1.78e-02 | 0.188 -> 0.059 |
| toy32_rms3 | 3.0 | 800  | 0.8406 | 0.7120 | 0.9838 | 1.0067 | 0.9665 | 1.0074 | 0.9842 | 7.78e-03 | 3.76e-02 | 0.423 -> 0.178 |

Caveat: rms1/rms3 got 800 steps, the rms=2.0 run 1500, so the rms=3.0 row is partly undertrained
(the rms=2.0 run was still at loss 5.0e-3 at step 800 vs 4.18e-3 at 1500).

- Added `runs/compare_32_vs_16.png` (32->64 vs the 16->32 reference, plus the sampler sweep),
  made by a throwaway script in the scratchpad; no project file was modified.
  Note for whoever re-plots: `k_high` in results.json is in physical h/Mpc, not k/k_Ny,c.

## 2026-09-08 — Stage 3.2b, sampler sweep on the rms=1.0 and rms=3.0 models

`--eval-only runs/toy32_rms{1,3}/model_ema.pt --rms-delta {1.0,3.0} --nsteps-sample N`, N = 1 2 4 8 16
-> `runs/toy32_rms{1,3}_eval_nN`. Motivation: at rms=2.0 the 1-step and 8-step results were
identical, but the rms=3.0 training run showed 1-step P/P=0.9665 vs 8-step 1.0067, so the
saturation point looks nonlinearity-dependent. See REPORT for the tables.

## 2026-09-08 — Stage 3b (KICKOFF_STAGE3B.md), after the external review

Note on the earlier Stage 3 runs: `runs/toy32_main`, `runs/toy32_rms{1,3}` and their eval
sweeps were produced BEFORE the review update, i.e. with the sphere window (the only option
at the time) and with the pre-review version of `octave_flow_toy.py`. `runs/toy32_main` is
the sphere/flow/physical point, but it is not strictly comparable with the new cube runs
because the script changed; B1 below re-runs that configuration with the current script.

### D — coarse-only 2LPT harmonics in the detail (review point 1)

- `python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128 --offset 0.5 --out runs/D_harm_rms08`
  -> operator check 3.63e-07 / 3.82e-07; nestedness 2.80e-07 / 3.14e-07; slopes unchanged
  (sphere & cube P_eps=+2.19/+2.18, P_div=+4.14/+4.18; haar +0.85/+0.64).
  rms(eps)/h_c: sphere 0.0215 / 0.0379, cube 0.0013 / 0.0037, haar 0.0202 / 0.0337
  -> the cube correction is ~16x smaller than the sphere one, as the review's Remark 2.4 implies.
  coarse-only harmonics: 32to64 P_harm/P_d=0.002, r(d,harm)=0.031, P_harm/P_nl=0.243, r=0.452
                         64to128 P_harm/P_d=0.005, r(d,harm)=0.061, P_harm/P_nl=0.237, r=0.446
  Reproduces the cloud reference (0.002-0.005, 0.24, 0.45) exactly.
- rms_delta sweep through a scratchpad snippet (import phase0_octaves, make_selftest,
  analyse_transition; no script edit), cube window, de-aliased, levels 32/64/128:

  | rms_delta | P_harm/P_d (32to64 / 64to128) | P_harm/P_nl | r(d_nl,harm) |
  |-----------|------------------------------|-------------|--------------|
  | 0.8 | 0.0018 / 0.0050 | 0.243 / 0.237 | 0.454 / 0.445 |
  | 1.5 | 0.0062 / 0.0167 | 0.243 / 0.237 | 0.454 / 0.445 |
  | 2.5 | 0.0165 / 0.0412 | 0.243 / 0.237 | 0.454 / 0.445 |

  CONTRADICTS the expectation in KICKOFF_STAGE3B.md that `P_harm/P_nl` grows with
  nonlinearity: it is exactly constant. In 2LPT both the coarse-only harmonics and the
  nonlinear part of the detail are pure second order in delta_lin, so the amplitude cancels
  in the ratio; only `P_harm/P_d` grows, and it grows as rms^2 (0.0018 -> 0.0062 -> 0.0165
  tracks (rms/0.8)^2 = 1 / 3.52 / 9.77 to within 5%) because P_d is dominated by the linear
  octave. The 24% figure is a structural constant of the 2LPT toy, not a measurement of how
  the term behaves in a real simulation; that needs real snapshots with --ic and --growth.

### Incident: two concurrent MPS jobs corrupt each other (my operator error, no code fault)

- First attempt at A1/A2 ran the two trainings CONCURRENTLY on the one GPU. Both broke:
  `runs/A_reg_phys` loss -> nan by step 150; `runs/A_flow_phys` loss 2.6e-2 (75) -> 4.99e-2
  (150) -> 1.14 (225) -> 18.2 (300) -> 321 (375), then
  `Error: command buffer exited with error status ... AGXG14XFamilyCommandBuffer / Apple M2 Max`,
  after which the loss printed 0.0 and the step time dropped from 3.4 to 1.27 s/step.
- Diagnosis (single-job controls, all with the current script):
  - `--nc 16 --nf 32 --steps 600 --base 16 --regression`, window cube AND sphere: both stable
    and nearly identical (loss 4.45e-3 vs 4.63e-3 at step 270) -> the cube window is not the cause.
  - `--nc 32 --nf 64 --steps 400 --window cube` single job: stable, loss 3.49e-2 -> 7.38e-3 at
    step 220, 1.88 s/step -> the 32->64 cube configuration is fine on its own.
  Root cause is GPU contention between two MPS processes, not the window, the coupling or the
  script. Probe runs kept as `runs/dbg_16_{cube,sphere}_reg`, `runs/dbg_32_cube_flow`.
- Consequence: all Stage 3b runs must be strictly sequential. Added `runs/stage3b_driver.sh`
  (new file, no existing file touched): sequential A1-A4, B1-B2, then the C eval sweep; it
  skips any run whose `results.json` already exists, so it survives a shutdown and restart.

### Paused at the owner's request

Stopped at 18:47 with A1 ~2 min in (its output directory was removed, nothing partial left).
Nothing from A/B/C has been produced yet. To resume after reboot:
  cd /Users/zhangxiaowen/AntigravityProjects/progressive-sr
  PYTHONUNBUFFERED=1 nohup runs/stage3b_driver.sh >> runs/stage3b.log 2>&1 &
Estimated 4 h 45 min for the six trainings (1.9 s/step, 1500 steps each) plus ~3 min for C.

## 2026-09-08 — repository initialised from the Cowork session (git 2.34 in the desktop-app VM)

- `git init -b main`, `.gitignore` excludes `.venv`, `Claude outputs/`, `runs/**/*.pt`, `*.npy`, logs;
  RUNLOG/REPORTs/results.json/summary.png ARE committed. Initial commit 9bebaab (85 files).
- Added `hpc/` (checklist, SLURM templates, `convert_snapshot.py`) and git-commit/argv recording in
  `results.json` (`git_commit`, `argv`) and `phase0_results.json` (`_git_commit`, `_argv`).
- Note: `octave_flow_toy.py` and `phase0_octaves.py` were replaced on disk while the Stage 3b
  driver was inside A1 (python had already loaded the old module, so A1 is unaffected); A2 onwards
  run the new file, which differs only by the commit/argv bookkeeping in results.json. The
  earlier sphere-window Stage 3 runs and A1 therefore have no `git_commit` field.
- KICKOFF_STAGE3B.md experiment D expectation corrected: in pure 2LPT `P_harm/P_nl` is a
  constant (both terms second order); the trend with nonlinearity is a real-data question.
- To push: create the GitHub repo, then `git remote add origin <url> && git push -u origin main`.

## 2026-09-08 19:23 — Stage 3b resumed after the reboot

`PYTHONUNBUFFERED=1 nohup runs/stage3b_driver.sh >> runs/stage3b.log 2>&1 &`, one job on the
GPU at a time. A1 (cube, flow, physical) is healthy: loss 3.49e-2 (1) -> 1.52e-2 (75) ->
8.46e-3 (150) at 1.90 s/step, i.e. on top of the old sphere run's 1.50e-2 / 8.33e-3 —
further confirmation that the earlier divergence was GPU contention, not the cube window.

Added `runs/make_figures.py` (new file, imports the two main scripts, changes neither):
full-range spectra of the displacement divergence and real-space slices (Lagrangian theta
and an Eulerian CIC density) for coarse / baseline / emulator / generative / truth.
Runs on the CPU by default so it can be used while a training job holds the GPU.
The CIC deposit self-checks mass conservation (262144.0 = 64^3, <delta> = -1.1e-16).

- `python runs/make_figures.py --ckpt runs/toy32_main/model_ema.pt --window sphere --tag sphere`
  -> `runs/fig_spectra_sphere.png`, `runs/fig_slices_sphere.png`.
  P_theta of the prolonged coarse field drops ~11 decades at k_Ny,c = 1.005 h/Mpc; the
  baseline's octave-band power falls from 1.0 to 0.78 of truth; emulator and generative both
  sit on 1.0; r(k) is ~1 below k_Ny,c for all, then stays ~1 for the emulator and drops to 0
  for the generative sample. To be repeated with the cube model (A1) for REPORT_3b.

## 2026-09-09 00:00 — Stage 3b complete (A, B, C); see runs/REPORT_3b.md

All six trainings sequential on MPS, 1.82-1.90 s/step, ~47 min each; C is eval-only.
Cube window unless stated. Baseline (cube): r=0.9176 P/P=0.8446 eps=1.05e-02 rms=0.187.

| run | emu r | emu P/P | r^2 | 1-step r | 1-step P/P | gen P/P | emu eps | gen eps | rms/h_f |
|-----|-------|---------|-----|----------|------------|---------|---------|---------|---------|
| A_flow_phys    | 0.9963 | 1.0090 | 0.9925 | 0.9965 | 0.9880 | 1.0182 | 1.32e-03 | 9.32e-03 | 0.055 |
| A_reg_phys     | 0.9976 | 0.9952 | 0.9952 | 0.9976 | 0.9952 | 1.0047 | 6.12e-04 | 8.98e-03 | 0.045 |
| A_flow_indep   | 0.9542 | 0.9656 | 0.9106 | 0.2212 | 0.0414 | 0.9735 | 1.16e-02 | 1.10e-02 | 0.146 |
| A_reg_indep    | 0.1399 | 0.0233 | 0.0196 | 0.1399 | 0.0233 | 0.0233 | 5.03e-03 | 5.03e-03 | 0.383 |
| B_flow_sphere  | 0.9948 | 1.0341 | 0.9897 | 0.9939 | 1.0419 | 1.0393 | 7.95e-03 | 1.58e-02 | 0.083 |
| B_reg_sphere   | 0.9979 | 0.9957 | 0.9958 | 0.9979 | 0.9957 | 0.9944 | 1.24e-03 | 1.76e-02 | 0.047 |

C (sampler steps on A_flow_phys): N=1 r=0.9965 P/P=0.9880 eps=1.58e-03 | N=2 0.9970 1.0005
1.38e-03 | N=4 0.9964 1.0058 1.32e-03 | N=8 0.9963 1.0090 1.32e-03 | N=16 0.9962 1.0100
1.32e-03. Saturated by N=2; P/P drifts away from 1 as steps are added.

Expectations from KICKOFF_STAGE3B: (2), (3), (4) all held, including the collapse of the
independent-coupling regression (r=0.14 on the r^2 line) and of the independent flow's
one-step evaluation (r=0.22, P/P=0.041). Regression + physical coupling BEATS the flow on
this toy (r 0.9976 vs 0.9963, rms 0.045 vs 0.055 h_f). Cube beats sphere on the generative
coarse-band eps_rel (1.7-2.0x) and on the window-independent rms.
DEVIATION: (2) was expected off the r^2 line but sits on it to 4 decimals; P/P=r^2 is the
locus of any MSE-optimal predictor (the no-network baseline is on it too) and the two lines
merge as r->1, so the diagnostic only discriminates at low r. Argued in REPORT_3b.md, not
explained away.
CAVEAT: B_flow_sphere (new script) gives emu eps 7.95e-03 / rms 0.083 where the pre-review
runs/toy32_main gave 2.55e-03 / 0.059 for the nominally identical configuration; baselines
agree to 4 decimals so the data path is unchanged. Single runs per configuration cannot
separate a different random draw from a real effect => eps_rel carries ~3x run-to-run
uncertainty; repeat seeds are the first item for the next round.

- `python runs/make_figures.py --ckpt runs/A_flow_phys/model_ema.pt --window cube --tag cube`
  -> runs/fig_spectra_cube.png, runs/fig_slices_cube.png
- runs/A_r2_plane.png (the four A models in the (r, P/P) plane, made by a scratchpad script)
- runs/REPORT_3b.md written.
