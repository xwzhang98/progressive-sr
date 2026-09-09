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

## 2026-09-09 01:00 — Stage 4: real snapshots from CMU Box

Owner supplied https://cmu.box.com/s/j2tkysa3a5iejzua991o2up5sy2ah2gs. The shared link is
public (no SSO needed for the download endpoint), so the files were fetched directly:
`curl -L "https://cmu.app.box.com/index.php?rm=box_download_shared_file&shared_name=<KEY>&file_id=f_<ID>"`.
Contents of the `sr_training_data` folder: dmo-64.tar.gz 177 MB, dmo-128.tar.gz 1.4 GB,
dmo-256.tar.gz 11.0 GB, dmo-512.tar.gz.part-aa/ab 46+43 GB, state_710.pt 165 MB.
Downloaded 64 and 128; 256 in progress; **512 not downloaded (owner's instruction)**.
Everything lands in `data/`, which was added to .gitignore BEFORE the first download.

Format: already map2map-style, no `hpc/convert_snapshot.py` needed. Each tarball is
`dmo-<N>/set{0..15}/PART_009/{disp,vel,cat,style}.npy`; 16 nested seeds per level, one
snapshot. `disp.npy` is float32 (3,N,N,N), `cat.npy` = concat(disp, vel), `style.npy` = [1.].
**Units are kpc/h, not Mpc/h**: disp rms per component 4519 (64) / 4537 (128), i.e. 4.5 Mpc/h
in a 100 Mpc/h box. All runs therefore use `--box 100000`, which keeps every dimensionless
quantity correct (k comes out in h/kpc; panel (a)'s y-axis label still says (Mpc/h)^3 — a
cosmetic mislabel, the script was not changed for it).

### Grid convention (the offset probe, done without ICs)

There are no IC snapshots in the tarballs, so the usual nestedness check is unavailable.
Substitute: the correct offset is the one minimising |R Psi_f - Psi_c|.
`python phase0_octaves.py --levels 64 128 --box 100000 --dis "data/dmo-{N}/set2/PART_009/disp.npy" --offset {0,0.5} --out runs/real_probe_off{0,0.5}`

| offset | rms(eps)/h_c sphere | cube | haar |
|--------|--------------------|------|------|
| 0      | 0.4204 | 0.4231 | 0.2432 |
| 0.5    | 0.2836 | 0.2745 | 0.2432 |

=> **offset 0.5**. Haar is offset-independent (real-space block average), as expected, and
its value is identical in both runs, which is a useful internal consistency check.
Nestedness confirmed independently: r(coarse, R fine) = 1.00000 at k = 0.04 k_Ny,c and
Wiener r = 0.99999 at low k, so set2 at 64 and at 128 really are the same seed.

### Phase 0 on the real 64/128 pair (set2, offset 0.5)

`python phase0_octaves.py --levels 64 128 --box 100000 --dis "data/dmo-{N}/set2/PART_009/disp.npy" --vel "data/dmo-{N}/set2/PART_009/vel.npy" --offset 0.5 --out runs/real_64to128_set2`

- correction: rms(eps)/h_c = 0.2836 sphere / 0.2745 cube / 0.2432 haar (displacements);
  0.1183 / 0.1199 / 0.1173 for the velocities.
- **low-k slopes are NOT the ideal k^2/k^4**: P_eps = -0.16, P_div,eps = +1.70 (sphere and
  cube alike; haar -0.27 / +1.51). Panel (a) shows P_eps essentially flat (white) across
  0.04-1.0 k_Ny,c. This is the discreteness floor of Sec. 5's caveat, and on this real
  64/128 pair it dominates the k^2 term completely, unlike the de-aliased 2LPT self-test
  (+2.19 / +4.14). In absolute terms it is still ~6 decades below P_Psi at the lowest k and
  only becomes comparable near k_Ny,c, where P_eps/P_coarse = 0.031 (0.41 k_Ny,c) -> 0.128
  (0.59) -> 0.372 (0.81) -> 0.571 (0.94).
- Wiener T(k) (the "best" linear R, Sec. 3.2): 1.00 to 0.25 k_Ny,c, then 0.95 at 0.50,
  0.90 at 0.66, 0.80 at 0.94, 0.50 at 1.06 k_Ny,c.
- conditional stats: var(d/h_f) = 0.2535 with excess kurtosis +3.16, and **multistream
  fraction 0.374** (13x the velocity field's 0.013). The 2LPT toy has 0.000 by construction,
  so this pair does probe the regime the notes say the toy cannot.
- panel (b) is meaningless here: with no IC the script substitutes the fine field itself,
  so r(d, linear octave) = 1 identically. Do not read it.

### Bug fixed (plotting only, no physics)

`plot_all` received the bookkeeping keys `_git_commit` / `_argv` that `main` writes into
`results`, because the filter only excluded `*_vel`:
`TypeError: string indices must be integers` at `r["correction"]["sphere"]`. This broke the
figure on EVERY phase0 run made after that bookkeeping was added (the analysis and the JSON
were fine). One-line fix, also skip keys starting with "_":
`plot_all({k: v for k, v in results.items() if not k.endswith("_vel") and not k.startswith("_")}, args.out)`
Verified by re-running the command above; `phase0_summary.png` is written again.

### Still open

- The IC-dependent diagnostics — (b) detail vs linear octave, and the coarse-only harmonics
  of review point 1 — need IC snapshots at z_init plus D(z_out)/D(z_ic). Not in the tarballs;
  ask the owner whether they exist.
- Which snapshot PART_009 is (redshift) and the cosmology are not recorded anywhere in the
  data; needed for the growth factor.

## 2026-09-09 01:40 — Stage 4 continued: 256 downloaded, series / seed scatter / alpha

Owner confirmed: no IC snapshots for now (they are on the cluster), and all of this data is
z = 0. Consequence to record: **real-data TRAINING is blocked, not just diagnostic (b)**.
`octave_flow_toy.py` needs the FINE-level IC of each pair — `load_real` asserts `ic_f` is
present, `Batcher.make` builds the source as `sc.band(ic_f * growth, "high")`, and
`fit_linear_power` measures P_lin from the same ICs. The coarse IC is never needed for
training (P Psi_c comes from the coarse snapshot); it is needed only for the coarse-only
harmonics diagnostic. Minimum ask for the cluster: the highest-resolution IC displacement
per seed, plus z_init and the cosmology for D(z=0)/D(z_init). Per-level ICs are better
because they re-enable the nestedness check.

### Full series 64 -> 128 -> 256 (set2, offset 0.5)

`python phase0_octaves.py --levels 64 128 256 --box 100000 --dis "data/dmo-{N}/set2/PART_009/disp.npy" --offset 0.5 --out runs/real_series_set2` (9.3 s)

| transition | rms(eps)/h_c sphere / cube / haar | slope P_eps | slope P_div | var(d/h_f) | kurt | multistream |
|------------|-----------------------------------|-------------|-------------|------------|------|-------------|
| 64to128    | 0.2836 / 0.2745 / 0.2432 | -0.16 | +1.70 | 0.2535 | +3.16 | 0.374 |
| 128to256   | 0.4063 / 0.3928 / 0.2432(*) | -0.66 | +1.32 | 0.4741 | +6.00 | 0.417 |

(*) haar at 128to256 is 0.2432 in the printout, identical to 64to128 — worth a second look,
it may be a coincidence of rounding or a reused value; not used in any conclusion here.
The dimensionless quantities DRIFT with level: the correction grows 43%, the detail variance
87%, the kurtosis 90%. At fixed z this is expected from Sec. 6 (going one level down is like
going forward in time, h_l/r_NL changes), and it quantifies how much work the style scalar
s_l has to do for weight sharing. It is not evidence against weight sharing by itself.

### Seed scatter, 64->128, all 16 sets (scratchpad snippet, no script change)

rms(eps)/h_c: cube mean 0.2743 std 0.0019 (**0.7%**), sphere 0.2836 / 0.0019 (0.7%),
haar 0.2429 / 0.0020; multistream frac 0.375 +- 0.001; k where Wiener T = 0.5:
1.075 +- 0.015 k_Ny,c. set2 is representative.
Note the contrast with the ~3x scatter flagged in REPORT_3b: the DATA measurements are
stable to 0.7%, so that earlier scatter was training/optimisation variance, not data variance.
Also note haar has the SMALLEST rms(eps) on real z=0 data (0.2429 vs cube 0.2743) — opposite
to the de-aliased 2LPT self-test. This does not overturn the "no Haar" convention: the notes
reject Haar because it injects a *random* aliasing component into the correction at linear
order (forcing the correction head to be generative), which rms(eps) does not measure. Here
the discreteness floor dominates and hides Haar's low-k mismatch (its slope is still the
worst, -0.27 vs -0.16).

### Sphere alpha sweep — run, then discarded as the wrong metric

alpha 0.6/0.7/0.8/0.9/1.0 gives rms(eps)/h_c = 0.3822/0.3438/0.3168/0.2968/0.2836.
Monotonic, but this does NOT answer "should the unreliable coarse modes near k_Ny,c be
discarded and regenerated": with alpha < 1 the coarse field's own power outside the sphere
is deliberately dropped and lands entirely inside eps, so the metric charges the method for
something it did on purpose. Recorded so it is not repeated.
The question is better answered by the coarse-vs-fine correlation, which is already measured:
r(coarse, R fine) = 0.985 at 0.41 k_Ny,c, 0.938 at 0.59, 0.804 at 0.81, 0.678 at 0.94. The
coarse modes near Nyquist are still strongly informative, not noise, so re-sampling them from
the prior (what the sphere does) discards real information — consistent with review point 4
and with the cube being the default. A real test needs a trained model, i.e. the ICs.

## 2026-09-09 02:00-03:00 — self-run MP-Gadget 32/64 with ICs (owner's suggestion)

Motivation: the Box snapshots have no ICs, which blocks real-data TRAINING entirely (the
source needs the fine IC octave). Running our own small pair gives N-body data — with shell
crossing, which the 2LPT toy structurally cannot produce — together with its ICs.

### Build (macOS arm64, M2 Max)

`brew install open-mpi gsl fftw pkg-config` (pulls gcc-16); MP-Gadget cloned to ~/src/MP-Gadget
(5.0.1.dev1_e85eb611fd). Two things had to be worked around, neither in this repo:
- `depends/install_pfft.sh` uses `wget`, absent here. Pre-placing the tarball with `curl` in
  `depends/` makes the script skip its download; MP-Gadget's own files were not edited.
- `Options.mk`: `MPICC = OMPI_CC=gcc-16 mpicc` builds MP-Gadget but breaks pfft's autoconf,
  which treats the whole string as the program name. Use `MPICC = mpicc` and build with
  `OMPI_CC=gcc-16 make -j8`. (Apple clang, which mpicc wraps by default, has no -fopenmp.)
Both binaries build clean. Param files kept in `sims/` (tracked); outputs in `data/` (ignored).

### Runs

MP-GenIC + MP-Gadget, Ngrid 32 and 64, same Seed 181170, BoxSize 100000 kpc/h (= the
production box), Omega0=0.2814 OmegaLambda=0.7186 h=0.697, z_init=99 -> z=0, class_pk_99.dat.
Wall time on 2 ranks x 6 threads: 21 s (32^3), 104 s (64^3).
`hpc/convert_snapshot.py` conventions found with `--check`: `--id-offset 1 --id-order C
--offset 0` gives rms|x-q|/h = 0.033 (32) / 0.066 (64) at z=99; the alternatives give
0.87 and 26. Note `--pos-unit 1 --box 100000` to stay in kpc/h like the production data.
z=0 displacement rms/component: 4364 (32) / 4432 (64) kpc/h, against 4519/4537 for the
production 64/128 — the self-run data sits at the same physical amplitude.
Growth D(z=0)/D(z=99) = 76.7439 for this cosmology.

Fixed in `hpc/convert_snapshot.py` (it was shipped untested): bigfile's `AttrSet` has no
`.items()`, only `keys()`. Two-line change to iterate over `head.keys()`; nothing else touched.

### MP-GenIC ICs are NOT exactly nested across resolution

`|R_cube IC_64 - IC_32| / |IC_32| = 1.439e-01` at offset 0 (2.458e-01 at offset 0.5), where
an exactly nested pair gives ~1e-6. But they are not independent either (uncorrelated fields
would give sqrt(2)). Resolving it in k:

| k/k_Ny,32 | 0.08 | 0.25 | 0.44 | 0.63 | 0.82 | 1.00 |
|-----------|------|------|------|------|------|------|
| r(R IC_64, IC_32) | 1.00000 | 0.99960 | 0.99541 | 0.97370 | 0.91230 | 0.75915 |
| P_R/P_32 | 0.9999 | 0.9992 | 0.9871 | 0.9417 | 0.8238 | 0.5995 |

So the white noise IS shared -- r = 1 to five decimals at low k -- and the two ICs diverge
progressively toward the COARSE Nyquist, i.e. each level's IC generator discretises
differently where the coarse grid is least able to represent the field. This is Sec. 2.3's
(A)-vs-(B) distinction appearing in the initial conditions themselves, and the production
data very likely has the same property (its r(coarse, R fine) is also 1.00000 at low k).
**This does not block the method**: the physical coupling needs only the FINE IC octave,
which is exact. It does mean the phase0 nestedness check cannot be used as an offset probe
on MP-GenIC output, and that "nested ICs" in the notes should be read as "shared white
noise", not "bitwise truncation".

### First Phase 0 on N-body data WITH ICs (32->64, offset 0, growth 76.7439)

`python phase0_octaves.py --levels 32 64 --box 100000 --dis "data/selfsim/dis_{N}.npy" --ic "data/selfsim/ic_dis_{N}.npy" --offset 0 --growth 76.7439 --out runs/selfsim_32to64`

- rms(eps)/h_c: sphere 0.1947, cube 0.1898, **haar 0.2893**; slopes P_eps/P_div =
  +0.14/+2.01 (sphere and cube), -2.03/-0.09 (haar).
- multistream frac 0.293, var(d/h_f) 0.1411, kurt +1.42.
- coarse-only harmonics: P_harm/P_d = 0.213, P_harm/P_nl = 0.201, but
  **r(d, harm) = -0.031 and r(d_nl, harm) = -0.019, i.e. no correlation at all.**

TO RE-CHECK BEFORE ANY OF THIS IS USED (written at 03:00, deliberately not interpreted):
1. The harmonics term has 20% of the detail's power but ZERO correlation with it. In the
   de-aliased 2LPT self-test the same diagnostic gave r = 0.45. Either the 2LPT harmonic
   prediction simply does not describe the detail of a z=0 N-body run (plausible: 29% of the
   volume is multi-stream and higher orders dominate), or the growth scaling / the use of a
   non-nested ic_c is corrupting it. Zero correlation with non-zero power is exactly what a
   wrong amplitude-and-phase prediction looks like, so do NOT read "20% of the detail is
   deterministic" out of this number until it is understood.
2. Haar is the WORST R here (0.2893) but was the BEST on the production 64/128 pair (0.2429
   vs cube 0.2743). Opposite orderings on two real datasets; the transitions differ (32->64
   vs 64->128) and so does the IC provenance. Needs the same transition on both.
3. The slopes here (+0.14/+2.01) are closer to the ideal 2/4 than the production pair's
   (-0.16/+1.70), even though this pair has the extra non-nested-IC error. Unexplained.

### Grid-offset conventions disagree between the two datasets

MP-GenIC's own ICs want offset 0 (rms|x-q|/h = 0.066 vs 0.87 at 0.5) -- a direct measurement
on the z=99 snapshot, which is the definitive test. The production `disp.npy` files instead
preferred offset 0.5 in the rms(eps) probe. Both can be true: those files were produced by
someone else's converter, which chose its own q convention. Worth asking the owner which
convention their conversion used, because a half-cell error is a real systematic in eps.

### Where this leaves training

Unblocked for the self-run pair: `octave_flow_toy.py --nc 32 --nf 64 --dis
"data/selfsim/dis_{N}.npy" --ic "data/selfsim/ic_dis_{N}.npy" --growth 76.7439` needs the
`{seed}` placeholder too, so a multi-seed set has to be generated first (one seed per
MP-GenIC run; 21 s + 104 s each, so 8 seeds is ~20 min). Not started.

## 2026-09-09 02:20 — overnight CPU analyses (GPU busy with the selfsim training)

### Correction to an earlier entry (append-only, so recorded here rather than edited)

1. The "Stage 4 continued" table above lists haar rms(eps)/h_c = 0.2432 for the 128to256
   production transition and flags it as suspicious. It was wrong: that grep did not include
   the `R=haar` line for 128to256 and I carried the 64to128 value over. The correct value is
   **0.3589**. The conclusion in that entry is unaffected (haar is still the smallest of the
   three at both production transitions).
2. The heading of the MP-Gadget entry says "02:00-03:00"; the actual clock times were
   01:00-01:30.

### Haar vs the spectral R: the ordering flip is (partly) a grid-convention artefact

Haar wins on BOTH production transitions and loses on selfsim, so it is dataset-dependent,
not transition-dependent:

| dataset | transition | offset | sphere | cube | haar | winner |
|---------|-----------|--------|--------|------|------|--------|
| production set2 | 64->128 | 0.5 | 0.2836 | 0.2745 | 0.2432 | haar |
| production set2 | 128->256 | 0.5 | 0.4063 | 0.3928 | 0.3589 | haar |
| selfsim s0 | 32->64 | 0.0 | 0.1947 | 0.1898 | 0.2893 | cube |

Analytically, the Haar block of fine cells 2j, 2j+1 has its centre at (j + (o+0.5)/2) h_c
while the coarse grid point is at (j + o) h_c; these coincide only for **o = 0.5**. At o = 0
the block centre is a quarter of a coarse cell off. The misalignment is NOT a constant
(<eps_haar> is ~1e-4 h_c in every dataset); it is a resampling error ~ delta . grad Psi.
Controlled test on the same selfsim data, shifting BOTH levels to cell centres with
`shift_field`:

  haar 0.2893 -> **0.2155** (-26%)   cube 0.1898 -> 0.1861 (-2%)

So Haar's rms(eps) is convention-sensitive at the 26% level and the spectral R is not; any
Haar-vs-spectral comparison must fix the convention first. This does NOT fully explain the
flip: even aligned, haar (0.2155) still loses to cube (0.1861) on selfsim while winning on
production. Settling it needs the same transition on both datasets, i.e. a 64->128 selfsim
pair (a 128^3 run, ~15 min/seed — not done tonight).

### (b) detail vs linear octave — the panel that is impossible on the production data

`runs/selfsim_32to64`, real N-body, z=0, 32->64, with real ICs:

| k/k_Ny,c | 1.00 | 1.25 | 1.50 | 1.75 | 1.88 |
|----------|------|------|------|------|------|
| r(d, linear octave) | 0.747 | 0.628 | 0.533 | 0.444 | 0.421 |

Band-averaged r = 0.5561, so r^2 = 0.32. **The linear-theory octave linearly explains only
32% of the detail variance at z=0.** T/D = 55.0/76.7 = 0.72: the detail's cross-power with
the octave is 72% of the linear extrapolation (README documents that T carries the growth
ratio, so T is not expected to be 1). Read this as "how much of the detail the hard-coded
linear part of Sec. 4(i) already covers", NOT as irreducible noise: under full physical
conditioning Var(x1 | x0, C) = 0, so the other 68% is the nonlinear map the network learns.

### The coarse-only harmonics anomaly is physics, not a bug — resolved by a redshift scan

Reran seed 181170 with `OutputList = 0.1,0.25,1.0` (a 2-minute rerun) and applied the
diagnostic at each output with the matching growth factor:

| z | growth | multistream | P_harm/P_nl | r(d_nl, harm) | rms(eps)/h_c cube | slope P_div,eps |
|---|--------|-------------|-------------|---------------|-------------------|-----------------|
| 9 | 9.9954 | 0.000 | 0.040 | **0.135** | 0.0181 | +2.12 |
| 3 | 24.8220 | 0.004 | 0.102 | **0.203** | 0.0481 | +2.06 |
| 0 | 76.7439 | 0.291 | 0.201 | **-0.018** | 0.1898 | +2.00 |

The correlation rises from z=9 to z=3 and collapses to zero at z=0, exactly where the
multistream fraction jumps from 0.004 to 0.291. So the 2LPT harmonic description of the
detail holds (weakly) while the flow is single-stream and dies once shell crossing sets in.
The z=0 value P_harm/P_nl = 0.201 has power but zero correlation: it is not a predictive
component there. **The 24% / r = 0.45 from the 2LPT self-test does not transfer to N-body**;
even at z=9 the correlation is only 0.135, because in N-body the other second-order terms
plus discreteness dilute the coarse-only piece. This closes item 1 of the re-check list.

Independent observation from the same scan: the low-k slope of P_div,eps is +2.0 to +2.1 at
ALL THREE redshifts (theory 4), including z=9 where the multistream fraction is exactly 0.
So the shortfall of the ideal k^4 law on N-body data is a DISCRETENESS effect, not a
nonlinearity effect — which is what Sec. 5's caveat says, now with the confound removed.
This also closes item 3 of the re-check list (the selfsim slopes are not "unexplained": they
are the discreteness floor, present from z=9 on).

## 2026-09-09 03:00 — BUG: --growth applied twice to the SAMPLED octave (fixed)

Found on the first real-data run: `runs/R_flow_phys` reported generative
`P/P_true = 7397` while sample-A-vs-sample-B was 1.0018, i.e. the two samples agreed with
each other but both were enormously too powerful.

Cause: `main()` fits the octave prior on growth-scaled ICs,
`sc.fit_linear_power([it["ic_f"] * args.growth for it in train])`, so `A_delta` — and hence
`sample_linear_octave()` — already carries the growth factor; `Batcher.make()` then did
`lin = sc.sample_linear_octave(...) * self.growth`, applying it a second time. The true-octave
branch, `lin = sc.band(icf * self.growth, "high")`, applies it once and is correct, as does
`fit_source_filters`, which uses `ic_f * growth`. So the sampled octave was growth = 76.7 times
too large in amplitude, growth^2 = 5890 too large in power; predicted 1.486 * 5890 = 8752
against the observed 7397 (the difference is the isotropic fit versus the true realisation's
spectral shape).

**Invisible on every toy run**, which all use the default `--growth 1.0`: no Stage 3 or 3b
result is affected. On real data it invalidates the generative lines only; emulator lines and
the training of any `--coupling physical` run are unaffected (they never call the sampled
branch). It WOULD have silently corrupted the training of `--coupling independent` runs.

Fix (one line, in `Batcher.make`): drop the second `* self.growth`, with a comment recording
why. Verified by re-running the R1 checkpoint through `--eval-only` on the CPU:
generative `P/P_true` 7397 -> **0.811**, generative rms/h_f 31.0 -> 0.564.

R1 (`runs/R_flow_phys`) re-evaluated after the fix, self-run N-body 32->64 at z=0
(multi-stream fraction 0.29); rms is now split by the coarse run's multi-stream mask:

| line | octave r | octave P/P | coarse r | coarse eps | rms/h_f (multi / single) |
|------|----------|------------|----------|------------|--------------------------|
| baseline x0 | 0.555 | 1.486 | 0.9464 | 1.08e-01 | 0.577 (0.658 / 0.540) |
| emulator, 8 Heun | 0.722 | 0.993 | 0.9711 | 5.64e-02 | 0.411 (0.490 / 0.373) |
| emulator, 1 Euler | 0.756 | **0.682** | 0.9715 | 5.50e-02 | 0.372 (0.441 / 0.339) |
| generative A | 0.165 | 0.811 | 0.9541 | 9.21e-02 | 0.564 (0.619 / 0.540) |
| sample A vs B | 0.190 | 1.002 | 0.9767 | 4.63e-02 | 0.476 |

Note the one-step evaluation has HIGHER r (0.756 vs 0.722) but much LOWER P/P (0.682 vs
0.993) — the power-deficit pattern Stage 5 predicts, and the opposite of the 2LPT toy where
1-step and 8-step were identical. This is still an intra-model comparison; the discriminating
number is R2, the independently trained regression, which is running.
Also note sample-A-vs-B octave r = 0.190 here versus ~0.008 on the toy: at z=0 a real part of
the octave-band detail is fixed by the coarse field through mode coupling, so two generative
realisations are not independent there.

Queue rearranged for Stage 5 (`runs/stage5_train.sh`), superseding the independent-coupling
R3/R4 I had queued before KICKOFF_STAGE5.md landed: R3/R4 are now the `--source-filter wiener`
runs, followed by seed repeats of R1/R2 for an error bar, then CPU re-evaluation of R1/R2.

## 2026-09-09 03:10 — Haar settled at a matched transition; Phase-0 panel (c) on real data

### Haar vs spectral R, same transition (64->128) and same convention

Using the extra 128^3 self-run made for this (seed 181170, 28 min):

| dataset | offset | sphere | cube | haar | winner | multistream |
|---------|--------|--------|------|------|--------|-------------|
| self-run 64->128 (corners) | 0.0 | 0.2757 | 0.2654 | 0.3936 | cube | 0.369 |
| self-run 64->128 shifted to centres | 0.5 | 0.2733 | **0.2629** | 0.3013 | cube | 0.364 |
| production 64->128 (centres) | 0.5 | 0.2836 | 0.2745 | **0.2432** | haar | 0.374 |

So the ordering flip is neither a transition effect nor fully a convention effect: at the
matched transition, matched convention and nearly equal multistream fraction, cube still wins
on the self-run data and haar still wins on the production data. The spectral values agree
between datasets to 4% (cube 0.2629 vs 0.2745); the whole difference is in Haar, which is 24%
better on the production data. The datasets differ in IC generator settings, the converter
that produced disp.npy, force softening, time stepping and the particle lattice, and I cannot
isolate which tonight.
This does NOT challenge the no-Haar convention: that rests on Haar injecting a *random*
linear-order aliasing component into the correction (forcing a generative correction head),
which rms(eps) does not measure. It does mean **rms(eps) comparisons are not portable between
datasets**, and any future R comparison must be made within one dataset.

### Panel (c): is the detail Gaussian given the local coarse jet?  -> `runs/fig_conditional_real.png`

| dataset | multi frac | var in / out | kurt in | kurt out | global kurt |
|---------|-----------|--------------|---------|----------|-------------|
| self-run 32->64 z=0 | 0.293 | 0.1717 / 0.1284 | 1.30 | 1.37 | 1.42 |
| production 64->128 z=0 | 0.374 | 0.3007 / 0.2253 | 2.68 | 3.43 | 3.16 |
| 2LPT self-test 32->64 | 0.000 | n/a / 0.0067 | n/a | 0.05 | 0.05 |

**CONTRADICTS the expectation stated in KICKOFF_STAGE5.md** ("inside them it is expected to be
far from Gaussian"): the excess kurtosis is *lower* inside the multi-stream patches than
outside, in both datasets (1.30 vs 1.37 and 2.68 vs 3.43). And the outside is not "close to
Gaussian" on the production pair either (3.43, against the stated ~1 criterion); only the
self-run pair is near it (1.37).

Hypothesis, NOT verified: the mask is `det(I + D_L) < 0` evaluated on the COARSE field, so it
misses structures already collapsed at the fine level but unresolved coarsely. Outside the
mask the detail is then a quiet background with rare large-|d| outliers -> heavy tails; inside
it, large |d| is typical, so the distribution is broad but not heavy-tailed relative to its own
variance. Testable by building the mask from the fine field; not done.
Note also that the mask separates variance only weakly: var_in/var_out is 1.34 (self-run) and
1.33 (production). Binning by the coarse delta_L does more (variance rises by 2x from the most
underdense to the densest bin) but leaves the kurtosis flat at 1.1-1.3 (self-run) / 2.4-4.5
(production), so conditioning on the local jet does not Gaussianise the detail on real data.

### Panel (c) follow-up: the coarse-mask hypothesis is REFUTED

Rebuilt the multi-stream mask from the FINE field (`det(I + D_f) < 0`) instead of the coarse
one, to test the hypothesis logged above:

| dataset | coarse-mask frac | fine-mask frac | fine-mask cells the coarse mask misses |
|---------|-----------------|----------------|----------------------------------------|
| self-run 32->64 | 0.293 | 0.369 | 62% |
| production 64->128 | 0.374 | 0.417 | 56% |

| dataset | mask | var in / out | kurt in / out |
|---------|------|--------------|---------------|
| self-run | coarse | 0.1717 / 0.1284 | +1.30 / +1.37 |
| self-run | fine | 0.1643 / 0.1275 | +1.22 / +1.48 |
| production | coarse | 0.3007 / 0.2253 | +2.68 / +3.43 |
| production | fine | 0.2856 / 0.2305 | +2.77 / +3.45 |

The first half of the hypothesis holds — the coarse mask does miss 56-62% of the cells that
are multi-stream at the fine level — but the conclusion does not follow: with the fine mask
the kurtosis is still LOWER inside than outside, and for the self-run pair the gap widens
(1.22 vs 1.48). So "the detail is less heavy-tailed inside multi-stream regions" is a robust
property of both datasets under either mask, not an artefact of using the coarse field.

Reading (interpretation, not proof): excess kurtosis is normalised by the variance squared.
Inside the patches the distribution is already broad, so the same outliers are not heavy tails
relative to its own width; outside, a quiet background with occasional large |d| looks very
heavy-tailed. Both masks separate the variance by only ~1.3x while the tails are relatively
heavier outside.

Consequence for Stage 5: its premise is that the flow's advantage should be localised in the
multi-stream patches. If the non-Gaussianity of the detail is not localised there, that premise
is weaker than stated. The decisive measurement is still the direct one — the multi/single
split of the rms error in R1-R4, which the updated script prints on every evaluation line.

### Wiener source path verified on CPU before spending GPU time (20-step smoke test)

`--source-filter wiener`, self-run 32->64, growth 76.7439. Fitted filters, printed at start-up:

    Tc(k/kNy,c) = 0.25:1.000  0.50:1.018  0.75:0.886  0.95:0.677
    G (k/kNy,c) = 1.05:0.703  1.30:0.494  1.60:0.382  1.90:0.329

Both match KICKOFF_STAGE5.md's stated expectations (Tc ~1 below 0.5 k_Ny,c falling to ~0.7 at
0.95; G ~0.4-0.6 across the octave, measured 0.33-0.70). And the predicted baseline change is
confirmed exactly: octave-band `P/P_true` = **0.331** against `r^2` = 0.555^2 = **0.308** (it
was 1.486 with the raw linear octave), and rms/h_f drops 0.577 -> 0.465. The filtered source
therefore sits on the `P/P = r^2` line, which is where the best linear prediction belongs.

## 2026-09-09 06:25 — Stage 5 results (R1-R4 + seed repeats); see runs/REPORT_5.md

Chain: R3_flow_wiener 03:19-04:06, R4_reg_wiener 04:06-04:52, R1b_flow_s1 04:52-05:38,
R2b_reg_s1 05:38-06:24, then re-evaluation of R1/R2. Self-run MP-Gadget 32->64, z=0,
multi-stream fraction 0.29, 8 train seeds / test seed 8.

| run | r | P/P | r^2 | eps | rms | multi | single | gen P/P | gen r |
|-----|---|-----|-----|-----|-----|-------|--------|---------|-------|
| baseline raw     | 0.5551 | 1.4860 | 0.3081 | 1.08e-01 | 0.577 | 0.658 | 0.540 | - | - |
| baseline wiener  | 0.5550 | 0.3310 | 0.3081 | 9.96e-02 | 0.465 | 0.538 | 0.431 | - | - |
| R1  flow phys s0 | 0.7220 | 0.9928 | 0.5213 | 5.64e-02 | 0.411 | 0.490 | 0.373 | 0.8109 | 0.165 |
| R1b flow phys s1 | 0.7261 | 0.9830 | 0.5272 | 5.73e-02 | 0.410 | 0.489 | 0.372 | 0.8048 | 0.171 |
| R3  flow wiener  | 0.6880 | 1.1266 | 0.4733 | 5.85e-02 | 0.434 | 0.515 | 0.396 | 0.8749 | 0.155 |
| R2  reg  phys s0 | 0.8192 | 0.6902 | 0.6711 | 3.13e-02 | 0.314 | 0.380 | 0.282 | 0.6516 | 0.198 |
| R2b reg  phys s1 | 0.8201 | 0.6878 | 0.6725 | 3.07e-02 | 0.313 | 0.379 | 0.281 | 0.6486 | 0.200 |
| R4  reg  wiener  | 0.8275 | 0.7035 | 0.6848 | 3.02e-02 | 0.309 | 0.374 | 0.278 | 0.6658 | 0.198 |

Seed spread is tiny (regression r 0.0008, flow r 0.0041) against a flow-vs-regression gap of
0.098, so every statement below is 20-100x the repeat scatter.

The predicted separation is REAL and is the opposite of the 2LPT toy: regression sits on the
P/P = r^2 line (0.690 vs 0.671) and the flow on P/P = 1 (0.983-1.127). But it is a trade-off,
not a win for the flow: the regression reaches r = 0.82 where the flow reaches 0.72, with 24%
smaller rms and half the coarse-band eps. Stage 5's "at similar r" does not hold.
The gap is NOT localised in the multi-stream patches — regression is better in both, by nearly
the same factor (multi/single ratio 1.35 regression, 1.31 flow). Caveat: the printed split is
of the rms error; the power deficit is spectral and is not split by region, so "the power
deficit is localised in multi-stream" is untested (needs masked spectra, not implemented).

Wiener source: shortens the path 19% (baseline rms 0.577 -> 0.465, P/P 1.486 -> 0.331 ~ r^2)
but does not improve the trained model — worse for the flow (r 0.688 vs 0.722, 8x the seed
scatter), inside the seed scatter for the regression. Recommend dropping it from round one.
Coarse-band residual matches 1 - r_cf^2 to 5% (0.1044 predicted / 0.1076 measured raw;
0.1040 / 0.0996 after Tc).
Generative mode under-produces for BOTH objectives (flow 0.81, regression 0.65) where the toy
gave 1.02 / 1.00: the "accurate deterministic map of a fresh octave" argument needs an accurate
map, and at r = 0.72-0.82 it is not.

### My own error, caught by an internal consistency check

The first re-evaluation of R2 omitted `--regression`, so a model trained only at t=0 was
integrated with 8 Heun steps: r = 0.709 instead of 0.819. It was exposed by the emulator line
disagreeing between the in-run and re-run evaluations for R2 but not for R1 — the growth fix
touches only the sampled branch, so the emulator lines had to be identical. Corrected; the
table above is the `--regression` evaluation. **Any `--eval-only` of a regression checkpoint
must pass `--regression`.**

runs/REPORT_5.md written.

## 2026-09-09 10:20 — why generative mode loses power: the sampler cannot make a transverse octave

Diagnostic asked for after REPORT_5 flagged the generative power deficit (flow 0.81,
regression 0.65, against 1.02/1.00 on the 2LPT toy) as the largest toy-to-real discrepancy.

### Mixing scan: eta = sqrt(1-a) eta_true + sqrt(a) eta_sample

Built by hand from the R1/R2 checkpoints (no script change), 2 noise realisations per point,
octave-band k-integrated ratios on the test box:

| a | source P/P | source r | flow P/P | flow r | reg P/P | reg r |
|---|-----------|----------|----------|--------|---------|-------|
| 0.00 | 1.5804 | 0.4740 | 0.9674 | 0.6456 | 0.5972 | 0.7621 |
| 0.25 | 1.5202 | 0.4202 | 0.9147 | 0.5917 | 0.5914 | 0.6837 |
| 0.50 | 1.4549 | 0.3517 | 0.8597 | 0.5222 | 0.5835 | 0.5913 |
| 0.75 | 1.3876 | 0.2559 | 0.8031 | 0.4228 | 0.5744 | 0.4713 |
| 1.00 | 1.3152 | 0.0035 | 0.7401 | 0.1506 | 0.5597 | 0.1841 |

Smooth and monotonic — **no off-manifold cliff**, so the review's "the learned velocity is
constrained only on the support of p_t" is not what is happening here. The decomposition is:

* **The regression's power deficit has nothing to do with generative mode.** It is 0.597 with
  the TRUE octave and 0.560 with a fully sampled one. It is the conditional mean being smooth,
  present in emulator mode already.
* **The flow's power is nearly right with the true octave (0.967) and falls to 0.740.** Of that
  23-point drop, **17 points are the source itself** and only ~6 come from losing the
  octave-truth correlation.

### The source prior is 17% low in power, and the reason is structural

The sampled octave has 0.829-0.833 of the true octave's variance, consistent across
realisations. Cause:

| field | longitudinal (curl-free) fraction |
|-------|-----------------------------------|
| IC displacement, all bands (z=99) | 0.9951 |
| IC displacement, coarse band | 0.9997 |
| **IC displacement, octave band** | **0.8281** |
| z=0 displacement, all bands | 0.9868 |
| **z=0 displacement, octave band** | **0.6922** |

`sample_linear_octave` builds `Psi = i k delta / k^2`, which is **purely longitudinal by
construction** (its docstring says "curl-free"). The true IC octave on this grid is only 82.8%
longitudinal, and 0.828 is exactly the measured power ratio 0.820-0.833. The sampler is not
mis-normalised; it structurally cannot produce the missing component.

The transverse part is concentrated entirely in the octave band (the coarse band is 99.97%
longitudinal) and grows with k (P_sampled/P_true = 0.92 at k_Ny,c falling to 0.83 at
1.88 k_Ny,c). That is the signature of **particle-lattice aliasing**, not physical vorticity:
the octave band reaches |k| = sqrt(3) k_Ny,f in the corners of the fine cube, and a displacement
parallel to the un-aliased k is not parallel to the aliased k. It is already 17% at z = 99,
where the flow is exactly potential, which rules out multi-streaming as the cause there; by
z = 0 it is 31%, where genuine multi-stream vorticity adds to it.

**This sharpens point 5 of the external review.** The review objected that the source is
degenerate (rank-one, longitudinal); the response was that the transverse components of the
true detail are functions of eta rather than extra randomness. On this data the *initial
condition octave itself*, as represented on the grid, is 17% transverse — so the sampled source
does not span the target's support at the level of the initial conditions, not merely at the
level of the detail.

### Not changed, needs a decision

Fixing this means changing the source distribution, which CLAUDE.md requires asking about.
Options: (a) sample the octave displacement directly from its measured per-component vector
power instead of going through delta — matches the power, drops the curl-free structure;
(b) keep the longitudinal part as now and add a transverse component with its measured
spectrum; (c) leave it and quote the 17% as a known floor on generative power.
Note that emulator mode is unaffected: it uses the true octave, transverse part included.
