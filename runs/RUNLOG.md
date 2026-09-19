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

## 2026-09-09 13:10 — the 64->128 self-run set is built; Phase 0 on it; 128^3 training cost

`sims/run_selfsim_128.sh 10` added the 128^3 level to all ten seeds (09:39-13:10, ~21 min per
seed), so `data/selfsim/s{n}/` now holds 32/64/128 cubes and their ICs from one realisation.

### Phase 0, self-run 64->128, z=0, offset 0, growth 76.7439

`python phase0_octaves.py --levels 64 128 --box 100000 --dis "data/selfsim/s0/dis_{N}.npy" --ic "data/selfsim/s0/ic_dis_{N}.npy" --offset 0 --growth 76.7439 --out runs/selfsim_64to128`

rms(eps)/h_c: sphere 0.2757, cube 0.2654, haar 0.3936; slopes -0.19 / +1.73 (spectral).
var(d/h_f) = 0.2420, kurt +2.80, multistream 0.369. Harmonics still uncorrelated
(r = -0.018) as at 32->64. IC nestedness 8.57e-02 (the shared-white-noise level, as expected).

### Level-to-level drift at fixed z (the weight-sharing question)

| | 32->64 | 64->128 | drift |
|---|--------|---------|-------|
| rms(eps)/h_c cube | 0.1898 | 0.2654 | +40% |
| var(d/h_f) | 0.1410 | 0.2420 | +72% |
| excess kurtosis | 1.42 | 2.80 | +97% |
| multistream frac | 0.291 | 0.369 | +27% |

Same pattern and similar size as the production series (64->128 -> 128->256). This is NOT
evidence against self-similarity: Sec. 6 says a level down at fixed z is a step forward in
time, so the drift is expected and is exactly what s_l must absorb. The real test is a
matched-sigma comparison; for n_eff ~ -2, halving h multiplies sigma by sqrt(2), so the
64->128 pair should be compared with 32->64 at z=0 when D(z)/D(0) = 1/sqrt(2), i.e. z ~ 0.45.
That needs one more 128^3 run with extra outputs (~28 min); not done yet.

Haar at the matched 64->128 transition, self-run: cube 0.2654 vs haar 0.3936 (cube wins), where
the production 64->128 has cube 0.2745 vs haar 0.2432 (haar wins). The flip survives.

### 128^3 training cost: batch 2 thrashes, batch 1 does not

| config | s/step |
|--------|--------|
| batch 2, base 24 | **127** (vm_stat showed 75 MB free: unified-memory thrashing, invisible in RSS) |
| batch 1, base 24 | **7.55** |

So 1500 steps is 53 h at batch 2 and 3.1 h at batch 1. Running at **batch 1**, which is the one
setting that differs from the 32->64 runs and must be quoted with any cross-level comparison.
Patch cropping (which CLAUDE.md requires asking about) is NOT needed at this size.
Baseline at 64->128 is much harder than at 32->64: octave r = 0.394 (against 0.555),
P/P_true = 1.292, rms 0.816 h_f (multi/single 0.896/0.764).

`runs/selfsim128_train.sh` launched: F1 flow + F2 regression, 1500 steps each, ~6.3 h total,
`--octave-sampler full` so the in-run generative lines use the corrected octave (training is
physical-coupling, so the sampler never enters it).

## 2026-09-09 15:45 — the 64->128 runs are pausable and resumable (tested, not assumed)

The first launch of `runs/selfsim128_train.sh` was NOT resumable: `octave_flow_toy.py` writes
`train_state.pt` only when `--max-seconds` fires or training completes, so a kill mid-run loses
everything. Rewritten to run each training as a sequence of `--max-seconds $CHUNK --resume`
invocations (CHUNK = 900 s by default), looping until `results.json` appears.

  touch runs/PAUSE     # current chunk finishes, saves, driver exits cleanly
  rm runs/PAUSE        # then relaunch the same nohup command to carry on

Verified end to end rather than assumed:
- chunk 1 ran to step 115, wrote train_state.pt (25 MB), chunk 2 printed "resumed from step
  115" and continued;
- `touch runs/PAUSE` while chunk 2 was running: it finished at step 233, saved, printed
  "PAUSED", and exited with zero python processes left;
- checkpoint contains step / model / ema / opt / log / base;
- `rm runs/PAUSE` + relaunch printed "resumed from step 233" and carried on.

Cost of chunking: ~40 s of start-up (loading nine 128^3 pairs plus the baseline evaluation) per
900 s chunk, i.e. 4.4%. A hard kill costs at most one chunk. At 7.7 s/step, 1500 steps is ~3.35 h
per run, ~6.7 h for the pair.
Note that `--resume` reseeds the batch sampler as `default_rng(seed + step0)`, so a chunked run
does not see the same batch order as an uninterrupted one; irrelevant for the comparison but
worth knowing if an exact rerun is ever needed.

## 2026-09-09 18:32 — paused at the owner's request

`touch runs/PAUSE`; the running chunk finished at step 1316/1500 of F1 (flow, 64->128),
saved `runs/F_flow_128/train_state.pt`, and the driver exited with no python processes left.
F2 (regression, 64->128) has not started. Nothing else was running.

To resume:
    cd /Users/zhangxiaowen/AntigravityProjects/progressive-sr
    rm runs/PAUSE
    PYTHONUNBUFFERED=1 nohup runs/selfsim128_train.sh >> runs/selfsim128_train.log 2>&1 &
F1 needs ~24 min more, then F2 takes ~3.4 h, so ~3.8 h to finish the pair.

Loss so far, 64->128 flow: 6.42e-1 (step 1) -> 3.29e-1 (234) -> 2.07e-1 (1198) -> 1.94e-1
(1275). About twice the 32->64 flow's trajectory at the same step, consistent with the harder
baseline there (octave r = 0.394 against 0.555, rms 0.816 against 0.577 h_f). No evaluation
numbers yet: they are only written when a run completes.

## 2026-09-09 19:55 — F1 (flow, 64->128) and the matched-sigma self-similarity test

### F1: flow + physical coupling, cube, self-run N-body 64->128 at z=0

`runs/F_flow_128`, 1500 steps, batch 1, `--octave-sampler full`, multistream 0.370.

| line | r | P/P | r^2 | coarse eps | rms | multi | single |
|------|---|-----|-----|------------|-----|-------|--------|
| baseline | 0.3940 | 1.2923 | 0.1552 | 1.249e-01 | 0.816 | 0.896 | 0.764 |
| emulator, 8 Heun | 0.5234 | 1.0719 | 0.2739 | 9.088e-02 | 0.683 | 0.766 | 0.629 |
| emulator, 1 Euler | 0.5829 | 0.6105 | 0.3398 | 8.131e-02 | 0.591 | 0.660 | 0.546 |
| generative | 0.1496 | 1.1203 | 0.0224 | 1.084e-01 | 0.831 | 0.905 | 0.784 |
| sample A vs B | 0.1475 | 0.9989 | - | 4.220e-02 | 0.766 | - | - |

The network gains LESS one level down: baseline->emulator is r 0.555 -> 0.722 (+0.167) and rms
-29% at 32->64, against r 0.394 -> 0.523 (+0.129) and rms -16% here. The accuracy/power split
inside the flow is sharper: 1-step r 0.583 / P/P 0.611 against 8-step 0.523 / 1.072, where at
32->64 it was 0.682 vs 0.993. Generative P/P = 1.120 with the corrected sampler.

**CAVEAT, and it limits the above: neither run is converged at 1500 steps.** Over the last
third the loss still falls by 10.4% (32->64) and 12.8% (64->128), so "the network gains less at
the finer level" may be partly an undertraining artefact, and the finer level is the one with
further to go. Do not quote the gain comparison without this. F2 (regression) is running.

### Matched-sigma self-similarity test (Sec. 6), the first on N-body data

sigma_lin measured directly from the ICs rather than assumed from n_eff: the ratio between the
two coarse cell sizes is 1.3741 (not sqrt(2) = 1.414), so the 64->128 pair matches 32->64 at
z=0 when D(z)/D(0) = 0.7278, i.e. **a = 0.60964, z = 0.640**, growth D(a)/D(z=99) = 55.8542.
One extra 64/128 pair was run to that output (`data/selfsim/ss`, 26 min).

| quantity | 32->64 @ z=0 | 64->128 @ z=0.640 | 64->128 @ z=0 | matched agreement |
|----------|--------------|-------------------|---------------|-------------------|
| rms(eps)/h_c cube | 0.1898 | **0.1887** | 0.2654 | **-0.6%** |
| ... sphere | 0.1947 | 0.1960 | 0.2757 | +0.7% |
| ... haar | 0.2893 | 0.2945 | 0.3936 | +1.8% |
| multistream frac | 0.2929 | **0.2924** | 0.3693 | **-0.2%** |
| k(T=0.5)/k_Ny,c | 1.0645 | **1.0630** | 1.0944 | **-0.1%** |
| var(d/h_f) | 0.1411 | 0.1330 | 0.2420 | -5.8% |
| k(T=0.8)/k_Ny,c | 0.8784 | 0.9383 | 0.9383 | +6.8% |
| excess kurtosis | 1.4216 | **2.1267** | 2.7975 | **+49.6%** |

**Self-similarity holds at the level of second moments and fails at the fourth.** Matching
sigma collapses drifts of +40%, +27% and +72% to well under 1% for the correction amplitude,
the multi-stream fraction and the Wiener half-power scale, and to 6% for the detail variance —
across a factor 2 in resolution and a factor 1.37 in sigma. But the excess kurtosis only comes
down from +97% to +50%: the *shape* of the detail distribution keeps a residual level
dependence that sigma does not capture.

Note this is not explained by the multi-stream fraction, which matches to 0.2% — so notes v3's
suggestion that s_l needs "sigma and the local multi-stream indicator" would not fix it either.
A plausible reading (hypothesis, not tested): kurtosis is dominated by the very smallest
resolved scales, which sit at the grid scale, a different physical scale at each level, so the
tails carry a level dependence that no single dimensionless amplitude can absorb.

Operationally, for weight sharing: conditioning on sigma(h_l, z) looks sufficient for everything
that sets the amplitude of the correction and the detail, which is what the loss is dominated
by; a model that also has to match the tails will need something more.

## 2026-09-09 20:10 — queued runs/extend3k.sh: 1500 -> 3000 steps on all four runs

Reason: at 1500 steps none of the four is converged (loss still falling 10.4% / 12.8% over the
last third), and the finer level -- the one that looked worse -- has further to go, so
"the network gains less one level down" is not quotable yet.

Uses `--resume`, so only the extra 1500 steps are paid for. Each run directory is copied to
`<name>_3k` first with `results.json` removed, so the 1500-step numbers stay on disk alongside.
Order: 32->64 first (1.5 h for the pair, and needed for the cross-level comparison to be fair),
then 64->128 (6.4 h). All four evaluate with `--octave-sampler full` so the generative lines are
comparable; training is physical-coupling, so the sampler never enters it. Same PAUSE protocol.

Waits for the GPU, so it starts when F2 finishes (~23:20) and should end ~07:15.

## 2026-09-09 23:02 — F2 done: the full flow-vs-regression comparison at 64->128

| | r | P/P | r^2 | coarse eps | rms | multi | single | gen P/P |
|---|---|-----|-----|------------|-----|-------|--------|---------|
| **32->64** baseline | 0.5551 | 1.4860 | 0.3081 | 1.08e-01 | 0.577 | 0.658 | 0.540 | - |
| flow | 0.7220 | 0.9928 | 0.5213 | 5.64e-02 | 0.411 | 0.490 | 0.373 | **0.995** |
| regression | 0.8192 | 0.6902 | 0.6711 | 3.13e-02 | 0.314 | 0.380 | 0.282 | **0.650** |
| **64->128** baseline | 0.3940 | 1.2923 | 0.1552 | 1.25e-01 | 0.816 | 0.896 | 0.764 | - |
| flow | 0.5234 | 1.0719 | 0.2739 | 9.09e-02 | 0.683 | 0.766 | 0.629 | **1.120** |
| regression | 0.7160 | 0.5162 | 0.5127 | 4.52e-02 | 0.490 | 0.555 | 0.447 | **0.485** |

(1500 steps; 32->64 at batch 2, 64->128 at batch 1. The generative column is with
`--octave-sampler full` at both levels: the 32->64 models were re-evaluated on the CPU for
this, since they had originally been scored with the longitudinal-only sampler.)

**The separation intensifies one level down; it is not an artefact of the coarse 32->64 step.**
The regression's r advantage grows from 0.097 to 0.193, and its power deficit deepens from
0.690 to 0.516 while the flow stays on P/P = 1 at both levels (0.993, 1.072). At 64->128 the
regression again lands exactly on the r^2 line (P/P 0.5162 against r^2 0.5127). For the
progressive chain this points the wrong way for the regression: at 256^3 and 512^3 the split
should be sharper still.

**Generative mode is where it becomes decisive.** With the corrected sampler, the flow gives
0.995 and 1.120 of the true octave power at the two levels while the regression gives 0.650 and
**0.485** — at 64->128 it produces less than half the power, and it degrades with resolution.
So the two statements that now have evidence at two independent levels are: *in emulator mode
the regression is more accurate everywhere*, and *in generative mode the regression is not
usable*, with both gaps widening as the transition gets finer.

Still to check tonight: all four runs are being extended to 3000 steps
(`runs/extend3k.sh`), because at 1500 the loss is still falling 10-13% over the last third and
the finer level has further to go. Until that finishes, the level-to-level comparison of the
*size of the network's gain* stays provisional; the r-vs-P/P split above is a much more robust
feature and is unlikely to move.

## 2026-09-10 06:46 — the 3000-step extensions finished; see runs/REPORT_6.md

E1 32->64 flow 23:20-00:00, E2 32->64 reg 00:00-00:38, E3 64->128 flow 00:38-03:43,
E4 64->128 reg 03:43-06:46. All resumed from the 1500-step train_state.pt.

| | r | P/P | r^2 | eps | rms | multi | single | gen P/P | final loss |
|---|---|-----|-----|-----|-----|-------|--------|---------|------------|
| 32->64 baseline | 0.5551 | 1.4860 | 0.3081 | 1.08e-01 | 0.577 | 0.658 | 0.540 | - | - |
| 32->64 flow 3000 | 0.7327 | 0.9759 | 0.5368 | 5.87e-02 | 0.409 | 0.491 | 0.370 | 0.9808 | 8.51e-02 |
| 32->64 reg 3000 | 0.8289 | 0.7214 | 0.6871 | 3.00e-02 | 0.307 | 0.373 | 0.275 | 0.6797 | 7.91e-02 |
| 64->128 baseline | 0.3940 | 1.2923 | 0.1552 | 1.25e-01 | 0.816 | 0.896 | 0.764 | - | - |
| 64->128 flow 3000 | 0.5478 | 1.0703 | 0.3001 | 9.50e-02 | 0.679 | 0.764 | 0.624 | 1.1055 | 1.83e-01 |
| 64->128 reg 3000 | 0.7293 | 0.5391 | 0.5319 | 4.31e-02 | 0.480 | 0.545 | 0.438 | 0.5096 | 2.15e-01 |

Convergence settled: 1500 -> 3000 moves r by +0.010 to +0.024 and changes no ordering.

### RETRACTED: "the network gains less one level down"

Flagged last night as possibly an undertraining artefact. At 3000 steps the gain in r over
baseline is flow +0.178 (32->64) vs +0.154 (64->128) — a 13% shrink — but regression
+0.274 vs **+0.335**, i.e. it GROWS one level down. The blanket claim was wrong. What is true:
the relative rms improvement falls for both (flow -29% -> -17%, regression -47% -> -41%).

### The Eulerian contradiction of last night is explained and tested

Density statistics (CIC, mean 1) on the test box:

| field | max | 99.9 pct | mass frac d>10 | d>100 |
|-------|-----|----------|----------------|-------|
| coarse | 2297 | 63.1 | 0.374 | 0.104 |
| baseline | 1526 | 53.1 | 0.315 | 0.063 |
| regression | **5619** | 71.4 | 0.428 | **0.178** |
| flow | 2360 | 63.5 | 0.381 | **0.107** |
| truth | 3480 | 77.5 | 0.433 | **0.152** |

The regression OVER-concentrates the densest structures (peak 61% above truth, 17% too much
mass above delta=100) and the flow UNDER-concentrates (30% too little). Mechanism: a
conditional-mean displacement is too smooth in LAGRANGIAN space, so particles that should have
dispersed inside a collapsed region stay together and the caustic is thinner and denser than it
should be. Low Lagrangian octave power and high Eulerian small-scale power are the same fact.
=> **the Lagrangian octave-band P/P is not sufficient to judge a model**; and at moderate
overdensity (d>10) the regression is actually closer to truth than the flow (0.428 vs 0.381
against 0.433). One test box, no window deconvolution or shot-noise subtraction: model-to-model
ratios are meaningful, absolute numbers are not.

Also added: `runs/make_figures.py` now takes `--dis/--ic/--growth/--regression` for real data
(it was synthetic-only); `runs/fig_128_density.png` (density slices, full slab and zoom) and
`runs/fig_128_density_spectra.png` (Eulerian density spectrum, ratio, r_delta) written.

## 2026-09-10 09:40 — the two power orderings are inverted (owner's observation, quantified)

Eulerian density P_delta/P_true at k_Ny,c = 2.01 h/Mpc: coarse 0.677, baseline 0.323,
regression 1.919, flow 0.714, flow-generative 1.109. At 4 h/Mpc: 0.255 / 0.021 / 2.571 /
0.223 / 0.681.

Against the Lagrangian octave band the ordering INVERTS: baseline 1.292 -> 0.323,
flow 1.070 -> 0.714, regression 0.539 -> 1.919.

The flow's Eulerian density power tracks the COARSE FIELD (0.714 vs 0.677 at k_Ny,c, and below
it at 3 h/Mpc: 0.411 vs 0.437), so getting the Lagrangian octave power right added essentially
no Eulerian small-scale power. Eulerian high-k density needs the phases — caustics at the right
Lagrangian positions — and at r = 0.55 half of what the flow adds is uncorrelated. The baseline
is the extreme case: octave power 29% too HIGH, Eulerian density power 0.32, i.e. a random
linear octave smears the structure the coarse run already had.

Consequences recorded in REPORT_6 section 4b:
- "the flow has the right power" must always be qualified as "in the Lagrangian octave band";
  in the Eulerian density no model here is near 1;
- the GENERATIVE flow has the best Eulerian power of the three (1.109 at k_Ny,c) because its
  octave is full-amplitude, even though its r vs truth is ~0;
- an Eulerian metric in the training loop moves from "refinement" to "clearest next step".

## 2026-09-10 10:00 — the Eulerian failure is general; added an optional CIC loss term

### The 32->64 pair shows the same inversion, and it worsens with resolution

Eulerian density P_delta/P_true, 32->64, 3000-step models (k_Ny,c = 1.01 h/Mpc):

| k/k_Ny,c | 0.75 | 1.00 | 1.5 | 2.0 |
|----------|------|------|-----|-----|
| coarse | 0.852 | 0.747 | 0.511 | 0.343 |
| baseline | 0.638 | 0.452 | 0.170 | 0.061 |
| regression | 1.253 | **1.435** | 1.866 | 2.170 |
| flow | 0.997 | **0.963** | 0.777 | 0.551 |

Density tail: regression max 2000 vs truth 1234 (+62%), mass above delta=100 0.0616 vs 0.0524
(+18%); flow max 1108 (-10%), 0.0401 (-23%). Mass above delta=10: regression 0.2968 against
truth 0.2972, flow 0.2748.

So the pattern is not specific to 64->128, and the flow's Eulerian deficit **worsens with
resolution**: at k_Ny,c it is 0.963 (32->64) and 0.714 (64->128), tracking r falling 0.73 -> 0.55.

### `--cic-weight`: an Eulerian term in the loss (owner approved; halo mass function explicitly not)

Added `cic_density()` (differentiable CIC deposit of q + Psi, mean 1) and `cic_loss()` (MSE on
log(1+delta)); the training loop adds `cic_weight * cic_loss(x1_pred, x1)` with
`x1_pred = x_t + (1-t) v`, which is the velocity's own prediction of the endpoint and works for
the regression parameterisation too. Default 0, so nothing changes unless asked.

Verified rather than assumed:
- torch CIC vs the numpy one in make_figures.py: max |diff| 4.5e-06, mass exactly N^3, mean 1;
- gradients finite and non-zero; identical inputs give exactly 0;
- the default path is bit-identical (16->32 20-step smoke test still gives loss 3.0888e-02 and
  baseline r=0.915 / P/P=0.835);
- runs on MPS (scatter_add is supported): 2.11 s/step against 1.83 without, i.e. +15%.

lambda calibrated against the two terms at 32->64 convergence (flow term ~8.5e-2, CIC term at
the baseline ~8.3e-2): lambda = 1.0 makes them comparable, 0.3 makes the CIC term ~30% of the
flow term. `runs/cic_train.sh` runs both from scratch, 3000 steps, ~1.8 h each.
The log-compression in cic_loss is a choice, not a derivation, and is worth revisiting: it
weights underdense cells more, while the failure being targeted is in the peaks.

## 2026-09-10 11:06 — --jac-weight: the purely Lagrangian competitor to the CIC loss (owner approved)

Owner's question: can the Eulerian information be supplied without ever using a density?
Answer: in single-stream regions, yes, exactly — J(q) = det(I + dPsi/dq) is 1/rho_Eul at the
particle's position, and it is a local, fully differentiable functional of Psi alone. The
failure modes seen today are visible in J directly: the regression's over-concentration is
J -> 0 too fast, the flow's under-concentration is J not small enough. In multi-stream regions
(29-37% here) J constrains each stream separately instead of their sum — arguably a stronger
constraint than the CIC density, and the honest caveat is that it is not identical information.

Implementation (`jac_loss`): spectral derivatives via sc.gradients (consistent with every other
derivative in the code), hand-written 3x3 determinant (differentiable everywhere, no linalg
backend needed on MPS), MSE on **asinh(J)** — linear around J = 0, which is exactly the
caustics the Lagrangian MSE cannot see; log-like in the void tail; smooth through the
multi-stream sign change, where log|J| would diverge at every caustic. The compression is a
knob, not a derivation (same status as log1p in cic_loss).

Verified: manual det vs np.linalg.det max|diff| 5.7e-06; J<0 fraction of the test-box truth =
0.370, reproducing phase0's fine-level multi-stream fraction; identical inputs -> exactly 0;
gradients finite (rms 2.9e-05); default path bit-identical (16->32 smoke test 3.0888e-02);
MPS fine at 2.00 s/step (+9%, cheaper than CIC's +15%).

lambda calibration: jac term at the baseline = 2.041 against the flow term's ~0.085 at
convergence, so **0.04** makes them comparable and **0.012** makes the jac term ~30% — the
same logic as the CIC pair's 1.0 / 0.3. (A smoke test at jac-weight 1.0 shows the total loss
rising — the term would dominate by 25x; calibration matters.)

Queue now: C1 cic 1.0 (resumes from step 1274) -> C2 cic 0.3 -> C3 jac 0.04 -> C4 jac 0.012,
3000 steps each, all `--octave-sampler full`, ~6 h total. The comparison at the end is
four-way: no extra term / CIC x2 / jac x2, all against the same baseline and truth, on
Lagrangian AND Eulerian metrics.

## 2026-09-10 11:20 — CIC runs skipped at the owner's direction; straight to the Jacobian term

Owner's reasoning, recorded: (1) CIC is resolution-limited — the deposit kernel smooths at the
grid scale, so structure inside a cell is invisible to the loss no matter the weight; (2) their
own earlier tests of a CIC-type loss (the Paper-IV lag2eul line) found it improves the power
spectrum but not the small scales. The Jacobian term has no deposit kernel — J(q) is pointwise
in q at the field's native resolution — so it does not share that ceiling, and it is the one
worth spending GPU on.

C1 (cic 1.0) abandoned at step ~1650 and deleted; C2 never started. `--cic-weight` remains in
the script as an option. Queue is now C3 (jac 0.04) -> C4 (jac 0.012), 3000 steps each,
~3.5 h total.

## 2026-09-10 12:30 — Stage 7 Part 1 (CPU): Eulerian evaluation + constructive tests; see runs/REPORT_7a.md

- `python eulerian_metric.py`: mass 0, pullback 0, first-order 9.6e-3, kernel 0.245/0.038/0.009,
  Jacobi 1.1e-4. scatter_add and linalg.det fine on torch 2.14 CPU. Nothing patched.
- New `runs/eval_eulerian.py` (reads a run's stored args, rebuilds the test box, writes
  <run>/eulerian.json+png). Validated: the 1500-step checkpoints reproduce REPORT_6 §4b
  exactly (0.7136 / 1.9193 / 0.323 / 0.677); Gaussian kernel-fraction control gives 1.00.
- Ran on R_flow_phys_3k, R_reg_phys_3k, F_flow_128_3k, F_reg_128_3k; tables in REPORT_7a.

Headline outcomes of the four predictions:
- T1 smearing: baseline CONFIRMED (constructed 0.341-0.343 vs measured 0.323 at 64->128);
  flow REFUTED — measured 0.680 far above its independent-noise construction 0.407. The flow's
  residual is not independent noise.
- T2 shrinkage: REFUTED — every band-limited shrink gives a DEFICIT (0.54-0.66 at k_Ny,c),
  regression measures 1.898; and the family is non-monotonic the wrong way (a 0.5->0.9 gives
  0.660->0.542) because the constructions sit on the uncorrected coarse band: detail phased
  with the TRUE coarse band decoheres against the coarse run's misplaced structures. The
  coarse-band correction is a first-order actor in the Eulerian budget, absent from the note's
  Sec. 3 model. "Coherent-only ~1 at a=0.57" also refuted (0.645).
- T3 kernel fraction: REFUTED in an informative direction — E fractions 14-30, not ~1.
  Mechanism isolated: with uniform-density positions the flow's 17.8 collapses to 1.14, so the
  excess is the density weighting correlating with a residual that is coherent exactly where
  particles pile up (coherent halo mislocation). Good news for Q_E: a large visible target.
  Q_J fractions 0.24-0.58 (<1): the residual is smoother in gradients than Gaussian.
- T4 single-stream deposit: REFUTED — the regression's excess barely moves (1.887 vs 1.898 at
  64->128; 1.415 vs 1.435 at 32->64). Not localised in the coarse-mask multi-stream patches,
  consistent with the coarse mask missing 56-62% of fine-level multi-stream cells.
- J-quantile directions CONFIRMED (noise broadens -14.1/21.4, shrink narrows -5.4/12.8,
  truth -6.8/15.2).

Two incidental findings:
- More Lagrangian training worsened the flow's Eulerian power (0.714 -> 0.680 at k_Ny,c from
  1500 to 3000 steps) while octave r rose. The two objectives now measurably trade.
- The corrected (full) octave sampler LOWERED generative Eulerian power: longitudinal sampler
  1.109 at k_Ny,c (REPORT_6) vs full sampler 0.56-0.59. The sampled transverse component is
  uncorrelated displacement noise -> pure Eulerian smearing, the same mechanism as T1. The
  Lagrangian and Eulerian verdicts on the sampler fix point in opposite directions.

C3 (jac 0.04) still training on the GPU throughout; nothing here touched the training script.

## 2026-09-10 18:40 — Stage 7 Part 2: the biased jac runs (C3/C4), evaluated

C3/C4 finished at 14:42 (3000 steps each, lambda = 0.04 / 0.012, --octave-sampler full).
Final printed loss (total, flow-MSE + lambda*jac, not separable post hoc from the log — a gap
the Part 3 code fixes by printing both terms): R 8.51e-2, C3 9.09e-2, C4 8.64e-2.

| | oct r | oct P/P | rms (m/s) | gen P/P | svs P/P | P_d/P @kNyc | @1.5 | mass>100 | kernE | gen P_d @kNyc |
|---|---|---|---|---|---|---|---|---|---|---|
| R  flow (no term) | 0.7327 | 0.9759 | 0.409 (.491/.370) | 0.9808 | 1.0003 | 0.963 | 0.777 | 0.0401 | 13.9 | 0.860 |
| C3 jac 0.04 | 0.7376 | 0.9692 | 0.400 (.480/.362) | 0.9680 | 0.9973 | **1.002** | **0.845** | 0.0407 | 14.3 | **0.912** |
| C4 jac 0.012 | 0.7350 | 0.9760 | 0.407 (.487/.368) | 0.9793 | 0.9973 | 0.979 | 0.806 | 0.0400 | 14.3 | 0.880 |

(truth mass>100 = 0.0524)

Read with §5.5/§5.7d in mind, as instructed:
- The biased term WORKS on this box and it is monotonic in lambda: Eulerian power at k_Ny,c
  0.963 -> 0.979 -> 1.002, at 1.5 k_Ny,c 0.777 -> 0.806 -> 0.845. And with NO Lagrangian cost —
  r actually rises (0.7327 -> 0.7376) and rms falls (0.409 -> 0.400).
- The bias signature the note predicts is present but small: generative-mode octave P/P drifts
  0.9808 -> 0.9680 and sample-vs-sample 1.0003 -> 0.9973 (about 1%). At this lambda and level
  the drift is far below the Eulerian gain.
- NOT §5.7e's failure mode: P_delta did move, so the h_f-grid J is not merely matching
  roughness here.
- The kernel fraction E did NOT drop (13.9 -> 14.3): the jac term closed the k-integrated
  power deficit without reducing the density-weighted coherent residual; r_delta unchanged
  (0.975). So the term fixes the SPECTRUM, not the halo-mislocation error — consistent with
  Q_J having no relabelling kernel and the residual living exactly there.
- J quantiles moved toward truth but remain narrower (printed above); mass>100 nearly
  unchanged (0.0407 vs truth 0.0524) — the 1-halo compactness deficit is not fixed by this term.

## 2026-09-10 19:20 — Stage 7 Part 3: the admissible metrics are in the training script; queue launched

`--loss-metric {lag,qe,qj}` (Q_E on the flow-matching error at coarse-field Eulerian positions;
Q_J with the coarse field as state, p=2, eps=0.1), `--lambda-e` (auto-equalised at step 1 —
measured on the real 32->64 box: L0_qe = 0.039, L0_qj = 0.0100, with baseline terms
L_mse = 3.29e-1, L_qe = 8.37, L_qj = 33.0), `--euler-positions {coarse,interp}`,
`--eulerian-inputs` (log(1+delta_c) pulled back to q; cin=13; flag rides in the checkpoint).
The forms never see x1. Default path bit-identical (3.0888e-02); qe 1.94 s/step, qj 2.22 on MPS.

Two bugs caught by the smoke tests, both mine:
1. The eulerian-inputs channel patch targeted pre-wiener text and silently replaced nothing
   (str.replace matches zero occurrences without error). Caught because the cin=13 smoke test
   crashed; fixed with a direct edit. Lesson recorded: patch-by-string on a file someone else
   updates must verify the count.
2. --eval-only restored the checkpoint's eulerian_inputs flag AFTER the batcher and test box
   were built, feeding a cin=13 model 12-channel inputs. Caught by running the eval-only smoke
   to completion (the 'loaded weights' line alone had looked fine); flag now restored right
   after argument parsing.

Queue `runs/stage7_train.sh` (E_flow_qe, J_flow_qj, E_flow_qe_in, then E_reg_qe, J_reg_qj;
3000 steps each, ~10 h) launched at commit HEAD; results.json records the hash.

## 2026-09-10 23:20 — the generative Eulerian gap IS the sampler; GenIC-oracle test

Owner's directive: r < 1 is acceptable, but the generative (random-octave) P ratio must be 1,
as the GAN series demonstrated. Overnight diagnostic on where the generative Eulerian deficit
comes from.

Mechanism hunts first, both REFUTED and recorded:
- "transverse = super-Nyquist aliasing": GenIC only samples modes within its own grid, and a
  per-mode ik delta/k^2 field is longitudinal at every grid k including the corners.
- "transverse = a discrete gradient kernel, deterministic per mode": fitted the vector kernel
  K_i(k) = Psi_hat_i/delta_hat on one GenIC seed and applied it to a second. Coarse band
  transfers (residual 1.7e-2) but the octave does NOT (residual 8.0): the octave displacement
  is not a per-mode function of the recorded ICDensity (which is likely a CIC-deposited
  diagnostic, not the sampling field). PrePosition is the undisplaced lattice, not Zel-only —
  a first test built on that assumption was discarded.

So the empirical route: use MP-GenIC itself as the sampling oracle (random seed -> GenIC,
0.2 s at 64^3 -> octave band -> growth). By construction this draws from the exact joint prior.
Ladder on the SAME checkpoint (R_flow_phys_3k, 8 Heun), all CPU:

| octave sampler | out Lag P/P | Eul P_d/P @kNyc | @1.5 | @2 |
|---|---|---|---|---|
| (D) longitudinal-only | 0.738 | 1.152 | 1.168 | 1.071 |
| (C) independent transverse (current --octave-sampler full) | 0.996 | 0.860 | 0.617 | 0.387 |
| (G) GenIC oracle, seed 7001 | 0.936 | 0.957 | 0.756 | 0.543 |
| (G) GenIC oracle, seed 7002 | 0.937 | 0.962 | 0.761 | 0.547 |
| (T) true octave (= emulator reference) | 0.954 | 0.963 | 0.777 | 0.551 |

**With the true joint prior, generative mode equals emulator mode to 0.005 at every probed
scale** (two independent oracle seeds agree). The entire generative-vs-emulator Eulerian gap
was the sampler's wrong JOINT statistics: the independently-sampled transverse component is
uncorrelated displacement noise (pure smearing), while the correct prior's "transverse" part
is phase-locked and builds structure. Note the inversion AGAIN: (C) has the best Lagrangian
marginal (0.996) and the worst Eulerian tail; (D) is Lagrangian-deficient but Eulerian
OVERSHOOTS (1.15) — under-powered octave = less smearing = over-compaction.

Consequences:
1. The road to gen P_delta = 1 splits cleanly: (i) sample with the IC generator's own joint
   statistics (oracle now; implementable as the generator's kernel later — the per-mode fit
   says it is NOT a simple kernel, so this needs the generator's actual sampling recipe);
   (ii) close the remaining model residual 0.96 -> 1, which is emulator-mode work — exactly
   what the Q_E runs now training target.
2. --octave-sampler full (independent transverse) is the WORST of the three for Eulerian
   statistics despite the best Lagrangian marginal; it should not be the generative default on
   real data until replaced by a joint-statistics sampler.

## 2026-09-10 23:40 — BUG in the first qe/qj runs: lambda_e drifted across resume chunks (fixed, reruns queued)

`--lambda-e` is derived at step 1 (v = 0, so the two terms are the baseline terms) and was
NOT persisted in train_state.pt: every 900-s chunk re-derived it from its own first batch with
a partially trained model. E_flow_qe's lambda drifted 0.0442 -> 0.297 (6.7x) over the run;
J_flow_qj's 0.0100 -> 0.0067. The affected runs are kept as `runs/E_flow_qe_drift`,
`runs/J_flow_qj_drift`, `runs/E_flow_qe_in_drift` (the last killed mid-run) — they are
"growing-lambda variants", not the designed experiment.

For the record, the drift runs' Lagrangian numbers (3000 steps):
- E_flow_qe_drift (lambda -> 0.297): r=0.6738 (down from 0.7327), P/P=1.113, rms=0.465,
  gen P/P=1.146 — late-training over-weighted qe visibly degrades the Lagrangian side.
- J_flow_qj_drift (lambda -> 0.0067): r=0.7464 (UP from 0.7327, the best flow r so far),
  rms=0.392 (best), P/P=0.956, gen P/P=0.948 — promising even in drifted form.

Fix: lambda_e now saved in train_state.pt and restored on --resume (derived once, at the true
step 1). Verified with a two-chunk CPU run: the second chunk prints "resumed" and no new
lambda line. Clean queue relaunched from scratch for all five runs.

## 2026-09-11 07:31 — Stage 7 Part 3 complete; see runs/REPORT_7.md

Clean queue (pinned lambda) finished: E_flow_qe 23:22-01:00, J_flow_qj -02:38, E_flow_qe_in
-04:16, E_reg_qe -05:51, J_reg_qj -07:31. eval_eulerian extended to read the eulerian_inputs
flag from the checkpoint (it had cin=12 hardwired; the watcher's eval of E_flow_qe_in crashed
on the shape mismatch and took the two regression evals down with it — rerun separately).

Headline (full table and verdicts in REPORT_7):
- **flow + Q_J (lambda 0.0100) improves BOTH sides at once**: Lagrangian r 0.7327 -> 0.7460
  and rms 0.409 -> 0.393 (best flow values so far), Eulerian P_delta/P 0.963 -> 1.042 at
  k_Ny,c, 0.777 -> 0.918 at 1.5, 0.551 -> 0.701 at 2. Cost: octave marginal P/P 0.976 -> 0.949.
- **flow + Q_E hurts the Eulerian tail** (0.551 -> 0.418 at 2 k_Ny,c) and barely moves its own
  kernel fraction (13.9 -> 12.3): expectation (a) refuted for qe, achieved by qj; (b) inverted
  (qj wins everywhere, including single-stream T4 1.049 vs 0.913); the note's 5.7c kernel
  ranking does not describe what limits these networks.
- (c) confirmed: the regression's excess survives qe/qj (qe trims the tail 2.17 -> 1.78).
- (d) refuted: --eulerian-inputs changes nothing measurable at this size.
- **qj + GenIC-oracle sampler: generative P_delta/P = 1.036 / 0.887 / 0.683 at (1/1.5/2)
  k_Ny,c, coinciding with emulator mode (1.042 / 0.918 / 0.701)**. Combined with last night's
  oracle result, the owner's requirement — sampled P ratio = 1 with r < 1 — is met at k_Ny,c
  at this level; the remaining tail deficit is shared with emulator mode (model residual).
  Figure: runs/fig8_qj_closure.png.

## 2026-09-11 20:50 — Stage 8: multi-transition weight sharing (owner approved) + the 128 qj reference

Owner: "可以两个一起训练，32-64和64-128" — the CLAUDE.md ask-first item "multi-transition
(style scalar) training" is hereby approved and started.

`runs/multi_train.py` (new file; imports the frozen training script's classes): one
UNet3D(cin=12) trained on BOTH transitions with alternating steps (32->64 at batch 2,
64->128 at batch 1), style scalar s_l = ln(sigma_c) measured from each level's training ICs
(0.6362 / 0.9392 — physical, extrapolates to future levels, unlike a level tag), qj loss with
per-level lambda equalised at each level's own first step and pinned (0.0099 / 0.0076),
--octave-sampler full, chunked/resumable (verified: resume does not re-derive lambdas).
Per-level evaluation at the end -> results_32to64.json / results_64to128.json with Lagrangian
and Eulerian numbers.

Queue `runs/stage8_train.sh`: M_qj_multi (3000 steps/level, ~9 h) then J_flow_128
(single-level qj at 64->128, 3000 steps, ~7 h — both the harder-level test of Stage 7's
Eulerian gain and the apples-to-apples reference for the multi model's 128 side).
The key comparisons when done:
  multi@32->64  vs J_flow_qj      (does sharing cost the small level anything?)
  multi@64->128 vs J_flow_128     (does sharing cost the big level anything?)
  J_flow_128    vs F_flow_128_3k  (does qj's Eulerian gain survive one level up?)

## 2026-09-11 22:00 — owner's review: the aliasing verdict is OVERTURNED; corrections and follow-ups

**CORRECTION of RUNLOG 2026-09-11 23:20 (the mechanism hunt).** The owner ran the controlled
test I did not: fixing generator/seed/box/cosmology and varying only the generation mesh,
16^3 particles get ~0 octave transverse power when Nmesh = Ngrid and 14.74% when
Nmesh = 2 Ngrid. Verified in source: `genic/params.c:199` sets `Nmesh = 2*Ngrid` when the
parameter is 0, and all our runs used the default (the logs read "Nmesh 0 # Default").
So the transverse octave component IS aliasing — of the 2x-finer GENERATION mesh onto the
particle lattice. My "refuted: super-Nyquist aliasing" entry tested the wrong hypothesis
(super-Nyquist of the particle grid, generation assumed on-grid), and my per-mode kernel test
failed because ICDensity is not the generation-mesh field. Both husks recorded there stand
corrected by this entry. `zeldovich.c` additionally uses a finite-difference `diff_kernel`,
but the owner's Nmesh=Ngrid -> ~0 result shows the mesh doubling is the dominant mechanism.

**Consequences adopted:**
- The refined-grid sampler (generate on 2N, subsample at the lattice) is back on the table as
  the principled standalone prior; the GenIC oracle remains the reference.
- The coarse level's own IC (Ngrid=Nc, Nmesh=2Nc) contains folded contributions from
  [k_Ny,c, 2 k_Ny,c] — which IS the octave band. The coarse state may "know" part of the
  octave at z_init already; the conditional prior p(eta | coarse) is then not the
  unconditional Gaussian. To be measured (fixed white noise across levels).
- P_n = A(1-r^2) P_true is the corrected residual-variance formula; it is what my T1 actually
  used (it reproduces the note's Sec. 3.1 table), so no T1 number changes — but the
  implication is now stated correctly: A = r^2 does NOT imply n = 0 or uniform halo shrinkage
  by r; the regression's over-concentration still needs an independent structural diagnostic.
- Q_J on a uniform background penalises the divergence only (tr[adj(I) de] = div e); purely
  transverse errors sit in its null space. "Q_J = k^2 gradient penalty" claims are hereby
  narrowed; divergence-vs-full-gradient is an ablation to run.
- det(I+D) > 0 does not imply single-stream: sign(J) is a PARITY. Every "multistream fraction"
  in this log and the reports is the negative-parity fraction of J on the stated grid; the
  labels overstate what is measured. To be renamed at next code touch.
- Evaluation debts acknowledged from the review: test seeds beyond the development box
  (seed 8 was reused throughout; seed 9 is on disk and unused), deposit-grid convergence at
  2 k_Ny,c (that probe sits AT the deposit grid's Nyquist), unified checkpoint-config restore,
  and the missing chained 32->64->128 evaluation (the four-cell table:
  specialist/shared x true-coarse/chained).

Reading list logged for follow-up: Adaptive Flow Matching (ICML 2025; learned source),
Spatiotemporal Pyramid Flow Matching (CVPR 2026; stage-to-stage distribution handoff), WSGM
(cross-scale conditional normalisation), RFMSR 2026-07 / PixelIR 2026-08 (deterministic base +
stochastic residual), Cosmo3DFlow (invertible wavelet representation; inverse task, not a
baseline).

## 2026-09-11 23:30 — review follow-ups executed (CPU, alongside Stage 8 training)

### The sampler recipe and the cross-level prior (the review's top priority)

TEST 1 — "generate at 2N, subsample" recipe: octave-band r(ic_64, subsample(ic_128)) =
0.9622 / 0.9622 / 0.9623 over three seeds. The recipe class is right; the 0.04 shortfall and
the higher transverse fraction (0.207 vs 0.170) of the subsampled field are the EXTRA fold
layer (the 128 IC was itself generated at Nmesh=256). A faithful standalone sampler must
generate at exactly 2N. Until then the GenIC oracle stays the generative prior.

TEST 2 — the coarse IC already knows most of the folded octave: correlating the coarse IC's
content beyond the shared band (rms 8.5 kpc/h, comparable to the octave's 9.7) with the
alias-fold of the fine octave gives r = 0.8493 / 0.8499 / 0.8487 over three seeds.
**At z_init the coarse state already carries r^2 ~ 72% of the folded octave.** "The octave is
the only new randomness" is violated at the IC level by the generator's Nmesh = 2 Ngrid; the
conditional prior p(eta | coarse) is not the unconditional Gaussian. This is also a likely
part of the earlier cross-level IC non-nestedness (r = 0.76 at the coarse Nyquist).

### Independent test box (seed 9, never used in development) — the headline ordering survives

| 32->64, emulator P_delta/P at (1/1.5/2) k_Ny,c | dev box (s8) | fresh box (s9) |
|---|---|---|
| flow + qj | 1.042 / 0.918 / 0.701 | 0.959 / 0.796 / 0.610 |
| flow lag-only | 0.963 / 0.777 / 0.551 | 0.921 / 0.730 / 0.539 |
| regression | 1.435 / 1.866 / 2.170 | 1.231 / 1.385 / 1.552 |

qj > lag at every k on both boxes; the regression's excess is on both. Absolute values carry
~0.05-0.1 box-to-box scatter (the truth's own delta>100 mass differs: 0.0524 vs 0.0452), and
the kernel fractions halve on the fresh box (13.8 -> 5.9) — dominated by a few dense objects.
Headline claims must quote both boxes from now on. (eval_eulerian gained --test-seed and
--deposit-factor; outputs suffixed eulerian_s<seed>_d<factor>.json.)

### Deposit-grid convergence — the 2 k_Ny,c probe is fine

Factor-2 deposit (128^3 grid for the 64^3 field) moves the probes by <= 2%:
qj 1.049/0.935/0.707 vs 1.042/0.918/0.701; lag 0.969/0.792/0.558 vs 0.963/0.777/0.551.
The ratio cancels the CIC window as assumed; the review's concern is checked and closed.

Stage 8 (M_qj_multi) continues on the GPU throughout; J_flow_128 queued behind it.
Next session: the four-cell table (specialist/shared x true-coarse/chained) once both land.

## 2026-09-12 19:40 — Stage 8 results: the four-cell table (specialist/shared x direct/chained)

M_qj_multi finished 05:35 (6000 alternating steps, lambdas 0.0099/0.0076, s_l = 0.636/0.939);
J_flow_128 finished 12:10 (3000 steps, batch 1, lambda auto 0.0076). All emulator-mode
numbers below use true octaves at both levels, test seed 8; "chained" replaces the 64->128
step's coarse input with the 32->64 model's OUTPUT (runs/eval_chain.py, new).

### Single-level, per level (with the lag-only references)

| 32->64 | r | P/P | rms | gen P/P | Eul emu @(1,1.5,2)kNyc |
|---|---|---|---|---|---|
| specialist J_flow_qj | 0.7460 | 0.9488 | 0.393 | 0.9404 | 1.042 / 0.918 / 0.701 |
| SHARED | 0.7416 | 0.9332 | 0.395 | 0.9232 | 0.995 / 0.862 / 0.655 |

| 64->128 | r | P/P | rms | gen P/P | Eul emu @(1,1.5,2)kNyc |
|---|---|---|---|---|---|
| lag-only F_flow_128_3k | 0.5478 | 1.0703 | 0.679 | 1.1055 | 0.680 / 0.439 / 0.285 |
| specialist J_flow_128 | 0.5671 | 0.9345 | 0.644 | 0.9546 | 0.894 / 0.603 / 0.395 |
| SHARED | 0.6029 | 0.8356 | 0.602 | 0.8440 | 1.045 / 0.790 / 0.574 |

### The four-cell table at 128 (the review's priority deliverable)

| vs truth at 128 | true coarse (direct) | chained (generated coarse) |
|---|---|---|
| specialist | r=0.5671 rms=0.644 Eul 0.894/0.603/0.395 | r=0.5135 rms=0.872 Eul 0.780/0.409/0.205 |
| SHARED | r=0.6029 rms=0.602 Eul 1.045/0.790/0.574 | r=0.5460 rms=0.848 Eul 0.842/0.497/0.272 |

Findings:
1. **qj's gains survive the level**: specialist-128 beats lag-only on every metric
   (Eulerian at k_Ny,c 0.680 -> 0.894, mass>100 toward truth, r +0.02).
2. **Weight sharing is free at the easy level and WINS at the hard level**: at matched
   per-level steps the shared model beats the 128 specialist in BOTH columns (direct r +0.036,
   Eulerian at k_Ny,c 1.045 vs 0.894; chained r 0.546 vs 0.514, Eulerian 0.842 vs 0.780).
   Positive cross-level transfer, not just no-cost sharing. Cost: octave-band marginal P/P
   (0.836 vs 0.935) and generative P/P (0.844) at 128 — the power-undershoot side-effect of
   qj is larger for the shared model.
3. **Chaining costs r ~0.05-0.06 and roughly halves the Eulerian tail per level**
   (specialist 0.395 -> 0.205, shared 0.574 -> 0.272 at 2 k_Ny,c; rms +35-41%). Error
   accumulation through the coarse pathway is the next structural target — the note's
   rollout-fine-tuning item (CLAUDE.md ask-first) now has its motivating number.
4. The shared chained cell — ONE operator, applied twice, 32->64->128 — reaches r=0.546,
   Eul 0.842 at k_Ny,c against the direct specialist's 0.894: the progressive chain works,
   with quantified degradation. This is the first end-to-end progressive result of the project.

## 2026-09-12 20:30 — seed-9 four-cell table and the three-level figure

Both approved items delivered while the Q_J ablations train.

### The four-cell table on BOTH boxes (at 128, emulator octaves; s8 / s9)

| | direct r | chained r | direct Eul@1 | chained Eul@1 | chained Eul@2 |
|---|---|---|---|---|---|
| specialist | 0.5671 / 0.5784 | 0.5135 / 0.5251 | 0.894 / 0.867 | 0.780 / 0.690 | 0.205 / 0.215 |
| SHARED | 0.6029 / 0.6160 | 0.5460 / 0.5586 | 1.045 / 0.983 | 0.842 / 0.777 | 0.272 / 0.291 |

Every Stage-8 conclusion survives the fresh box: the shared model beats the specialist in all
four cells on both seeds (direct r +0.036/+0.038, chained r +0.033/+0.034), the chaining cost
is stable (Delta r ~ -0.053 to -0.057, Eulerian tail roughly halves), and the box-to-box
scatter of each cell is ~0.01 in r. The four-cell table is now double-boxed and quotable.

runs/fig9_chain.png: one shared operator applied twice, 32^3 -> 64^3 -> 128^3, density slabs
against the truths. The generated 64 is visually indistinguishable from truth-64; the
generated 128 (built on the GENERATED 64) has the filament network in the right places with
slightly muted small-scale contrast against truth-128 -- consistent with the measured
r = 0.546-0.559 / Eul@1 ~ 0.8.

Q_J ablations (AB_qj_p0 / p3 / div / grad) continue on the GPU, ~03:30 finish.

## 2026-09-13 03:00 — Q_J ablations complete: every component of the full form earns its place

Four variants at 32->64, 3000 steps, lambda auto-equalised per form, vs J_flow_qj (adj, p=2)
and the lag-only reference. Full table (Lagrangian | Eulerian emulator | generative):

| run | r | P/P | rms | gen P/P | Eul @1 | @1.5 | @2 | gen Eul @1 | m>100 |
|---|---|---|---|---|---|---|---|---|---|
| lag-only | 0.7327 | 0.9759 | 0.409 | 0.9808 | 0.963 | 0.777 | 0.551 | 0.860 | 0.0401 |
| **adj p=2** | **0.7460** | 0.9488 | **0.393** | 0.9404 | 1.042 | 0.918 | 0.701 | 0.969 | 0.0422 |
| adj p=0 | 0.7417 | 0.9674 | 0.395 | 0.9576 | 1.000 | 0.844 | 0.613 | 0.924 | 0.0412 |
| **adj p=3** | 0.7452 | 0.9555 | 0.394 | 0.9471 | **1.048** | **0.923** | **0.710** | **0.983** | 0.0428 |
| div p=2 | 0.7390 | 0.9608 | 0.399 | 0.9577 | 1.009 | 0.889 | 0.703 | 0.925 | 0.0417 |
| grad p=2 | 0.7373 | 0.9690 | 0.402 | 0.9628 | 0.994 | 0.855 | 0.645 | 0.897 | 0.0406 |

Verdicts (Eulerian tail is the discriminator):
1. **The caustic weight matters and saturates by p=2**: p=0 loses 0.07-0.09 across the tail
   (0.613 vs 0.701 at 2 k_Ny,c); p=3 is marginally better than p=2 (+0.01) — not worth a
   default change, but p in [2,3] is the plateau.
2. **The adj(A) state-geometry coupling earns ~0.03**: div (same weight, A -> I) sits at
   1.009/0.889 against adj's 1.042/0.918 at (1, 1.5) k_Ny,c.
3. **The full gradient is WORSE than the divergence** (0.994/0.855/0.645): the reviewer's
   null-space concern — transverse errors escaping div's kernel — does not pay off in
   practice; penalising transverse gradient components spends capacity on error directions
   that are not the harmful ones. The harmful residual is divergence-like.
4. Lagrangian numbers are nearly flat across variants (r 0.737-0.746); the ranking lives
   entirely in the Eulerian tail, one more instance of the two-space split.

Ranking: adj p=3 >= adj p=2 > div > adj p=0 > grad > lag-only. The full Q_J design is
validated component-wise; every simplification loses something. Keep adj p=2 as the default
(p=3 optional). Reviewer item 3 is closed for this level; the cross-level check exists via
J_flow_128 (adj p=2, Eulerian 0.894 at k_Ny,c vs lag-only 0.680).

## 2026-09-13 12:30 — owner's second review absorbed; nsteps convergence; two-box ablations; Stage 10 launched

Owner's review points adopted:
- The four-cell headline is now quoted as TWO-BOX AVERAGES (their numbers confirmed mine).
- The key chaining clue is on record: Lagrangian octave power is nearly unchanged by chaining
  (0.856 -> 0.850) while the density tail halves (0.579 -> 0.282) — the chain problem is
  coarse-error/detail coupling, NOT displacement power. No single-mechanism attribution yet.
- Boundary stated: everything chained so far is the EMULATOR chain (true octaves at both
  levels). The generative chain and 128->256 extrapolation are unverified.
- My "slightly muted small-scale contrast" description of fig9 is retracted as an
  understatement: the chained tail is ~28% of the true density power at 2 k_Ny,c.
- Easy-level verdict refined: shared vs specialist at 32->64 is s8-weaker/s9-stronger,
  i.e. "roughly equal on the easy level, benefits on the hard level".

### ODE integration-step convergence (fixed J_flow_qj checkpoint, seed 8)

| nsteps | Lag r | Eul @1 | @1.5 | @2 |
|---|---|---|---|---|
| 1 | 0.7756 | 1.235 | 1.347 | 1.362 |
| 2 | 0.7667 | 1.127 | 1.106 | 0.970 |
| 4 | 0.7518 | 1.054 | 0.951 | 0.750 |
| 8 | 0.7460 | 1.042 | 0.918 | 0.701 |
| 16 | 0.7435 | 1.042 | 0.909 | 0.687 |

n=8 vs n=16 differ by <= 0.014: the reported numbers are integration-converged. Notable: the
qj model's ONE-step evaluation is density-EXCESSIVE (1.24-1.36, regression-like), converging
from above — opposite sign to the plain flow's one-step deficit.

### Ablations at TWO-BOX averages (s8/s9), Eulerian emulator

| variant | @1 avg | @2 avg |
|---|---|---|
| lag-only | 0.942 | 0.545 |
| adj p=0 | 0.973 | 0.593 |
| grad p=2 | 0.962 | 0.620 |
| adj p=2 | 1.000 | 0.656 |
| adj p=3 | 1.009 | 0.665 |
| div p=2 | 0.979 | 0.671 |

Refinement of yesterday's single-box verdicts: the caustic weight (p >= 2) remains the clear
winner over p=0, and grad remains the weakest weighted form at @1 — but div is TIED with adj
at @2 (0.671 vs 0.656; the 0.03 adj advantage lives at @1 only). With ~0.05 box scatter, the
honest ranking is {adj p2/p3, div} > {p0, grad} > lag-only, and adj p=2 stays the default.

### Stage 10 queued (both owner-approved)

- RFT_mix50: rollout fine-tuning per the owner's protocol — frozen upstream (M_qj_multi at
  32->64) generates predicted 64^3 for all training boxes; the downstream level fine-tunes
  from the shared checkpoint with P(predicted coarse)=0.5, target/octave always the true 128
  of the same realization; 1500 steps, LR 1e-4. Limitation noted in-code: augmentation is off
  (the predicted-coarse copy must stay aligned with its target).
- RFT_mix0: the control — identical fine-tuning with true coarse only, to separate "mixing
  helps" from "extra low-LR steps help".
- RF_resflow: frozen R_reg_phys_3k as deterministic base + a fresh residual flow from
  x0' = B to the truth, qj loss, 3000 steps. The base removes 77% of the squared path
  (step-1 L_mse 0.076 vs 0.33 from the physical source).
- eval_chain gained --ckpt-b so the fine-tuned downstream can be chained under the frozen
  upstream. ~9 h total; evaluation plan: the four-cell row for each FT variant on both boxes.

## 2026-09-13 20:20 — rollout fine-tuning round 1: the control isolates the signal

Three-row table at 128, seed 8 (emulator octaves; direct = true coarse, chained = generated):

| model | direct r / Eul@1 / @2 | chained r / Eul@1 / @2 |
|---|---|---|
| shared, no FT | 0.6029 / 1.045 / 0.574 | 0.5460 / 0.842 / 0.272 |
| + FT mix0 (control) | 0.5839 / 0.839 / 0.420 | 0.5276 / 0.697 / 0.198 |
| + FT mix50 | 0.5831 / 0.893 / 0.442 | 0.5373 / 0.760 / 0.224 |

Verdicts:
1. **The fine-tuning REGIME itself is damaging**: the true-coarse-only control loses as much
   as or more than the mixed run everywhere (direct Eul@1 1.045 -> 0.839). Causes on the
   table: augmentation off (a documented limitation of round 1 — the predicted-coarse copy
   must transform with its target and the minimal driver skipped that), single-level
   fine-tuning (forgetting the 32->64 half), batch 1, 1500 steps of LR 1e-4 over 8 boxes.
2. **Within the regime, the mixing HELPS on every single number** (mix50 > mix0: chained
   Eul@1 0.760 vs 0.697, chained r 0.537 vs 0.528, and even direct 0.893 vs 0.839). The
   owner's protocol signal is real and positive; round 1 buried it under regime damage.
3. Round 2 spec (not launched): paired augmentation for (predicted coarse, target), keep BOTH
   levels training during FT with mixing only at the 128 level, fewer steps / lower LR.

RF_resflow (learned base + residual flow) training since 20:00, ~22:15 finish.

## 2026-09-14 13:20 — RF_resflow evaluated on both boxes: the project's best model; REPORT_8

RF_resflow finished 2026-09-13 22:05 (3000 steps, lambda auto 0.0159). Seed-9 evaluation via
a scratchpad script (base = frozen R_reg_phys_3k, flow from x0' = B to truth, 8 Heun):

| | Lag r (s8/s9) | Lag P/P | Eul @1 | @1.5 | @2 |
|---|---|---|---|---|---|
| base | .829/.842 | .721/.745 | 1.435/1.231 | 1.866/1.385 | 2.170/1.552 |
| resflow emulator | .795/.806 | .996/1.029 | 1.082/1.006 | 1.029/0.889 | 0.844/0.726 |
| resflow generative | .190/.204 | .947/.976 | 1.106/0.997 | 1.046/0.866 | 0.879/0.704 |

Two-box emulator Eulerian 1.044/0.959/0.785 vs J_flow_qj's 1.000/0.878/0.656, with r 0.80 vs
0.75: better at every probe on both boxes. The base's Eulerian excess is repaired from above
(2.17 -> 0.84), not smeared from below; generative mode ~1 at k_Ny,c with the independent-T
sampler. The Stage 5-7 accuracy/power trade-off is substantially dissolved by the two-stage
design. Caveats: single level, not chained, 2x parameters (not parameter-matched), rms 0.353
between base 0.307 and single-flow 0.393. runs/REPORT_8.md written.

## 2026-09-14 21:20 — owner verified the residual flow; Stage 11 launched (approved 1+2)

Owner's independent verification recorded: two-box density ratios reproduced (emulator
under-power at 2 k_Ny,c shrinks 34% -> 21%, generative 48% -> 21%), r_delta also improves
(0.905 -> 0.919 at the top probe, seed 8), 16-step integration keeps the gain (0.844 ->
0.833). Boundaries kept on the record: (1) mass in high-density cells still 13%/19% low on
the two boxes — physical statistics not fully recovered; (2) single level only, no 64->128
or chain for the residual flow, no new rollout results; (3) the prior question stands —
independent-T sampler throughout, power improvement does not prove the conditional prior.
Parameter accounting for all comparisons: base+flow = 3.11M vs 1.56M single-stage.

Launched:
- runs/stage11_train.sh -> RF_resflow_128: base = frozen F_reg_128_3k, residual flow at
  64->128, 3000 steps, batch 1 (~9 h; resflow_train.py gained --nc/--nf/--base-ckpt/--batch).
- CPU: oracle-prior generative evaluation on the existing RF_resflow (32->64) — does the
  exact joint prior push the resflow's generative Eulerian to its emulator level, as it did
  for the single-stage flow?

## 2026-09-14 21:45 — the residual flow is prior-robust (oracle evaluation)

Sampler ladder on RF_resflow (32->64, seed 8), generative Eulerian at (1/1.5/2) k_Ny,c:
independent-T 1.106/1.046/0.879; GenIC-oracle 1.10/1.03/0.86 and 1.083/1.026/0.862;
emulator reference 1.082/1.029/0.844. All within 0.02-0.03 of each other.

**The prior sensitivity that governed the single-stage flow (independent-T 0.860 vs oracle
0.957 vs emulator 0.963 at k_Ny,c) has essentially vanished in the two-stage design.** The
sampled octave's imperfect joint statistics pass through the conditional-mean base first,
which is insensitive to the octave's fine structure; the residual flow then operates on the
base's output. Practical consequence: the residual flow does not need the exact-prior/oracle
sampler at this level for these metrics. The owner's boundary stands in principle (a correct
conditional prior is still the right target), but the sensitivity is an order smaller — a
paper-worthy robustness property of the architecture.

RF_resflow_128 training continues overnight (~06:30).

## 2026-09-15 14:10 — RF_resflow_128: the two-stage design holds at the hard level, double-boxed

RF_resflow_128 (base = frozen F_reg_128_3k, residual flow, 3000 steps, batch 1, lambda auto
0.0121) finished 05:43. Both boxes:

| 64->128 | r (s8/s9) | Lag P/P | Eul @1 | @1.5 | @2 |
|---|---|---|---|---|---|
| base (= F_reg_128_3k) | .729/.741 | .539/.568 | 1.898/1.607 | 2.298/2.088 | 2.627/2.549 |
| **resflow emulator** | **.677/.689** | .970/1.023 | **1.018/0.991** | 0.849/0.830 | **0.636/0.605** |
| resflow generative | .239/.245 | .931/.987 | 1.007/0.987 | 0.835/0.824 | 0.622/0.599 |

Against the 64->128 field (two-box emulator Eulerian averages):
lag-only 0.647/0.421/0.271 | qj specialist 0.881/0.612/0.399 | shared 1.014/0.792/0.579 |
**resflow 1.005/0.840/0.621** — best r by a wide margin (0.68 vs shared's 0.61, single-stage
flows' 0.55-0.57) AND the best Eulerian tail, with the octave marginal at ~1. The base's
2.6x density excess is repaired from above, exactly as at 32->64.

Prior robustness holds at this level too: generative == emulator to 0.01-0.02 at every probe
(with the independent-T sampler). Parameter note stands: 3.11M combined.

The two-stage residual-flow design is now the default-operator candidate for the progressive
chain: better than every single-stage variant at both levels, on both boxes, in both modes.
Next (not launched): its chain row (specialist resflow 32->64 exists, resflow-128 exists —
eval_chain needs a resflow pair mode), a shared/cross-scale version, rollout round 2 on the
resflow pair.

## 2026-09-15 (evening) — owner's chain evaluation of the resflow pair; claims tightened

Owner computed the resflow chain row independently (my in-repo reproduction was still running
and is superseded by their numbers; eval_chain gained a --pair resflow mode for the record).
Two-box averages, 128^3 final, true octaves at both levels:

| model | input | r_Psi | density P ratio @1/@1.5/@2 |
|---|---|---|---|
| shared | single, true 64 | 0.609 | 1.014/0.792/0.579 |
| shared | chained | 0.552 | 0.809/0.501/0.282 |
| resflow | single, true 64 | 0.683 | 1.004/0.839/0.621 |
| **resflow** | **chained** | **0.610** | **1.150/0.913/0.587** |

- The resflow CHAIN matches the old shared SINGLE-LEVEL displacement correlation (0.610 vs
  0.609), and the chained top-probe power doubles vs the shared chain (0.282 -> 0.587),
  retaining 95% of its own single-level value.
- NEW ISSUE: mid-frequency overshoot at k_Ny,c in the chain — seed 8 gives 1.268, seed 9
  1.032; large box-to-box variance.
- **The phase problem persists, now quantified via coherent power** P_coh/P_true =
  (P_pred/P_true) r_delta^2 (density r, not displacement r): at the top probe the resflow
  goes 0.621 x 0.890^2 = 0.492 (single) -> 0.587 x 0.708^2 = 0.295 (chained) — total power
  kept, but ~40% of coherent power still lost; part of the chained power is incoherent.
  My earlier "phase contagion likely remains" is supported; "two-stage helps chained power
  only marginally" is corrected — the POWER improvement is real, its coherence is not.

Claims tightened per the owner:
1. "Default-operator candidate" stands; "leads on all metrics" retracted — the bare
   regression still wins displacement r and rms, and qj's two-box first probe at 32->64 is
   closer to 1.
2. "Prior-robust" is limited to the measured POWER statistics: one generative draw per box;
   conditional variance/covariance/diversity untested. The randomness path (IC octave ->
   base -> deterministic flow) may compress true conditional variation — to be checked with
   fixed-C multi-draw evaluations.
3. Parameters counted per CHAIN: two specialists = 6.22M (not 3.11M); a shared base+flow
   would be 3.11M for the whole chain. And s8/s9 are box seeds, not independent trainings.

Math note adopted: the FM target of stage 2 is u* = E[X - B | x_t, t, C] with EXPLICIT
conditioning on C (the note's "C is a function of x_t" phrasing to be fixed); single-level
path shortening does not imply compositional stability — the downstream operator's response
to upstream error is the object to analyse.

Owner's next batch (order fixed): (1) evaluation layer — chain logic into the formal
evaluator, 8->16-step checks on the hard level and the chain, a both-levels-sampled
generative chain, fixed-C diversity checks; (2) short rollout pilot on the resflow pair —
freeze upstream AND base, tune the downstream flow only, no-FT/mix0/mix50 rows, paired
augmentation restored, and the CRITICAL adaptation: recompute B for whichever coarse field
is chosen and use X - B as the velocity target (the round-1 driver started from the physical
source and must not be reused as-is); (3) the shared resflow + a same-base/second-stage-
REGRESSION control. Literature: CorrDiff/AFM (ensemble calibration), PixelIR/RFMSR (controls
first, distillation later), MP-PDE (rollout stability as input-distribution shift).

## 2026-09-15 — Stage 12 launched: rollout pilot round 2 on the resflow pair (owner's spec)

runs/rollout_ft2.py implements the corrected protocol: frozen upstream resflow chain AND
frozen base128; only the downstream residual flow is tuned (warm start RF_resflow_128);
the base is re-evaluated on whichever coarse enters the batch and the velocity target is
X - B (round 1's physical-source pipeline is NOT reused); paired cubic-group augmentation of
(chosen coarse, target, IC); lambda reused from RF_resflow_128 (0.01214), 800 steps at 5e-5.
Queue: RFT2_mix50 then RFT2_mix0 (control); rows for the report are no-FT / mix0 / mix50.
eval_chain gained --flow-b (evaluate a tuned downstream flow under the frozen upstream) and
now reports r_delta and the coherent power (P/P x r_delta^2) per probe — the owner's
decomposition is in the formal evaluator. Targets for the pilot: reduce the chain's
mid-frequency overshoot (1.27 on s8), raise coherent power (0.295 at the top probe), keep
the single-level numbers.

## 2026-09-15 22:00 — research directive step 2: the ideal projected label FAILS as a density target

`runs/eval_y64.py` (reuses the owner's verified interlaced-CIC + deconvolution estimator and
complete-shell averager; common 256^3 ANALYSIS mesh; native particle counts). Restriction
verified as a true Fourier restriction first: retained complex coefficients of Y64 match
those of Psi_128 to 6.8e-8 / 7.2e-8 (max relative), i.e. the label is exactly
Y64 = R_{128->64} Psi_128 under the project conventions.

Density spectra vs the SAME 128^3 reference, band 0 < k < k_Ny,64 (old-IC s8/s9 data — but
note the Y64-vs-128 comparison is IC-clean by construction, since Y64 derives from Psi_128):

| s8 | P/P at kNy,32 | 0.75 kNy,64 | 0.9 kNy,64 | r@0.9 | 1%/5% band [h/Mpc] |
|---|---|---|---|---|---|
| native 64^3 sim | 0.976 | 0.952 | 0.964 | 0.977 | 0.32 / 1.51 |
| **Y64 (ideal label)** | **1.223** | **1.321** | **1.344** | 0.981 | 0.20 / 0.38 |

(s9: native 0.987/0.981/0.987, band to 0.32/1.95; Y64 1.110/1.140/1.165, band to 0.26/0.57.)

Two findings, both prior to any network:
1. **Displacing 64^3 particles by the band-limited restriction of the 128^3 displacement
   OVER-CONCENTRATES: 11-34% density excess across the band.** Removing the octave detail
   removes exactly the small-scale spreading that keeps caustics finite — the same mechanism
   as the regression's over-concentration, now demonstrated as a property of the LABEL
   itself. A perfect predictor of Y64 would inherit this excess. delta[R Psi] != R delta[Psi],
   quantified.
2. **The native 64^3 simulation's density is already within 2-5% of the 128^3 density over
   the whole band** (1% band to ~0.3 h/Mpc, 5% to 1.5-2.0). The coarse run's deficiency in
   this band is displacement-level phase near its Nyquist, not band density power. The naive
   premise "the 64 run is a poor proxy for the 128 density below k_Ny,64" is false at the
   power level.

Directive question 2 gets a data answer: Y64 as a TRAINING LABEL is exact for the
displacement band by definition, but as a route to 128-level density it is worse than the
native 64 run — any 64^3-particle representation that aims at 128-level density must carry
compensating small-scale displacement content, not the bare restriction. Box scatter is large
(Y64 excess 1.34 vs 1.17 at 0.9 kNy,64) and this is old-IC data for the native row: the
strictly-nested rerun (in progress) re-checks both rows on clean ICs.

## 2026-09-15 22:30 — strict trio evolved; IC error vs dynamical error separated; clean-IC Y64 confirmed

MP-Gadget accepted the hand-written strictly-nested BigFile ICs unmodified; the s5000 trio
(32/64/128, Zel-only ICs, nesting 1.8e-7) evolved to z=0 and converted.

### The first clean separation of IC error from dynamical error

| pair | IC rel-RMS | z=0 rel-RMS | z=0 r@0.9 kNy,c |
|---|---|---|---|
| 32->64 STRICT | 1.8e-07 | 0.126 | 0.821 |
| 32->64 old s8 | 1.4e-01 | 0.136 | 0.784 |
| 64->128 STRICT | 1.8e-07 | 0.093 | 0.743 |
| 64->128 old s8 | 8.6e-02 | 0.094 | 0.712 |

**Making the ICs exactly nested changes the final-state cross-level mismatch by only ~0.01
in rel-RMS and 0.03-0.04 in near-Nyquist correlation.** The z=0 non-nesting is overwhelmingly
DYNAMICAL (the fine run resolves octave modes whose backreaction alters the shared band, plus
discreteness/softening) — not inherited from the generator. Directive question 1 answered:
the IC construction accounts for a small fraction of the final coarse-band mismatch; and the
project's model results on old-IC data are not materially contaminated at the final-state
level. Caveat: the strict trio is Zel-only (no 2LPT, ~1e-2 of the IC displacement at z=99, a
documented physical difference from the GenIC pipeline).

### Clean-IC Y64 check (nested s5000): both step-2 conclusions reproduce

native64 vs ref128 density: 0.947/0.961/0.954 at (kNy32, 0.75, 0.9 kNy64), r@0.9=0.977;
Y64 vs ref128: **1.179/1.303/1.370** (coefficient identity 1.0e-7). The label's
over-concentration is NOT an old-IC artifact; the native run's 5%-level band agreement holds
on clean data too (1%/5% bands to 0.32/0.94 h/Mpc).

## 2026-09-15 23:20 — directive step 3: existing models against the 128^3 density reference

Same estimator/mesh/shells as step 2; s8/s9 (old-IC, flagged); emulator mode (true octave).

| vs 128^3 density | s8: kNy32 / 0.75 / 0.9 kNy64 | s9 | r@0.9 (s8/s9) |
|---|---|---|---|
| native 64^3 sim | 0.976/0.952/0.964 | 0.987/0.981/0.987 | 0.977/0.975 |
| regression | 1.393/1.733/1.914 | 1.214/1.344/1.430 | 0.907/0.914 |
| resflow | 1.062/0.996/0.900 | 0.996/0.879/0.783 | 0.918/0.915 |
| Y64 (ideal label) | 1.223/1.321/1.344 | 1.110/1.140/1.165 | 0.981/0.987 |

Readings for the report:
1. **Under the 128 reference, the native 64 run beats every model and the ideal label in this
   band, in both power (2-5%) and phase (r 0.975+).** Expected — the models were trained to
   reproduce the 64 run — but it makes the directive's premise concrete: the headroom below
   k_Ny,64 in band POWER is < 5%; the real 64-vs-128 gap lives in phase and at k > k_Ny,64.
2. resflow is the closest model (its Q_J training pulls the density toward its own 64-level
   reference, which is itself near the 128 one); the regression's excess (1.2-1.9) is roughly
   the Y64 excess compounded with its own over-concentration.
3. Main band capped at k < k_Ny,64 per the directive; 64^3-particle densities do carry
   content beyond it (points alias), where all 64-representations differ most from 128 — a
   documented boundary, not measured here.

## 2026-09-15 23:35 — fig11 + REPORT_9 draft

`scratchpad fig_y64.py -> runs/fig11_y64_verdict.png`: density P/P_128 for native64 vs Y64
(s8 solid / s9 dashed / strict-IC dotted) with regression/resflow probe markers and the
r_delta panel; both particle Nyquists marked; analysis mesh 256^3 stated in the title.
`runs/REPORT_9.md` drafted: directive steps 1-3 + five answers + minimal-experiment
recommendation (Y64-base + native-64 residual flow). RFT2 three-row table pending the
mix0 training + auto-eval (waiter armed on 'STAGE12 TRAIN DONE'); will be appended as §3b
before morning delivery.

## 2026-09-15 23:55 — no-FT baseline rows refreshed with full metrics (CPU, alongside GPU training)

`eval_chain.py --pair resflow --test-seed 8|9` (no --flow-b): the pre-existing
chain_resflow_s8.json predated the r_delta/coherent metrics and s9 was missing; both rewritten
by the same script that will measure the RFT2 rows (comparability). Baseline for the verdict:
s8 chained Eul [1.268, 0.994, 0.614], r_d [0.908, 0.824, 0.723], coherent [1.046, 0.675, 0.321];
s9 chained Eul [1.032, 0.832, 0.560], coherent [0.831, 0.533, 0.269]; direct rows unchanged
within noise (r 0.677/0.689).

## 2026-09-16 01:00 — RFT2 round-2 verdict (rollout pilot closed)

Training: RFT2_mix50 (2026-09-15, 800 steps) and RFT2_mix0 control (800 steps, ended 00:06);
both chunked --max-seconds 900 --resume, one GPU job at a time; eval_chain on CPU, four rows
(runs/chain_RFT2_{mix50,mix0}_s{8,9}.json), baseline = re-measured no-FT rows (23:55 entry).

| chained | r s8/s9 | Eul@(kNy64,1.5kNy64,kNy128) s8 | s9 | coh@kNy128 |
|---|---|---|---|---|
| no-FT  | .602/.617 | 1.268/0.994/0.614 | 1.032/0.832/0.560 | .321/.269 |
| mix0   | .601/.616 | 1.244/0.953/0.579 | 1.010/0.796/0.522 | .303/.248 |
| mix50  | .596/.610 | 1.161/0.856/0.498 | 0.952/0.714/0.443 | .251/.203 |

Direct mode: no-FT r .677/.689 Eul 1.018/0.991 -> mix50 .669/.681 Eul 0.883/0.904 (damped).
Criteria: overshoot reduced (real: mix0 isolates ~1/4 as drift), coherent NOT raised (down
everywhere), r_delta unchanged to 3rd digit in every cell -> the mixing benefit is pure power
damping toward the r^2 line, no phase repair; direct regime measurably damaged. Round-2 with
all owner fixes = clean negative; downstream rollout FT closed, chain error goes upstream.
REPORT_9.md sec 3b written.

## 2026-09-16 14:40 — HPC handoff prepared (owner: run A -> B -> C1 one at a time, all pushed)

Portability pass for the cluster session: `compute_spectra.py` moved verbatim into the repo
root (only ROOT made repo-relative; provenance in its docstring); scratchpad analysis scripts
rescued into runs/ (fig_y64, phase3_models, fig_chain, fig_chainfull, nsteps, rf_oracle,
rf_s9, rf128_s9) and all runs/*.py absolute paths rewritten repo-relative (verified: import
smoke test + ast.parse on every file; eval_y64 --help runs). CLAUDE.md HPC-maintenance line
updated; .claude/ gitignored. `HANDOFF.md` written: state, approved plan A/B/C1, reference
numbers, conventions/gotchas, rsync manifest (git carries no weights/.npy), open threads.

## 2026-09-16 — HPC bring-up (Phase A step 1: env + selftests + smoke test)

Cluster: PSC (hildafs), repo `/hildafs/projects/phy200018p/xzhangn/progressive_sr/progressive-sr`, HEAD ff105ad.
Env (owner's choice): conda `torch206` = Python 3.11.11, torch 2.8.0+cu128, numpy 2.2.4, scipy 1.15.2,
matplotlib 3.10.1, bigfile importable. GPU allowance (owner): ONE A100 at a time, TWIG-GPU or HENON-GPU;
CPU work on RM (28 cores / 128 GB nodes). Login node (CPU, shared) used for the checks below.

- `python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128 --offset 0.5 --out st_dealias`
  -> operator check |RP x - x|/|x| = 3.23e-07; slopes P_eps/P_div,eps = 2.19/4.14 (32to64), 2.18/4.18
  (64to128); Haar 0.85/2.78; nestedness 2.8e-07 / 3.2e-07. Matches reference 2.2 / 4.15.
- `python phase0_octaves.py --selftest --levels 32 64 128 --offset 0.5 --out st_alias`
  -> aliased slopes 0.19/2.15 and -0.71/1.30 (white floor, as documented).
- `python octave_flow_toy.py --nc 16 --nf 32 --steps 20 --base 16 --rms-delta 2.0 --device cpu --out <scratch>/cpu_probe`
  -> baseline r=0.915459 P/P=0.835218 rms err/h_f=0.184311, multistream 0.166: BIT-IDENTICAL to
  reference_results/toy16to32_cube_flow_physical.json (data synthesis + x0 path unchanged). Step-1 loss
  3.3257e-02 bit-identical; from step 2 the losses differ at the 1e-3 level (x86 + torch 2.8 backward /
  optimizer kernels vs the laptop); step 20 = 3.0882e-02 vs the laptop's 3.0888e-02. Timing 3.3 s/step on
  the shared login node (vs 0.57 s native Mac CPU) — not a benchmark. Numbers acceptable; the
  "bit-identical" criterion is a same-machine regression test, not a cross-platform one.
- Data found (owner): `/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology/
  dmo-{64,128,256,512}/set{0..15}/{IC,PART_009}/{disp,vel,style}.npy`, already map2map layout
  float32 (3,N,N,N) in **kpc/h** (`--box 100000`), PART_009 = a=1 (z=0), IC = a=0.01 (z=99) but only
  at the 64 and 512 levels. GenIC params (sim_scripts/dmo-100MPC/...): Seed = 3242400+set, identical
  across the four levels of a set (16 same-seed nested series, not one!), Nmesh unset (= GenIC default
  2*Ngrid, i.e. the folded-content caveat of RUNLOG 2026-09-15 applies), same cosmology as sims/
  (Omega0 0.2814, OmegaL 0.7186, h 0.697, z_init 99) -> GROWTH = 76.7439 carries over.
  Consequence for the plan: C1 (production 64/128 pairs, 16 seeds) already EXISTS as data; only the
  conversion/offset audit (A.2) and Phase 0 (A.3) remain before B.
- `hpc/slurm_phase0.sh` rewritten for this layout/env (env-var overrides, 512 IC as the single top-level
  --ic, lower-level ICs cube-truncated inside phase0).

## 2026-09-16 — Phase A step 2: offset audit of the PSC series + multi-octave R/P phase fix

Provenance: `runs/offset_audit_psc/` (probe scripts + their stdout; probe_axes_prefix ran BEFORE the fix).

Probe scripts (scratch, results below): restrict the fine field to 64^3 (cube window) and compare with
the native 64^3 field per k-shell: r(k) and the cross-spectrum phase arg<A B*>. Caveat learned: modes
along the full (non-rfft) axes sum Hermitian pairs, so their phase is identically 0 — only the rfft
(z) axis and the shell average carry the signal.

- z=0, R[128->64] vs native 64 run, set0: offset 0.5 -> shell phase |<0.7 deg| up to 0.94 k_Ny,64,
  r(k) = 1.000/0.9995/0.996/0.982/0.950/0.897/0.807/0.684 (8 bins to k_Ny,64);
  offset 0 -> phase grows linearly, -22 deg at 0.94 k_Ny,64. **OFFSET = 0.5 for this series**
  (consistent with HANDOFF: 0.5 for the laptop's Box-downloaded production boxes).
- IC, R[512->64] vs native 64 IC: rel-rms 0.24 (offset 0) / 0.21 (offset 0.5) — NOT ~1e-6 and not
  discriminating by itself; per-shell it is r(k) 1.000 ... 0.84 at 0.94 k_Ny,64 (after the fix
  below), i.e. the GenIC Nmesh=2*Ngrid folding of the 64-level IC (RUNLOG 2026-09-15), not an offset.
- BUG found and fixed in `phase0_octaves.py` (`restrict_spectral`, `prolong_spectral`): the offset
  re-referencing phase exp(+i k o h_f) assumed one octave; for Nf/Nc = m it must be
  exp(+i k o (m-1) h_f). Symptom: R[512->64] and R[256->64] at offset 0.5 carried a residual z-axis
  phase equal to a 0.375 h_64 / 0.25 h_64 shift (measured -27/-18 deg at 0.44 k_Ny vs predicted
  -29.5/-19.7). Only phase0's single-top-level `--ic` path was affected (all other callers use offset
  0 or one octave). After the fix: direct R[512->64] == chained R[512->256->128->64] to 3.1e-7,
  RP = I at m=4/8 with offset 0.5 to 3e-7, z-axis phases ~0 for 128/256/512->64; the dealias
  selftest (m=2) is IDENTICAL line-for-line before/after.
- `hpc/slurm_phase0.sh`: OFFSET default 0.5, partition RM, torch206 python, 512 IC as top-level --ic.

## 2026-09-16 — Phase A.4 on the PSC series, set0 (job 1215144, RM, 2m41s, 16 GB): convergence + Y64/Y128

`sbatch hpc/slurm_ylabel.sh` = `python runs/eval_ylabel.py --sets 0 --nc 64 128 --convergence` (commit b21d739).
Estimator = compute_spectra.density_coeff/Shells verbatim, common 256^3 analysis mesh, offset 0.5 applied
as a global +0.5 h_N shift per level before the deposit (so all levels sit at their physical positions).
Density P/P against the reference, probes at (k_Ny,c/2, 0.75 k_Ny,c, 0.9 k_Ny,c), r_delta at 0.9 k_Ny,c:

(a) convergence, band k < k_Ny,64 (reference = native 512):
| run        | P/P @kNy32 / 0.75kNy64 / 0.9kNy64 | min..max in band | r@0.9kNy64 | 1% / 5% frontier [h/Mpc] |
|------------|-----------------------------------|------------------|------------|--------------------------|
| native 128 | 0.993 / 0.992 / 0.992             | 0.989 .. 0.999   | 0.9969     | 0.76 / 1.95 (= band end) |
| native 256 | 0.999 / 0.999 / 1.000             | 0.998 .. 1.002   | 0.9996     | 1.95 / 1.95              |
Same up to k_Ny,128 (reference 512): native 128 P/P 0.993 @kNy64, 0.983 @1.5kNy64, 0.983 @0.9kNy128
(r 0.9956/0.986/0.977); native 256: 0.9995/0.9988/0.999 (r 0.9995/0.998/0.997).
-> VERDICT: in the 64 band the 128 run is converged to <=1.1% in power and r>=0.996 against 512, and
256 to <0.3%. "128 is a reference, not truth" is UPGRADED for the 64 band: 128 is truth to 1%, so the
Y64 label excess (+10..19%) and the native-64 deficit (-3..-4%) are both real w.r.t. converged truth.
For the 128 band the 256 run is the reference at the 0.1% level; the 128 run itself sits 1.7% low.

(b) labels (reference = native 2c):
| candidate for level c | P/P @kNy(c/2) / 0.75kNy,c / 0.9kNy,c | r@0.9kNy,c | 1% / 5% frontier | disp r(label,native) mean/last |
|-----------------------|--------------------------------------|------------|------------------|--------------------------------|
| native 64             | 0.957 / 0.958 / 0.973                | 0.977      | 0.32 / 1.95      | —                              |
| Y64 = R Psi_128       | 1.097 / 1.154 / 1.188                | 0.986      | 0.26 / 0.57      | 0.916 / 0.656                  |
| native 128            | 0.994 / 0.984 / 0.984                | 0.980      | 1.57 / 4.02      | —                              |
| Y128 = R Psi_256      | 1.295 / 1.422 / 1.483                | 0.979      | 0.26 / 0.63      | 0.894 / 0.572                  |
-> the HANDOFF prediction holds: the projected label over-concentrates MORE one level up (+30..48% vs
+10..19%), tracking the multi-stream fraction (phase0: 0.42 of the Lagrangian volume at 128->256 vs
~0.3 at 64->128). Laptop numbers for Y64 (selfsim s8/s9, IC-clean 1.18..1.37) bracket the PSC set0 value.
The native 64 run is 3-4% low in-band (laptop: 0.95..0.99) and the native 128 run 1-2% low in its band:
the "headroom is phase and k > k_Ny,c, not band power" reading stands at both levels.
Phase B (Y64-base) success criterion unchanged: in-band P/P in [0.95, 1.00] AND r_delta@0.9kNy64 > 0.95.
Caveats: one set (set0); the 1% frontier of Y is 0.26 h/Mpc at both levels (the excess starts at the
largest scales of the band, not at the Nyquist), which is the same signature as on the laptop.

## 2026-09-16 — Phase A.3: Phase 0 on the PSC series, set0, 64->128->256->512 (job 1214902, RM r011, 424 s)

`sbatch hpc/slurm_phase0.sh` (commit 66ca7ce + cwd fix): `phase0_octaves.py --levels 64 128 256 512 --box 100000
--dis .../dmo-{N}/set0/PART_009/disp.npy --ic .../dmo-512/set0/IC/disp.npy --vel ... --offset 0.5 --growth 76.7439
--window cube` -> `runs/phase0_real_set0_1214902/` (phase0_results.json, phase0_summary.png; npz gitignored).
NOTE: the "nestedness/convention check" printed by this run (2.3e-7 / 0) is TRIVIAL here because the lower-level
ICs are cube-truncated from the single 512 IC inside phase0; the real offset audit is the entry above.
Velocity transitions were analysed too (`*_vel` in the json; not tabulated here).

Cube window, displacement, values interpolated at fixed k/k_Ny,c (first-ever 128->256 and 256->512 numbers):
| transition | multistream frac | rms eps/h_c | P_eps/P_c @0.5/0.75/0.94 kNy,c | r(c,Rf) @0.94 | Wiener T @0.5/0.75/0.94 | detail r^2(lin) @1.1/1.5/1.95 kNy,c | T_det/G @1.1/1.95 | P_d/(G^2 P_lin) @1.1/1.5/1.95 | detail kurt (in/out MS) | 2LPT harmonics P/P_d |
|---|---|---|---|---|---|---|---|---|---|---|
| 64->128  | 0.374 | 0.272 | 0.066/0.288/0.572 | 0.677 | 0.940/0.867/0.796 | 0.288/0.139/0.056 | 0.472/0.262 | 0.77/0.78/1.24 | 2.85 (2.42/3.06) | 0.286 |
| 128->256 | 0.418 | 0.390 | 0.086/0.338/0.607 | 0.645 | 0.946/0.878/0.809 | 0.158/0.068/0.025 | 0.372/0.209 | 0.88/1.01/1.76 | 5.38 (4.61/6.00) | 0.367 |
| 256->512 | 0.443 | 0.573 | 0.112/0.390/0.661 | 0.601 | 0.949/0.879/0.803 | 0.076/0.031/0.014 | 0.288/0.164 | 1.09/1.35/1.94 | 9.76 (8.62/10.85) | 0.379 |

Reading:
* 64->128 reproduces the laptop's selfsim facts on the production series: P_eps/P_c = 0.572 at 0.94 k_Ny,c
  (CLAUDE.md: 0.57), r^2 of the linear octave ~0.3 at the bottom of the octave falling to 0.06 at the top,
  multi-stream fraction 0.37 (30-40%), Wiener gain G = T/growth < 1 everywhere (0.47 -> 0.26).
* Coarse-band statistics are nearly LEVEL-INVARIANT: the coarse run's error at 0.94 k_Ny,c grows only 0.57 ->
  0.61 -> 0.66, and the Wiener T(k) of the coarse band is the same curve at all three levels (0.80 at 0.94
  k_Ny,c). Good news for one weight-shared 2x operator on the correction.
* The DETAIL band is not level-invariant: the linear octave explains 0.29 -> 0.16 -> 0.08 of it at the octave
  bottom (0.056 -> 0.025 -> 0.014 at the top); the conditional detail becomes heavy-tailed (kurtosis 2.8 ->
  5.4 -> 9.8, inside AND outside the multi-stream mask); the coarse-only 2LPT harmonics hold 29 -> 38% of
  the detail power but stay uncorrelated with it (r ~ -0.02 .. -0.005). P_d/(G^2 P_lin): the linear octave
  overshoots the detail power at 64->128 (0.77 at the octave bottom) but undershoots it from 128->256 up
  (1.0 -> 1.9 at the top of the 256->512 octave) -- the source amplitude for the higher transitions needs
  the Wiener/measured filter, linear theory is wrong in both directions depending on level.
* Multi-stream variance ratio in/out: 1.33, 1.28, 1.22 -- the detail is only moderately larger inside halos;
  the level dependence is in the tails (kurtosis), not the variance.
Consequence for the weight-shared design: the style scalar must carry the level (sigma-matching, RUNLOG
2026-09-09) AND the source filter must be per level; expect the flow-vs-regression gap to widen with level.

## 2026-09-16 — Phase B prerequisites: converter validated bitwise, dmo-32 sets 1-15 launched, GPU held

- `hpc/convert_snapshot.py <raw>/dmo-64/set0/output/{IC,PART_009} --id-offset 1 --id-order C --offset 0.5
  --pos-unit 1` reproduces the owner's PSC `.npy` (disp AND vel, IC and z=0) with max|a-b| = 0 -> the raw
  MP-Gadget outputs (`sim_output/dmo-100MPC/int_redshift_same_cosmology`) and the PSC series are the same
  data under the same convention (kpc/h, q = (i+0.5) h, IDs 1-based C order). Wrong id-order gives 26 h.
- 32^3 level: set0 existed (run 2026-02-24, Seed 3242400, Nmesh 64 = 2 Ngrid, same params otherwise);
  sets 1..15 created in `sim_scripts/.../dmo-32/set{1..15}` (Seed = 3242400+k, verified equal to the
  64/128/256/512 seeds of every set) and launched as one RM job (1217024, `submit_sets1to15.job`).
  set0 converted -> `data/psc/dmo-32/set0/{IC,PART_009}/{disp,vel}.npy`; `data/psc/dmo-{64..512}` are
  symlinks to the PSC directories, so one pattern `data/psc/dmo-{N}/set{seed}/PART_009/disp.npy` covers
  all levels. 32 vs R_cube[64->32] at offset 0.5: corr 0.991 (z=0) / 0.990 (IC), rel-rms 0.13/0.14
  (the GenIC folding level, as for 64 vs 128).
- GPU: `hpc/slurm_hold_gpu.sh` holds one A100 on HENON-GPU (qos henon-gpu; job 1216965 on henon-gpu01,
  2 days); experiments go in with `srun --jobid=1216965 --overlap --exact --ntasks=1 --cpus-per-task=8
  --gres=gpu:1 ...`. Owner's instruction: use only this allocation, never the TWIG/other HENON ones.
- Laptop base recipe recovered from runs/R_reg_phys_3k/results.json: nc 32 nf 64, train seeds 0..7,
  test seed 8 (s9 second box), base 24, 3000 steps, source_filter none, --regression; resflow_train.py
  = frozen base + residual flow (qj, lambda auto), train seeds 0..7, test 8.

## 2026-09-16 — Phase B tooling (no training yet; waiting for the 32^3 sets 1-15)

- GPU path validated on the held A100 (srun into 1216965): 16->32 20-step smoke test, baseline and step-1
  loss bit-identical to CPU, step 20 = 3.0945e-02 (GPU kernels), 0.22 s/step.
- `octave_flow_toy.py --label-from N`: training target := R_cube[N->nf] of the level-N run (offset-aware,
  via restrict_spectral); the test truth stays the native nf run; dis_c/ic_f untouched. Smoke test on PSC
  set0 (32->64, --regression --label-from 128, 40 steps): runs, 0.14 s/step at base 24 -> 3000 steps ~7 min.
  PSC set0 baseline x0 (true octave, no filter): octave r=0.552 P/P=1.533, coarse-band r=0.9455,
  P_eps/P=0.110, multi-stream 0.296 (vs selfsim: r 0.58-ish; the linear octave overshoots by 53% here).
- `runs/resflow_train.py --data psc|selfsim --train-seeds --test-seeds`: dataset table (pattern + offset),
  and it now dumps the test-box fields (`field_{truth,base,emulator,generative}.npy`, gitignored) for
- `runs/eval_vs_ref.py <run> --set S`: density of those fields vs the native 128 of the same set with the
  verified estimator, offset-aware; self-check with the native 64 as "prediction" reproduces the A.4 row
  (0.957/0.958/0.973, r 0.977) exactly.
- `hpc/run_phaseB.sh`: the four runs B0 (control base, native label) / B1 (Y64 base) / B2 (resflow on B0) /
  B3 (resflow on B1) + EVAL, one at a time through `srun --jobid=$JOBID`. Train sets 0-7, test set 8
  (mirrors the laptop split), 3000 steps, base 24, source_filter none (laptop recipe R_reg_phys_3k).

## 2026-09-16 — 32^3 sets 1-15 done and converted; Phase B launched (commit bb2f501, GPU hold job 1216965)

- Job 1217024 (RM, 1 node, 8 ranks): 15 sets in ~15 min. Converted with the validated convention into
  `data/psc/dmo-32/set{k}/{IC,PART_009}`. IC rms|x-q|/h = 0.0334 for every set; z=0 rms 2.39-2.56 h_32.
  Seed alignment check per set: corr(dis32, R_cube[64->32] dis64) = 0.990-0.991 for all 16 sets, cross-set
  control (set k vs set k+1) in [-0.32, +0.30] -> ALL OK.
- Phase B launched: `JOBID=1216965 bash hpc/run_phaseB.sh B0 B1 B2 B3 EVAL` (train sets 0-7, test set 8,
  3000 steps, base 24, batch 2, source_filter none, qj residual flow, lambda auto at step 1):
  B0 runs/R_reg_psc (native label) -> B1 runs/R_reg_psc_y64 (--label-from 128) -> B2 runs/RF_resflow_psc
  (on B0) -> B3 runs/RF_resflow_psc_y64 (on B1) -> eval_vs_ref.py on B2/B3 vs native 128 of set 8.
  Pre-registered success (HANDOFF §2 B): B3 in-band density P/P in [0.95, 1.00] AND r_delta@0.9 k_Ny,64
  > 0.95, beating both end-members (native 64: 0.957/0.958/0.973 on set0; Y64 label: 1.10-1.19).

## 2026-09-16 17:15 — Phase B on the PSC series, part 1: B0/B1/B2 done (commit bb2f501, A100 in hold job 1216965)

Train sets 0-7, test set 8, 32->64, base 24, 3000 steps, batch 2, source_filter none. Timings on the A100:
regression 0.12 s/step (6 min), residual flow 0.16 s/step (8 min). Harness note: the launcher process was
killed by a session teardown at B3 step 1; B3+EVAL relaunched detached (`setsid nohup`), B0-B2 unaffected.
Test box set8: multi-stream fraction 0.300; x0 (true octave, no filter) octave r=0.550 P/P=1.527.

Lagrangian octave band vs the NATIVE 64 test run (emulator = 1 Euler step for the bases, 8 Heun for the flows):
| run | label / target | train loss | r | P/P | rms err/h_f (MS/SS) |
|---|---|---|---|---|---|
| B0 R_reg_psc (control base)    | native 64          | 8.44e-2 | 0.830 | 0.721 | 0.308 (0.372/0.276) |
| B1 R_reg_psc_y64 (Y64 base)    | R_cube[128->64]    | 6.33e-2 | 0.777 | 0.688 | 0.335 (0.398/0.305) |
| B2 RF_resflow_psc (flow on B0) | native 64, lam 0.0156 | 1.01e-1 | 0.796 | 1.001 | 0.354 |
| B2 generative                  |                    |         | 0.206 | 0.956 | 0.584 |
B1 is judged against a truth it was not trained on, so its lower r/P/P is expected; its loss is 25% lower
because the projected label is smoother than the native run.

Eulerian density of the 64^3 test fields (`eval_vs_ref.py`, verified estimator, offset 0.5), band k < k_Ny,64:
| field (set8) | reference | P/P @kNy32 / 0.75kNy64 / 0.9kNy64 | r@0.9kNy64 | band min/max | 1%/5% frontier |
|---|---|---|---|---|---|
| native 64 (truth)      | native 128 | 1.000 / 1.012 / 1.015 | 0.974 | 0.990/1.029 | 0.50 / 1.95 |
| B0 base                | native 128 | 1.191 / 1.299 / 1.397 | 0.924 | 1.001/1.447 | 0.26 / 0.44 |
| B2 resflow emulator    | native 128 | 0.958 / 0.829 / 0.746 | 0.926 | 0.718/1.012 | 0.44 / 1.07 |
| B2 resflow generative  | native 128 | 0.966 / 0.834 / 0.741 | 0.876 | 0.707/1.021 | 0.57 / 1.07 |
Same fields vs the native 64 truth (resflow_train's own probes at k_Ny,32 / 1.5 k_Ny,32 / k_Ny,64):
base 1.204/1.309/1.478, emulator 0.967/0.825/0.678, generative 0.976/0.829/0.682 — laptop RF_resflow
(selfsim) had emulator 1.082/1.029/0.844 (s8) and 1.006/0.889/0.726 (s9), base 1.43-2.17: the PSC control
sits at the low end of the laptop's two-box range; same qualitative picture (base over-concentrates,
the residual flow repairs it from above and overshoots into a deficit at the top of the band).
NB: set8's native 64 is within 1-3% of 128 in-band (set0 was 3-4% low) — box-to-box scatter of the
"native 64 end-member" is of the same size as its deficit; quote both boxes when this matters.

## 2026-09-16 17:25 — Phase B verdict: the Y64-base does NOT change the residual flow's density (set8)

B3 RF_resflow_psc_y64 (residual flow on the Y64-trained base B1, target = native 64; lambda auto 0.0155,
3000 steps, loss 9.97e-2). Lagrangian octave band vs native 64: base r=0.777 P/P=0.688; emulator r=0.770
P/P=1.007 rms 0.369; generative r=0.199 P/P=0.983.

Eulerian density vs the CONVERGED native 128 of set8, k < k_Ny,64 (eval_vs_ref.py):
| field (set8)               | P/P @kNy32 / 0.75kNy64 / 0.9kNy64 | r@0.9kNy64 | band min/max | 1%/5% frontier |
|----------------------------|-----------------------------------|------------|--------------|----------------|
| native 64 (truth)          | 1.000 / 1.012 / 1.015             | 0.974      | 0.990/1.029  | 0.50 / 1.95    |
| B0 base (native label)     | 1.191 / 1.299 / 1.397             | 0.924      | 1.001/1.447  | 0.26 / 0.44    |
| B1 base (Y64 label)        | 1.196 / 1.283 / 1.368             | 0.924      | 1.001/1.399  | 0.14 / 0.38    |
| B2 flow on B0, emulator    | 0.958 / 0.829 / 0.746             | 0.926      | 0.718/1.012  | 0.44 / 1.07    |
| B3 flow on B1, emulator    | 0.956 / 0.819 / 0.734             | 0.921      | 0.694/1.017  | 0.76 / 1.07    |
| B2 generative              | 0.966 / 0.834 / 0.741             | 0.876      | 0.707/1.021  | 0.57 / 1.07    |
| B3 generative              | 0.951 / 0.808 / 0.709             | 0.873      | 0.668/1.014  | 0.50 / 1.01    |

Pre-registered criterion (HANDOFF §2 B): in-band P/P in [0.95, 1.00] AND r_delta@0.9 k_Ny,64 > 0.95.
B3: FAILS on both (0.73 at the top of the band; r 0.921). B2 (control) fails identically. Verdict:
* The label swap is invisible at the density level: B1's base excess equals B0's (1.37 vs 1.40 at
  0.9 k_Ny,64, identical r 0.924) although B1 was trained toward a label that itself over-concentrates by
  10-19% -> the base's over-concentration is conditional-mean shrinkage (notes: a<1 over-compacts
  multi-stream patches), not a label property; and the residual flow, whose own target is the native 64
  run, lands at the same deficit (0.73-0.75) and the same r (0.92) whichever base it sits on.
* Hence "supervision alone" (a better label for the base) is REFUTED as the route to the reference's
  density in-band; the HANDOFF's consequence applies: the representation route gets priority.
* What the flow does and does not fix: it removes the base's +40% excess (from above, as on the laptop)
  and overshoots into a -25% deficit at 0.9 k_Ny,64, while r_delta stays at the base's 0.92 (native 64:
  0.974). Both models' 5% frontier is 1.07 h/Mpc = 0.53 k_Ny,64; the native 64 run's is the band end.
  The missing quantity is phase coherence at k > 0.5 k_Ny,64 (the octave region where the linear octave
  explains r^2 < 0.15 and the conditional is heavy-tailed, Phase 0 table), not band power.
* Caveats: one box (set8); Lagrangian r 0.796 vs 0.770 says the Y64 base is also slightly worse in the
  displacement metric; generative-mode density is within 0.03 of emulator at both models (independent-T
  sampler, prior-robust as on the laptop).
Parameter accounting: base+flow 3.11M (1.56M + 1.56M) for B2/B3, 1.56M for B0/B1.

## 2026-09-16 21:10 — Diagnostic (owner option 3): where the r_delta ceiling sits — at the BASE, uniformly in space

`python runs/diag_rceiling.py runs/RF_resflow_psc --set 8` (CPU; verified estimator; subset deposits by the
coarse-run multi-stream mask upsampled, eval_eulerian T4 convention; multi-stream fraction 0.300).
r_delta(k) vs native 128 at k/k_Ny,64 = 0.2 ... 1.0 (step 0.1) and P/P(k):
| field      | r: 0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0                | P/P: 0.2 ... 1.0 |
|------------|----------------------------------------------------------------|------------------|
| native 64  | 1.000 .999 .998 .996 .992 .988 .980 .974 .967                  | 0.99 .99 1.00 1.00 1.01 1.02 1.01 1.01 1.03 |
| B0 base    | 0.999 .995 .991 .983 .974 .960 .942 .924 .908                  | 1.03 1.11 1.17 1.19 1.24 1.28 1.35 1.40 1.45 |
| B2 emulator| 0.999 .995 .991 .984 .975 .961 .943 .926 .908                  | 1.00 1.01 1.01 0.96 0.92 0.87 0.80 0.75 0.72 |
| B2 generat.| 0.998 .988 .976 .961 .945 .925 .897 .876 .854                  | 1.00 1.02 1.02 0.97 0.93 0.87 0.80 0.74 0.71 |
Per-mask r at 0.5/0.75/0.9 k_Ny,64: base MS .983/.950/.922 SS .980/.942/.913; emulator MS .983/.949/.919
SS .981/.941/.910; native 64: MS .995/.984/.974 SS .994/.977/.961. Lagrangian rms err (h_f) all/MS/SS:
base 0.308/0.372/0.276, emulator 0.354/0.430/0.315. J quantiles (1/5/25/50/75/95/99%): truth
[-2.92 -1.28 -0.16 0.17 1.37 6.14 11.05], base [-1.68 -0.68 -0.06 0.07 1.18 5.79 10.45] (compressed),
emulator [-3.05 -1.32 -0.17 0.17 1.38 6.13 10.98] (restored); J<0 fraction truth/base/emu 0.375/0.367/0.374;
mass at delta>100: truth 0.056, base 0.071, emulator 0.047, generative 0.048.
Findings:
1. **The residual flow adds NO phase information**: its r_delta(k) equals the base's at every k to the third
   digit (0.926 vs 0.924 at 0.9 k_Ny,64, 0.908 vs 0.908 at k_Ny). It only rescales power (1.45 -> 0.72 at
   k_Ny,64) and restores the J distribution and the multi-stream fraction. The r ceiling is the base's, i.e.
   the conditional mean's, i.e. the information limit of the inputs (P Psi_c, D_ij, true IC octave).
2. **The ceiling is spatially uniform**: inside and outside the multi-stream mask the base/emulator r differ
   by < 0.01 (MS 0.919 vs SS 0.910 at 0.9 k_Ny,64); the native 64 run also loses coherence equally in both
   (0.974/0.961). Not a halo/multi-stream phenomenon.
3. **It sits in the octave band**: r >= 0.99 for k < 0.4 k_Ny,64 and starts falling exactly at k_Ny,32 = 0.5
   k_Ny,64, where the model has to synthesise the detail (Phase 0: linear octave r^2 <= 0.29, coarse run's
   own error 0.57 at 0.94 k_Ny,c). The emulator has the TRUE IC octave and still reaches only r_lag 0.83,
   so the missing ~30% of detail variance is not the IC octave: it is the part of the detail that the
   coarse run's O(1)-wrong Nyquist band does not determine (the "chain error goes upstream" finding).
4. **The density deficit is the price of incoherent added power**: Lagrangian P/P = 1.00 at r_lag = 0.80
   means 36% of the octave-band variance is noise w.r.t. the truth; per the notes it damps P_delta by
   exp(-k^2 sigma_n^2), giving the 0.72 at k_Ny,64 (base: no noise, but shrinkage -> +45% and 0.071 mass at
   delta > 100). A sampler from the TRUE conditional would have P_delta/P_true = 1 at the same r; ours does not
   -> the flow's conditional is too Gaussian in the density sense (adds displacement power that does not
   form the missing structure), even though its J quantiles are right.
Implications for C1/C2 (the representation route): a bigger/better base can only move the ceiling through
the input information (receptive field over the coarse run's error band); the flow's job is the conditional's
density coherence, which the qj form alone does not enforce — the Eulerian quadratic form Q_E (approved) on
the residual flow is the natural next knob, with the generative-mode density as the readout.
Provenance: runs/RF_resflow_psc/diag_rceiling.{json,png}; runs/diag_rceiling.py.

## 2026-09-16 22:30 — C1 on the PSC series (production split: train sets 0-13, test 14), 64->128

`hpc/run_phaseC.sh` (commit 9a99970; --octave-sampler full as in the laptop 64->128 runs; batch 1, base 24,
3000 steps). A100 timings: 64->128 regression 0.78 s/step (40 min), residual flow 0.90 s/step (45 min).
| run (set14, 64->128)                  | r_lag | P/P_lag | rms/h_f | Eul vs native 128 @ (kNy64, 1.5, 2) | laptop (s8, selfsim) |
|---------------------------------------|-------|---------|---------|-------------------------------------|----------------------|
| C1a F_reg_128_psc (base, 1 Euler)     | 0.728 | 0.542   | 0.485   | 1.645 / 2.189 / 2.603               | 0.729 / 0.539 / 0.480 |
| C1b RF_resflow_128_psc emulator       | 0.677 | 0.993   | 0.570   | 0.978 / 0.821 / 0.602               | 0.677 / 0.970 / 0.563 |
| C1b generative                        | 0.240 | 0.957   | 0.772   | 0.982 / 0.819 / 0.591               | 0.239 / 0.931 / 0.761 |
-> the laptop 64->128 numbers are reproduced on the production series to the third digit (r), so the
selfsim -> PSC move costs nothing and 14 training seeds instead of 8 change nothing at this capacity.
lambda auto (C1b): see train.log. Multi-stream fraction of the 64 test run (set14): 0.374.

Diagnostic `runs/diag_rceiling.py runs/RF_resflow_128_psc --set 14 --nc 128` (reference = native 256, converged
to 0.1% in the 128 band, A.4): r_delta at k/k_Ny,128 = 0.2..1.0:
| field      | r: 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0                        | P/P @0.9 | MS/SS r@0.9 | m(d>100) |
|------------|---------------------------------------------------------------|----------|-------------|----------|
| native 128 | 1.000 .999 .998 .996 .993 .990 .985 .979 .973                | 0.97     | .978/.974   | 0.161    |
| C1a base   | 0.998 .994 .983 .969 .955 .935 .918 .898 .876                | 2.30     | .897/.893   | 0.193    |
| C1b emu    | 0.999 .995 .986 .974 .961 .942 .926 .904 .882                | 0.69     | .900/.896   | 0.142    |
| C1b gen    | 0.998 .993 .981 .968 .952 .930 .910 .884 .858                | 0.68     | .881/.873   | 0.142    |
Same three facts as at 32->64, one level harder: the flow's r_delta(k) tracks the base's (+0.006 at 0.9
k_Ny,128 here, i.e. essentially no phase gain), the ceiling is spatially uniform (MS/SS within 0.005), and it
starts at the coarse Nyquist (r >= 0.99 below 0.4 k_Ny,128). The base's density excess is now +130% at 0.9
k_Ny,128 (shrinkage in a 37%-multi-stream volume) and the flow's deficit -31%; J<0 fraction truth/base/emu
0.418/0.412/0.419 (restored by the flow again). Provenance: runs/RF_resflow_128_psc/diag_rceiling.{json,png}.

## 2026-09-16 23:10 — Owner's course notes §1 + §4 launched (martingale structure in resolution)

Owner reviewed the course<->research mapping (Wiener process / filtration / martingale / L2 projection)
and chose two items. Session's corrections recorded there: (i) two ceilings must be distinguished —
Var(fine | F_n) with F_n = sigma(modes k < k_n) bounds the GENERATIVE mode (coarse-only information);
the EMULATOR has the true octave, so its ceiling (r_lag 0.83, r_delta 0.92; identical for base and flow,
spatially uniform, RUNLOG 21:10) is the learnability of the chaotic IC -> z=0 map given the coarse run's
O(1) Nyquist error, not an information limit; (ii) cube vs sphere windows both give exactly independent
Gaussian increments (disjoint k-sets), the correlated-step issue belongs to real-space tophat filters;
(iii) the excursion-set walk is defined on the linear field, so it does not transfer to the z=0 SR field.

§1 — TRUE conditional samples from the simulator (`runs/resample_octave_ic.py --set 14 --nsamples 8`):
real 128 IC of set14, coarse band (64-cube) kept, octave (128-cube minus 64-cube, Nyquist planes zero)
replaced by unit-modulus random phases with the shell-mean amplitude A(k) (the real IC's per-mode
|delta|/A spreads by 0.27 — GenIC's UnitaryAmplitude is not exact on the 2LPT displacement; the shell
POWER is matched). Zel'dovich at z=99, q = (i+1/2) h, IDs 1-based C, header copied from the real IC;
v = 0.52776104 Psi measured on the set14 64 IC (residual 8e-8: the real ICs are pure Zel'dovich in
velocity). Controls: ctrl_exact (real Psi through the writer), ctrl_zel (real modes re-Zel'dovich'd,
|dPsi| = 4.2% rms: 2LPT + Nyquist planes + folded content). Coarse band identical across samples to 2e-7.
MP-Gadget: `sim_scripts/.../dmo-128-resample/{ctrl_exact,ctrl_zel,oct0..7}` (dmo-128/set14 params,
OutputList = 1 only), RM job 1233922, 14 ranks. Outputs -> `sim_output/.../dmo-128-resample/set14/`.
Analysis to follow: per-shell Var(fine | F_n) and predictable fraction, the empirical conditional mean vs
the learned base F_reg_128_psc, and P_delta/P_true, r_delta of each TRUE conditional sample vs the real run
(the reference for the flow's generative deficit, RUNLOG 21:10 finding 4).

§4 — chain martingale test (`runs/chain_martingale.py --set 14`, CPU, running): 32->64->128 with the
production-split resflow pairs, correction-band regression slopes beta/gamma/kappa/gamma_D per shell and
the orthogonality check E|e_chain|^2 vs E|e1|^2 + E|eps2|^2; fields dumped for eval_vs_ref vs native 256.

## 2026-09-16 23:50 — §4 result: the chain error is NOT a martingale-difference sum; it is the persistence of the upstream low-k error

`python runs/chain_martingale.py --set 14 --out runs/chain_psc14` (CPU, 35 min; commit 76b5a3d). 32->64->128 with the
production-split resflow pairs, emulator mode at both steps, test set 14. Per-shell regression slopes in the
CORRECTION band (64 band of the 128 grid; R = cube restriction 128->64; e1 = step-1 output - true 64;
eps2 = (R chained - R true128) - e1; e_D = R direct - R true128):
| k/kNy,64 | beta (pass-through of e1) | gamma (eps2 on e1) | kappa (eps2 on step-1 output) | gamma_D (e_D on true 64) | rho (eps2 on e_D) | P_e1 | P_eps2 | P_e_chain | P_e_D |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.850 | -0.154 | -0.006 | -0.001 | 0.85 | 1.31e12 | 3.49e11 | 1.26e12 | 1.97e11 |
| 0.50 | 0.810 | -0.224 | -0.039 |  0.002 | 0.77 | 6.55e11 | 2.66e11 | 6.28e11 | 1.21e11 |
| 0.75 | 0.473 | -0.592 | -0.158 |  0.003 | 0.81 | 2.31e11 | 2.02e11 | 1.59e11 | 7.06e10 |
| 0.91 | 0.322 | -0.751 | -0.265 |  0.000 | 0.83 | 1.76e11 | 1.81e11 | 9.28e10 | 5.51e10 |
| 0.97 | 0.278 | -0.792 | -0.299 |  0.004 | 0.84 | 1.68e11 | 1.76e11 | 7.77e10 | 5.09e10 |
Band totals (k < 0.97 kNy,64): E|e1|^2 = 4.46e16, E|eps2|^2 = 2.82e16, sum 7.28e16, E|e_chain|^2 = 3.67e16
(2<e1,eps2> = -3.61e16), direct step E|e_D|^2 = 1.07e16. Step-1 output vs true 64: P_err/P_true = 0.02 / 0.22 /
0.43 / 0.60 / 0.67 at 0.25 / 0.5 / 0.75 / 0.91 / 0.97 kNy,64 (r 0.990 / 0.888 / 0.781 / 0.694 / 0.659), P/P 0.96-1.00.
Octave band of the 128 grid, error power / P_true at 1.09 / 1.5 / 1.91 kNy,64: direct 0.35 / 0.65 / 0.93,
chained 0.47 / 0.79 / 1.06. Density vs native 256 (eval_vs_ref, k < kNy,128): direct 0.969 / 0.824 / 0.688 at
(kNy,64, 0.75, 0.9 kNy,128), r_delta@0.9 = 0.904; CHAINED 1.033 / 0.849 / 0.687, r_delta@0.9 = 0.716 (laptop s9 chain:
1.032 / 0.832 / 0.560 at kNy64 / 1.5 kNy64 / kNy128 — same picture).
Reading:
1. gamma_D ~ 0 at every k: given its OWN input the 64->128 step is conditionally unbiased (its error is a
   martingale difference w.r.t. the input, in the linear-regression sense). That is exactly why it cannot repair
   an upstream error it cannot tell from signal: at k < 0.5 kNy,64 the step-1 error passes through with beta =
   0.81-0.85 and the innovation is small (kappa ~ 0).
2. Near the coarse Nyquist (0.75-0.97 kNy,64) beta drops to 0.47-0.28 and gamma to -0.6..-0.8: this is the step's
   correction band, where it is trained to overwrite the coarse run's O(1) Nyquist error, so it overwrites the
   upstream error too (70% repaired). The chain error is SUB-additive there (E|e_chain|^2 < E|e1|^2).
3. Therefore the chain does NOT accumulate as a random walk (sqrt N) nor as an unbiased sum: the coarse-band chain
   error is 3.4x the direct step's own error, and it is DOMINATED by the persistence of step 1's error at
   k < 0.5 kNy,64 (P_err/P = 0.02-0.22 there, r 0.99-0.89), which every later step will carry unchanged (beta ~ 0.85
   per step -> geometric persistence, error grows with the number of steps in which that band is "coarse band").
   rho = 0.8: the innovation on a chained input is the direct step's own error — the steps make correlated
   errors, another reason the martingale accounting fails. Consequence for 64->512: the 64-band error of the
   first step is the floor of the whole chain; fixing it upstream (the 32->64 step's r = 0.66 at its Nyquist
   -> 0.89 at half its Nyquist) buys more than anything downstream, as the RFT2 verdict already said.
4. Phase loss compounds through the octave: r_delta at 0.9 kNy,128 falls from 0.904 (direct) to 0.716 (chained) —
   the new octave is synthesised from a coarse field whose own top band has r 0.66.
Provenance: runs/chain_psc14/chain_martingale.json, runs/chain_psc14/{direct,chained}/vs_ref.json, runs/chain_psc14.log.

## 2026-09-17 00:00 — §1 result: the simulator's own conditional samples — Var(fine | F_n), the two ceilings, and the density theorem

10 MP-Gadget 128^3 runs of set14 (RM array 1235607/1236177/1236641, ~30 min each on 14 ranks; MaxMemSizePerNode had
to be pinned to 80000 MB because the default 0.6 x node memory exceeds the job cgroup on 257 GB nodes): ctrl_exact,
ctrl_zel, oct0..7 (coarse band of the real IC kept, octave resampled; runs/resample_octave_ic.py). Converted with the
validated convention; `runs/condvar_analysis.py --set 14 --sim .../dmo-128-resample/set14 --model-run runs/RF_resflow_128_psc`
-> runs/condvar_set14.{json,png,log}. Reference = the real 128 run of set14; model = C1 (F_reg_128_psc base +
RF_resflow_128_psc), emulator mode = true octave, generative = sampled octave.

Lagrangian, 128 grid, per shell (k/k_Ny,64; 1..2 = the octave):
| quantity                                   | 0.50  | 0.90  | 1.10  | 1.25  | 1.50  | 1.75  | 1.95  |
|--------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
| Var(fine|F_n)/P_real  (unpredictable frac.) | 0.010 | 0.094 | 0.304 | 0.455 | 0.604 | 0.685 | 0.711 |
| r(conditional mean, real)  [8 samples]      | 0.994 | 0.944 | 0.811 | 0.697 | 0.564 | 0.475 | 0.419 |
| r(true sample, real)  mean                  | 0.990 | 0.904 | 0.695 | 0.539 | 0.387 | 0.297 | 0.249 |
| r(sample_i, sample_j)  mean                 | 0.990 | 0.906 | 0.696 | 0.542 | 0.394 | 0.304 | 0.261 |
| P_sample / P_real  mean                     | 1.000 | 0.998 | 1.000 | 0.994 | 0.996 | 0.983 | 0.961 |
| r(learned base, real)  [emulator: true octave] | 0.985 | 0.901 | 0.831 | 0.792 | 0.727 | 0.638 | 0.591 |
| r(learned base, conditional mean)           | 0.987 | 0.896 | 0.694 | 0.549 | 0.400 | 0.282 | 0.213 |
| r(flow emulator, real)                      | 0.981 | 0.876 | 0.793 | 0.748 | 0.671 | 0.572 | 0.519 |
| r(flow generative, real)                    | 0.971 | 0.769 | 0.537 | 0.376 | 0.239 | 0.146 | 0.105 |
| r(ctrl_exact, real)  [same IC rerun]        | 0.999 | 0.994 | 0.987 | 0.978 | 0.956 | 0.932 | 0.915 |
| r(ctrl_zel, real)                           | 0.999 | 0.989 | 0.975 | 0.958 | 0.920 | 0.873 | 0.841 |
Real-space: ctrl_exact vs real rel-rms 3.5% (corr 0.9994), ctrl_zel 4.7%, resampled octave 11% (corr 0.994).

Eulerian density vs the real 128 run, k < k_Ny,128, P/P and r_delta at (k_Ny,64 / 0.75 / 0.9 k_Ny,128):
| field                          | P_delta/P_true            | r_delta             |
|--------------------------------|---------------------------|---------------------|
| TRUE conditional samples (8)   | 1.006+-0.009 / 1.014+-0.014 / 1.008+-0.017 | 0.985 / 0.959 / 0.938 (+-0.001..0.004) |
| ctrl_exact                     | 1.000 / 1.000 / 1.001     | 1.000 / 0.999 / 0.998 |
| ctrl_zel                       | 0.994 / 1.001 / 1.003     | 0.999 / 0.997 / 0.994 |
| C1 base (regression)           | 1.638 / 2.139 / 2.365     | 0.972 / 0.932 / 0.904 |
| C1 residual flow, emulator     | 0.984 / 0.839 / 0.708     | 0.976 / 0.938 / 0.909 |
| C1 residual flow, generative   | 0.987 / 0.836 / 0.698     | 0.969 / 0.922 / 0.887 |

Findings:
1. **Var(fine | F_n) measured**: the coarse band is 99% predictable at 0.5 k_Ny,64 and 91% at 0.9 k_Ny,64 (the
   coarse run's own Nyquist error is mostly a deterministic function of the coarse modes); the octave is 70%
   predictable at its bottom and 29% at its top. The conditional mean given the coarse band alone reaches
   r = 0.81 -> 0.42 across the octave, a true conditional sample r = 0.70 -> 0.25, and the flow's generative
   samples r = 0.54 -> 0.10: the flow's generative mode uses LESS of the coarse information than the simulator's
   own conditional (it is under-informed, not over-dispersed: its P/P is 0.96 in the Lagrangian octave band).
2. **Two ceilings, now both measured**. Generative (F_n): the conditional-mean r above. Emulator (F_{n+1}, true
   octave): the learned base already beats the F_n conditional mean at the top of the octave (0.59 vs 0.42 —
   that is the IC octave's information), but the simulator's own reproducibility ceiling from the SAME IC is
   r = 0.987 -> 0.915 across the octave (ctrl_exact: rank count / sync points / chaos), so the emulator's 0.79 -> 0.52
   is a MODEL limit with 0.2-0.4 of headroom in r, not a physics floor; in real space the unlearnable floor is
   3.5% rms (multi-stream chaos), the emulator sits at 57%.
3. **The density theorem holds in the simulator**: every true conditional sample has P_delta/P_true = 1.01 +- 0.02 at
   all three probes while its r_delta is 0.94 at 0.9 k_Ny,128 — LOWER coherence than our emulator (0.909 is only
   0.03 below) yet no deficit at all. The flow's -30% at the same k is therefore not the price of r < 1; it is the
   flow's conditional being wrong in the density sense (its added octave power does not build the missing
   structure; RUNLOG 21:10 finding 4 confirmed against ground truth). The base's +137% is shrinkage, as before.
4. **Density is the robust observable, Lagrangian displacement inside halos is not**: the same-IC rerun differs by
   3.5% rms in Psi (r 0.915 at k_Ny,128) but by 0.1% in P_delta with r_delta 0.998. Per-realization Lagrangian
   metrics near the fine Nyquist compare against an object that is itself only reproducible to r ~ 0.92.
5. ctrl_zel: the Zel'dovich / Nyquist-plane / folded-content difference (4.2% in the IC) costs r 0.84 at the top of
   the octave and 4.7% rms at z=0 but nothing in density (0.994-1.003): the resampled runs are on the same footing
   as the real one for every density statement, and slightly disadvantaged (P 0.96 at the octave top) for
   Lagrangian ones.
What this changes: (a) the target for the generative flow is now numerical — r_delta 0.94 with P_delta/P = 1.00
at 0.9 k_Ny,128 exists and is reachable in principle; (b) the emulator's ceiling (0.91) is not the physics floor
(0.998 in density); (c) both the base's excess and the flow's deficit are properties of the learned conditional,
and the true conditional's samples are the natural training/validation targets for whatever fixes it
(e.g. the Q_E form on the residual flow, or training on multiple true conditional samples per coarse box).
Provenance: runs/condvar_set14.{json,png}, runs/condvar_set14.log; ICs and sims under sim_output/.../dmo-128-resample.

## 2026-09-17 00:50 — C2: ONE weight-shared two-stage operator for 32->64 AND 64->128 matches or beats the specialists with half the parameters

`runs/multi_resflow_train.py` (commit 5fddead; s_l = ln sigma_c = per-level style scalar, steps alternate between the
levels, 3000 per level, batch 2 / 1, lambda per level auto at its first step and pinned): C2a `--mode base`
(runs/M_reg_psc, HENON A100, 0.46 s/step avg, 46 min), then in parallel C2b `--mode resflow` on the frozen shared
base (runs/M_resflow_psc, HENON) and C2c `--mode reg2` (runs/M_reg2_psc, TWIG A100, owner allowed a second card
2026-09-16 23:30; hpc/slurm_hold_gpu_twig.sh, job 1234452), 0.55 s/step, 55 min each. Same production split as C1
(train 0-13, test 14). Specialists for comparison: S32a/S32b (runs/R_reg_psc14, RF_resflow_psc14) and C1a/C1b
(runs/F_reg_128_psc, RF_resflow_128_psc). lambda: shared resflow 0.0164 (32->64) / 0.0123 (64->128).

Lagrangian octave band vs the native fine run (emulator; generative in parentheses):
| model (set14)                 | 32->64: r / P/P / rms         | 64->128: r / P/P / rms        | params |
|-------------------------------|-------------------------------|-------------------------------|--------|
| specialist base (S32a / C1a)  | 0.831 / 0.707 / 0.307         | 0.728 / 0.542 / 0.485         | 2 x 1.56M |
| SHARED base (C2a)             | 0.839 / 0.714 / 0.299         | 0.737 / 0.552 / 0.479         | 1.56M  |
| specialist resflow (S32b/C1b) | 0.796 / 0.992 / 0.352 (0.204) | 0.677 / 0.993 / 0.570 (0.240) | 2 x 3.11M |
| SHARED resflow (C2b)          | 0.806 / 1.004 / 0.343 (0.205) | 0.685 / 0.982 / 0.561 (0.248) | 3.11M  |
| SHARED reg2 control (C2c)     | 0.855 / 0.742 / 0.288 (0.239) | 0.755 / 0.575 / 0.468 (0.306) | 3.11M  |

Eulerian density vs the CONVERGED reference (native 2c; eval_vs_ref, k < k_Ny,c), P/P at (k_Ny,c/2, 0.75, 0.9 k_Ny,c)
and r_delta at 0.9 k_Ny,c:
| model (set14)                 | 32->64 vs native 128                 | 64->128 vs native 256                |
|-------------------------------|--------------------------------------|--------------------------------------|
| native c run (truth)          | 0.986 / 0.987 / 0.998   r 0.969      | 0.985 / 0.982 / 0.972   r 0.979      |
| specialist base               | 1.207 / 1.332 / 1.455   r 0.904      | 1.614 / 2.101 / 2.299   r 0.898      |
| SHARED base                   | 1.205 / 1.368 / 1.518   r 0.905      | 1.624 / 2.129 / 2.342   r 0.900      |
| specialist resflow emulator   | 0.984 / 0.872 / 0.805   r 0.906      | 0.969 / 0.824 / 0.688   r 0.904      |
| SHARED resflow emulator       | 0.983 / 0.903 / 0.853   r 0.907      | 0.986 / 0.857 / 0.728   r 0.906      |
| SHARED resflow generative     | 0.975 / 0.885 / 0.831   r 0.852      | 0.989 / 0.848 / 0.715   r 0.885      |
| SHARED reg2 control emulator  | 1.239 / 1.486 / 1.701   r 0.905      | 1.690 / 2.322 / 2.633   r 0.900      |
Verdict:
1. Weight sharing across the two transitions costs nothing and gains a little at BOTH levels, for the base (r +0.008
   / +0.009) and for the residual flow (r +0.010 / +0.008, density deficit at 0.9 k_Ny,c reduced from -19.5% to
   -14.7% at 32->64 and from -31% to -27% at 64->128), with half the parameters of two specialist pairs. The
   sigma-matched style scalar (RUNLOG 2026-09-09) does its job; this is the operator the 64->512 chain should use.
2. The reg2 control separates the two-stage structure from the flow: a second regression stage raises r by 0.02 at
   both levels (the best r of any model here, 0.855 / 0.755) but pushes the density excess further UP (1.70 / 2.63
   at 0.9 k_Ny,c, worse than the base) — the second conditional mean shrinks again. The flow's entire contribution
   is the power (from +52% / +134% to -15% / -27%), with the same r_delta (0.905-0.907) as every regression:
   the r ceiling diagnostic (RUNLOG 21:10) holds for the shared operator too.
3. Against the §1 ground truth: the true conditional samples sit at P/P = 1.01 with r_delta 0.938 at 0.9 k_Ny,128;
   the best learned sampler (shared resflow, generative) is at 0.715 with r_delta 0.885. The gap is the flow's
   conditional, not information (RUNLOG 00:00).
Parameter accounting: shared base + shared flow = 3.11M for both levels vs 6.22M for the specialist pairs; reg2 = 3.11M.
Provenance: runs/M_{reg,resflow,reg2}_psc/results_{32to64,64to128}.json and */{32to64,64to128}/vs_ref.json.

## 2026-09-17 10:50 — Second box (set15) for the shared operator, and the hybrid-detail test: density power = COHERENCE between coarse band and detail

Correction first: the previous entry's/HANDOFF's suggestion "Q_E on the residual flow" is withdrawn — REPORT_7 already
refuted Q_E on the single-stage flow (P_delta moved DOWN 0.963 -> 0.905; Q_J won). `--cic-weight` was abandoned on the
owner's reasoning (RUNLOG 2026-09-10: deposit-kernel ceiling, Paper-IV experience); `--jac-weight` works monotonically
but modestly (0.777 -> 0.845 at 1.5 k_Ny,c). No new training was launched in this entry.

(1) Second box. `runs/multi_resflow_train.py --eval-only` (new flag, inference only on the HENON A100) on test set 15:
| shared operator, emulator     | 32->64 vs native 128 (P/P @kNy32/0.75/0.9 kNy64, r@0.9) | 64->128 vs native 256            |
|-------------------------------|----------------------------------------------------------|----------------------------------|
| native run, set14 / set15     | .986/.987/.998 r .969  /  .977/.976/.971 r .972          | .985/.982/.972 r .979 / .983/.978/.973 r .977 |
| shared base, set14 / set15    | 1.205/1.368/1.518 r .905 / 1.206/1.342/1.487 r .929      | 1.624/2.129/2.342 r .900 / 1.679/2.196/2.474 r .899 |
| shared resflow, set14 / set15 | .983/.903/.853 r .907  /  .975/.869/.801 r .935          | .986/.857/.728 r .906 / 1.010/.887/.769 r .912 |
| shared resflow gen., 14 / 15  | .975/.885/.831 r .852  /  .966/.834/.785 r .860          | .989/.848/.715 r .885 / 1.001/.866/.752 r .888 |
| shared reg2, set14 / set15    | 1.239/1.486/1.701 r .905 / 1.233/1.445/1.644 r .933      | 1.690/2.322/2.633 r .900 / (set15: Lagr. r .760 P/P .582, Eul vs 128 1.77/2.43/3.08) |
Lagrangian set15: shared base .844/.722/.293 and .743/.559/.472; shared resflow .815/1.008/.336 and .690/.995/.557.
-> every C2 statement holds on the second box; box-to-box scatter is +-0.05 in P/P and +-0.03 in r_delta.

(2) Hybrid-detail test (`runs/hybrid_detail_test.py --set 14 --model-run runs/RF_resflow_128_psc --sim .../dmo-128-resample/set14`;
training-free; fine grid 128, W = 64-cube; "low" = W band, "high" = (1-W) band; density vs the real 128 run,
P/P and r_delta at k_Ny,64 / 0.75 / 0.9 k_Ny,128; last columns: Lagrangian r and P/P of the detail vs the true detail):
| field                                  | P_delta/P_true        | r_delta            | detail r_lag | detail P/P |
|----------------------------------------|-----------------------|--------------------|------|------|
| T (sanity)                             | 1.000/1.000/1.000     | 1.000/1.000/1.000  | 1.00 | 1.00 |
| low(T) + NO detail                     | 1.164/1.134/1.063     | 0.986/0.964/0.948  | 0    | 0    |
| flow emulator, as is                   | 0.984/0.839/0.708     | 0.976/0.938/0.909  | 0.68 | 0.99 |
| low(T) + high(emulator)                | 0.933/0.752/0.622     | 0.993/0.982/0.972  | 0.68 | 0.99 |
| low(emulator) + high(T)                | 0.892/0.716/0.595     | 0.985/0.963/0.939  | 1.00 | 1.00 |
| low(T) + high(generative)              | 0.875/0.660/0.519     | 0.989/0.970/0.955  | 0.24 | 0.96 |
| base, as is                            | 1.638/2.139/2.365     | 0.972/0.932/0.904  | 0.73 | 0.54 |
| low(T) + high(base)                    | 1.336/1.461/1.477     | 0.992/0.980/0.971  | 0.73 | 0.54 |
| low(base) + high(T)                    | 1.030/0.927/0.825     | 0.987/0.967/0.946  | 1.00 | 1.00 |
| low(T) + PHASE-RANDOMISED high(T)      | 0.621/0.279/0.141     | 0.985/0.957/0.926  | 0.00 | 1.00 |
| low(T) + envelope-MODULATED random (sigma = 1 / 2 h_c) | 0.473/0.188/0.095 , 0.466/0.167/0.077 | 0.987/0.955/0.913 | 0.00 | 1.00 |
| true conditional sample oct0..2, as is | 1.00-1.02 / 1.00-1.03 / 1.01-1.02 | 0.985/0.958/0.937 | 0.37 | 0.99 |
| low(T) + high(oct_j)                   | 0.92/0.81-0.82/0.72-0.73 | 0.993/0.980/0.968 | 0.37 | 0.99 |
| low(oct_j) + high(T)                   | 0.92/0.81/0.72-0.73   | 0.994/0.980/0.968  | 1.00 | 1.00 |
(two PR/MOD realisations agree to 0.001.) Findings:
1. **The detail's job in the density is to CANCEL, not to add**: the true coarse band with no detail at all has an
   EXCESS (+6..16%) and r_delta 0.948 at 0.9 k_Ny,128 — higher than any model (0.909). The band-limited map is too
   sharp; the true detail takes that power out coherently (virialised halos of finite size).
2. **H1 confirmed and far stronger than expected**: a detail with exactly the right per-mode power but random
   phases destroys the density power (0.14 at 0.9 k_Ny,128). The flow (0.71) is therefore mostly coherent already;
   its deficit is the incoherent remainder.
3. **H2 REFUTED**: putting the random detail power where the true detail power sits (envelope modulation) makes it
   WORSE (0.08-0.10): the detail power lives in the halos, and incoherent displacement there smears exactly the
   particles that carry the power. The conditional-variance field is not the missing ingredient.
4. **H3 REFUTED, and the key fact**: the detail of a TRUE conditional sample — a genuine N-body solution with the
   right power, right non-Gaussianity, right everything — put on the true coarse band gives 0.92/0.81/0.73, i.e. the
   SAME -27% deficit as our flow, although its own coarse band differs from T's by only 1-9% in power (RUNLOG 00:00:
   Var/P 0.01-0.09 in the coarse band). And symmetrically low(oct_j)+high(T). The sample as a whole is at 1.01.
   => the density power is a property of the JOINT field: detail and coarse band must belong to the same solution
   at the percent level near the coarse Nyquist. A detail that is correct in distribution but not matched to the
   realised coarse band costs ~27%; the flow sits exactly at that level (0.71-0.73 "as is"), i.e. its detail is
   about as matched to its own coarse band as an independent conditional draw would be — and mixing the model's
   bands with the truth's is worse still (0.60-0.62), so the model's bands are partly matched to EACH OTHER.
5. r_delta in k < k_Ny,128 is carried by the coarse band: every hybrid with low(T) has r_delta 0.97 at 0.9 k_Ny,128
   whatever the detail (even random: 0.93), and the model "as is" loses its r_delta through its coarse band
   (low(emulator)+high(T): 0.939; emulator 0.909; base 0.904). The r ceiling of RUNLOG 21:10 is the CORRECTION band.
What this changes: the target is not "the right conditional for the detail" but "a field whose bands are mutually
consistent"; metrics to add to every evaluation: P_delta of low(model)+0 and of the band-swapped hybrids. The
coarse-band correction (r 0.90 at 0.9 k_Ny,c vs the 0.99 reproducibility) is where both the r_delta ceiling and,
through band mismatch, part of the power deficit originate — consistent with "fix it upstream" (RFT2, §4).
Provenance: runs/hybrid_set14.{json,log}; runs/M_{resflow,reg2}_psc_set15/.

Addendum (2026-09-17 10:55): shared reg2, set15, 64->128 vs native 256, emulator:  (the table row above had only the Lagrangian numbers when it was written).

## 2026-09-17 11:20 — Oracle-coarse diagnostic (32->64): the r_delta ceiling is 100% the correction band; the power deficit is only ~1/3 upstream

`--oracle-coarse` (commit a7b3948; DIAGNOSTIC, not deployable): the conditioning field is R_cube Psi_f (the fine run's own
coarse band = the Phase-B label Y) instead of the coarse run, in training AND test; everything else = S32a/S32b
(split 0-13/14, base 24, 3000 steps, batch 2, octave-sampler full; HENON A100, 0.12 / 0.16 s/step).
runs/O_reg_psc14 (regression base) -> runs/OF_resflow_psc14 (residual flow, qj, lambda auto).
| 32->64, set14                  | r_lag | P/P_lag | rms/h_f | P_delta/P vs native 128 @kNy32/0.75/0.9 kNy64 | r_delta@0.9 |
|--------------------------------|-------|---------|---------|-----------------------------------------------|-------------|
| native 64 run                  |   —   |    —    |    —    | 0.986 / 0.987 / 0.998                         | 0.969       |
| specialist base (coarse RUN)   | 0.831 | 0.707   | 0.307   | 1.207 / 1.332 / 1.455                         | 0.904       |
| ORACLE base                    | 0.894 | 0.790   | 0.204   | 1.148 / 1.279 / 1.385                         | 0.966       |
| specialist resflow (coarse RUN)| 0.796 | 0.992   | 0.352   | 0.984 / 0.872 / 0.805                         | 0.906       |
| shared resflow (coarse RUN)    | 0.806 | 1.004   | 0.343   | 0.983 / 0.903 / 0.853                         | 0.907       |
| ORACLE resflow, emulator       | 0.868 | 0.969   | 0.239   | 0.968 / 0.912 / 0.874                         | 0.967       |
| ORACLE resflow, generative     | 0.315 | 0.857   | 0.448   | 0.934 / 0.832 / 0.780                         | 0.946       |
(oracle x0: coarse band r = 1.0000, P_eps/P = 4e-9 as it must; after the base 1.4e-4.)
Reading against the pre-stated hypothesis ("if the density returns to ~1 the deficit is all upstream"):
1. r_delta: 0.906 -> 0.967, i.e. exactly the native 64 run's own coherence with 128 (0.969). The whole r_delta ceiling
   of every model in this project is the correction band (the coarse run's Nyquist error the model cannot undo);
   with a perfect coarse band the detail generator costs NO coherence in k < k_Ny,64. Hybrid finding 5 confirmed
   by training, not just by band-swapping.
2. Power: the deficit at 0.9 k_Ny,64 shrinks from -19.5% (specialist) / -14.7% (shared) to -12.6%. So the hypothesis is
   REFUTED as stated: only about one third of the deficit is upstream; two thirds remain with a PERFECT coarse band.
   They belong to the detail generator itself: r_lag 0.87 means 25% of its detail power is not matched to the (true)
   coarse band, and unmatched detail power smears (hybrid findings 2-4). The base (shrunk detail) overshoots by +38%,
   the flow (full power, part of it unmatched) undershoots by -13%: the truth needs full power AND matched phases.
3. Lagrangian: the oracle raises r_lag from 0.80 to 0.87 and halves the rms error (0.352 -> 0.239 h_f): the coarse run's
   error is the single largest error source of the 32->64 step in every metric.
Consequences: (a) upstream first — a better correction band buys all of r_delta and a third of the power; the label
R Psi_2c of Phase B is the right TARGET for a dedicated correction stage (Phase B tested it as a label for the whole
base, where it was diluted; as the conditioning field it is worth +0.06 in r_delta); (b) the remaining -13% needs the
detail to be phase-matched to the coarse band beyond what MSE/flow matching with the Q_J form delivers; the approved
`--jac-weight` moved the single-stage flow monotonically in the right direction (RUNLOG 2026-09-10) and has not been
tried on the residual flow or with an oracle coarse band — that is the cheapest next probe (15 min at 32->64).
Provenance: runs/O_reg_psc14/results.json, runs/OF_resflow_psc14/{results_resflow,vs_ref}.json.

## 2026-09-17 12:10 — Option 1 (owner): `--jac-weight` on the residual flow, 32->64 dose-response — real, monotonic, shallow

`runs/resflow_train.py --jac-weight w` (commit 526af7d): oft.jac_loss (MSE on asinh J) on the flow's endpoint prediction
x1p = x_t + (1-t) v, ON TOP of the Q_J form (lambda auto ~0.0156). Same frozen base (runs/R_reg_psc14), split 0-13/14,
3000 steps, batch 2, HENON A100 (0.16-0.17 s/step); control = runs/RF_resflow_psc14 (w = 0). runs/RFJ_resflow_psc14_j*.
Density vs native 128 (set14), P/P at k_Ny,32 / 0.75 / 0.9 k_Ny,64:
| jac-weight | w*L_jac / L_flow (end) | emulator              | r_delta@0.9 | generative            | r_lag | octave P/P_lag | rms/h_f |
|------------|------------------------|-----------------------|-------------|-----------------------|-------|----------------|---------|
| 0          | —                      | 0.984 / 0.872 / 0.805 | 0.906       | 0.984 / 0.868 / 0.799 | 0.796 | 0.992          | 0.352   |
| 0.012      | 0.013                  | 0.986 / 0.877 / 0.812 | 0.904       | 0.988 / 0.872 / 0.805 | 0.797 | 0.992          | 0.352   |
| 0.04       | 0.044                  | 0.985 / 0.872 / 0.807 | 0.907       | 0.987 / 0.869 / 0.799 | 0.797 | 0.992          | 0.352   |
| 0.12       | 0.13                   | 0.995 / 0.889 / 0.827 | 0.906       | 0.995 / 0.884 / 0.816 | 0.797 | 0.983          | 0.351   |
| 0.4        | 0.43                   | 0.994 / 0.890 / 0.828 | 0.907       | 0.998 / 0.890 / 0.826 | 0.799 | 0.978          | 0.349   |
| 1.2        | 1.26                   | 1.011 / 0.916 / 0.860 | 0.908       | 1.011 / 0.910 / 0.852 | 0.802 | 0.966          | 0.345   |
(generative octave P/P_lag: 0.966 / 0.964 / 0.966 / 0.957 / 0.950 / 0.938.)
Reading:
* The biased term works in the right direction and monotonically over two decades (below w ~ 0.05 it is inside the
  run-to-run noise, +-0.005), with no Lagrangian cost — r_lag and rms even improve slightly (0.796 -> 0.802, 0.352 ->
  0.345). The bias signature of notes 5.5/5.7d is there and small: the Lagrangian octave P/P drifts 0.992 -> 0.966
  (emulator) and 0.966 -> 0.938 (generative), and the density at k_Ny,32 starts to exceed 1 (1.011) at w = 1.2.
* But the lever is SHALLOW: ~+0.02 per decade of weight at 0.9 k_Ny,64. With the jac term already LARGER than the flow
  loss (w = 1.2) the deficit goes from -19.5% to -14.0%: a quarter of the gap, the same gain that weight sharing
  alone gives (shared resflow 0.853 with w = 0). Extrapolating the log-slope, closing the gap would need w ~ 10^3.
  On the laptop's single-stage flow (no Q_J form) w = 0.04 moved the same probe by +0.04..0.07; here the Q_J form
  has already taken that gain, and the jac loss adds little on top.
* r_delta does not move (0.904-0.908), as the oracle-coarse diagnostic predicts: it lives in the correction band.
Verdict: `--jac-weight` is not the fix for the density deficit of the residual flow. It is a harmless +0.02..0.05 that
can be stacked (w ~ 0.4-1.2), not a route to P_delta/P = 1. The oracle-coarse budget stands: all of r_delta and about
a third of the power are upstream (correction band); the remaining ~ -13% is detail/coarse phase matching that a
pointwise-J loss on the endpoint does not enforce.

## 2026-09-17 16:30 — Two probes (owner-approved plan of the session): correction-stage headroom = NONE; `--cic-weight` on the residual flow = WRONG SIGN

`hpc/run_cic_probe.sh` (commit 80b652c + launcher), 32->64, split 0-13/14, HENON A100, one run at a time.

(1) Headroom test for a dedicated correction stage: `octave_flow_toy.py --regression --loss-band low` (MSE on the coarse
band of the error only; runs/G_reg_psc14_lowband) vs the normal base (runs/R_reg_psc14):
| model                         | coarse band r | coarse band P_eps/P |
|-------------------------------|---------------|---------------------|
| x0 (no network)               | 0.9458        | 0.109               |
| base, loss on all k           | 0.9858        | 0.0280              |
| base, loss on the coarse band | 0.9862        | 0.0272              |
-> giving the network's whole capacity to the correction band buys 3% of the error power: NO-GO. The correction-band
error is not a capacity-sharing problem; it is what this network can infer from (P Psi_c, D_ij, octave). The oracle
bound (r_delta 0.967) is not reachable by dedicating a stage with the same inputs.

(2) `runs/resflow_train.py --cic-weight w --cic-compress {log1p,sqrt,lin} --cic-factor {1,2}`: MSE between compressed CIC
densities of q + x1_pred and q + x1 (eulerian_metric.cic_deposit, differentiable), on top of the Q_J form; frozen base
runs/R_reg_psc14; control = runs/RF_resflow_psc14. Density vs native 128 (set14), emulator, P/P at k_Ny,32/0.75/0.9 k_Ny,64:
| compress / mesh | w    | w*L_cic / L_flow (end) | P_delta/P             | r_delta@0.9 | r_lag | octave P/P_lag | generative P_delta/P @0.9 |
|-----------------|------|------------------------|-----------------------|-------------|-------|----------------|---------------------------|
| control         | 0    | —                      | 0.984 / 0.872 / 0.805 | 0.906       | 0.796 | 0.992          | 0.799                     |
| log1p, f1       | 3    | 0.14                   | 0.970 / 0.850 / 0.779 | 0.904       | 0.796 | 0.998          | 0.775                     |
| log1p, f1       | 30   | 1.3                    | 0.883 / 0.708 / 0.613 | 0.906       | 0.790 | 1.022          | 0.612                     |
| log1p, f2       | 1    | 0.36                   | 0.962 / 0.838 / 0.764 | 0.907       | 0.796 | 0.992          | 0.759                     |
| log1p, f2       | 10   | 3.3                    | 0.917 / 0.755 / 0.661 | 0.906       | 0.792 | 1.006          | 0.654                     |
| sqrt, f1        | 3    | 0.24                   | 0.906 / 0.745 / 0.656 | 0.905       | 0.791 | 1.008          | 0.653                     |
| sqrt, f1        | 30   | 2.1                    | 0.792 / 0.570 / 0.458 | 0.895       | 0.783 | 1.044          | 0.460                     |
| lin, f1         | 0.03 | 0.09                   | 0.824 / 0.609 / 0.501 | 0.895       | 0.789 | 1.028          | 0.504                     |
| lin, f1         | 0.3  | 0.88                   | 0.739 / 0.506 / 0.397 | 0.876       | 0.785 | 1.101          | 0.404                     |
Hypothesis of the session ("the deposit sees the band matching that pointwise J cannot; expect >= +0.10 at 0.9 k_Ny,64"):
REFUTED, with the opposite sign, monotonically in the weight, in every compression, and the more peak-weighted the
compression the worse (lin at w = 0.03, with the term only 9% of the flow loss, already costs -0.30). Mechanism, in
hindsight: a pointwise MSE on the density is itself a regression to the mean in Eulerian space — when a halo's position
and shape are uncertain, a sharp peak slightly misplaced is penalised twice, so the loss prefers SMEARED density. The
network obliges in the cheapest way: it ADDS incoherent displacement power (Lagrangian octave P/P rises to 1.02-1.10
while P_delta falls), exactly the smearing mechanism of the hybrid-detail test. The 2x mesh changes nothing of substance.
This is consistent with the owner's 2026-09-10 decision to drop the CIC loss; the session's argument for reopening it
overlooked the double-penalty effect. `--cic-weight` stays in the script as an option, default 0.

State of the loss-shaping ledger for the one-shot map (32->64, 0.9 k_Ny,64, control 0.805): Q_J form (in the control;
the winner of Stage 7), weight sharing +0.05, `--jac-weight` 1.2 +0.055, Q_E form negative (REPORT_7), CIC-MSE strongly
negative, Y64 label for the base 0, dedicated correction loss 0. Truth-level references: native 64 run 0.998, true
conditional samples 1.01 (64->128). No loss that is an expectation of a pointwise error closes the gap.
Provenance: runs/cic_probe.log, runs/G_reg_psc14_lowband/results.json, runs/RFC_resflow_psc14_*/{results_resflow,vs_ref}.json.

## 2026-09-17 17:40 — Capacity probe (base 48) and coarse-VELOCITY inputs at 32->64 (owner: "1和2一起做"): both real, both mild

Split 0-13/14, 3000 steps, batch 2, HENON A100; `hpc/run_vel_probe.sh` (commit d84942b). Velocity inputs (owner approved
2026-09-17, the "adding velocities" ask-first item, INPUT side only): 9 extra conditioning channels
d_i (v_c / aHf)_j - D_ij from the coarse run's peculiar velocity (prolonged like Psi_c; aHf = 0.0497884 km/s per kpc/h at
z=0 — checked on data: <v.Psi>/<Psi^2> = 0.049-0.052 at k < 0.07 k_Ny,64, so vel.npy is peculiar km/s at a=1), zero in
linear theory, augmented jointly with the displacements; default path bit-identical (20-step CPU test 3.0882e-02).
| regression base (set14)            | params | octave r | octave P/P | rms/h_f (MS/SS)     | coarse band r | coarse band P_eps/P |
|------------------------------------|--------|----------|------------|---------------------|---------------|---------------------|
| base 24 (runs/R_reg_psc14)         | 1.56M  | 0.831    | 0.707      | 0.307 (0.371/0.276) | 0.9858        | 0.0280              |
| base 24, loss on coarse band only  | 1.56M  | (0.640)  | (2.0)      | —                   | 0.9862        | 0.0272              |
| base 48 (runs/R_reg_psc14_b48)     | 5.85M  | 0.852    | 0.758      | 0.290 (0.353/0.259) | 0.9874        | 0.0249              |
| base 24 + velocity (runs/V_reg_psc14) | 1.56M | 0.836   | 0.712      | 0.302 (0.365/0.272) | 0.9865        | 0.0265              |
Residual flow on the velocity base, both stages with velocity inputs (runs/VF_resflow_psc14) vs the control pair:
| pair (set14)            | r_lag | octave P/P | rms   | P_delta/P vs native 128 @kNy32/0.75/0.9 kNy64 | r_delta@0.9 | generative P_delta/P, r_delta |
|-------------------------|-------|------------|-------|-----------------------------------------------|-------------|-------------------------------|
| control (no velocity)   | 0.796 | 0.992      | 0.352 | 0.984 / 0.872 / 0.805                         | 0.906       | 0.984/0.868/0.799, 0.85       |
| velocity inputs         | 0.803 | 0.994      | 0.347 | 0.949 / 0.821 / 0.754                         | 0.918       | 0.936/0.814/0.742, 0.867      |
| oracle coarse band      | 0.868 | 0.969      | 0.239 | 0.968 / 0.912 / 0.874                         | 0.967       | 0.934/0.832/0.780, 0.946      |
Reading:
* Capacity IS a lever, a mild one: 3.75x the parameters buy +0.021 in octave r, -11% in correction-band error power,
  -5.5% in rms. (The coarse-band-only loss bought -3%: it is total capacity, not its allocation, that matters.)
* The coarse velocity in this form is a smaller lever: +0.005 in r, -5% in correction-band error power. Through the
  residual flow it raises r_delta by +0.012 (0.906 -> 0.918; base 0.904 -> 0.917) — the first thing in this project,
  other than the oracle, that moves r_delta at all — but the density power does not improve (0.754 vs 0.805 at
  0.9 k_Ny,64; one run each, run-to-run noise ~0.01, so the drop is probably real and unexplained).
* Against the oracle bound (correction-band error -> 0 gives r_delta 0.967) both levers cover ~1/5 of the way in
  r_delta at best. The correction-band error of 2.5-2.8% is robust against capacity x3.75, loss allocation and the
  coarse velocity: most of it is genuinely not determined by the coarse run (the chaotic, IC-octave-dependent part of
  the coarse run's own Nyquist error), consistent with Var(fine | F_n) of RUNLOG 00:00 (1-9% across the band).
Not done (would complete the picture, owner's call): base 48 + velocity, and a residual flow on the base-48 base.
Provenance: runs/vel_probe.log, runs/{R_reg_psc14_b48,V_reg_psc14}/results.json, runs/VF_resflow_psc14/{results_resflow,vs_ref}.json.

## 2026-09-17 20:10 — Capacity x velocity x jac at 32->64 (owner: "可以继续"): CAPACITY is the strongest lever found; velocity inputs cost density power twice

`hpc/run_cap_vel_probe.sh` (commit 3ff1f53; `resflow_train --flow-base`). Split 0-13/14, 3000 steps, batch 2, HENON A100;
flow width = base width in every pair below. Density vs native 128 (set14), P/P at k_Ny,32 / 0.75 / 0.9 k_Ny,64, r_delta at 0.9:
| pair (base -> residual flow)          | params (base+flow) | r_lag | octave P/P | rms   | emulator P_delta/P     | r_delta | generative P_delta/P   | gen r_delta |
|---------------------------------------|--------------------|-------|------------|-------|------------------------|---------|------------------------|-------------|
| 24 -> 24 (control, RF_resflow_psc14)  | 1.56M + 1.56M      | 0.796 | 0.992      | 0.352 | 0.984 / 0.872 / 0.805  | 0.906   | 0.984 / 0.868 / 0.799  | 0.85        |
| 24 -> 24, jac 1.2                      | same               | 0.802 | 0.966      | 0.345 | 1.011 / 0.916 / 0.860  | 0.908   | 1.011 / 0.910 / 0.852  | 0.858       |
| 24 -> 24, velocity (VF_resflow_psc14) | same (+9 ch)       | 0.803 | 0.994      | 0.347 | 0.949 / 0.821 / 0.754  | 0.918   | 0.936 / 0.814 / 0.742  | 0.867       |
| shared 24 (M_resflow_psc, both levels)| 1.56M + 1.56M      | 0.806 | 1.004      | 0.343 | 0.983 / 0.903 / 0.853  | 0.907   | 0.975 / 0.885 / 0.831  | 0.852       |
| 48 -> 48 (RF48_resflow_psc14)         | 5.85M + 5.85M      | 0.820 | 0.991      | 0.333 | 1.001 / 0.936 / 0.892  | 0.915   | 1.005 / 0.933 / 0.887  | 0.860       |
| 48 -> 48, velocity (VF48)             | same (+9 ch)       | 0.825 | 0.986      | 0.327 | 0.970 / 0.873 / 0.827  | 0.915   | 0.981 / 0.887 / 0.846  | 0.856       |
| 48 -> 48, velocity + jac 1.2 (VFJ48)  | same               | 0.829 | 0.960      | 0.321 | 0.999 / 0.921 / 0.880  | 0.916   | 1.008 / 0.934 / 0.900  | 0.857       |
| oracle coarse band 24 -> 24 (bound)   | —                  | 0.868 | 0.969      | 0.239 | 0.968 / 0.912 / 0.874  | 0.967   | 0.934 / 0.832 / 0.780  | 0.946       |
Bases alone: 24: r 0.831 / coarse P_eps/P 0.0280; 24+vel 0.836 / 0.0265; 48 0.852 / 0.0249; 48+vel 0.857 / 0.0235.
Findings:
1. **Capacity is the strongest single lever found in this project**: 3.75x parameters take the emulator density at
   0.9 k_Ny,64 from 0.805 to 0.892 (+0.087, more than jac 1.2 (+0.055) or weight sharing (+0.05)), r_lag +0.024, rms -5%,
   and r_delta +0.009 — and it beats the oracle-coarse pair's DENSITY (0.874) while staying far below its r_delta (0.967).
   No sign of saturation between 24 and 48; the dose-response above 48 is not measured.
2. **Velocity inputs cost density power, reproducibly**: -0.05 at width 24 and -0.065 at width 48, both in emulator and
   (at 24) generative mode, while r_lag rises (+0.005) and r_delta rises (+0.01). Two runs, same sign and size: real,
   unexplained. Hypothesis to test later: the coarse velocity inside halos is virial noise, d_i(v/aHf)_j - D_ij is
   O(10) there, and the network learns to pull those regions toward the base's (shrunk) solution.
3. jac 1.2 on top of velocity + 48 recovers +0.053 (0.827 -> 0.880) with the usual -2.6% Lagrangian P/P drift; the best
   GENERATIVE density of the project is VFJ48 (0.900 at 0.9 k_Ny,64, r_delta 0.857), the best EMULATOR density RF48
   (0.892, r_delta 0.915). Neither beats 0.90; the true conditional's 1.01 at r_delta 0.94 (64->128) stands.
Recommendation (owner's call): (a) measure the capacity dose-response one step further (width 96 at 32->64; width 48
at 64->128 with the shared operator) before any architecture change; (b) drop velocity inputs unless the halo-noise
hypothesis is disproved; (c) production candidate = shared two-stage operator at width 48 (+ jac ~1 for generative use).
Provenance: runs/cap_vel_probe.log; runs/{RF48,VF48,VFJ48}_resflow_psc14/{results_resflow,vs_ref}.json; runs/V48_reg_psc14/results.json.

## 2026-09-17 22:05 — Session note + RFJ48 (base 48 + jac 1.2, no velocity): best density so far

Session hygiene: the previous session instance kept running after the harness restart and, on its own, launched
`hpc/run_cap2_probe.sh` (commit 003b380) with RFJ48 and a base-96 regression (20:04-20:41, "cap2 probe done"); the
current session did not see that and re-launched a base-96 regression under the same name at 21:59 (25 min of GPU
duplicated; results identical by construction, seed 0). No two trainings ever ran concurrently (checked: one step).
runs/RFJ48_resflow_psc14 (frozen base runs/R_reg_psc14_b48, flow width 48, --jac-weight 1.2, split 0-13/14):
  Lagrangian r 0.824, octave P/P 0.967, rms 0.327; density vs native 128 at k_Ny,32 / 0.75 / 0.9 k_Ny,64:
  emulator 1.029 / 0.979 / 0.945 (r_delta 0.915), generative 1.030 / 0.971 / 0.935 (r_delta 0.863).
-> capacity 48 + jac 1.2 WITHOUT velocity is the best pair of the project on set14 (0.945 at 0.9 k_Ny,64 vs 0.892
for 48 alone, 0.880 for 48+velocity+jac): the two positive levers stack almost additively (+0.087, +0.053), and the
velocity channels' -0.065 is confirmed a third time by subtraction. Density at k_Ny,32 now overshoots by 3% (the jac
bias signature). Provenance: runs/RFJ48_resflow_psc14/{results_resflow,vs_ref}.json.

## 2026-09-18 02:00 — Capacity saturates between width 48 and 96 at 32->64; the shared width-48 operator gains most at 64->128; set15 confirms

`hpc/run_cap2_probe.sh` (commit aa84f7c), HENON hold 1279890 (5 days). Split 0-13/14 unless stated.
(1) Width 96, 32->64 (runs/R_reg_psc14_b96 22.7M params, runs/RF96_resflow_psc14): base r 0.862, coarse P_eps/P 0.0232, rms 0.281;
    pair r_lag 0.833, emulator density 0.990 / 0.925 / 0.891 (r_delta 0.920), generative 0.990 / 0.935 / 0.906 (0.851).
    Capacity dose-response at 0.9 k_Ny,64: width 24 0.805 -> 48 0.892 -> 96 0.891: SATURATED in density between 48 and 96,
    while r_lag (0.796 -> 0.820 -> 0.833) and r_delta (0.906 -> 0.915 -> 0.920) still creep up. Width 48 is the sweet spot.
(2) Shared two-level operator at width 48 (runs/M48_reg_psc -> runs/M48_resflow_psc; 5.85M + 5.85M for BOTH levels):
| shared operator, emulator (generative) | 32->64 vs native 128: P/P @kNy32/0.75/0.9 kNy64, r_delta@0.9 | 64->128 vs native 256 (@kNy64/0.75/0.9 kNy128) |
|----------------------------------------|---------------------------------------------------------------|------------------------------------------------|
| width 24 (M_resflow_psc)               | 0.983 / 0.903 / 0.853  r 0.907  (gen 0.831, 0.852)             | 0.986 / 0.857 / 0.728  r 0.906  (gen 0.715, 0.885) |
| width 48 (M48_resflow_psc)             | 0.999 / 0.937 / 0.900  r 0.929  (gen 0.918, 0.859)             | 1.036 / 0.939 / 0.830  r 0.921  (gen 0.834, 0.896) |
    Lagrangian: 32->64 r 0.834 / P/P 0.982 / rms 0.316; 64->128 r 0.719 / 0.978 / 0.531 (width 24: 0.806 and 0.685).
    -> at 64->128 (the harder level, 37% multi-stream) width 48 buys +0.10 in density and +0.034 in r_lag — more than at
    32->64 — and r_delta 0.929 at 32->64 is the best of any non-oracle model. The shared width-48 operator is the
    production candidate (RUNLOG 20:10 recommendation confirmed).
(3) Second box (set15, eval-only): RF48 emulator 0.986 / 0.889 / 0.852 (r_delta 0.939; set14 0.892 / 0.915), generative 0.871;
    VFJ48 (48 + velocity + jac) 1.001 / 0.912 / 0.879 (0.940; set14 0.880 / 0.916), generative 0.892. Box-to-box scatter in
    density +-0.04 (set15's native 64 sits at 0.971 vs 0.998 on set14), r_delta consistently +0.02 higher on set15: every
    ranking from set14 holds.
Ledger at 0.9 k_Ny,64 (32->64, set14, emulator density / r_delta): control 0.805/0.906; shared24 0.853/0.907; jac1.2 0.860/0.908;
48 0.892/0.915; 96 0.891/0.920; 48+jac1.2 (RFJ48) 0.945/0.915; shared48 0.900/0.929; oracle 0.874/0.967; truth-level 1.0/0.97.
Provenance: runs/cap2_probe.log; runs/{R_reg_psc14_b96,RF96_resflow_psc14,M48_reg_psc,M48_resflow_psc,RF48_resflow_psc14_set15,VFJ48_resflow_psc14_set15}/.

## 2026-09-18 09:40 — M48J (shared width-48 operator + jac 1.2): jac shifts the whole band up, it does not fix the tilt; data-tree incident fixed

(1) runs/M48J_resflow_psc (`multi_resflow_train.py --mode resflow --base 48 --jac-weight 1.2`, frozen shared base runs/M48_reg_psc,
split 0-13/14, HENON hold 1279890). Emulator (generative) density vs the converged reference, P/P at k_Ny,c/2 / 0.75 / 0.9 k_Ny,c:
| shared width-48 operator | 32->64 vs native 128                              | 64->128 vs native 256                             |
|--------------------------|---------------------------------------------------|---------------------------------------------------|
| M48 (no jac)             | 0.999 / 0.937 / 0.900  r_delta 0.929 (gen 0.918)  | 1.036 / 0.939 / 0.830  r_delta 0.921 (gen 0.834)  |
| M48J (jac 1.2)           | 1.018 / 0.968 / 0.937  r_delta 0.929 (gen 0.954)  | 1.124 / 1.046 / 0.924  r_delta 0.920 (gen 0.934)  |
Lagrangian octave: 32->64 r 0.836 P/P 0.965 rms 0.312 (M48: 0.834 / 0.982 / 0.316); 64->128 r 0.725 P/P 0.917 rms 0.516
(M48: 0.719 / 0.978 / 0.531); generative octave P/P 0.934 and 0.878.
Reading: at 32->64 jac adds +0.02/+0.03/+0.04 (as for the specialist RFJ48); at 64->128 it lifts the whole band by a nearly
UNIFORM +0.09..0.11, overshooting at k_Ny,64 (+12%) and 0.75 k_Ny,128 (+5%) while 0.9 k_Ny,128 is still -8%, and the bias
signature is three times larger (Lagrangian octave P/P 0.978 -> 0.917; the J-loss is larger where multi-streaming is larger).
The TILT of the density ratio across the band (about -0.2 from k_Ny,64 to 0.9 k_Ny,128 at 64->128, with or without jac) is
untouched: jac sets the level, not the shape. A single jac weight shared across levels is too strong at 64->128; if used, it
should be per level (~1.2 at 32->64, <= 0.4 at 64->128). Production candidate stays M48 (jac optional, per level).

(2) Data-tree incident (found and fixed by the session). On 2026-09-16 20:59 the 128^3 IC conversion wrote 16 new directories
`IC/{disp.npy,disp_meta.json}` INTO the owner's PSC tree (`cosmo_sr/2-data/train/int_redshift_same_cosmology/dmo-128/set*/`),
because `data/psc/dmo-128/setK` were symlinks to those directories. Nothing was overwritten (the 128 level had no IC before);
dmo-64/256/512 were never touched. Fix (2026-09-18 09:28): the 16 IC directories were moved (md5-verified) into the repo's own
gitignored tree, `data/psc/dmo-128/setK/` is now a real directory holding `IC/` plus a `PART_009` symlink to the owner's run;
the owner's set directories again contain only PART_009 (their directory mtimes now read 2026-09-18). All scripts keep the
same paths. Rule for the future: per-level data under data/psc are real directories; only PART_009 is ever symlinked.

## 2026-09-18 12:00 — First autoregressive chain 32->64->128->256 with the shared width-48 operator; 128->256 ZERO-SHOT

`hpc/run_chain_shared.sh` -> `runs/chain_shared.py` (commit b3f606e; GPU 09:34-11:03 in HENON hold 1279890, full 256^3 boxes
in fp32, peak 35.2 GB) + RM job 1280611 (`hpc/slurm_eval_chain.sh`, 37 min: density vs the native 2c run with the verified
estimator; 256-level fields vs native 512 on a 512^3 mesh). Models: M48 (runs/M48_reg_psc + M48_resflow_psc) and M48J
(same base + M48J_resflow_psc); trained levels 32->64 (s 0.655), 64->128 (s 0.958); 128->256 zero-shot with the extrapolated
s = ln sigma_c = 1.215 (measured from the set-12/13 256^3 ICs; the measurement reproduces the checkpoint s at the trained
levels to 1e-4). Every step emulator (true IC octave) and generative (own sampled octave); "from M" = chained from true M^3.
Trained levels reproduce the models' own evaluations exactly (32->64 r 0.834, 64->128 r 0.719 on set14).

Density P/P at (k_Ny,c/2, 0.75, 0.9 k_Ny,c) vs native 2c, r_delta at 0.9 k_Ny,c; M48, set14 | set15:
| level / input                     | emulator set14                 | emulator set15                 | generative set14               |
|-----------------------------------|--------------------------------|--------------------------------|--------------------------------|
| native run (truth) 128 vs 256     | 0.985 / 0.982 / 0.972  r .979  | 0.983 / 0.978 / 0.973  r .977  | —                              |
| 64->128 direct (true 64)          | 1.036 / 0.939 / 0.830  r .921  | 1.054 / 0.977 / 0.881  r .924  | 1.055 / 0.951 / 0.836  r .892  |
| 64->128 chained from 32           | 1.035 / 0.933 / 0.823  r .758  | 1.062 / 0.965 / 0.884  r .788  | 1.069 / 0.935 / 0.818  r .599  |
| native run (truth) 256 vs 512     | 0.993 / 0.990 / 0.988  r .978  | 0.995 / 0.988 / 0.983  r .977  | —                              |
| 128->256 base (true 128), 0-shot  | 2.416 / 3.346 / 3.919  r .893  | 2.587 / 3.603 / 4.220  r .869  | —                              |
| 128->256 direct, 0-shot           | 1.679 / 1.795 / 1.767  r .905  | 1.784 / 1.890 / 1.848  r .884  | 1.684 / 1.789 / 1.752  r .899  |
| 128->256 chained from 64          | 1.765 / 1.866 / 1.779  r .731  | 1.922 / 1.989 / 1.936  r .737  | 1.804 / 1.873 / 1.771  r .668  |
| 128->256 chained from 32          | 1.721 / 1.728 / 1.645  r .411  | 1.840 / 1.857 / 1.736  r .503  | 1.660 / 1.675 / 1.549  r .262  |
M48J (jac 1.2): 64->128 direct 1.124/1.046/0.924 and chained 1.138/1.047/0.932 (set14), 1.144/1.090/0.988 and 1.163/1.081/0.988
(set15); 128->256 direct 1.754/1.886/1.850 (set14), 1.871/1.999/1.954 (set15) -- jac makes the zero-shot excess WORSE.
Lagrangian octave (emulator, M48 set14): 32->64 r .834 P/P .982; 64->128 direct .719/.978, chained .650/.960; 128->256 direct
.633/.666, chained-from-64 .530/.654, chained-from-32 .469/.646; correction-band P_eps/P .026 / .049 (.131 chained) / .070
(.193, .299 chained). Generative octave P/P at 128->256: 0.64.
Findings:
1. **Across the TRAINED levels the chain is power-stable**: chaining 32->64->128 leaves the 128-level density spectrum unchanged
   (0.823 vs 0.830, 0.884 vs 0.881 at 0.9 k_Ny,128; generative likewise) — the upstream error only costs phase coherence
   (r_delta 0.92 -> 0.76-0.79). Old width-24 specialist chain (RUNLOG 2026-09-16 23:50): 0.687 and r_delta 0.716.
2. **Zero-shot 128->256: phase extrapolates, power does not.** r_delta 0.905 / 0.884 (as good as the trained levels) but the
   density OVERSHOOTS by +68..85% across the band: the flow adds only 2/3 of the octave power (Lagrangian P/P 0.67 vs 0.98 at
   the trained levels), so the field stays base-like (the base alone is +140..320% here: the conditional mean is far more
   shrunk at this level). Chaining into the zero-shot level adds no further power error (1.65-1.94).
3. **Style-scalar scan (runs/s_scan.py, set 13, 128->256, emulator)**: flow octave P/P 0.555 / 0.617 / 0.679 / 0.752 at
   s = 0.655 / 0.958 / 1.215 / 1.5 (base 0.37 -> 0.46), r_lag 0.60 -> 0.64 then flat: monotonic but only ~+0.26 per unit s;
   P/P = 1 would need s ~ 2.4, twice as far beyond the trained range as the physical value. Extension to s = 2, 2.5, 3 with
   density running.
Verdict: the weight-shared operator transfers phase information to an unseen level but not the level-dependent detail
VARIANCE; the style scalar alone does not carry it. Going beyond 128 requires training data at 128->256, i.e. crops (full
256^3 training does not fit: inference alone peaks at 35 GB) — patch cropping is on the owner's ask-before list.
Provenance: runs/chain_shared.log, runs/slurm_evalchain_1280611.log, runs/chain_{M48,M48J}_set{14,15}/{chain.json,vs_ref_*.json},
runs/s_scan_M48_set13/s_scan.json.

## 2026-09-18 13:00 — Owner APPROVED patch cropping (ask-before item) for 128->256; two design probes first

(1) s-scan completed (runs/s_scan_M48_set13{,_b}; density vs native 512, set 13; runs/slurm_sscan_eval_1281620.log):
| s     | flow octave P/P_lag, r_lag | coarse P_eps/P | P_delta/P @kNy128/0.75/0.9 kNy256 | r_delta@0.9 |
|-------|----------------------------|----------------|-----------------------------------|-------------|
| 0.655 | 0.555, 0.599               | 0.074          | 1.317 / 1.041 / 0.820             | 0.910       |
| 0.958 | 0.617, 0.628               | 0.068          | 1.587 / 1.580 / 1.459             | 0.910       |
| 1.215 | 0.679, 0.638  (physical)   | 0.069          | 1.689 / 1.832 / 1.805             | 0.905       |
| 1.5   | 0.752, 0.638               | 0.075          | 1.669 / 1.799 / 1.774             | 0.897       |
| 2.0   | 0.875, 0.619               | 0.098          | 1.421 / 1.275 / 1.119             | 0.871       |
| 2.5   | 0.985, 0.587               | 0.136          | 1.106 / 0.768 / 0.580             | 0.821       |
| 3.0   | 1.074, 0.552               | 0.184          | 0.828 / 0.447 / 0.292             | 0.769       |
-> no s gives a flat density spectrum: raising s tilts it (low-k excess with high-k deficit) and costs coherence (r_delta
0.91 -> 0.77, correction-band error x2.7). The level cannot be reached by re-conditioning; it must be trained.

(2) Crop probe (`runs/crop_probe.py`, set 13, M48 at 128->256, runs/crop_probe/crop_probe.json): network output on a periodic
crop vs the same region of the full-box output, relative rms by Chebyshev distance d from the crop face:
| crop | base d = 0 / 4 / 16 / 48 | flow (t = 0.5) d = 0 / 4 / 16 / 48 | training-step peak memory (width 48, batch 1) |
|------|--------------------------|------------------------------------|-----------------------------------------------|
| 128  | 0.66 / 0.38 / 0.37 / 0.38 | 0.91 / 0.33 / 0.32 / 0.37         | 22.1 GB                                       |
| 160  | 0.66 / 0.37 / 0.34 / 0.34 | 0.86 / 0.30 / 0.28 / 0.27         | 40.9 GB                                       |
| 192  | 0.67 / 0.43 / 0.41 / 0.42 | 0.91 / 0.36 / 0.35 / 0.37         | OOM                                           |
The local face contamination (circular padding wraps opposite crop faces) is gone within ~4 cells, but a 30-40% difference
remains EVERYWHERE, including the crop centre: a GLOBAL effect — GroupNorm normalises over the whole input volume, so the
statistics (and the face voxels' contribution to them) differ between a crop and the box. Consequence: a model trained on full
boxes cannot be run on crops and vice versa; training and inference at the crop level must use the same crop geometry.
Design adopted (standard overlap-tile, no model change): 128->256 trained on 128^3 fine crops (halo 16, loss on the 96^3
interior; 32->64 and 64->128 stay full-box); all spectral operations (prolongation, D_ij, source octave, J/adj of the Q_J state)
on the full box, then cropped; the Q_J error field is cos^2-tapered to zero inside the halo before its spectral gradient and
the form is averaged over the interior only. Inference at that level: 128^3 tiles on the circularly padded full box, each
tile contributes only its central 64^3 (>= 32 cells from its faces, i.e. deeper inside than any training-loss voxel).

## 2026-09-18 15:10 — C3 launched: shared width-48 operator trained on THREE levels, 128->256 on 128^3 crops

Implementation (commit below): `tiling.py` (periodic crops, cos^2 halo taper, TiledNet = overlap-tile inference: circular pad by
(C-S)/2, C^3 tiles, central S^3 core kept); `runs/multi_resflow_train.py --crops 0 0 128 --halo 16 --tile-core 64` (per-level
crop side, 0 = full box; spectral ops and the Q_J state J/adj on the full box, then cropped; MSE on the interior; Q_J on a crop:
error tapered across the halo, spectral gradient on the crop, form averaged over the interior; `--jac-weight` refused on crop
levels); tiled evaluation at crop levels; `runs/chain_shared.py` reads each level's crop geometry from the checkpoints.
Checks (`checks/tiling_check.py`): TiledNet and crop_periodic reproduce a local circular-conv network's full-box output EXACTLY
(max rel diff 0); the crop Q_J reproduces the full-box form's interior mean to 2.6-3.7% for an octave-band error field
(pointwise 19-21%; the spectral-derivative kernel is non-local, a larger halo does not help) — it is a slightly different but
still ADMISSIBLE form (positive semi-definite, state-dependent only), and lambda is equalised per level anyway.
Smoke tests (2 train sets, 6 steps/level): base and resflow run, peak GPU 21.4 / 22.3 GB; lambda at the crop level 0.0069;
tiled 256^3 flow sampling (64 tiles x 17 network passes) ~4.5 min per prediction.
Launched `JOBID=1279890 hpc/run_c3.sh`: C3a runs/M48c_reg_psc -> C3b runs/M48c_resflow_psc (split 0-13/14, 3000 steps per
level, batches 2/1/1) -> chain 32->64->128->256 on sets 14 and 15 -> RM density job. Expected ~7 h.
Restart 2026-09-18 17:25: the first C3a attempt ran CPU-bound (1.93 s/step, GPU utilisation 37%): the numpy cube-group
augmentation of a 256^3 training sample costs 1.4 s. Crop levels now skip it and augment the CROPS on the GPU instead
(`tiling.augment_crops`: vectors, rank-2 tensors D_ij and adj, scalar J), which is exactly equivalent because every scaffold
operation commutes with the cube group (checks/tiling_check.py augment: max rel diff <= 7.7e-6 for x0, x1, Pc, D, J, adj over
6 group elements = float32 FFT round-off). Full-box levels keep the numpy augmentation unchanged. First attempt deleted.

## 2026-09-19 01:10 — C3 result: the three-level shared operator (128->256 trained on crops) fixes the 256-level density

Runs (commit 5c2caf1; HENON hold 1279890): runs/M48c_reg_psc (17:00-18:58, 0.77 s/step) -> runs/M48c_resflow_psc (18:58-21:44;
lambda 0.0167 / 0.0124 / 0.0104 for 32->64 / 64->128 / 128->256) -> runs/chain_M48c_set{14,15} (tiled inference at 128->256,
~34 min per set) -> density vs the native 2c run (hpc/slurm_eval_chain.sh; the RM job 1291167 sat 2 h in the queue on priority
(151 jobs pending) and was cancelled; the same script ran on the idle CPUs of the HENON hold allocation, runs/evalchain_c3_hold.log).
Density P/P at (k_Ny,c/2, 0.75, 0.9 k_Ny,c) vs native 2c, r_delta at 0.9 k_Ny,c; emulator unless marked:
| level / input                 | M48c set14                    | M48c set15                    | zero-shot M48 set14 / set15 (RUNLOG 12:00)    |
|-------------------------------|-------------------------------|-------------------------------|-----------------------------------------------|
| 32->64 direct                 | 0.997 / 0.927 / 0.891  r .924 | 1.023 / 0.969 / 0.937  r .939 | 0.999 / 0.937 / 0.900 r .929 ; .904 r .940     |
| 64->128 direct                | 1.069 / 0.987 / 0.869  r .920 | 1.081 / 1.029 / 0.938  r .918 | 1.036 / 0.939 / 0.830 r .921 ; .881 r .924     |
| 64->128 from 32               | 1.098 / 0.982 / 0.872  r .750 | 1.177 / 1.119 / 1.026  r .776 | 1.035 / 0.933 / 0.823 r .758 ; .884 r .788     |
| 128->256 base                 | 2.366 / 3.364 / 4.001  r .890 | 2.529 / 3.629 / 4.340  r .863 | 2.416 / 3.346 / 3.919 ; 2.587 / 3.603 / 4.220  |
| 128->256 direct               | 1.169 / 1.054 / 0.939  r .905 | 1.206 / 1.081 / 0.959  r .887 | 1.679 / 1.795 / 1.767 r .905 ; 1.784 / 1.890 / 1.848 r .884 |
| 128->256 direct, generative   | 1.159 / 1.035 / 0.915  r .902 | 1.198 / 1.062 / 0.934  r .884 | 1.684 / 1.789 / 1.752 ; 1.790 / 1.882 / 1.828  |
| 128->256 from 64              | 1.312 / 1.208 / 1.041  r .718 | 1.411 / 1.282 / 1.147  r .718 | 1.765 / 1.866 / 1.779 ; 1.922 / 1.989 / 1.936  |
| 128->256 from 32              | 1.321 / 1.136 / 0.974  r .392 | 1.552 / 1.344 / 1.144  r .481 | 1.721 / 1.728 / 1.645 ; 1.840 / 1.857 / 1.736  |
| 128->256 from 32, generative  | 1.221 / 1.058 / 0.898  r .232 | 1.540 / 1.299 / 1.086  r .330 | 1.660 / 1.675 / 1.549 ; 1.887 / 1.847 / 1.694  |
(native 256 vs 512: 0.993 / 0.990 / 0.988, r .978 and 0.995 / 0.988 / 0.983, r .977.)
Lagrangian octave (emulator): 32->64 r .826/.834 P/P .996/1.001; 64->128 .711/.716, .968/.971; 128->256 direct .590/.592,
.927/.936 (zero-shot .633/.636, .666/.675); from 64 .486/.490, .895/.905; from 32 .429/.435, .871/.886.
Findings:
1. **Training the third level on crops works**: the 256-level density goes from +68..85% across the band (zero-shot) to
   1.17-1.21 / 1.05-1.08 / 0.94-0.96 — the best top-of-band density at ANY level so far — with r_delta unchanged (0.905 / 0.887).
   The trade-off is the usual one: octave power 0.67 -> 0.93 at the cost of octave r 0.63 -> 0.59.
2. **Adding the third level does not hurt the others**: 32->64 within +-0.03 of M48, 64->128 slightly closer to 1 at the top
   (0.87 / 0.94 vs 0.83 / 0.88). One weight-shared operator now covers 32->64->128->256 (5.85M + 5.85M parameters).
3. **Residual problem 1 — an excess at the coarse Nyquist that grows with level**: at the band's low probe (k = k_Ny of the
   coarse grid = half the fine Nyquist) the direct steps give 1.00/1.02 -> 1.07/1.08 -> 1.17/1.21 for 32->64 -> 64->128 -> 128->256. This is the correction band's edge, where the coarse run's own error is O(1) (Phase 0: P_eps/P_c
   0.57-0.66 at 0.94 k_Ny,c).
4. **Residual problem 2 — chained inputs add power at that edge**: into 256 from 64 / from 32 the low probe rises to 1.31-1.41 /
   1.32-1.55 (direct 1.17-1.21), in both modes, more on set15. Unlike the chain into 128 on set14 (power-stable), the chain into
   256 is not: the model's own output is a different kind of coarse input than a native run (its top band carries the flow's
   unmatched detail rather than the coarse run's Nyquist deficit), and the next step's correction does not undo it. This is the
   rollout distribution shift; RFT2 (RUNLOG 2026-09-15) showed rollout fine-tuning damps power without repairing phase — here
   power is exactly what is wrong, so a rollout-aware step is back on the table (ask-before item, owner's call).

## 2026-09-19 01:40 — Owner: "先1后2，内存映射读取": (1) rollout-aware fine-tuning of the shared operator, then (2) 256->512 with mmap

(1) `runs/rollout_shared.py` = the RFT2 protocol (owner's spec 2026-09-15) on the shared three-level M48c: frozen shared base and
frozen upstream predictions cached once from M48c (emulator mode): pred64 (32->64 from true 32), pred128_64 (64->128 from true
64), pred128_32 (two steps from true 32); ONLY the shared residual flow is tuned, warm-started from M48c_resflow_psc; lambda per
level reused (0.0167 / 0.0124 / 0.0104); per step the coarse input is native with probability 1 - mix, else a cached prediction
(64->128: pred64; 128->256: pred128_64 or pred128_32 50/50; 32->64 always native); base re-evaluated on the chosen coarse, target
X - B; paired augmentation (numpy before make() at full-box levels, tiling.augment_crops at the crop level); 800 steps per level
at 5e-5, EMA 0.99; crop level keeps C3's geometry. Smoke test (2 sets, mix 1.0): cache, mixing and crop path run.
Arms (one training per GPU): RS_mix50 on HENON (1279890), RS_mix0 control on a TWIG A100 (owner allowed a second card), each
followed by chain_shared on sets 14/15 and the density evaluation on its own allocation's CPUs (`hpc/run_rollout_shared.sh`).
Rows: no-FT (M48c, RUNLOG 01:10) / mix0 / mix50.

## 2026-09-19 03:45 — RS_mix50 (rollout-aware fine-tune of the shared flow): the chained excess is gone, but power is damped hard

runs/RS_mix50 (01:44-02:27, ~1.0 s/step, 800 steps per level at 5e-5; inputs per level 32: 800 native; 64: 402 native / 398 pred64;
128: 415 native / 385 pred128) -> runs/chain_RS_mix50_set{14,15} -> runs/evalchain_rs_mix50.log (density on the hold allocation CPUs).
TWIG hold 1295912 never left the priority queue; the mix0 control therefore runs after mix50 on HENON (dispatcher, 03:41).
Density P/P at (k_Ny,c/2, 0.75, 0.9 k_Ny,c) vs native 2c, r_delta at 0.9 k_Ny,c, emulator:
| level / input      | no-FT M48c set14       | RS_mix50 set14         | no-FT M48c set15       | RS_mix50 set15         |
|--------------------|------------------------|------------------------|------------------------|------------------------|
| 32->64 direct      | .997/.927/.891 r .924  | .983/.899/.864 r .924  | 1.023/.969/.937 r .939 | 1.011/.947/.910 r .935 |
| 64->128 direct     | 1.069/.987/.869 r .920 | .990/.853/.723 r .909  | 1.081/1.029/.938 r .918 | 1.001/.892/.775 r .910 |
| 64->128 from 32    | 1.098/.982/.872 r .750 | 1.000/.835/.708 r .731 | 1.177/1.119/1.026 r .776 | 1.062/.927/.799 r .754 |
| 128->256 direct    | 1.169/1.054/.939 r .905 | .922/.675/.531 r .892 | 1.206/1.081/.959 r .887 | .941/.686/.539 r .876 |
| 128->256 from 64   | 1.312/1.208/1.041 r .718 | .934/.639/.466 r .681 | 1.411/1.282/1.147 r .718 | .989/.664/.493 r .693 |
| 128->256 from 32   | 1.321/1.136/.974 r .392 | .929/.583/.418 r .360 | 1.552/1.344/1.144 r .481 | 1.046/.660/.461 r .415 |
Generative mode follows the emulator (e.g. 128->256 direct .905/.651/.506 set14). Lagrangian octave at 128->256 direct: r .579/.580,
P/P .937/.944 (no-FT .590/.592, .927/.936): the displacement power is unchanged, the DENSITY power at high k is not.
Reading (verdict pending the mix0 control, running): the coarse-Nyquist excess of chained inputs is removed (1.31-1.55 -> 0.93-1.05)
but the whole band is pulled down, strongest at the top (0.94 -> 0.53 at 0.9 k_Ny,256 direct; 0.87 -> 0.72 at 0.9 k_Ny,128), and
r_delta falls slightly everywhere — the RFT2 signature again (power damping toward the r^2 line, no phase repair), now at three
levels. With equal Lagrangian octave power the density loss at the top means the tuned flow's detail is LESS matched to the coarse
structure (hybrid-test mechanism, RUNLOG 2026-09-17 10:50). Even 32->64, which only ever saw native inputs, moved (-0.03): shared-
weight drift. The mix0 control decides how much of this is mixing and how much is fine-tuning drift.
