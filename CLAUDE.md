# progressive-sr — project memory for Claude Code

## What this is
Research prototype for **progressive 2x super-resolution of cosmological N-body
simulations in the Lagrangian description** (particles on the initial grid,
displacement fields `Psi(q)`). One weight-shared 2x step, applied autoregressively:
64^3 -> 128^3 -> 256^3 -> 512^3 in a 100 Mpc/h box. It extends the AI-assisted SR
series (Li+21, Ni+21, Zhang+24 III & IV, Paper V in prep) of which the owner
(Xiaowen Zhang, CMU McWilliams Center) is an author.

Method ("octave flow"), fixed unless the owner says otherwise:
- Restriction `R` = a 0/1 Fourier window (a projector, `RP = I` exactly), prolongation
  `P` = Fourier zero-padding with the grid-offset phase. Default window = the **coarse
  cube** (`--window cube`): nothing of the coarse run is discarded and the octave is
  exactly the new initial-condition modes. The isotropic sphere (`--window sphere`) is an
  ablation only: it discards the coarse corners and re-samples them, which is an
  approximation (notes/octave_flow_derivation.pdf, Remark 2.4 and Sec. 5).
  Details live in `(1 - W)`; the coarse band of the residual is the "correction".
- Conditional flow matching with the **physical coupling**:
  `x0 = P Psi_c + Psi_lin[eta]`, `x1 = Psi_f`, where `eta` is the true IC octave in
  training, and a Gaussian octave with the measured linear P_delta(k) at sampling.
- Network sees only translation-invariant, dimensionless inputs: the residual
  `x_t - P Psi_c` (units of h_f) and the coarse deformation tensor `D_ij = d_i Psi_j`.
- No GAN, no learned `P`, no Haar (Haar puts a deterministic linear-order mismatch
  into the correction; verified numerically, see README).

The theory is in `notes/octave_flow_derivation.pdf` (English, v2 after review) and
`notes/octave_flow_physics.pdf` (Chinese, physics-style); the project overview is
`notes/progressive_sr_brainstorm.md`. Read them before changing anything physical.
Known subtleties (all in the notes, Sec. "What is proven"): the detail band also
contains deterministic harmonics of the coarse modes; the one-step (regression) limit
is not necessarily power-deficient under full physical conditioning; the Markov
property holds for the full nested state, only approximately for reduced states.

## Files
- `phase0_octaves.py` — measurements on real/synthetic nested resolution series
  (correction spectra, Wiener T(k), detail vs linear octave, conditional statistics).
  Has `--selftest` (aliased 2LPT levels) and `--selftest --selftest-dealias`.
- `octave_flow_toy.py` — the end-to-end PyTorch prototype (imports phase0_octaves).
  Spectral ops run on CPU when the network is on MPS (`--fft-device`).
- `README.md` — commands, expected numbers, output formats.
- `KICKOFF_PROMPT.md` (Stages 0–4) and `KICKOFF_STAGE3B.md` (review-driven experiments:
  objective × coupling, cube vs sphere, sampler steps, coarse-only harmonics).
- `hpc/` — cluster checklist (`README.md`), SLURM templates (Phase 0; resumable training with
  `--max-seconds/--resume` self-resubmission), and `convert_snapshot.py` (MP-Gadget BigFile ->
  map2map cubes with the ID -> grid mapping and an IC-snapshot sanity check).
- `reference_results/` — figures/JSON from verified runs in the cloud sandbox:
  selftest slopes 2.2 / 4.15 (theory 2 / 4); toy 16->32, 600 steps CPU:
  baseline r=0.917, P/P=0.83 -> emulator r=0.994, P/P=0.993; generative P/P=1.02.

## Conventions (do not silently change)
- Fields are `float32 (3, N, N, N)` Lagrangian displacements in Mpc/h, map2map layout.
- Lagrangian grid positions `q = (i + offset) h`; `--offset` must match the IC
  generator (find it with the nestedness check in `phase0_octaves.py`: ~1e-6 = right).
- All network quantities are in units of the fine grid spacing `h_f`; `D_ij` is
  dimensionless; box is periodic (circular padding).
- Nested ICs: lower resolutions are cube truncations of the same seed.
- Keep `phase0_octaves.py` and `octave_flow_toy.py` importable from the same directory.

## Environment
- macOS; use a fresh venv or conda env; `pip install -r requirements.txt`.
- Prefer `--device mps` for the network; if any op is unsupported on MPS
  (conv3d / avg_pool3d / circular pad on old torch), fall back to `--device cpu`
  rather than rewriting the model.
- HPC cluster is currently under maintenance: everything here must run on the laptop.
  Real snapshots (64/128/256/512, same seed) are on the cluster; only run the real-data
  paths if local copies exist.

## Working agreements
- Run the selftests first and compare with the reference numbers before any change.
- Record every run's command line and metrics in `runs/RUNLOG.md` (append-only) and
  keep `results.json`/`summary.png` per run under `runs/<name>/`.
- Ask before: adding velocities, multi-transition (style scalar) training, patch
  cropping, rollout fine-tuning, Eulerian CIC loss, or any change to R/P/W.
- A direct regression baseline (same inputs, same network, MSE on x1 - x0, one step)
  is the right comparison for any claim about the multi-step flow; do not describe
  multi-step gains as "restoring variance".
- Small, reviewable diffs; no reformatting of the existing files.
- Git: commit before launching a sweep (`results.json` records the short hash); never commit
  weights, `train_state.pt`, `.npy` data or logs (see `.gitignore`); `runs/RUNLOG.md`,
  `results.json` and `summary.png` are committed.
- Never run two GPU trainings concurrently on one device (see RUNLOG, 2026-09-08 incident).
