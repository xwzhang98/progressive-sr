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
  On real N-body data use `--source-filter wiener` (notes v3): `x0 = P[Tc Psi_c] + G Psi_lin[eta]`
  with the Wiener/propagator filters measured on the training set — the best linear
  prediction; linear theory is its perturbative limit.
- Network sees only translation-invariant, dimensionless inputs: the residual
  `x_t - P Psi_c` (units of h_f) and the coarse deformation tensor `D_ij = d_i Psi_j`.
- No GAN, no learned `P`, no Haar (Haar puts a deterministic linear-order mismatch
  into the correction; verified numerically, see README).

The theory is in `notes/octave_flow_derivation.pdf` (English, v2 after review) and
`notes/octave_flow_physics.pdf` (Chinese, physics-style); the project overview is
`notes/progressive_sr_brainstorm.md`. Read them before changing anything physical.
Real-data facts that changed the notes (v3, Sec. "What the real N-body data changes"): at
z=0 the linear octave explains only r^2~0.3 of the detail and overshoots its power (G<1);
the coarse run is O(1) wrong near its Nyquist (P_eps/P_c = 0.57 at 0.94 k_Ny,c); 30-40% of the
Lagrangian volume is multi-stream, which is where regression and flow are expected to differ;
MP-GenIC ICs share white noise but are not bitwise nested near k_Ny,c; 2LPT harmonics are
uncorrelated with the z=0 detail. Report every metric split by the multi-stream mask.
Eulerian density (notes/eulerian_power.pdf, Chinese): the Lagrangian octave-band P/P=1 line and the
Eulerian P_delta/P_true=1 line are different lines that meet only at r=1. In the octave band
d_model = a d_true + n with sigma_n^2 = (P/P_true - r^2) Var(d); independent noise damps P_delta by
exactly exp(-k^2 sigma_n^2) (verified), shrinkage a<1 over-compacts multi-stream patches (1-halo
excess). Generative flow: P_delta/P_true=1 is a theorem in the exact limit; a conditional sampler
has it in expectation even at r<1; a conditional mean (regression) never does in multi-stream
regions. Two kinds of Eulerian-aware loss exist and must not be confused: (i) the *quadratic forms*
Q_E (CIC-gradient deposit at coarse/interpolant positions) and Q_J (Jacobi-linearised
det(I + dPsi/dq), purely Lagrangian) in `eulerian_metric.py` (`--loss-metric qe|qj`) — they depend
on the state, not on x_1, so the flow-matching minimiser is unchanged (any positive-definite form
depending on (x_t, C, t) does that) and only the finite network's residual is steered, into the
density-invisible kernel for Q_E; (ii) the *nonlinear* losses on the x-prediction
(`--cic-weight`, `--jac-weight` with asinh(J)), which bias the velocity by lambda (1-t) times a
term that contains x_1 (notes §5.5, §5.7d) — legitimate for the regression (Paper-IV style), a
controlled bias for the flow, and their generative-mode numbers are where the bias shows. Always
report P_delta/P_true(k), r_delta(k), the delta>100 mass fraction and the J quantiles next to the
Lagrangian numbers. On the h_f grid J is a clean 1/rho only in single-stream regions; inside
halos it measures the roughness of Psi, not a per-stream Jacobian.
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
- `eulerian_metric.py` — CIC deposit / gradient deposit / pullback, the quadratic forms
  `eulerian_form` (Q_E) and `jacobian_form` (Q_J), the `eulerian_inputs` channel;
  `python eulerian_metric.py` self-test.
- `checks/smear_check.py` — numerical check of the smearing formula on a multi-stream Zel'dovich field.
- `README.md` — commands, expected numbers, output formats.
- `KICKOFF_PROMPT.md` (Stages 0–4), `KICKOFF_STAGE3B.md` (review-driven experiments),
  `KICKOFF_STAGE5.md` (wiener source, multi-stream split), `KICKOFF_STAGE6.md` (Eulerian tests
  T1–T4 and the Eulerian metric runs), `KICKOFF_STAGE7.md` (eval_eulerian.py tooling while the
  jac runs train; then the qe/qj runs).
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
  cropping, rollout fine-tuning, new nonlinear losses on the flow's x-prediction beyond the
  approved `--cic-weight`/`--jac-weight`, or any change to R/P/W. The quadratic forms of
  `eulerian_metric.py` are allowed (see above).
- A direct regression baseline (same inputs, same network, MSE on x1 - x0, one step)
  is the right comparison for any claim about the multi-step flow; do not describe
  multi-step gains as "restoring variance".
- Small, reviewable diffs; no reformatting of the existing files.
- Git: commit before launching a sweep (`results.json` records the short hash); never commit
  weights, `train_state.pt`, `.npy` data or logs (see `.gitignore`); `runs/RUNLOG.md`,
  `results.json` and `summary.png` are committed.
- Never run two GPU trainings concurrently on one device (see RUNLOG, 2026-09-08 incident).
