# HANDOFF — laptop session → HPC session (2026-09-16)

For a fresh Claude session on the cluster after `git pull`. Read together with `CLAUDE.md`
(rules there still bind). This file is the session memory: where the project stands, what is
decided, what runs next, and every portability fact the laptop session learned the hard way.
Laptop HEAD at handoff: `7fcfb10` + this commit.

## 1. State in one paragraph

The full story to date is `runs/REPORT_9.md` (read it first) + `runs/RUNLOG.md` (append-only,
every run). Headlines: (i) two-stage **residual flow** (frozen regression base + flow on the
residual, Q_J metric) is the best model family at both 32→64 and 64→128; (ii) **RFT2 verdict
(closed)**: fine-tuning the downstream flow on rollout inputs damps power but repairs no
phase — chain error must be fixed upstream; (iii) **Y64 verdict**: the ideal projected label
R Ψ₁₂₈ over-concentrates (density +11–37% vs the 128 reference, IC-clean), while the native
64³ run is within 2–5% in-band — headroom is *phase* and k > k_Ny,64, not band power;
(iv) **IC audit**: `nested_ic.py` achieves 1.8e-7 nesting (GenIC's Nmesh=2·Ngrid folding was
the main culprit), but exact nesting changes the z=0 mismatch by only ~0.01 — the cross-level
error is dynamical, so strict ICs are a *diagnostic* tool, not a training-data requirement.

## 2. The approved plan (owner, 2026-09-16): A → B → C1, strictly one at a time

Owner's words: "都留着吧，我们一个一个测试，记得都放到GitHub repo里然后push掉".
Run a phase, commit+push results, report, THEN start the next. Everything (scripts, RUNLOG,
results.json, figures) goes into the repo; weights/.npy/logs never do.

**A — cluster bring-up + reference audit (measurement only; start immediately)**
  1. Env per `hpc/README.md` §1 (CUDA torch wheel; pin versions in RUNLOG). Run BOTH
     selftests and compare `reference_results/` (selftest slopes 2.2/4.15; 20-step CPU 16→32
     reference loss 3.0888e-02 must be bit-identical on CPU).
  2. Convert the real 64/128/256/512 same-seed series (`hpc/convert_snapshot.py`; check ID
     convention on the IC snapshot first; offset via nestedness check — production data was
     0.5 on the laptop's Box-downloaded boxes; ~1e-6 = right).
  3. Phase 0 on the full series (`hpc/slurm_phase0.sh`; 512³ needs ~20 GB, ask ≥64 GB):
     first 256/512-level Wiener T(k), detail-vs-linear r², multi-stream fractions.
  4. NEW, from the Y64 verdict: (a) **convergence test** — 128 vs 256 vs 512 density in
     k < k_Ny,64 (upgrades/demotes "128 is a reference, not truth"); (b) **Y128 = R Ψ₂₅₆ vs
     native 128** via `runs/eval_y64.py` generalized one level up — prediction: the label
     excess grows with multi-streaming. Both reuse `compute_spectra.py` (now at repo root).
**B — the one label experiment (owner approved as part of the sequence)**
  Y64-base: retrain the 32→64 regression base with label R Ψ₁₂₈ (via `restrict_spectral`),
  keep the residual flow's target = native 64 run. Pre-registered success: in-band density
  P/P ∈ [0.95, 1.00] AND r_δ@0.9 k_Ny,64 > 0.95 (beat both end-members). Recipe = the
  RF_resflow recipe (runs/resflow_train.py) with the base's target swapped; λ auto-equalised
  at step 1 then pinned (the J_flow_qj convention). Needs `data/selfsim` transferred (see §5).
  If the flow cannot absorb the label excess → supervision alone is refuted; representation
  route gets priority.
**C1 — production training data (after A conventions verified)**
  Production pipeline (2LPT GenIC, its native Nmesh=2·Ngrid — physical realism, NOT
  nested_ic) 64/128 pairs, 16–32 seeds + 1–2 held-out test seeds; a couple of 128/256 boxes.
  Strict-IC trios (nested_ic.py, Zel-only) only as a small diagnostic set. Then C2 (later,
  separate decision): 64→128 full-box training incl. the weight-shared resflow variant
  (owner pre-approved two-transition training earlier); 128→256+ needs patch cropping or the
  map2map port — ask first (both on the ask-before list).

## 3. Reference numbers the new session must not re-derive (all in RUNLOG/REPORT_9)

* Selftest slopes 2.2 / 4.15 (theory 2/4). Toy 16→32 600-step CPU: baseline r=0.917 →
  emulator r=0.994, P/P=0.993; generative P/P=1.02.
* 32→64 (s8/s9): resflow r=0.795/0.806, Eulerian ≈1; 64→128: r=0.677/0.689 direct.
* No-FT chain baselines (re-measured with r_δ/coherent, `runs/chain_resflow_s{8,9}.json`):
  chained Eul@(kNy64,1.5kNy64,kNy128) = 1.268/0.994/0.614 (s8), 1.032/0.832/0.560 (s9).
* Y64 excess 1.18–1.37 (strict IC 1.370 @0.9k_Ny,64); native 64 in-band 0.95–0.99.
* nested_ic acceptance 1.8e-7; strict-vs-old z=0 mismatch 0.126/0.136 (32→64), 0.093/0.094.
* Growth D(z=0)/D(z=99) = 76.7439 (this cosmology; recompute per hpc/README for new runs).
  C_VEL = 0.52776101 km/s per kpc/h at z=99. Fields on disk in kpc/h (`--box 100000`).
* Parameter accounting: two specialists 6.22M vs 1.56M single-stage — always state it.

## 4. Conventions and gotchas (the expensive lessons)

* **One training per GPU, ever** (RUNLOG 2026-09-08 incident). SLURM: exclusive GPU.
* Chunked training: `--max-seconds` + `--resume`; `train_state.pt` carries model/EMA/opt/λ_e
  (λ_e MUST persist across chunks — drift bug already fixed, don't reintroduce). PAUSE
  protocol: `touch runs/PAUSE` stops the queue loop gracefully.
* Eval flags must mirror training flags (`--regression`, `--eulerian-inputs` restored from
  checkpoint BEFORE batcher build — both were real bugs).
* Emulator vs generative: growth is applied ONCE to the sampled octave (double-application
  bug fixed). Octave sampler: GenIC-oracle/full sampler closed the gen-emu gap.
* Offset: 0 for self-run/nested_ic data, 0.5 for production Box data — always confirm via
  the phase0 nestedness check, never assume.
* Estimator discipline: `compute_spectra.py` (repo root, moved verbatim from the owner's
  reviewed analysis; only ROOT made repo-relative) — interlaced CIC + window deconvolution,
  common 256³ analysis mesh, complete shells only, no shot-noise subtraction. Use it for
  every density claim; probes (k_Ny,c, 1.5 k_Ny,c, 2 k_Ny,c); quote BOTH boxes, never average.
* Report style the owner expects: honest contradiction flagging (no explaining-away),
  "current high-resolution reference" not "truth", per-mask (multi-stream/parity) splits,
  every claim next to its control (mix0-style), one minimal next experiment with its
  hypothesis. RUNLOG entry per run with command line + commit hash before GPU queues.
* zsh: `for X in "a b"` does NOT word-split; `timeout` doesn't exist on macOS; long jobs via
  the harness's background mechanism, not nohup.

## 5. What git does NOT carry — transfer manifest (run from the laptop)

Weights, `train_state.pt`, `.npy`, `data/` are gitignored by design. The cluster needs:

```
# training/eval fields for phase B and any laptop-comparison (2.3G + 355M):
rsync -av data/selfsim data/nestedic <user>@<cluster>:<repo>/data/
# optional, only for comparing against laptop-trained models (~6 MB each):
rsync -av runs/{R_reg_phys_3k,RF_resflow,F_reg_128_3k,RF_resflow_128,J_flow_qj}/model_ema.pt \
      --relative <user>@<cluster>:<repo>/
```

Phase A needs NO transfer (the raw snapshots already live on the cluster). Checkpoints are
optional: B trains from scratch; the laptop numbers above serve as the comparison row.

## 6. Open threads (do not silently drop)

* C2 shared/weight-shared resflow + base+regression-stage2 control — owner's pending item,
  after C1.
* Generative-chain diversity checks and 8→16-step checks on the hard level — partially done,
  owner's eval-layer batch item 1.
* 2LPT for nested_ic (currently Zel-only, documented ~1e-2 difference) — only if strict
  trios are ever used for training, which is currently NOT the plan.
* `.claude/` is local session state, now gitignored; scratchpad scripts were rescued into
  `runs/` (fig_y64, phase3_models, fig_chain, fig_chainfull, nsteps, rf_oracle, rf_s9,
  rf128_s9) with paths made repo-relative — they are the provenance of every committed figure.

## 7. Cluster session 2026-09-16 — Phase A DONE (details: RUNLOG 2026-09-16 entries)

* Env `torch206` (torch 2.8 cu128); selftests match; smoke test baseline bit-identical. GPU: one A100 via the
  owner's held allocation (`srun --jobid=<hold job> --overlap --exact --gres=gpu:1 ...`), ask which is free.
* Data: the PSC series is 16 same-seed sets x 64/128/256/512 (`cosmo_sr/2-data/train/int_redshift_same_cosmology`,
  map2map .npy, kpc/h, offset **0.5**, IC only at 64/512; raw BigFile with all ICs in `sim_output/dmo-100MPC/...`).
  C1's production data therefore already exists (16 seeds of 64/128 + 256/512).
* Bug fixed: multi-octave offset phase in `restrict_spectral`/`prolong_spectral` (one-octave path unchanged).
* A.3 Phase 0 set0 (RUNLOG table): coarse-band stats level-invariant (P_eps/P_c 0.57/0.61/0.66 at 0.94 k_Ny,c,
  Wiener T = 0.80 at all levels), detail band not (linear r^2 0.29 -> 0.16 -> 0.08; kurtosis 2.8 -> 9.8;
  multistream 0.37 -> 0.44).
* A.4 (RUNLOG table): 128 is converged to 1% vs 512 in the 64 band (upgrade of "128 is a reference");
  Y128 = R Psi_256 over-concentrates +30..48% vs +10..19% for Y64 -- the label excess grows with level.
* NEXT = Phase B (Y64-base retrain) as specified in §2; needs a GPU and the 32-level data (`data/selfsim`
  rsync from the laptop, or convert the raw `dmo-32` BigFile outputs with `hpc/convert_snapshot.py`).

## 8. Cluster session 2026-09-16 — Phase B DONE (RUNLOG 17:15 and 17:25 entries)

* Data: 32^3 sets 1-15 simulated (seeds verified = other levels), `data/psc/` unified layout, converter
  validated bitwise. Phase B = B0 control base / B1 Y64 base / B2 / B3 residual flows on PSC sets 0-7 -> 8.
* VERDICT: B3 fails the pre-registered criterion exactly like the control B2 (density vs native 128 in the
  64 band: 0.96/0.82/0.73 at k_Ny,32 / 0.75 / 0.9 k_Ny,64, r 0.92; B2: 0.96/0.83/0.75, r 0.93). The base's
  +40% excess is shrinkage, not the label; the flow's -25% deficit and r = 0.92 do not depend on the base.
  "Supervision alone" refuted -> representation route (C1/C2) has priority, per §2.
* NEXT (owner's decision): C1 production 64->128 on the 16-set PSC series is already possible (data exists);
  C2 = weight-shared 32->64 + 64->128 resflow with the level in the style scalar (owner pre-approved
  two-transition training); the one open diagnostic worth doing first is WHERE the r_delta ceiling (0.92)
  sits — a per-mask (multi-stream/parity) split of the B2 fields and an r(k) curve, using eval_eulerian.py.

## 9. Cluster session 2026-09-16/17 — options 1+2+3 and the course-notes §1/§4 DONE (RUNLOG 21:10 .. 00:50)

* Diagnostic (option 3): the residual flow adds NO phase (its r_delta(k) equals the base's at every k), the ceiling is
  spatially uniform and starts at the coarse Nyquist; the flow only rescales power. Holds at 32->64 and 64->128.
* C1 (production split 0-13/14, 64->128): base + residual flow reproduce the laptop numbers to the third digit.
* C2: ONE weight-shared two-stage operator (shared regression base + shared residual flow, s_l = ln sigma_c) matches or
  beats the specialists at both levels with half the parameters (`runs/multi_resflow_train.py`; M_reg_psc,
  M_resflow_psc); the reg2 control (M_reg2_psc) shows a second regression stage raises r by 0.02 and worsens the
  density excess — the flow's job is the power only.
* §4 chain test (`runs/chain_martingale.py`): each step is conditionally unbiased given its OWN input (gamma_D ~ 0),
  so it passes upstream low-k errors through (beta 0.85 at k < 0.5 k_Ny,64) and repairs them only in its correction
  band (beta 0.28 near k_Ny,64); chain error = persistence of the first step's error, not a random walk; r_delta of the
  chained 128 vs 256 drops to 0.72 (direct 0.90).
* §1 true conditional samples (`runs/resample_octave_ic.py` -> 10 MP-Gadget runs -> `runs/condvar_analysis.py`,
  sim_output/.../dmo-128-resample/set14): Var(fine|F_n)/P = 0.30 -> 0.71 across the octave; the same-IC rerun is
  reproducible only to r 0.915 at k_Ny,128 in Psi but 0.998 in density; TRUE conditional samples have
  P_delta/P_true = 1.01 +- 0.02 at r_delta 0.94 — the flow's -27..-30% deficit is its conditional being wrong, not
  the price of r < 1. The emulator's r ceiling (0.91) is a model limit (physics floor 0.998 in density).
* GPUs: HENON hold job 1216965 (kept), TWIG hold job 1234452 (released after C2c). One training per GPU.
* NEXT (owner's call): the flow's conditional is the target. CORRECTION 2026-09-17: the Q_E form is NOT a
  candidate — REPORT_7 already refuted it on the single-stage flow (qe moved P_delta DOWN, 0.963 -> 0.905; qj won);
  the session's earlier suggestion overlooked that. Candidates that stand: training/validating against the true
  conditional samples (30 min per 128^3 run), the shared operator for the 64->512 chain, and first the
  training-free localisation `runs/hybrid_detail_test.py` (what in the octave detail carries the density).
  Per-box scatter: quote set14 AND set15 before any claim.

## 10. Cluster session 2026-09-17 — what the deficit is, and what does NOT fix it (RUNLOG 10:50 .. 16:30)

* Second box (set15) confirms every C2 statement (scatter +-0.05 in P/P, +-0.03 in r_delta).
* Hybrid-detail test (`runs/hybrid_detail_test.py`): the true coarse band with NO detail has an excess (+6..16%) and
  r_delta 0.95; phase-randomised true detail destroys the power (0.14); even the detail of a TRUE conditional sample on
  a 1-9%-different coarse band costs -27% = the flow's deficit. Density power = mutual coherence of the bands.
* Oracle-coarse (`--oracle-coarse`): r_delta 0.906 -> 0.967 (= the native run's own): the r_delta ceiling is 100% the
  correction band; but only ~1/3 of the power deficit is upstream (-19.5% -> -12.6%).
* Loss-shaping ledger at 32->64 (0.9 k_Ny,64, control 0.805): weight sharing +0.05, `--jac-weight` 1.2 +0.055 (shallow,
  ~+0.02 per decade), Q_E negative (REPORT_7), `--cic-weight` strongly NEGATIVE in every compression (double penalty ->
  the net adds incoherent displacement power), dedicated correction-band loss 0 (capacity is not the limit), Y64 label 0.
* Therefore: no expectation-of-a-pointwise-error loss closes the gap, and the correction band is limited by the INPUT
  information, not by capacity or loss. Candidates that remain, all the owner's call: (a) more input information for
  the correction band — the coarse run's VELOCITIES are the obvious one (ask-before list); (b) a capacity/receptive-field
  probe for the detail coherence (cheap, not yet done); (c) accept the two-stage shared operator + jac-weight ~1 as the
  production model and move to the 64->512 chain (needs patch cropping: ask-before list).
* GPU: HENON hold job 1216965 expires ~2026-09-18 16:30; TWIG released.
