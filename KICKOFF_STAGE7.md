# Stage 7 — while C3/C4 run: Eulerian evaluation tooling and the constructive tests; afterwards the quadratic-form runs (paste into the local Claude Code session)

---

State: `runs/cic_train.sh` is running C3 (`--jac-weight 0.04`) then C4 (`--jac-weight 0.012`) at 32->64, 3000 steps, `--octave-sampler full`, in 900-s chunks with `--resume` — so the script **re-imports `octave_flow_toy.py` every 15 minutes**. Rule for this whole stage: **do not edit `octave_flow_toy.py` (or `phase0_octaves.py`) until `########## CIC TRAIN DONE` appears in `runs/cic_train.log`.** Everything before that is CPU-only and lives in new files. One GPU job at a time, as always.

Read first: `notes/eulerian_power.pdf` (§5.7 is about the J-loss you implemented: the diagnostic is right; `asinh(J)` on the x-prediction is the same kind of biased term as `--cic-weight`, §5.5; the admissible form is the Jacobi-linearised quadratic form `Q_J`, and it has no relabelling kernel, unlike `Q_E`; on the `h_f` grid `J` is `1/rho` only in single-stream regions), then `KICKOFF_STAGE6.md` for the definitions of T1–T4 and of the `qe`/`qj` flags. This prompt supersedes the ordering in Stage 6. Append to `runs/RUNLOG.md`.

**Part 1 — now, CPU only.**

1. `python eulerian_metric.py` on the Mac (CPU). Expected: mass conservation 0, first-order identity ~1e-2, kernel test 0.25/0.041/0.016, Jacobi identity ~1e-4. Record. If `scatter_add`/`torch.linalg.det` misbehave on your torch, say which and do not patch around it silently.

2. New file `runs/eval_eulerian.py` (CPU; reuse the checkpoint/test-box logic of `runs/make_figures.py`, do not import anything that would change the training script). For a run directory it rebuilds the test box, produces the emulator (8 Heun), 1-step, and generative predictions (plus baseline `x0`, coarse `P Psi_c`, truth), and writes `<run>/eulerian.json` and `<run>/eulerian.png` with:
   * `P_delta(k)/P_delta,true(k)` and `r_delta(k)` from CIC on the fine grid (`eulerian_metric.cic_deposit` or the numpy CIC), tabulated at `k_Ny,c`, `1.5 k_Ny,c`, `2 k_Ny,c`;
   * mass fraction at `delta > 10` and `delta > 100`, max density;
   * `J = det(I + dPsi/dq)` (spectral gradients, same as `jac_loss`) quantiles 1/5/25/50/75/95/99 % and the fraction `J < 0`, for prediction and truth;
   * the kernel fractions of the residual `eps = Psi_pred - Psi_true` at the **true** Eulerian positions: `Q_E(eps)/mean(eps^2)` (`eulerian_form`, no k-weight) and `Q_J(eps)/mean(eps^2)` (`jacobian_form`, p=2, eps=0.1, state = truth), each divided by the same ratio for a Gaussian random field with the residual's power spectrum (3 realisations);
   * the same `P_delta` ratio from a deposit of **single-stream particles only** (coarse-field `det(I + D^L) < 0` mask upsampled x2, same mask for every field) — this is T4.
   Run it on `F_flow_128_3k`, `F_reg_128_3k` (64->128) and on the 3000-step `R_flow_phys`, `R_reg_phys` (32->64); one table per level in the RUNLOG. The 64->128 numbers must reproduce REPORT_6 §4b (0.714 / 1.919 / 0.323 at `k_Ny,c`) — if they do not, the new script is wrong, not the report.

3. Constructive tests T1 and T2 of Stage 6 on the 64->128 test box, and again at 32->64 (cheap): `Psi_c + a d_true (+ n)` with `(a, sigma_n)` recomputed from each level's own `r`, `P/P`, `Var(d)`; overplot on the measured curves from step 2. Predictions to confirm or contradict: flow curve ≈ `exp(-k^2 sigma_n^2) x P_delta[Psi_c + a d_true]` to ~10 % up to `k_Ny,c`; regression curve ≈ `P_delta[Psi_c + 0.73 d_true]`, excess rising with k; coherent-only ratio at `a ≈ 0.57` already close to 1. Figure `runs/fig_eulerian_tests.png`. Also compute the J quantiles of the constructed fields: the note says noise **broadens** the J distribution and shrinkage **narrows** it (by `a^3` inside multi-stream patches) — check the direction on real data.

4. `runs/REPORT_7a.md`: the two evaluation tables, the T1/T2 figure with predictions next to measurements, the kernel fractions (expected ~1 for all current models — the residual is in general position), and the T4 split (expected: the regression's excess disappears in the single-stream deposit, the flow's deficit shrinks but stays). Flag every contradiction rather than explaining it away.

**Part 2 — when C3/C4 have finished** (`CIC TRAIN DONE`): `runs/eval_eulerian.py` on `runs/C_flow_jac04` and `runs/C_flow_jac012`; one table against the 3000-step `R_flow_phys` (same level, same steps, same sampler) with three column groups: Lagrangian (octave `r`, `P/P`, rms multi/single), Eulerian (`P_delta/P_true` at `k_Ny,c` and `1.5 k_Ny,c`, `r_delta`, `delta > 100` mass fraction, J quantiles, both kernel fractions), and **generative mode** (octave-band `P/P`, sample-vs-sample `P/P`, generative `P_delta/P_true`). Read it with §5.5/§5.7 in mind: the jac term is a nonlinear loss on the x-prediction, so its emulator numbers may improve while the *generative-mode* numbers drift away from `R_flow_phys`'s — that drift is the bias, report it as such; if the predicted J quantiles match the truth's but `P_delta` does not move, that is §5.7e (J at `h_f` inside halos is roughness, not density). Also report the `lambda` actually used and the two loss terms at the end of training.

**Part 3 — code, only after Part 2** (small, additive, `octave_flow_toy.py`; the default path must stay bit-identical — 16->32 20-step smoke test loss `3.0888e-02` before anything else runs on the GPU):
   * `--loss-metric {lag,qe,qj}` (default `lag`), `--lambda-e`, `--euler-positions {coarse,interp}`, `--jac-p` (2), `--jac-eps` (0.1), exactly as in Stage 6 B1: `e = v - (x1 - x0)`; `qe` adds `lambda_e * eulerian_form(e, pos)` with `pos` from the coarse field (or `x_t.detach()`), `qj` adds `lambda_e * jacobian_form(e, state, p, eps, fft_device)`. These forms never see `x1`; keep it that way. Print both loss terms at step 1 and at the baseline; the default `lambda_e` equalises them at step 1 — record `L0_qe`, `L0_qj`.
   * `--eulerian-inputs` (Stage 6 B2), flag stored in the checkpoint.
   * Then the queue, 32->64, 3000 steps, `--octave-sampler full`, chunked like `cic_train.sh`, strictly sequential:
     `runs/E_flow_qe` (`--loss-metric qe --lambda-e L0_qe`), `runs/J_flow_qj` (`--loss-metric qj --lambda-e L0_qj`), `runs/E_flow_qe_in` (`qe` + `--eulerian-inputs`), then `runs/E_reg_qe` and `runs/J_reg_qj` (regression) if time allows.
   * Evaluate each with `runs/eval_eulerian.py`; final table = R_flow_phys / C3 / C4 / E_flow_qe / J_flow_qj / E_flow_qe_in (+ regressions), same three column groups.

Expectations to test, not to assume: (a) `qe`/`qj` leave the generative-mode numbers where `R_flow_phys` has them (no bias) while moving emulator `P_delta` up and the `Q_E` kernel fraction below 1; (b) `qj` matches `qe` on the single-stream deposit and does less on the multi-stream part and on the `Q_E` kernel fraction; (c) the regression's Eulerian excess does **not** go away under `qe`/`qj` (the minimiser is still the conditional mean) — if it does, say so loudly; (d) `--eulerian-inputs` raises `r` in multi-stream patches more than in single-stream ones. `runs/REPORT_7.md` at the end: tables, figures, five sentences tied to (a)–(d) and to the T1/T2 outcome.

Rules: physics conventions unchanged; no edits to the training script while `cic_train.sh` runs; small additive diffs; RUNLOG append-only; commit before each GPU queue (results.json records the hash); one GPU job at a time.

---
