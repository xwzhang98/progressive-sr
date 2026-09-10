# Report 7a — Eulerian evaluation tooling and the constructive tests (Stage 7 Part 1)

CPU-only, while `runs/cic_train.sh` trains C3/C4 on the GPU. New: `runs/eval_eulerian.py`
(per-run Eulerian evaluation -> `<run>/eulerian.json` + `.png`), `runs/fig_eulerian_tests.png`
(T1/T2). The training script was not touched.

## 0. Validation

* `python eulerian_metric.py` self-test: mass conservation 0, pullback identity 0,
  first-order identity 9.6e-3, kernel test 0.245/0.038/0.009 (expected 0.25/0.041/0.016 — the
  last bin has 54 pairs), Jacobi identity 1.1e-4. `scatter_add` and `linalg.det` behave on
  torch 2.14 CPU. Two harmless numpy warnings (a divide-by-zero inside the self-test's own
  field construction, and a deprecation about `irfftn(s=...)`); nothing patched.
* `runs/eval_eulerian.py` on the **1500-step** checkpoints reproduces REPORT_6 §4b exactly:
  flow 0.7136, regression 1.9193, baseline 0.323, coarse 0.677 at `k_Ny,c`. The script is
  consistent with the report.
* Kernel-fraction machinery control: a Gaussian "residual" gives 0.998 (Q_E) and 1.013 (Q_J).

## 1. The evaluation tables (3000-step models, `--octave-sampler full` generative)

**64→128** (`k_Ny,c` = 2.01 h/Mpc; probes at 1 / 1.5 / 2 `k_Ny,c`):

| | `P_δ/P` @1 | @1.5 | @2 | `r_δ`@1 | mass δ>100 | kernel E | kernel J | T4 single-stream @1 |
|---|---|---|---|---|---|---|---|---|
| coarse | 0.677 | 0.437 | 0.255 | — | 0.104 | — | — | — |
| baseline | 0.323 | 0.085 | 0.021 | — | 0.063 | — | — | — |
| flow emu | 0.680 | 0.439 | 0.285 | 0.975 | 0.112 | **16.5** | 0.54 | 0.686 |
| flow gen | 0.562 | 0.332 | 0.200 | — | — | — | — | 0.567 |
| reg emu | 1.898 | 2.298 | 2.627 | 0.971 | 0.181 | **29.6** | 0.24 | **1.887** |
| reg gen | 1.892 | 2.288 | 2.605 | — | — | — | — | 1.884 |
| truth | 1 | 1 | 1 | 1 | 0.152 | — | — | 1 |

**32→64** (`k_Ny,c` = 1.01 h/Mpc):

| | `P_δ/P` @1 | @1.5 | @2 | `r_δ`@1 | mass δ>100 | kernel E | kernel J | T4 @1 |
|---|---|---|---|---|---|---|---|---|
| flow emu | 0.963 | 0.777 | 0.551 | 0.973 | 0.040 | **13.9** | 0.58 | 0.971 |
| flow gen | 0.860 | 0.617 | 0.387 | — | — | — | — | 0.865 |
| reg emu | 1.435 | 1.866 | 2.170 | 0.975 | 0.062 | **23.1** | 0.42 | **1.415** |
| truth | 1 | 1 | 1 | 1 | 0.052 | — | — | 1 |

Two incidental findings from the tables alone:

* **More Lagrangian training worsened the flow's Eulerian power**: 0.714 (1500 steps) →
  0.680 (3000) at 64→128, while octave `r` rose 0.523 → 0.548. The Lagrangian loss is not
  merely blind to the Eulerian failure; at this stage it trades against it.
* **The corrected octave sampler made generative Eulerian power *worse***: the 1500-step
  longitudinal-sampler measurement gave 1.109 at `k_Ny,c` (REPORT_6), the full sampler gives
  0.56–0.59. The transverse component restores Lagrangian octave power, but as a *sampled*
  (uncorrelated) field it acts as independent displacement noise, i.e. pure Eulerian smearing.
  Same mechanism as T1 below; the Lagrangian and Eulerian verdicts on the sampler fix point in
  opposite directions.

## 2. T1 — smearing: confirmed for the baseline, refuted for the flow

`(a, σ_n²)` recomputed from each level's own octave `(r, P/P)` and `Var(d)` via
`a = r √(P/P)`, `σ_n² = (P/P)(1−r²) Var(d)` (this is the combination that reproduces the
note's §3.1 table):

| 64→128 | a | σ_n² | measured `P_δ/P`@`k_Ny,c` | constructed (2 reals) | exp(−k²σ_n²)×coh |
|---|---|---|---|---|---|
| baseline | 0.448 | 0.275 | **0.323** | 0.341 / 0.343 | 0.34 |
| flow | 0.567 | 0.189 | **0.680** | 0.407 / 0.407 | 0.41 |

(32→64: baseline measured 0.452 vs constructed 0.51; flow measured 0.963 vs constructed 0.617.)

The baseline **is** the independent-noise model: construction and analytic damping land within
6% (13% at 32→64). The flow is **not**: it beats its own construction by 0.27 at 64→128 and
0.35 at 32→64. Its residual is therefore substantially *not* independent noise — consistent
with the kernel fractions below. The note's eq. (6) is verified as a formula (baseline) and
its application to the flow in §3.3 overestimates the damage.

## 3. T2 — shrinkage: refuted, and the reason is structural

No band-limited shrinkage reproduces the regression. At 64→128, `Psi_c + a·d_true` gives
`P_δ/P(k_Ny,c)` = 0.660 / 0.599 / 0.542 for a = 0.5 / 0.73 / 0.9 — all *deficits*, while the
regression measures **1.898**; mass at δ>100 is 0.097–0.107 against the regression's 0.181.
The predicted "excess rising with k, near the a = 0.73 line" does not appear at any a.

Two structural facts emerge:

* **The shrink family is non-monotonic in the wrong direction**: *more* true detail (a 0.5→0.9)
  gives *less* Eulerian power at `k_Ny,c` (0.660→0.542), even though a = 1 with the true coarse
  band would give exactly 1. The constructions sit on the *uncorrected* coarse band `P Ψ_c`;
  detail phased against the *true* coarse band decoheres against the coarse run's misplaced
  structures. **The coarse-band correction is a first-order actor in the Eulerian budget**, and
  the note's §3 model (which manipulates only the octave) has no term for it.
* Consequently the kickoff's "coherent-only ratio close to 1 at a ≈ 0.57" is also refuted:
  measured 0.645.

The regression's excess therefore does not come from band-limited damping of the true detail.
The compaction that produces it (§3.4's `a³` argument) acts on the *full internal displacement*
of collapsed patches — which a band-limited scaling of `d_true` cannot imitate. T2 as designed
cannot test §3.4; it tests (and refutes) a band-limited version of it.

## 4. T3 — kernel fractions: refuted in an informative direction

Predicted ~1 ("residual in general position"). Measured, at the true Eulerian positions:

| | kernel E | kernel J |
|---|---|---|
| flow 32→64 / 64→128 | 13.9 / 16.5 | 0.58 / 0.54 |
| regression 32→64 / 64→128 | 23.1 / 29.6 | 0.42 / 0.24 |

The Gaussian control gives 1.00, and with the positions replaced by the undisplaced grid
(uniform density weighting) the flow's 17.8 collapses to **1.14**. So the excess is entirely
the density weighting correlating with the residual: the error is *coherent exactly where
particles pile up* — coherent halo mislocation, amplified by ρ in the deposit. The residual is
*more* Eulerian-visible than a random field, not less.

Two consequences: (i) there is a large Eulerian-visible error component for `Q_E` to act on —
if the fraction had been ~1 (let alone <1), the Q_E program would have had little to push;
(ii) the number that must *drop* after Part C now has a long way to fall, and "drop below 1"
is a much stronger success criterion than the kickoff's phrasing implied. `Q_J` fractions are
already < 1: the network's residual is smoother in gradients than a Gaussian, so the Jacobian
form sees relatively less of it — first quantitative hint of §5.7c's prediction that `Q_J`
constrains less of the actual residual than `Q_E`.

## 5. T4 — multi-stream split: refuted

Predicted: the regression's excess disappears in the single-stream-only deposit. Measured
(same coarse-mask q-cut for every field): 64→128 regression **1.887** single-stream vs 1.898
full; 32→64 **1.415** vs 1.435. The flow's deficit likewise barely moves (0.686 vs 0.680;
0.971 vs 0.963).

The excess is *not* localised in the coarse-field multi-stream patches. This echoes the
panel-(c) finding (RUNLOG 2026-09-09): the coarse mask misses 56–62% of fine-level multi-stream
cells, and the over-concentrated structures the regression builds live largely in patches the
coarse mask calls single-stream. Any theory statement that localises the conditional-mean
failure "in the multi-stream patches" needs the fine-level mask, at least.

## 6. J-quantile direction check — confirmed

Noise broadens, shrinkage narrows, on real data: truth 1/99% quantiles (−6.8, 15.2) at
64→128; flow-type construction (−14.1, 21.4); shrink a = 0.73 (−5.4, 12.8). Matches §5.7a.

## 7. What Part 1 changes about Part 3

* `Q_E` has a large visible target (kernel fraction ≫ 1), so the qe run is well motivated.
* The coarse-band correction carries more of the Eulerian budget than the note's model
  admits; `--euler-positions coarse` (positions from `P Ψ_c`) is the right default anyway, but
  the interpretation of improvements should watch the coarse band, not only the octave.
* The regression's excess is not multi-stream-localised (T4), so expectation (b) for Part 3 —
  qj matching qe on the single-stream deposit — may not discriminate much; the fine-mask split
  would.
* Everything here is one test box per level; no error bars on the Eulerian numbers yet.
