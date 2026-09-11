# Report 7 — the admissible metrics, the sampler prior, and the closure of the Eulerian gap

Overnight 2026-09-10/11, 32→64 self-run set, 3000 steps each, `--octave-sampler full`,
one GPU job at a time, commit recorded per run. Companion: REPORT_7a (Part 1 tooling and
constructive tests). Figures: `runs/fig8_qj_closure.png` (headline),
`runs/fig_eulerian_tests.png`, per-run `eulerian.png`.

## 0. An incident first: the first qe/qj runs had a drifting lambda (fixed, rerun)

`--lambda-e` is derived once at step 1 (v = 0, so the terms are the baseline terms) but was
not persisted in `train_state.pt`: every 900-s resume chunk re-derived it from a partially
trained model. E_flow_qe's lambda grew 0.0442 → 0.297 and its `r` fell to 0.674. The affected
runs are kept as `*_drift` variants; lambda is now saved/restored (verified with a two-chunk
test) and all five runs were retrained clean. Incidentally the drift runs bracket the lambda
response: qe at effective ~0.3 degrades everything Lagrangian; qj at ~0.007–0.010 already
improves `r`.

## 1. The seven-run table (all clean, pinned lambda)

Lagrangian (octave band) | Eulerian (CIC density) | generative:

| run | λ | `r` | `P/P` | rms | m/s | `P_δ/P`@1 | @1.5 | @2 | m>100 | kernE | gen Lag `P/P` | gen `P_δ`@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow, lag only | — | 0.7327 | 0.976 | 0.409 | .491/.370 | 0.963 | 0.777 | 0.551 | 0.0401 | 13.9 | 0.981 | 0.860 |
| flow + qe | .0442 | 0.7154 | 1.007 | 0.425 | .508/.386 | 0.905 | 0.665 | 0.418 | 0.0366 | 12.3 | 1.011 | 0.827 |
| **flow + qj** | .0100 | **0.7460** | 0.949 | **0.393** | .470/.356 | **1.042** | **0.918** | **0.701** | 0.0422 | 13.8 | 0.940 | **0.969** |
| flow + qe + inputs | .0442 | 0.7141 | 1.014 | 0.427 | .507/.389 | 0.898 | 0.664 | 0.416 | 0.0358 | 12.9 | 1.022 | 0.816 |
| reg, lag only | — | 0.8289 | 0.721 | 0.307 | .373/.275 | 1.435 | 1.866 | 2.170 | 0.0616 | 23.1 | 0.680 | 1.449 |
| reg + qe | .0442 | 0.8180 | 0.725 | 0.316 | .383/.283 | 1.376 | 1.676 | 1.782 | 0.0585 | 21.0 | 0.689 | 1.368 |
| reg + qj | .0100 | 0.8256 | 0.701 | 0.311 | .377/.279 | 1.447 | 1.921 | 2.253 | 0.0610 | 22.6 | 0.662 | 1.428 |

(truth m>100 = 0.0524; end-of-training term balance: qe L_mse 9.1e-2 vs λ·L_qe 2.5e-2;
qj L_mse 8.4e-2 vs λ·L_qj 4.7e-2.)

## 2. Verdicts on the four expectations

**(a) "qe moves emulator `P_δ` up, kernel fraction well below 1, generative unmoved" —
REFUTED for qe, ACHIEVED by qj.** qe moved `P_δ` *down* at every probed k (0.963→0.905 at
`k_Ny,c`, 0.551→0.418 at 2`k_Ny,c`) and the kernel fraction barely moved (13.9→12.3, nowhere
near <1). Admissibility held — the generative Lagrangian numbers stayed put — but the intended
Eulerian benefit failed. The qj run delivered exactly what (a) asked of qe: `P_δ` up at every
k, and simultaneously the best Lagrangian `r` (0.7460) and rms (0.393) of any flow run so far.

**(b) "qj matches qe in single-stream, does less in multi-stream and on the kernel fraction" —
INVERTED.** qj beats qe everywhere: T4 single-stream deposit 1.049 vs 0.913, multi-stream rms
0.470 vs 0.508, and neither run moved the Q_E kernel fraction. The note's §5.7c kernel
argument does not describe what limits these networks.

**(c) "the regression's excess does not go away under qe/qj" — CONFIRMED**, with one nuance:
qe did trim the regression's tail noticeably (2.170→1.782 at 2`k_Ny,c`, m>100 0.0616→0.0585)
while qj left it unchanged. The conditional-mean compaction is intrinsic, as §4.2 says.

**(d) "eulerian-inputs raises multi-stream `r` more than single-stream" — REFUTED.** The
channel changed nothing measurable (every number within noise of plain qe; multi-stream rms
0.507 vs 0.508). At this size the network evidently does not need the pulled-back density.

## 3. Why qj works where qe does not (reading, backed by Part 1)

Part 1's T1 showed the flow's harmful residual behaves as *incoherent small-scale displacement
noise* (its Eulerian effect is smearing), spread everywhere — not concentrated where Q_E's
density weighting looks. Q_J is a gradient-based form (`tr[adj(A) ∂e]`, an effective k²
weight) with caustic weighting `1/(J²+ε²)`: it penalises exactly the small-scale incoherent
component, everywhere. Q_E's deposited form both concentrates on dense regions and admits
cancellations (its kernel); the kernel fractions show no run pushed its residual there, so in
practice qe acted as a mild extra damping. The note's prediction ranked the two forms by their
kernels; the data ranks them by what they do to the noise component that actually matters.

## 4. The sampler prior: generative = emulator once the prior is right

(From last night, RUNLOG 23:20, extended to the qj model today.) Sampling the octave with
MP-GenIC itself as the oracle — the exact joint prior, including the phase-locked "transverse"
lattice component — makes generative mode coincide with emulator mode on the same checkpoint:

| generative `P_δ/P` | @`k_Ny,c` | @1.5 | @2 |
|---|---|---|---|
| lag-only + independent-T (start of Stage 7) | 0.860 | 0.617 | 0.387 |
| qj + independent-T | 0.969 | 0.796 | 0.552 |
| **qj + GenIC oracle** | **1.036** | **0.887** | **0.683** |
| qj emulator (upper reference) | 1.042 | 0.918 | 0.701 |

Two oracle seeds agree to 0.003. **The generative Eulerian ratio at `k_Ny,c` is now 1.04 —
the goal the owner set ("r < 1 is fine, the sampled P ratio must be 1, as the GAN series
showed") is met at this scale**, and the remaining tail deficit (0.68 at 2`k_Ny,c`) is shared
with emulator mode, i.e. it is model residual, not generative machinery. The two fixes are
independent and multiplicative: qj closes emulator→truth, the correct prior closes
generative→emulator.

The independent-transverse sampler (`--octave-sampler full`) is hereby demoted: it has the
best Lagrangian marginal and the worst Eulerian tail of the three samplers — do not use it for
generative evaluation on real data. The oracle is not yet a production sampler (it calls the
IC generator; fine in practice at 0.2 s per 64³ draw, but the note's "implement the
generator's kernel" route failed — the octave displacement is not a per-mode function of the
recorded ICDensity, so a faithful standalone sampler needs the generator's actual recipe).

## 5. Where this leaves the method statement

* **Flow + Q_J + correct prior** is the first configuration that satisfies the project's
  generative requirement (P ratio ≈ 1 in both spaces at `k_Ny,c`) while *also* improving the
  Lagrangian numbers. Costs: octave-band marginal `P/P` 0.976→0.949 and gen Lag `P/P` 0.940 —
  a mild Lagrangian power undershoot that a small λ sweep may fix.
* **Regression stays the emulator champion** (`r` 0.829, rms 0.307) and stays Eulerian-broken;
  qe trims but does not fix it. Use it where the true octave is available and pointwise
  accuracy is the goal; never for generation.
* The Q_E form, the eulerian-inputs channel, and (from Part 2) the biased jac term are all
  dominated by qj on this box: qe hurts the tail, the inputs channel does nothing, the biased
  term fixes the spectrum but drifts the generative mode and fixes neither peaks nor kernel.

## 6. Next

1. λ sweep around qj (0.005 / 0.01 / 0.02) to recover the last few % of Lagrangian marginal.
2. qj at 64→128 (batch 1) — does the tail improvement survive the harder level?
3. Productise the prior: either on-the-fly GenIC draws in the sampler path, or reverse the
   generator's recipe properly (per-mode kernel is refuted; next candidate is its actual
   potential/gradient pipeline).
4. The 2LPT toy's `--rms-delta` path and CLAUDE.md still describe the pre-Stage-7 world;
   after the owner reviews this report, the method statement and defaults need one pass.
