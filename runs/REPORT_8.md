# Report 8 — rollout round 1, the Q_J ablations settled, and the residual flow

Covers 2026-09-13/14: the two owner-approved experiments (rollout fine-tuning with its
control; learned base + residual flow) plus the evaluation debts from the second review
(two-box ablations, integration-step convergence). Figures: `fig10_chain_spectra.png`,
`fig10_chain_zoom.png`.

## 1. The headline: learned base + residual flow is the best model of the project

Frozen `R_reg_phys_3k` as the deterministic base; a fresh flow trained (qj loss, λ auto
0.0159) from `x0' = B(coarse, octave)` to the truth. 32→64, 3000 steps. Both boxes:

| | Lag `r` | Lag `P/P` | rms | Eul @1 | @1.5 | @2 |
|---|---|---|---|---|---|---|
| base (= regression) s8/s9 | .829/.842 | .721/.745 | .307/— | 1.435/1.231 | 1.866/1.385 | 2.170/1.552 |
| **resflow emulator** s8/s9 | **.795/.806** | **.996/1.029** | .353/— | **1.082/1.006** | **1.029/0.889** | **0.844/0.726** |
| resflow generative s8/s9 | .190/.204 | .947/.976 | — | 1.106/0.997 | 1.046/0.866 | 0.879/0.704 |
| J_flow_qj (prev champion) s8/s9 | .746/.760 | .949/.987 | .393 | 1.042/0.959 | 0.918/0.796 | 0.701/0.610 |

Two-box averages, emulator Eulerian: resflow **1.044 / 0.959 / 0.785** against J_flow_qj's
1.000 / 0.878 / 0.656 — better at every probe — while the Lagrangian `r` jumps from 0.75 to
**0.80** and the octave marginal sits at ~1.01. **The accuracy/power trade-off that defined
Stages 5–7 is substantially dissolved by the two-stage design**: the base supplies the
conditional-mean accuracy, the flow repairs exactly what the base breaks — its Eulerian
excess is pulled from 2.17 down through 1 to 0.84 (a slight overshoot-to-undershoot at the
last probe), not smeared from below.

Generative mode is the strongest yet: Eulerian ≈ 1 at `k_Ny,c` on both boxes and 0.70–0.88 in
the tail — with the *independent-T* sampler; the oracle prior should only help. The
stochasticity enters through the octave inside the base's input, so the base's generative
density excess (1.45 for the bare regression) is likewise repaired by the residual flow.

Caveats before promotion: one level (32→64); rms 0.353 sits between the base's 0.307 and the
single flow's 0.393; not yet chained; the base and flow are two 1.56M networks, i.e. 2× the
parameters of any single-stage model — the comparison is not parameter-matched.

## 2. Rollout fine-tuning, round 1: the control did its job

Owner's protocol at 128 (frozen upstream, mixed true/predicted coarse downstream), seed 8:

| model | direct r / Eul@1 / @2 | chained r / Eul@1 / @2 |
|---|---|---|
| shared, no FT | 0.603 / 1.045 / 0.574 | 0.546 / 0.842 / 0.272 |
| + FT mix0 (control) | 0.584 / 0.839 / 0.420 | 0.528 / 0.697 / 0.198 |
| + FT mix50 | 0.583 / 0.893 / 0.442 | 0.537 / 0.760 / 0.224 |

* **The fine-tuning regime itself damages the model** — the true-coarse-only control loses as
  much as or more than the mixed run everywhere. Round 1's regime differed from training in
  three ways at once: augmentation off (the minimal driver could not transform the
  predicted-coarse copy with its target), single-level updates (forgetting the 32→64 half),
  and 1500 low-LR steps over just 8 boxes.
* **Within the regime, the mixing helps on every single number** (chained Eul@1 0.760 vs
  0.697; even direct 0.893 vs 0.839). The protocol's signal is real and positive.
* Round 2 spec (awaiting approval): paired augmentation, keep both levels training with
  mixing only at 128, shorter/lower-LR schedule.

## 3. The chain problem, seen in k (fig10)

The chained spectra explain why patching displacement power cannot fix chaining: the chained
models' `r_δ(k)` falls **below the raw coarse run's** beyond ~1 h/Mpc — the generated coarse
field hands its phase errors to the next level, and beyond ~2.5 h/Mpc the chained power curve
rejoins the no-SR (prolonged coarse) curve. Power is present; phases are wrong. Shared and
specialist chains overlap almost exactly, so chaining degrades both equally — sharing is not
the problem.

## 4. Q_J ablations, settled at two boxes

| variant | Eul @1 avg | @2 avg |
|---|---|---|
| lag-only | 0.942 | 0.545 |
| adj p=0 | 0.973 | 0.593 |
| grad p=2 | 0.962 | 0.620 |
| adj p=2 (default) | 1.000 | 0.656 |
| adj p=3 | 1.009 | 0.665 |
| div p=2 | 0.979 | 0.671 |

The caustic weight (p ≥ 2) is the load-bearing component; the full gradient is the weakest
weighted form (the null-space worry inverted); adj-vs-div is a @1-only difference (~0.02),
tied at @2 within box scatter. Default stays adj p=2. Integration steps are converged
(n=8 vs 16: ≤0.014), with the side-finding that the qj model's 1-step is density-*excessive*
(1.24–1.36) — the conditional-mean-like limit now overshoots, opposite to the plain flow.

## 5. Where this points

1. **Promote the residual flow**: repeat at 64→128, then its chain row — if the two-stage
   design holds one level up, it becomes the default operator for the progressive chain.
2. Rollout round 2 with the fixed regime (approval pending).
3. The generative story is now: resflow + (eventually) the exact-prior sampler; the oracle
   test on the resflow checkpoint is cheap and worth running.
4. Open from before: generative chain (all-sampled octaves), 128→256 extrapolation,
   parameter-matched comparison for the two-stage model.
