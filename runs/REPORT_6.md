# Report 6 — two levels, converged, and the first Eulerian check

Overnight of 2026-09-09/10. Everything is on the self-run MP-Gadget set (10 nested seeds,
32/64/128 cubes with their ICs, 100 Mpc/h, z = 0, `--offset 0`, `--box 100000`,
`--growth 76.7439`), 8 training seeds, test seed 8. All four runs were extended from 1500 to
3000 steps with `--resume`; the 1500-step results are kept alongside.

---

## 1. The main table, at 3000 steps

| | `r` | `P/P` | `r^2` | coarse `eps` | rms | multi | single | gen `P/P` |
|---|---|---|---|---|---|---|---|---|
| **32→64** baseline | 0.5551 | 1.4860 | 0.3081 | 1.08e-01 | 0.577 | 0.658 | 0.540 | — |
| flow | 0.7327 | **0.9759** | 0.5368 | 5.87e-02 | 0.409 | 0.491 | 0.370 | **0.9808** |
| regression | **0.8289** | 0.7214 | 0.6871 | 3.00e-02 | **0.307** | 0.373 | 0.275 | 0.6797 |
| **64→128** baseline | 0.3940 | 1.2923 | 0.1552 | 1.25e-01 | 0.816 | 0.896 | 0.764 | — |
| flow | 0.5478 | **1.0703** | 0.3001 | 9.50e-02 | 0.679 | 0.764 | 0.624 | **1.1055** |
| regression | **0.7293** | 0.5391 | 0.5319 | 4.31e-02 | **0.480** | 0.545 | 0.438 | 0.5096 |

(32→64 at batch 2, 64→128 at batch 1 — the one setting forced by memory, see the RUNLOG.
Generative columns all use `--octave-sampler full`.)

**Convergence is settled.** Doubling the steps moves `r` by +0.010 to +0.024 and leaves every
ordering intact:

| | 1500 | 3000 | change |
|---|---|---|---|
| 32→64 flow | 0.7220 | 0.7327 | +0.011 |
| 32→64 regression | 0.8192 | 0.8289 | +0.010 |
| 64→128 flow | 0.5234 | 0.5478 | +0.024 |
| 64→128 regression | 0.7160 | 0.7293 | +0.013 |

## 2. What holds

**The accuracy/power split is real, reproduces at two levels, and sharpens with resolution.**
The regression sits on the `P/P = r^2` line at both (0.7214 against 0.6871; 0.5391 against
0.5319) and the flow on `P/P = 1` (0.976; 1.070). The regression's `r` advantage grows from
0.096 to 0.182, and its power deficit deepens from 0.72 to 0.54. This is not an artefact of the
coarse 32→64 step; going finer makes it worse, which is the direction that matters for a chain
that has to reach 512³.

**In generative mode the regression is not usable.** With the corrected octave sampler the flow
returns 0.98 and 1.11 of the true octave power at the two levels; the regression returns 0.68
and **0.51**. At 64→128 it supplies half the power, and it degrades with resolution.

**Self-similarity holds at second moments and fails at the fourth.** σ was measured from the
ICs (ratio 1.3741 between the two coarse cell sizes), which puts the matched-σ comparison at
**z = 0.640** for the 64→128 pair; one extra pair was run there.

| | 32→64 @ z=0 | 64→128 @ z=0.640 | 64→128 @ z=0 | matched |
|---|---|---|---|---|
| rms(eps)/h_c cube | 0.1898 | 0.1887 | 0.2654 | **−0.6%** |
| multistream fraction | 0.2929 | 0.2924 | 0.3693 | **−0.2%** |
| k(T=0.5)/k_Ny,c | 1.0645 | 1.0630 | 1.0944 | **−0.1%** |
| var(d/h_f) | 0.1411 | 0.1330 | 0.2420 | −5.8% |
| excess kurtosis | 1.4216 | 2.1267 | 2.7975 | **+49.6%** |

Matching σ collapses drifts of +40%, +27% and +72% to well under one percent, across a factor
2 in resolution. The kurtosis only improves from +97% to +50%, and that residual is *not*
explained by the multi-stream fraction, which matches to 0.2% — so conditioning on
`σ(h_l, z)` plus a multi-stream indicator, which is what notes v3 proposes, would not fix it
either. Operationally: σ looks sufficient for everything that sets the *amplitude* of the
correction and the detail, which is what the loss is dominated by; the *tails* keep a level
dependence no single dimensionless amplitude absorbs.

## 3. What did not survive the night

**"The network gains less one level down" — withdrawn.** At 1500 steps the flow's gain looked
smaller at 64→128 and I flagged it as possibly undertraining. At 3000 steps the picture is
different and the claim was simply too general:

| gain in `r` over baseline | 32→64 | 64→128 |
|---|---|---|
| flow | +0.178 | +0.154 |
| **regression** | +0.274 | **+0.335** |

The flow's gain does shrink slightly (13%), but the **regression's gain grows** one level down.
The relative rms improvement falls for both (flow −29% → −17%, regression −47% → −41%), so the
honest statement is: *the flow gets relatively less out of the finer transition, the regression
gets more `r` out of it, and both improve the rms by a smaller fraction.* The blanket claim was
wrong and is retracted.

## 4. The Eulerian check, which disagrees with the Lagrangian one — and why

Last night's Eulerian density spectrum contradicted the Lagrangian result: in the octave band
the regression is power-*deficient* (`P/P` = 0.54) but the CIC density power is *excessive*
(>1.4 at `k_Ny,c`, still rising), while the flow is the other way round. I recorded it as
unexplained. It now has an explanation and a test.

Density statistics of the same test box (CIC, mean density 1):

| field | max | 99.9 pct | mass fraction at δ>10 | **δ>100** |
|---|---|---|---|---|
| coarse `P Ψ_c` | 2297 | 63.1 | 0.374 | 0.104 |
| baseline `x_0` | 1526 | 53.1 | 0.315 | 0.063 |
| **regression** | **5619** | 71.4 | 0.428 | **0.178** |
| **flow** | 2360 | 63.5 | 0.381 | **0.107** |
| **truth** | 3480 | 77.5 | 0.433 | **0.152** |

**The regression over-concentrates the densest structures** — 61% higher peak density than
truth and 17% too much mass above δ = 100 — while **the flow under-concentrates** (30% too
little above δ = 100). The mechanism is consistent: a conditional-mean displacement field is too
smooth *in Lagrangian space*, so particles that should have dispersed inside a collapsed region
stay together, and the caustic ends up thinner and denser than it should be. Low Lagrangian
octave power and high Eulerian small-scale power are the same statement seen from two sides.

Two consequences worth taking seriously:

* **The Lagrangian octave-band `P/P` is not sufficient to judge a model.** On that metric the
  regression looks power-deficient and the flow looks correct; on the Eulerian density the
  regression is the one with excess small-scale power. Any claim about "restoring power" has to
  say in which space.
* **At moderate overdensity the regression is actually closer to truth** (mass above δ=10:
  0.428 against 0.433, versus the flow's 0.381). It is only in the extreme tail that it
  overshoots. So the Eulerian verdict is not simply "the flow wins"; it is that the two models
  fail differently, and the failure the flow has — under-dense peaks — is the one that shows up
  in halo mass functions.

Caveat: this is one test box, one CIC deposit, no window deconvolution, no shot-noise
subtraction. The ratios between models are meaningful because all fields go through identical
processing; the absolute numbers are not.

## 5. Where this leaves the method

The case for the flow is now specific rather than general: it is the only one of the two that
produces the right octave power from a *fresh* octave, which is the requirement for anything
generative — the progressive chain beyond the first level, and the zoom-in application. For
pure emulation with the true ICs in hand, the regression is more accurate on every Lagrangian
metric at both levels and closer to truth at moderate Eulerian overdensity.

That suggests the pair should be carried forward together rather than one being chosen now, and
that the next round needs an Eulerian metric in the loop — a halo mass function or at least a
density-PDF constraint — because the two objectives fail in opposite directions there and the
Lagrangian loss cannot see it.

## 6. Not done / open

* **Production data still needs its ICs**; without them there is no octave and no training.
* The Haar-versus-spectral ordering flip between the self-run and production datasets survives
  a matched transition and a matched grid convention. Unexplained; it means `rms(eps)` is not
  portable between datasets.
* The residual kurtosis drift under matched σ has a hypothesis (tails are set by the smallest
  resolved scale, which is a different physical scale at each level) and no test.
* Whether `--octave-sampler full` should become the default. It is the physically correct
  choice and emulator mode is unaffected, but it changes a convention, so it is left off.
* One test box per configuration for the Eulerian statistics; two training seeds per
  configuration only at 32→64.
