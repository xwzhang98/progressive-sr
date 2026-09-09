# Stage 5 — the first real N-body results

Data: a self-run MP-Gadget set, 10 nested seeds at 32^3 and 64^3, 100 Mpc/h box,
Omega0 = 0.2814, z_init = 99 -> z = 0, built overnight because the Box production snapshots
ship without initial conditions and the physical coupling needs the fine IC octave.
Multi-stream fraction of the test box: **0.29** (the 2LPT toy has 0 by construction).
Training: 8 seeds, test seed 8, 1500 steps, base 24, batch 2, MPS, ~46 min per run,
strictly one GPU job at a time. Lengths in kpc/h (`--box 100000`), particles on cell corners
(`--offset 0`), `--growth 76.7439`.

---

## The R1–R4 table

Octave-band unless marked; `eps` is the coarse-band `P_eps/P`; rms in units of `h_f`.

| | `r` | `P/P_true` | `r^2` | coarse `eps` | rms | rms multi | rms single | gen `P/P` | gen `r` |
|---|---|---|---|---|---|---|---|---|---|
| baseline, raw linear octave | 0.5551 | 1.4860 | 0.3081 | 1.08e-01 | 0.577 | 0.658 | 0.540 | — | — |
| baseline, Wiener-filtered | 0.5550 | 0.3310 | 0.3081 | 9.96e-02 | 0.465 | 0.538 | 0.431 | — | — |
| **R1** flow, physical (seed 0) | 0.7220 | **0.9928** | 0.5213 | 5.64e-02 | 0.411 | 0.490 | 0.373 | 0.8109 | 0.165 |
| **R1b** flow, physical (seed 1) | 0.7261 | **0.9830** | 0.5272 | 5.73e-02 | 0.410 | 0.489 | 0.372 | 0.8048 | 0.171 |
| **R3** flow, Wiener (seed 0) | 0.6880 | **1.1266** | 0.4733 | 5.85e-02 | 0.434 | 0.515 | 0.396 | 0.8749 | 0.155 |
| **R2** regression, physical (seed 0) | **0.8192** | 0.6902 | 0.6711 | 3.13e-02 | **0.314** | 0.380 | 0.282 | 0.6516 | 0.198 |
| **R2b** regression, physical (seed 1) | **0.8201** | 0.6878 | 0.6725 | 3.07e-02 | **0.313** | 0.379 | 0.281 | 0.6486 | 0.200 |
| **R4** regression, Wiener (seed 0) | **0.8275** | 0.7035 | 0.6848 | 3.02e-02 | **0.309** | 0.374 | 0.278 | 0.6658 | 0.198 |

Seed-to-seed spread (the thing REPORT_3b said had to be measured before any of this could be
read) is negligible here — `r` differs by 0.0008 between the two regression seeds and 0.0041
between the two flow seeds, against a flow-vs-regression gap of 0.098. Every difference
discussed below is 20–100x the repeat scatter.

Source filters fitted on the training pairs:
`Tc(k/k_Ny,c) = 1.000, 1.018, 0.886, 0.677` at 0.25/0.50/0.75/0.95 and
`G(k/k_Ny,c) = 0.703, 0.494, 0.382, 0.329` at 1.05/1.30/1.60/1.90 — both as notes v3 expects.

![panel (c)](fig_conditional_real.png)

---

## The headline: the predicted trade-off is real, but it is a trade-off, not a win

**Regression is more accurate; the flow has the right power.** Across three regression runs
and three flow runs:

- regression `r` = 0.819–0.828, rms = 0.309–0.314, coarse `eps` = 3.0e-2
- flow `r` = 0.688–0.726, rms = 0.410–0.434, coarse `eps` = 5.7e-2
- regression `P/P_true` = 0.688–0.704, against `r^2` = 0.671–0.685 — **on the `r^2` line**
- flow `P/P_true` = 0.983–1.127 — **on the `P/P = 1` line**

This is exactly the pattern notes v3 predicts for an information-limited conditional, and it
is the opposite of the 2LPT toy, where the regression had `P/P = 0.995` and beat the flow on
every metric (REPORT_3b). The multi-stream regime does separate the two objectives.

**But the separation is not free, and Stage 5's phrasing understates the cost.** The prompt
predicted the flow would keep `P/P ≈ 1` *"at similar r"*. It does not: the regression reaches
`r = 0.82` where the flow reaches `0.72`, and its rms error is 24% smaller (0.31 vs 0.41 h_f).
The flow buys correct second-order statistics by giving up real point-wise accuracy. Which of
those matters is a modelling decision, not something these numbers settle.

**The gap is not localised in the multi-stream patches.** Stage 5 expected regression and flow
to agree in the single-stream part. They do not: the regression is better in both parts and by
almost the same factor (multi 0.380 vs 0.490, single 0.282 vs 0.373; the multi/single ratio is
1.35 for regression and 1.31 for the flow). Note the caveat — the printed split is of the rms
error, and the power deficit is a spectral quantity that is not split by region, so *"the
power deficit is localised in the multi-stream patches"* is not tested by anything measured
here. Testing it needs a masked power spectrum, which the script does not compute.

---

## Answers to the five questions

**(a) How much does the Wiener/propagator source shorten the path, and does it help?**
It shortens it by 19%: baseline rms 0.577 -> 0.465 h_f, and it moves the baseline from
`P/P = 1.486` to `0.331 ≈ r^2 = 0.308`, which is where the best linear prediction belongs.
**It does not improve the trained model.** For the flow it is worse (R3 `r` = 0.688, rms 0.434
against R1's 0.722 / 0.411 — 8x the seed scatter, so real); for the regression it is inside
the seed scatter (R4 0.8275 / 0.309 against R2b 0.8201 / 0.313). So the shorter path is not
the bottleneck at this resolution; the network already absorbs the linear rescaling.

**(b) Is the regression-vs-flow gap localised in the multi-stream patches?** No — see above.

**(c) Is the coarse-band residual the size predicted by `1 - r_cf^2`?** Yes, to within 5%.
With `r_cf` = 0.9464 the prediction is 0.1044 against a measured baseline `eps_rel` of 0.1076
(ratio 1.03); after `Tc` the prediction is 0.1040 against 0.0996 (ratio 0.96). The trained
models cut this to 3.0e-2 (regression) and 5.7e-2 (flow), i.e. the correction head removes
70% and 47% of a residual whose scale is set by the coarse run's own decorrelation.

**(d) What contradicts notes v3 / the Stage 5 expectations.** Four things, all recorded in the
RUNLOG rather than smoothed over:

1. **"at similar r"** — the flow's `P/P ≈ 1` costs 0.10 in `r` and 24% in rms, as above.
2. **Panel (c): the detail is *less* heavy-tailed inside the multi-stream patches, not more.**
   Excess kurtosis inside/outside is 1.30/1.37 (self-run) and 2.68/3.43 (production). I
   hypothesised this was an artefact of building the mask from the coarse field; rebuilding it
   from the fine field refutes that — the coarse mask does miss 56–62% of fine-level
   multi-stream cells, but the ordering does not change (1.22/1.48 self-run, 2.77/3.45
   production). Both masks separate the variance by only ~1.3x. Binning by the local coarse
   `delta_L` doubles the variance from the most underdense to the densest bin but leaves the
   kurtosis flat at 1.1–1.3 (self-run) / 2.4–4.5 (production), so conditioning on the local
   jet does **not** Gaussianise the detail on real data.
3. **The 2LPT coarse-only harmonics do not transfer to N-body.** A redshift scan (z = 9, 3, 0
   on one pair) gives `r(d_nl, harm)` = 0.135 -> 0.203 -> **-0.018**, collapsing exactly where
   the multi-stream fraction jumps from 0.004 to 0.291. At z = 0 the term carries 20% of the
   nonlinear detail power at *zero* correlation with it, so it is not a predictive component
   there; the toy's 24% at `r` = 0.45 is a property of pure 2LPT, not of the simulation.
4. **Generative mode under-produces power for both objectives.** Flow `P/P` = 0.81, regression
   0.65, where the 2LPT toy gave 1.02 and 1.00. The review's argument — an accurate
   deterministic map of a fresh Gaussian octave has the right distribution — needs the map to
   be accurate; at `r` = 0.72–0.82 it is not, and both models damp the sampled octave. This is
   the single biggest gap between the toy and the real data.

**(e) What the first cluster run should be.** 64 -> 128 on the production series, the
**regression with the physical coupling** as the primary and the flow as the paired ablation,
both at the same seeds, because (i) the regression is better on every accuracy metric here and
is the cheaper object to train, (ii) the flow is the only one with correct octave power, so the
pair is what makes the trade-off measurable at production resolution, and (iii) `--source-filter
wiener` should be dropped from the first round — it costs a fit and buys nothing measurable.
Seeds: the production set has 16, so 12 train / 2 validation / 2 test, and **at least two
training seeds per configuration**, which is what made tonight's table readable. Before any of
it, the production ICs are needed: without them there is no octave and no training at all.

---

## Two corrections to my own overnight work

**A bug I introduced into the results, found and fixed.** `--growth` was applied twice to the
sampled octave: `fit_linear_power` is given `ic_f * growth`, so `A_delta` already carries it,
and `Batcher.make` multiplied again. The sampled octave was 76.7x too large in amplitude and
5890x in power — R1's first generative line read `P/P_true = 7397`. Invisible on every toy run
(all use `--growth 1`), so no Stage 3 or 3b number is affected; on real data it invalidated only
generative lines, but it would have silently corrupted the training of any `--coupling
independent` run. Fixed, verified (7397 -> 0.811), and R1/R2 re-evaluated.

**A mistake in my re-evaluation, caught by an internal consistency check.** I first re-ran R2
without `--regression`, which integrates 8 Heun steps through a model trained only at t = 0,
and got `r` = 0.709 instead of 0.819. The emulator line should not depend on the growth fix at
all, and for R1 it did not — that discrepancy is what exposed it. The table above uses the
correct evaluation. Any future `--eval-only` of a regression checkpoint must pass
`--regression`.

## Not done

- Masked power spectra, needed to test whether the power deficit itself is localised in the
  multi-stream patches (question (b) as it was meant, rather than as the rms split can answer).
- Production-data training: still blocked, no ICs.
- The Haar-vs-spectral ordering flip between datasets survives a matched transition and a
  matched grid convention (cube 0.2629 vs haar 0.3013 self-run; cube 0.2745 vs haar 0.2432
  production). The spectral values agree between datasets to 4%; the whole difference is Haar.
  Unexplained, and it means rms(eps) is not portable between datasets. It does not touch the
  no-Haar convention, which rests on the random linear-order aliasing that rms(eps) never
  measures.

---

## Addendum, after reading notes v3 §"What the real N-body data changes" in full

Three checks the report above did not make. The first two confirm notes v3; the third is a
concrete contradiction already stated in the body, repeated here against the notes' wording.

### G(k) is the Gaussian damping the notes claim, but not at the displacement dispersion

Notes v3 identifies `G` as the Lagrangian propagator whose leading form is the
`exp(-k^2 sigma_Psi^2 / 2)` damping of Lagrangian resummation. Testable: invert each shell,
`sigma_eff(k) = sqrt(-2 ln G) / k`, which is constant if and only if the form is Gaussian.

| k/k_Ny,c | 1.00 | 1.25 | 1.50 | 1.75 | 1.94 |
|----------|------|------|------|------|------|
| G(k) | 0.703 | 0.527 | 0.417 | 0.350 | 0.321 |
| sigma_eff [kpc/h] | 835 | 900 | 877 | 823 | 774 |

**sigma_eff = 857 +- 41 kpc/h, constant to 4.8% across the whole octave band** — so the shape
is Gaussian to good accuracy (with a slight rise-then-fall that a pure Gaussian would not have).
**But the scale is not sigma_Psi**: the full 1D displacement dispersion is 4446 kpc/h, so
sigma_eff / sigma_Psi = 0.193, i.e. 0.55 fine cells.

That is physically sensible — the bulk displacement lives in the coarse band and translates an
octave-wavelength patch almost uniformly, so it cannot decorrelate octave modes; only the
differential displacement can. It is worth pinning down exactly which dispersion sigma_eff is,
because **if G can be predicted rather than fitted, the source can be built at a level where no
training pairs exist** — which is exactly the situation at the end of a progressive chain, and
a prerequisite for weight sharing across levels.

### 1 - r^2 of the source matches the notes' 0.69

Notes v3 states that the fraction of the detail variance no linear map can supply is 0.69 on
this pair. Measured independently in Phase 0 panel (b): band-averaged `r` = 0.5561, so
`1 - r^2` = **0.679**. Consistent.

### "the single-stream fraction is where regression and flow should coincide" — they do not

The notes make this prediction twice, and the R1-R4 rms split refutes it directly: in the
single-stream part the regression error is 0.282 h_f against the flow's 0.373, a 24% gap that
is 60x the seed-to-seed scatter. The regression is better in the multi-stream part too (0.380
vs 0.490). Whatever separates the two objectives, it is not confined to the multi-stream
patches — consistent with panel (c), where the excess kurtosis of the detail is *lower* inside
the patches than outside under either the coarse or the fine mask.
The notes' prediction about *power* — `r` below 1 with `P/P_true` at 1 — does hold globally
(flow 0.98-1.13, regression 0.69 on the r^2 line). Whether the power deficit itself is
localised in the multi-stream patches remains untested: it needs a masked power spectrum,
which no script here computes.

### One suggestion in notes v3 that has not been tried

"in emulator mode the cleanest input is the whole fine initial condition rather than its
octave (Paper IV's choice), which removes chi' from the unknowns" — the non-nestedness residual
chi' is real on this data (IC `r` = 0.76 at the coarse Nyquist). Feeding the full fine IC
instead of its octave band is a cheap experiment and would separate "the model cannot use the
octave" from "the octave it is given is incomplete". Not run.

### Phase-0 panel (c) per-bin tables (the figure is `fig_conditional_real.png`)

Self-run 32->64, z=0, quantile bins of the coarse `delta_L = -tr D`:

| bin centre `delta_L` | -3.6 | -1.8 | -0.8 | -0.35 | 0.07 | 0.56 | 1.12 | 1.89 | 4.48 |
|---|---|---|---|---|---|---|---|---|---|
| var(d/h_f) | 0.086 | 0.106 | 0.118 | 0.132 | 0.142 | 0.155 | 0.167 | 0.176 | 0.152 |
| excess kurtosis | 1.25 | 1.20 | 1.16 | 1.21 | 1.24 | 1.11 | 1.12 | 1.20 | 1.72 |

Production 64->128, same binning: var rises 0.153 -> 0.323 and falls back to 0.251, with excess
kurtosis 4.50 -> 2.42 -> 2.75. In both datasets the variance responds strongly to the local
`delta_L` (a factor 2) while the kurtosis stays flat and well above 1, so conditioning on the
local jet does not Gaussianise the detail — the "close to Gaussian outside the multi-stream
patches" criterion of KICKOFF_STAGE5 fails on the production pair (kurtosis 3.43 outside) and
is only marginal on the self-run pair (1.37).

---

## Addendum 2: question (b) answered, and the chi' experiment pre-checked and dropped

### The chi' suggestion is not worth a training run

Notes v3 suggests feeding the whole fine initial condition instead of its octave, so that the
non-nestedness residual `chi' = W delta_f - delta_c` stops being unknown. Measured before
spending GPU time on it, on the test box:

| | rms | r(model error, chi') | P_chi / P_err |
|---|---|---|---|
| chi' as a displacement | 0.380 h_f | — | — |
| R1 flow, coarse-band error | 0.262 h_f | **-0.170** | 1.13 |
| R2 regression, coarse-band error | 0.196 h_f | **-0.184** | 2.06 |

`chi'` is large — bigger than the residual error itself — but the error is almost orthogonal to
it, so a linear fit to `chi'` would remove only `r^2 ≈ 3%` of the error variance. If the models
were failing because they cannot see `chi'`, the correlation would be near ±1. **Dropped.**
(The sign is mildly interesting: the error is *anti*-correlated with `chi'`, i.e. the models
over-correct in that direction rather than under-correct.)

### (b) revisited: the power deficit IS deeper in the multi-stream patches, but not confined to them

The body of this report said question (b) was untested because a masked power spectrum was not
available. A masked *spectrum* is in fact the wrong tool — a 29%-filling mask convolves the
spectrum badly. The k-integrated version is well defined in real space with no convolution:
the octave-band variance ratio of prediction to truth, evaluated inside and outside the coarse
multi-stream mask.

| model | `P/P` all | `P/P` multi | `P/P` single | `r` all | `r` multi | `r` single |
|-------|-----------|-------------|--------------|---------|-----------|-----------|
| flow | 0.9674 | 0.9412 | 0.9821 | 0.6456 | 0.5715 | 0.6862 |
| regression | 0.5972 | **0.5125** | **0.6445** | 0.7621 | 0.7026 | 0.7934 |

(These are k-integrated over the octave band and so weight high k more heavily than the
shell-averaged `P/P_true` of the main table; the ordering is the same.)

**The weak form of the notes' claim holds:** the regression's power deficit `1 - P/P` is 0.487
inside the multi-stream patches against 0.355 outside — 37% deeper inside. The flow's deficit
is small in both (0.059 / 0.018).

**The strong form fails:** notes v3 says twice that "the single-stream fraction is where
regression and flow should coincide". In the single-stream part the flow is at 0.982 and the
regression at 0.645 — a 36% power deficit in the region where the two were predicted to agree.

**And the deficit tracks difficulty, not stream count.** The regression's multi/single ratio is
1.37 for the power deficit and 1.35 for the rms error — the same factor. That is what one
expects if the multi-stream mask is a proxy for "how hard this patch is" rather than the seat of
a distinct mechanism. It weakens the case for splitting the model by a multi-stream mask
(the regression-in-single-stream / generative-in-multi-stream division of Sec. 10): on this data
the objective matters everywhere, not only where the flow has crossed.
