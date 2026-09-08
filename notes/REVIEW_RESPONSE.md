# Response to the external review of `octave_flow_derivation` (v1 → v2)

Date: 2026-09-08. Reviewer's points are numbered as in the review; "accepted" means the
notes (both the English derivation and the Chinese physics version) and, where relevant,
the code and CLAUDE.md were changed.

## 1. Eq. (22): real-space sign and the missing `S[δ_c]` in the detail — accepted
* Sign: with `Ψ^(2) = D_2 ∇φ^(2)`, `∇²φ^(2) = S`, `D_2 = −3/7`, the real-space form is
  `Ψ^(2) = −(3/7) ∇∇⁻² S`; the Fourier form `+(3/7) i k Ŝ / k²` was already right (∇⁻² ↔ −1/k²).
  Fixed in both notes; introduced `T_2 := D_2 ∇∇⁻²`.
* Missing term: `d^(2) = (I − P_W) T_2 [S[δ_c] + B(δ_c,η) + S[η]]`. The coarse-only term
  cancels in the correction (coarse band) but not in the detail: two coarse modes can add
  up to a mode outside the coarse cube. The reviewer's example (cutoff 4; p=(3,2,0),
  q=(3,−2,0), p+q=(6,0,0), K=144/169) is reproduced in the notes. Consequence stated
  explicitly: the detail contains a deterministic part driven by the coarse initial
  conditions; "the detail is the new randomness" is a first-order statement only.

## 2. Divergence vs Eulerian density; ensemble form of the k² law — accepted
* `−ik·Ψ̂^(2)` is now called `ϑ^(2)` (divergence of the second-order displacement), with the
  remark that the second-order Eulerian density has the extra `−½ k_i k_j (Ψ_i^(1)Ψ_j^(1))^`
  term and that the full `F_2` kernel also vanishes as `O(k²)`, so the density's
  octave–octave power still falls as `k⁴` (separate calculation, not done here).
* The `k²`/`k⁴` statement is now made as a power-spectrum statement via Wick's theorem,
  `P_S(k) = ½ ∫ d³p/(2π)³ K² P_η P_η → C k⁴`, using the exact identity
  `K(p, k−p) = k²(1−μ²)/|k−p|²`. The realisation-level statement is kept in the
  weaker form "the variance of Q̂(k) is k-independent for k ≪ k_Ny,c".
* Qualifications added: the mixed term is at least as suppressed (shrinking phase space
  near the cutoff), and neither higher orders nor discreteness errors are claimed to share
  the exponent.

## 3. Physical coupling vs one-step regression — accepted (the most important change)
* Under full physical conditioning, `η` is recoverable from `x_0 − PΨ_c` and, if `C`
  determines `δ_c`, `x_1 = S_f[δ_c + η]` is a deterministic function of `(x_0, C)`:
  `Var(x_1 | x_0, C) = 0`. The ideal one-step regressor is exact; chaos does not change
  this in exact arithmetic. The claim "one-step regression necessarily loses power" was
  removed.
* `P_pred/P_true = r²` is now stated relative to an information set `I` that leaves
  residual uncertainty (coarse-only conditioning, independent coupling, patch-local view,
  finite capacity). The meaningful contrast is "regression without the octave (or with
  the independent coupling) → line P/P = r²" versus "generative model whose randomness is
  the octave → line P/P = 1".
* The sentence "the multi-step flow adds back the unpredictable part of the velocity" was
  removed: `u_t` is itself a conditional mean, the ODE is deterministic, all randomness
  enters through `x_0`.
* The toy numbers are re-read: one-step `r = 0.993` but `P/P = 0.959 ≠ r² = 0.986`, so the
  one-step model is not the optimal regressor; the multi-step gain is a
  parameterisation/optimisation effect to be compared with a direct regression baseline.
  `octave_flow_toy.py --regression` was added for that purpose (same network, trained at
  t = 0 only, one Euler step at evaluation).
* One nuance kept on our side: in a real N-body code, round-off amplified by chaotic
  dynamics makes the map effectively non-deterministic inside halos. This is a statement
  about the simulator, not about the mathematics of the conditional, and is now phrased
  that way.

## 4. Push-forward and Markov statements for reduced states; sphere cleaning — accepted
* The push-forward is stated for a conditioning `C` that determines `δ_c`; the general
  case is written as the mixture
  `P(Ψ_f ∈ A | C) = ∫∫ 1_A(S_f[δ_c+η]) γ(dη) p(dδ_c | C)`. The reduction
  `(Ψ_c, v_c) → Ψ_c → P_WΨ_c → local patch` is named explicitly, and injectivity of the
  full map is no longer used to justify anything about reduced states.
* The Markov property is stated for the full nested state; for reduced states it is a
  modelling approximation unless the reduction is a sufficient statistic.
* Sphere window: the corner modes `χ` are part of `δ_c`, and `P_W S_c[δ_W + χ]` is
  informative about `χ` through mode coupling, so re-sampling them from the prior is an
  approximation. **The cube window is now the theoretical default** (`--window cube` in
  both scripts; CLAUDE.md updated); the sphere is kept as an ablation.

## 5. Degenerate (rank-one, longitudinal) source — partially addressed (the review text
was truncated at this point)
* Added to the notes: conditional on `C`, the source is supported on a `7N_c³`-dimensional
  subspace, but so is the conditional target (push-forward of the same scalar octave), so a
  regular flow maps one manifold onto the other; transverse components of the true detail
  are functions of `η`, not extra randomness. What the degeneracy does imply is that the
  learned velocity is constrained only on the support of `p_t`, so off-manifold errors are
  not self-correcting (argument for an Eulerian loss term and for multi-step integration).
* If the truncated part of the review raised a further objection (e.g. about existence or
  regularity of a flow between singular measures), it is not yet addressed.

## Numbers obtained after the revision (cloud sandbox, 16->32, 600 CPU steps, cube window)

* Direct regression baseline with the physical coupling (`--regression`): octave-band
  `r = 0.996`, `P/P_true = 0.990`, rms error `0.057 h_f`; flow with 8 Heun steps:
  `0.993 / 0.992 / 0.070`; the flow's own one-step evaluation: `0.972`. So on this smooth
  2LPT toy the regression is at least as good as the flow — exactly what the corrected
  Section 6.5 predicts under full physical conditioning. The earlier "one step 0.959 vs
  eight steps 0.993" comparison was a comparison inside one flow model, not against a
  regression baseline, and should not be read as evidence for the flow.
* Regression in generative mode (fresh octave): `P/P_true = 1.03`, as expected for an
  accurate deterministic map of a fresh Gaussian octave.
* Independent coupling: flow keeps `P/P_true = 1.00` in emulator mode and `1.06` in
  generative mode, but `r` drops from `0.993` to `0.973` (it transports the octave it is
  given, like the rescaling in the 1D Gaussian example, without using the octave-to-detail
  relation); its one-step evaluation collapses (`r = 0.38`, `P/P = 0.045`).
* Coarse-only 2LPT harmonics (`phase0_octaves.py`, de-aliased self-test, rms_delta 0.8):
  `P_harm/P_d = 0.002–0.005` of the detail, `P_harm/P_nl = 0.24` of the nonlinear part of
  the detail (`r = 0.45`). The term the review pointed out is real and measurable; its
  size on real data is now a Phase-0 number (needs `--ic` and `--growth`). The local
  rms_delta sweep (0.8 / 1.5 / 2.5) showed `P_harm/P_nl` exactly constant: in pure 2LPT
  both numerator and denominator are second order, so the 24% is a structural constant of
  the toy, not a nonlinearity trend (my earlier expectation in KICKOFF_STAGE3B was wrong).

Consequence for the method statement: "flow matching from the linear octave" is the
right generative objective, but the baseline it must beat is the Paper-IV-style
regression with the octave initial conditions as input, and on smooth data it may not
beat it. The case for the flow is the reduced-information regime (multi-stream regions,
finite receptive fields, real N-body chaos), which the 2LPT toy cannot probe.

## Not changed
* Section 2 operator identities, the linear-theory results, the exact kernel identity, the
  flow-matching theorem and loss decomposition, the scale-free level–time correspondence.
