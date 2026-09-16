# Report 9 — the projected-label question: IC audit, ideal-target test, and model re-evaluation

Research directive of 2026-09-15 ("32→64 output should approach the 128³ reference in its
representable band"). Everything here is diagnostics, IC repair, small fixed-seed runs and
re-evaluation — no retraining, no architecture change. Figure: `runs/fig11_y64_verdict.png`.
Data provenance: s8/s9 are the existing (old-IC) boxes and are flagged as such; `nestedic/
s5000` is the new strictly-nested trio. 128³ is called the high-resolution *reference*
throughout, not a converged truth; the 128/256 convergence pair needed to upgrade that
language is listed at the end.

## 1. IC audit and repair (directive step 1)

**Numerical audit** (same seed 424242, GenIC variants; `data/nestedic/audit/`):

| comparison | rel-RMS | corr |
|---|---|---|
| default pipeline: ic(32, Nmesh 64) vs R ic(64, Nmesh 128) | 1.46e-1 | 0.9896 |
| Nmesh = Ngrid both levels: ic(32,32) vs R ic(64,64) | 3.12e-2 | 0.99951 |
| Nmesh = Ngrid: ic(64,64) vs R ic(128,128) | 1.36e-2 | 0.99991 |
| mixed: ic(32,64) vs R ic(64,64) | 1.44e-1 | 0.9898 |

The dominant non-nesting is the **coarse level's own folded super-Nyquist content** from the
`Nmesh = 2*Ngrid` default (`genic/params.c:199`; the mixed case ≈ the default case proves the
attribution). At Nmesh = Ngrid the per-shell amplitude transfer between levels is **T =
1.0000 at every shell**, so the remaining 1.4–3.1e-2 is a phase-level boundary effect of the
mesh-dependent mode assignment — my earlier finite-difference-kernel hypothesis is refuted by
that measurement. `UnitaryAmplitude = 1` (fixed amplitude, random phase) is the generator's
convention and is preserved below.

**Repair**: `nested_ic.py` — one Hermitian mother field on the top grid (amplitudes exactly
A(k) from a GenIC template at Nmesh = Ngrid; seeded unit-modulus phases), per-level strict
cube truncation *before* particle sampling (Nyquist planes zeroed, project conventions),
Zel'dovich displacement evaluated spectrally on each lattice (band-limited → alias-free),
v = 0.52776101·Ψ (measured from the template to 2.5e-8 — pure Zel'dovich at z=99), BigFile
output with template headers. **Acceptance: |R Ψ₁₂₈ − Ψ₆₄|/rms = 1.8e-7** (float precision;
six orders below GenIC), shell spectrum matches the template to 1.9e-7. Documented physical
difference: Zel-only (no 2LPT, ~1e-2 of the IC displacement at z=99). MP-Gadget evolves the
hand-written ICs unmodified; the s5000 trio (32/64/128, one seed) ran to z=0.

**The separation the strict trio buys** — IC error vs dynamical error at z=0:

| pair | IC rel-RMS | z=0 rel-RMS | z=0 r@0.9 k_Ny,c |
|---|---|---|---|
| 32→64 strict / old | 1.8e-7 / 1.4e-1 | 0.126 / 0.136 | 0.821 / 0.784 |
| 64→128 strict / old | 1.8e-7 / 8.6e-2 | 0.093 / 0.094 | 0.743 / 0.712 |

**Exactly nested ICs change the final-state cross-level mismatch by only ~0.01 in rel-RMS and
0.03–0.04 in near-Nyquist correlation.** The z=0 non-nesting is overwhelmingly dynamical.

## 2. The ideal-target test (directive step 2)

Y64 = R₁₂₈→₆₄ Ψ₁₂₈ verified as an exact Fourier restriction (retained complex coefficients
match to ≤1.0e-7 on all three datasets). Densities via the owner's verified interlaced-CIC +
window-deconvolution estimator, one common 256³ analysis mesh, native particle counts,
complete shells, main band 0 < k < k_Ny,64. The Y64-vs-128 comparison is IC-clean by
construction; it was additionally repeated on the strict trio.

| P_δ/P₁₂₈ | k_Ny,32 | 0.75 k_Ny,64 | 0.9 k_Ny,64 | r@0.9 | 1%/5% band [h/Mpc] |
|---|---|---|---|---|---|
| native 64³ (s8) | 0.976 | 0.952 | 0.964 | 0.977 | 0.32 / 1.51 |
| native 64³ (s9) | 0.987 | 0.981 | 0.987 | 0.975 | 0.32 / 1.95 |
| native 64³ (strict IC) | 0.947 | 0.961 | 0.954 | 0.977 | 0.32 / 0.94 |
| **Y64 (s8)** | 1.223 | 1.321 | 1.344 | 0.981 | 0.20 / 0.38 |
| **Y64 (s9)** | 1.110 | 1.140 | 1.165 | 0.987 | 0.26 / 0.57 |
| **Y64 (strict IC)** | 1.179 | 1.303 | 1.370 | 0.977 | 0.20 / 0.44 |

Two facts, both established without any network and reproduced on clean ICs:

1. **The ideal projected label over-concentrates: 11–37% density excess across the band.**
   Removing the octave detail removes the small-scale spreading that keeps caustics finite —
   the mechanism behind the regression's excess is a property of the *label*. A perfect
   predictor of Y64 inherits it. δ[RΨ] ≠ Rδ[Ψ], quantified. (Per the directive's own caution:
   this shows *this displacement label* does not automatically deliver the density; it does
   not show that no 64³ representation can.)
2. **The native 64³ run's density is already within 2–5% of the 128³ reference over the whole
   band, with r_δ ≥ 0.975.** The coarse run's deficiency below k_Ny,64 is *phase* near its
   Nyquist and content *above* k_Ny,64 — not band power.

## 3. Existing models under the 128³ reference (directive step 3)

Emulator mode (true 32→64 octave; no 64→128 IC modes are provided — the models never had
them), s8/s9, same estimator; training used seeds 0–7 only, so nothing here was fit on the
test boxes.

| vs 128³ density | s8: k_Ny,32 / 0.75 / 0.9 k_Ny,64 | s9 | r@0.9 (s8/s9) |
|---|---|---|---|
| native 64³ | 0.976/0.952/0.964 | 0.987/0.981/0.987 | 0.977/0.975 |
| regression | 1.393/1.733/1.914 | 1.214/1.344/1.430 | 0.907/0.914 |
| resflow | 1.062/0.996/0.900 | 0.996/0.879/0.783 | 0.918/0.915 |
| Y64 | 1.223/1.321/1.344 | 1.110/1.140/1.165 | 0.981/0.987 |

Under the new reference the ordering changes: **the native 64 run beats every model and the
ideal label in-band**; resflow is the closest model; the regression's excess is roughly the
label excess compounded with its own over-concentration. This is expected — the models were
trained toward the 64 run — and it does not overturn the earlier model-vs-model conclusions,
which used the 64 run as the reference by design. The old displacement metrics are retained
in the per-run results as auxiliary diagnostics.

## 4. Answers to the directive's five questions

**(1) Which deviations are now attributed to the IC construction, and which are not?**
Attributed: the *initial* cross-level mismatch (1.4e-1 → 1.8e-7 after repair; dominated by
Nmesh = 2·Ngrid folding, residually by mesh-dependent mode assignment). NOT attributable to
ICs: the bulk of the *final-state* coarse-band mismatch (0.126 vs 0.136; 0.093 vs 0.094) and
the near-Nyquist decorrelation — these are resolution dynamics. The Y64 label excess is also
IC-independent (reproduced at 1.8e-7 nesting).

**(2) Is the projected 128³ displacement a suitable 64³ label for the physical goal?**
As a displacement-band target it is exact by definition; as a route to 128-level *density* it
is counterproductive as-is: its own density carries an 11–37% in-band excess, worse than the
native 64 run the models currently target. Any use of Y64 as a label must be paired with
either a compensating small-scale displacement component or an explicit density-side
objective; otherwise training toward it moves the density *away* from the reference.

**(3) Is there a displacement/density trade-off requiring extra supervision or a different
representation?** Yes, and it is now demonstrated network-free: at fixed 64³ particle count,
band-perfect displacement (Y64) and reference-level density are incompatible as stated. The
project's Q_J term is precisely a density-side supervision and the resflow's density is the
closest among models; whether a *representation* change (e.g. carrying sub-band content or
mass re-weighting) beats supervision remains open and is not presupposed.

**(4) If the label changes, how must 64→128 be trained?** The second stage's input
distribution changes from "native-64-like states" to "Y64-like (band-limited, caustic-thin)
states". Its training pairs must be regenerated with the *same* intermediate-state
distribution the first stage will actually emit (the Stage-12 rollout machinery — mixing
predicted intermediates with paired augmentation and re-computed base — is exactly the
template for this), and its coarse-conditioning statistics (D_ij distributions) shift
accordingly. Until then, chaining a Y64-trained first stage into the current 64→128 models
would compound the label's density excess with the downstream models' own biases.

**(5) Do super-64-cutoff IC modes require extra conditioning?** Yes in principle and now
quantified twice over: (i) the coarse *run* is influenced by its generator's folded
super-Nyquist modes (the audit); (ii) at z=0 the retained band of the 128 run depends on
octave modes the 64 representation never sees (the dynamical mismatch of §1, and the earlier
finding that the coarse IC carries r≈0.85 of the folded octave). A 64³ output that tracks the
128 run's retained band beyond r≈0.98 therefore needs either the fine-level IC octave as
input (emulator mode already has it for 32→64 but NOT for the 64→128 modes) or an explicit
stochastic channel for it; this bounds what any deterministic 32→64 map can achieve against
a 128 reference.

## 5. Boundaries and reproducibility

128³ is the current high-resolution reference; upgrading "reference" to "converged" needs a
128/256 pair on strict ICs (one seed suffices for the band k < k_Ny,64; ~4 h of MP-Gadget at
256³ on this machine — not launched). Strict trio is Zel-only ICs; s8/s9 rows use the old
generator. Two boxes ≠ confidence intervals; box scatter in the Y64 excess is large
(1.34 vs 1.17 at 0.9 k_Ny,64). All commands, seeds and configs are in the RUNLOG; artifacts:
`nested_ic.py`, `runs/eval_y64.py`, `data/nestedic/{audit,tmpl_*,s5000}`,
`runs/y64_*.json`, `runs/fig11_y64_verdict.png`. Code state committed and pushed at each step.

## 6. Recommended minimal next training experiment

**One run: retrain the 32→64 *regression base* with the label switched to Y64, keeping the
residual flow's target as the native 64 run** — i.e. base learns the (exactly defined,
deterministic) band-limited 128 projection, and the residual flow learns native-64 minus
that base. Hypothesis under test: the base's Y64 excess is *absorbable by the residual flow*
(which the resflow architecture already showed it can do for the regression's excess, pulling
2.17 → 0.84 from above), yielding an output that keeps Y64's phase advantage (r_δ = 0.98 vs
the models' 0.91) while the flow restores the density level. Cost: one 3000-step regression
+ one 3000-step flow at 32→64 (~4 h GPU). Decision criterion: in-band density P/P within the
native-64 envelope (0.95–1.00) *and* r_δ@0.9 k_Ny,64 > 0.95 — i.e. strictly better than both
current end-members. If the flow cannot absorb the label excess, question (3) is answered in
the negative for supervision alone, and the representation-change route gets priority.
