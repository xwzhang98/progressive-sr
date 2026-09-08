# Phase 0 — Lagrangian octave measurements (no training)

`phase0_octaves.py` is a standalone numpy/scipy script (matplotlib for the figure).
It expects map2map-style Lagrangian displacement fields, `float32` arrays of shape
`(3, N, N, N)` in Mpc/h, one per level, from the same nested seed.

## Run on the 64/128/256/512 series

```bash
# 1. find the grid convention of the IC generator (offset 0 or 0.5 cells)
python phase0_octaves.py --levels 64 128 --box 100 \
    --dis "/path/{N}/dis.npy" --ic "/path/{N}/ic_dis.npy" --offset 0   --out probe0
python phase0_octaves.py --levels 64 128 --box 100 \
    --dis "/path/{N}/dis.npy" --ic "/path/{N}/ic_dis.npy" --offset 0.5 --out probe05
#    -> the run whose "nestedness/convention check" prints ~1e-6 has the right offset

# 2. full series
python phase0_octaves.py --levels 64 128 256 512 --box 100 \
    --dis "/path/{N}/dis.npy" --ic "/path/{N}/ic_dis.npy" --offset <0|0.5> --alpha 1.0 \
    --vel "/path/{N}/vel.npy" --out phase0_z0
```

`--ic` can be a per-level pattern (best: enables the nestedness check) or one
top-level file (lower-level ICs are then built by cube truncation).  If the IC
displacement is stored at z_init, that is fine: r(k) in (b) is normalisation-free
and T(k) simply carries the growth ratio D(z)/D(z_init).

Memory at 512^3: a few fields of 1.6 GB each plus the FFT work arrays, ~15–20 GB;
run on a node.  Runtime is dominated by ~15 FFTs of 512^3 per transition.

## Outputs (`--out`)

* `phase0_results.json` — per transition:
  * `correction[R]`: `k`, `P_eps`, `P_coarse`, `P_Rfine`, `r_coarse_Rfine`,
    `Pdiv_eps`, `Pdiv_coarse`, low-k slopes (theory: 2 and 4 for spectral R), `rms_eps_over_h_c`
  * `wiener`: `T = P_{c x f}/P_f` and `r` on the fine grid
  * `detail_vs_linear`: `k`, `P_d`, `P_lin`, `r`, `T` for the octave band only
  * `detail_vs_coarse_harmonics` (needs a coarse IC): the coarse modes' own 2LPT
    harmonics outside W, `(I-P_W) T2 S[delta_c]`; `power_fraction` of the detail and
    `power_fraction_of_nonlinear` of the detail minus its linear octave, with r(k)
  * `conditional`: variance / excess kurtosis of `d/h_f` in quantile bins of the
    coarse `delta_L = -tr D`, of the traceless |S|, and in/out of `det(I+D)<0`
  * `ic_nestedness_rel_rms`
* `transition_<tag>.npz` — the correction spectra as arrays
* `phase0_summary.png` — four panels (a)(a')(b)(c)

## Self-tests (synthetic nested 2LPT, no data needed)

```bash
python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128 --offset 0.5 --out st_dealias
python phase0_octaves.py --selftest                    --levels 32 64 128 --offset 0.5 --out st_alias
```

* `--selftest-dealias`: every level is an ideal continuum coarse-graining.  The
  script must report `|RP x - x| ~ 1e-7`, nestedness `~3e-7`, and low-k slopes
  `P_eps ≈ 2`, `P_div,eps ≈ 4` for the spectral R (measured 2.2 / 4.15), while Haar
  is flat at low k — the deterministic linear-order mismatch discussed in the note.
* plain `--selftest`: coarse levels are computed on their own grids, so quadratic
  products alias, mimicking the discreteness error of a real LR run.  The low-k
  slope of eps then drops to ~0 (a small white floor) — the caveat in §5 of the note.

# Octave-flow prototype (`octave_flow_toy.py`)

A laptop-sized, end-to-end implementation of the recommended method (one 2x step,
flow matching from "prolonged coarse + linear octave" with the physical coupling).
It needs `phase0_octaves.py` in the same directory (data synthesis, spectra) and
PyTorch (CPU, `--device mps` on a Mac, or CUDA).

```bash
python octave_flow_toy.py --nc 16 --nf 32 --steps 300 --base 16          # ~2-3 min on CPU
python octave_flow_toy.py --nc 32 --nf 64 --steps 1500 --device mps      # Mac, ~20-40 min
python octave_flow_toy.py --nc 64 --nf 128 --dis "/d/{seed}/{N}/dis.npy" \
    --ic "/d/{seed}/{N}/ic.npy" --growth <D(z)/D(zinit)> --train-seeds 0 1 2 --test-seeds 3
```

What it does, in order: synthesises nested 2LPT boxes (`--rms-delta` sets the
nonlinearity; `--dealias` for ideal coarse levels), measures the linear P_delta(k)
from the training ICs (so the sampled octave has the right power and is curl-free),
prints the no-network baseline (x0 alone), trains the U-Net with FiLM(t) on the
flow-matching loss with cubic-group augmentation, then evaluates on a held-out seed:

* `emulator mode` — source built from the TRUE IC octave: r(k) and P/P_true in the
  octave band, correction quality in the coarse band (`P_eps/P`), rms error in h_f.
  The 1-Euler-step row is the pure regression limit.
* `generative mode` — source built from a SAMPLED Gaussian octave: P/P_true should
  be ~1 in the octave band while r vs truth is ~0 (different realisation) and
  r(sample A, sample B) is ~0 in the octave band but 1 in the coarse band.

Outputs: `results.json`, `model_ema.pt`, `summary.png` (loss, r(k), P ratio).
`--regression` trains the same network as a direct one-step regressor (t=0 only) and
evaluates it with one Euler step: the fair baseline for any multi-step claim.
`--coupling independent` builds the training source from a fresh sampled octave (same
marginals, uninformative source); with `--regression` this is the "conditional mean
given the coarse field" model that sits on the line P/P_true = r^2.
`--window cube|sphere` (default cube) chooses the restriction R; cube keeps every coarse
mode so the sampled octave is exactly the new IC modes, sphere is the isotropic ablation.
`--max-seconds S` stops training gracefully after S seconds and saves `train_state.pt`;
`--resume` continues from it (same `--out`), so long runs can be done in chunks.
`--eval-only <model_ema.pt>` skips training and re-runs the evaluation (use it for
`--nsteps-sample` sweeps; keep the same `--nc/--nf/--rms-delta/--test-seeds`).
On the reference 16->32 model: P/P_true in the octave band is 0.959 / 0.978 / 0.993
for 1 / 2 / 8 Heun steps, r stays 0.993-0.994.

Things deliberately left out of the toy (they are the next steps once the mechanics
are trusted on real 64/128 pairs): velocities, the scale/style scalar s_l across
several transitions (the hook is the `s` input), patch cropping for 256^3+, rollout
fine-tuning, the Eulerian CIC loss.
