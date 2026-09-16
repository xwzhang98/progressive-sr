#!/usr/bin/env python3
"""The network-free ideal-target comparison (research directive 2026-09-15, step 2).

Question: with the candidate label Y64 = R_{128->64} Psi_128 (proper Fourier restriction,
project conventions: cube window, Nyquist planes zeroed, offset 0), does a 64^3 particle set
displaced by Y64 reproduce the 128^3 particle density spectrum for k < k_Ny,64?
Not presupposed, because delta[R Psi] != R delta[Psi].

Measurement discipline: reuses the verified interlaced-CIC + window-deconvolution estimator
and complete-shell averager from the owner's analysis
(~/Documents/Codex/2026-09-11/zhe/progressive-sr-spectra-2026-09-15/compute_spectra.py);
one common 256^3 ANALYSIS mesh (not a simulation), native particle counts preserved;
displacement spectra on native grids, complete shells only; particle Nyquists marked
separately from the analysis-mesh Nyquist.

Outputs per seed/dataset: coefficient-identity check of the restriction, displacement and
density P(k), ratios and r(k) vs the 128^3 reference, error power, 1%/5% agreement bands
(main band 0 < k < k_Ny,64, Nyquist-boundary shells excluded by the complete-shell rule).

  python runs/eval_y64.py --data selfsim --seeds 8 9        # old-IC data (flagged)
  python runs/eval_y64.py --data nestedic/s5000 --seeds 5000  # strictly nested rerun
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/zhangxiaowen/AntigravityProjects/progressive-sr")
sys.path.insert(0, "/Users/zhangxiaowen/Documents/Codex/2026-09-11/zhe/progressive-sr-spectra-2026-09-15")
import phase0_octaves as p0                     # noqa: E402
from compute_spectra import Shells, density_coeff  # noqa: E402  (verified estimator, reused)

L = 100000.0  # kpc/h on disk; compute_spectra works in Mpc/h internally (L=100)


def load(data, seed, n):
    if data == "selfsim":
        return np.load(f"data/selfsim/s{seed}/dis_{n}.npy")
    return np.load(f"data/{data}/dis_{n}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="selfsim")
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 9])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.data.replace("/", "_")
    g64, g128 = p0.Grid(64, L), p0.Grid(128, L)
    W = p0.cube_window(g128, g64.kny)
    sh = Shells(256)
    kny64 = np.pi * 64 / 100.0   # h/Mpc
    kny32 = np.pi * 32 / 100.0
    out = {}
    for seed in args.seeds:
        p64 = load(args.data, seed, 64)
        p128 = load(args.data, seed, 128)
        # ---- the label: proper Fourier restriction --------------------------------------
        Y64 = p0.restrict_spectral(p128, g128, g64, W, 0.0)
        # coefficient identity on retained modes
        FY = p0.rfftn(Y64.astype(np.float32))
        F128 = p0.rfftn(p128.astype(np.float32)) * W
        ax, c_ax, az = p0._coarse_slices(128, 64)
        sub = F128[:, ax][:, :, ax][:, :, :, az] * (64 / 128) ** 3
        FYs = FY[:, c_ax][:, :, c_ax][:, :, :, az]
        num = np.abs(FYs - sub).max()
        den = np.abs(sub).max()
        coeff_err = float(num / den)
        # ---- densities on the common analysis mesh --------------------------------------
        fields = {"native64": p64 / 1000.0, "Y64": Y64 / 1000.0, "ref128": p128 / 1000.0}
        coeffs = {}
        for nm, f in fields.items():
            c, merr = density_coeff(f.astype(np.float64))
            coeffs[nm] = c
        res = {"coeff_identity_max_rel": coeff_err}
        ref = coeffs["ref128"]
        for nm in ("native64", "Y64"):
            pw = sh.powers(coeffs[nm], ref)
            k = pw["k"]
            band = (k > 0) & (k < kny64)
            ratio = pw["P"][band] / np.maximum(pw["Pb"][band], 1e-30)
            r = pw["r"][band]
            kk = k[band]
            def frontier(tol):
                bad = np.where(np.abs(ratio - 1) > tol)[0]
                return float(kk[bad[0] - 1]) if bad.size and bad[0] > 0 else (float(kk[-1]) if not bad.size else 0.0)
            res[nm] = dict(k=kk.tolist(), Pratio=ratio.tolist(), r=r.tolist(),
                           err_power=(pw["P"][band] + pw["Pb"][band] - 2 * pw["cross"][band]).tolist(),
                           within1pc=frontier(0.01), within5pc=frontier(0.05),
                           ratio_at_kny32=float(ratio[np.argmin(np.abs(kk - kny32))]),
                           ratio_at_075kny64=float(ratio[np.argmin(np.abs(kk - 0.75 * kny64))]),
                           ratio_at_09kny64=float(ratio[np.argmin(np.abs(kk - 0.9 * kny64))]),
                           r_at_09kny64=float(r[np.argmin(np.abs(kk - 0.9 * kny64))]))
        # displacement-side check of the label (native grids, complete shells via p0)
        s_ = p0.spectra_k(g64, p0.rfftn(Y64.astype(np.float32)),
                          p0.rfftn(p64.astype(np.float32)))
        res["disp_Y64_vs_native64_r_mean"] = float(np.mean(s_["r"]))
        out[str(seed)] = res
        print(f"[{tag} s{seed}] coeff identity max rel = {coeff_err:.2e}")
        for nm in ("native64", "Y64"):
            r_ = out[str(seed)][nm]
            print(f"  {nm:9s} vs ref128 density: P/P at kNy32={r_['ratio_at_kny32']:.3f}, "
                  f"0.75kNy64={r_['ratio_at_075kny64']:.3f}, 0.9kNy64={r_['ratio_at_09kny64']:.3f} "
                  f"| r@0.9kNy64={r_['r_at_09kny64']:.3f} | 1%/5% bands to k={r_['within1pc']:.2f}/{r_['within5pc']:.2f} h/Mpc")
    with open(f"runs/y64_{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote runs/y64_{tag}.json")


if __name__ == "__main__":
    main()
