#!/usr/bin/env python3
"""Phase A.4 on the PSC series (HANDOFF 2026-09-16, plan A item 4): the network-free label and
convergence measurements one level up from runs/eval_y64.py, on the real 16-set series.

(a) convergence: density P(k) of the native 128/256/512 runs in the band k < k_Ny,64, each
    against the highest level present (P/P_top and r) -- decides whether "128 is a reference,
    not truth" gets upgraded or demoted;
(b) Y_c = R_{2c->c} Psi_2c (cube window, Nyquist planes zeroed, the series' grid offset) vs the
    native c run, both against the 2c reference: P/P, r_delta, 1%/5% frontiers -- eval_y64
    generalised to any c in {64, 128, 256}. Prediction (HANDOFF): the label excess grows with
    multi-streaming, i.e. from 64 to 128.

Estimator discipline: `compute_spectra.density_coeff` / `Shells` reused verbatim (interlaced CIC,
window deconvolution, common 256^3 analysis mesh, complete shells). The series has grid offset
0.5 (RUNLOG 2026-09-16): the restriction takes it through `restrict_spectral`, and the deposit --
which places particle (i,j,k) at i*h -- gets it by adding offset*h_N to every level's
displacement, so that all levels sit on the same physical positions (a level-dependent global
shift would otherwise fake a phase error of k*h_2c/2 in r(k)).

  python runs/eval_ylabel.py --sets 0 --nc 64 128            # both label experiments
  python runs/eval_ylabel.py --sets 0 --convergence           # (a) only, needs the 512 deposit
Outputs runs/ylabel_psc_set<s>.json (+ merged runs/ylabel_psc.json over the given sets).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0                          # noqa: E402
from compute_spectra import Shells, density_coeff    # noqa: E402  (verified estimator, reused)

DATA = "/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology"
L_KPC = 100000.0   # on disk
L_MPC = 100.0      # compute_spectra works in Mpc/h


def load(s, n, snap="PART_009"):
    a = np.load(f"{DATA}/dmo-{n}/set{s}/{snap}/disp.npy")
    assert a.shape == (3, n, n, n) and np.isfinite(a).all()
    return a.astype(np.float32)


def deposit(psi_kpc, n, offset, cache):
    """density coefficients on the common mesh; particle i at (i + offset) h."""
    key = id(psi_kpc)
    if key in cache:
        return cache[key]
    f = psi_kpc.astype(np.float64) / 1000.0 + offset * (L_MPC / n)
    c, merr = density_coeff(f)
    cache[key] = c
    return c


def band_stats(pw, kmax, probes):
    k = pw["k"]
    band = (k > 0) & (k < kmax)
    kk = k[band]
    ratio = pw["P"][band] / np.maximum(pw["Pb"][band], 1e-30)
    r = pw["r"][band]

    def frontier(tol):
        bad = np.where(np.abs(ratio - 1) > tol)[0]
        if not bad.size:
            return float(kk[-1])
        return float(kk[bad[0] - 1]) if bad[0] > 0 else 0.0

    res = dict(k=kk.tolist(), Pratio=ratio.tolist(), r=r.tolist(),
               within1pc=frontier(0.01), within5pc=frontier(0.05))
    for name, kp in probes.items():
        j = int(np.argmin(np.abs(kk - kp)))
        res[f"ratio_at_{name}"] = float(ratio[j])
        res[f"r_at_{name}"] = float(r[j])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, nargs="+", default=[0])
    ap.add_argument("--nc", type=int, nargs="*", default=[64, 128], help="coarse levels for the label test")
    ap.add_argument("--convergence", action="store_true", help="also (a): 128/256/512 vs top in k < k_Ny,64")
    ap.add_argument("--top", type=int, default=512)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--out", default="runs/ylabel_psc")
    args = ap.parse_args()
    sh = Shells(256)
    kny = {n: np.pi * n / L_MPC for n in (32, 64, 128, 256, 512)}
    merged = {}
    for s in args.sets:
        t0 = time.time()
        fields, coeffs, cache = {}, {}, {}
        need = set()
        for nc in args.nc:
            need |= {nc, 2 * nc}
        if args.convergence:
            need |= {128, 256, args.top}
        for n in sorted(need):
            fields[n] = load(s, n)
            print(f"[set{s}] deposit native {n}^3 ...", flush=True)
            coeffs[n] = deposit(fields[n], n, args.offset, cache)
            print(f"[set{s}]   done ({time.time() - t0:.0f} s)", flush=True)
        res = {"offset": args.offset, "git_commit": None}
        try:
            import subprocess
            res["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                               text=True, timeout=5).stdout.strip() or None
        except Exception:
            pass
        if args.convergence:
            top = coeffs[args.top]
            conv = {}
            for n in (128, 256):
                if n >= args.top:
                    continue
                pw = sh.powers(coeffs[n], top)
                conv[f"native{n}_vs_{args.top}"] = band_stats(
                    pw, kny[64], {"kny32": kny[32], "075kny64": 0.75 * kny[64], "09kny64": 0.9 * kny[64]})
                # the same in the next band up, for the record (128 vs top up to k_Ny,128)
                conv[f"native{n}_vs_{args.top}_to_kny128"] = band_stats(
                    pw, kny[128], {"kny64": kny[64], "15kny64": 1.5 * kny[64], "09kny128": 0.9 * kny[128]})
                c = conv[f"native{n}_vs_{args.top}"]
                print(f"[set{s}] (a) native{n} vs {args.top}: P/P at kNy32={c['ratio_at_kny32']:.3f} "
                      f"0.75kNy64={c['ratio_at_075kny64']:.3f} 0.9kNy64={c['ratio_at_09kny64']:.3f} | "
                      f"r@0.9kNy64={c['r_at_09kny64']:.4f} | 1%/5% to k={c['within1pc']:.2f}/{c['within5pc']:.2f}",
                      flush=True)
            res["convergence"] = conv
        for nc in args.nc:
            nf = 2 * nc
            gc, gf = p0.Grid(nc, L_KPC), p0.Grid(nf, L_KPC)
            W = p0.cube_window(gf, gc.kny)
            Y = p0.restrict_spectral(fields[nf], gf, gc, W, args.offset)
            print(f"[set{s}] deposit Y{nc} = R Psi_{nf} ...", flush=True)
            cY = deposit(Y, nc, args.offset, cache)
            ref = coeffs[nf]
            probes = {f"kny{nc // 2}": kny[nc // 2], f"075kny{nc}": 0.75 * kny[nc], f"09kny{nc}": 0.9 * kny[nc]}
            lab = {}
            for name, c in ((f"native{nc}", coeffs[nc]), (f"Y{nc}", cY)):
                lab[name] = band_stats(sh.powers(c, ref), kny[nc], probes)
                r_ = lab[name]
                print(f"[set{s}] (b) {name:9s} vs ref{nf} density: P/P at kNy{nc // 2}={r_[f'ratio_at_kny{nc // 2}']:.3f}, "
                      f"0.75kNy{nc}={r_[f'ratio_at_075kny{nc}']:.3f}, 0.9kNy{nc}={r_[f'ratio_at_09kny{nc}']:.3f} | "
                      f"r@0.9kNy{nc}={r_[f'r_at_09kny{nc}']:.3f} | 1%/5% to k={r_['within1pc']:.2f}/{r_['within5pc']:.2f}",
                      flush=True)
            # displacement-side r between the label and the native run (native grid, complete shells)
            s_ = p0.spectra_k(gc, p0.rfftn(Y.astype(np.float32)), p0.rfftn(fields[nc].astype(np.float32)))
            lab["disp_r_label_vs_native_mean"] = float(np.mean(s_["r"]))
            lab["disp_r_label_vs_native_last"] = float(s_["r"][-1])
            res[f"label_{nc}to{nf}"] = lab
        merged[str(s)] = res
        with open(f"{args.out}_set{s}.json", "w") as f:
            json.dump(res, f, indent=1)
        print(f"[set{s}] wrote {args.out}_set{s}.json ({time.time() - t0:.0f} s)", flush=True)
    with open(f"{args.out}.json", "w") as f:
        json.dump(merged, f, indent=1)


if __name__ == "__main__":
    main()
