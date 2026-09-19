#!/usr/bin/env python3
"""Density of a model's 64^3 (or c^3) test-box prediction against the CONVERGED higher-resolution
reference of the same set (Phase B success criterion, HANDOFF 2026-09-16 §2: in-band density
P/P_ref in [0.95, 1.00] AND r_delta@0.9 k_Ny,c > 0.95 for k < k_Ny,c, reference = native 2c run;
A.4 showed the 2c run is converged to 1% in that band for c = 64).

Reads the `field_*.npy` dumps written by runs/resflow_train.py (box units, kpc/h, offset 0.5 for
the PSC series) and evaluates truth / base / emulator / generative with the verified estimator
(compute_spectra.density_coeff + Shells, common 256^3 mesh, complete shells), exactly as
runs/eval_ylabel.py does for the label candidates.

  python runs/eval_vs_ref.py runs/RB_resflow_psc --set 8 [--nc 64] [--data psc]
Writes <run>/vs_ref.json and prints one line per field.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_spectra import Shells, density_coeff    # noqa: E402  (verified estimator, reused)
import importlib.util                                  # runs/ is not a package: load eval_ylabel by path
_spec = importlib.util.spec_from_file_location("eval_ylabel", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ylabel.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
band_stats = _m.band_stats

L_MPC = 100.0
DATASETS = {"psc": ("data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", 0.5),
            "selfsim": ("data/selfsim/s{seed}/dis_{N}.npy", 0.0)}


def deposit(psi_kpc, n, offset, mesh=256):
    f = psi_kpc.astype(np.float64) / 1000.0 + offset * (L_MPC / n)
    return density_coeff(f, mesh=mesh)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--set", type=int, required=True, help="test set/seed of the run")
    ap.add_argument("--nc", type=int, default=64, help="grid of the predicted field")
    ap.add_argument("--data", default="psc", choices=sorted(DATASETS))
    ap.add_argument("--fields", nargs="+", default=["truth", "base", "emulator", "generative"])
    ap.add_argument("--mesh", type=int, default=256, help="common analysis mesh (use 512 for 256^3 predictions vs native 512)")
    ap.add_argument("--tag", default=None, help="write vs_ref_<tag>.json instead of vs_ref.json")
    ap.add_argument("--ref-level", type=int, default=None,
                    help="grid of the reference run (default 2 nc; = nc when no higher-resolution run exists, e.g. 512)")
    args = ap.parse_args()
    pat, off = DATASETS[args.data]
    nc = args.nc
    nf = args.ref_level or 2 * nc
    kny = {n: np.pi * n / L_MPC for n in (nc // 2, nc)}
    probes = {f"kny{nc // 2}": kny[nc // 2], f"075kny{nc}": 0.75 * kny[nc], f"09kny{nc}": 0.9 * kny[nc]}
    sh = Shells(args.mesh)
    ref_field = np.load(pat.replace("{N}", str(nf)).replace("{seed}", str(args.set)))
    ref = deposit(ref_field, nf, off, args.mesh)
    for run in args.runs:
        out = {"set": args.set, "nc": nc, "reference": f"native {nf}", "offset": off, "mesh": args.mesh}
        for nm in args.fields:
            path = os.path.join(run, f"field_{nm}.npy")
            if not os.path.exists(path):
                continue
            f = np.load(path)
            assert f.shape == (3, nc, nc, nc), (path, f.shape)
            st = band_stats(sh.powers(deposit(f, nc, off, args.mesh), ref), kny[nc], probes)
            out[nm] = st
            print(f"[{os.path.basename(run.rstrip('/'))} set{args.set}] {nm:10s} vs native{nf}: "
                  f"P/P @kNy{nc // 2}={st[f'ratio_at_kny{nc // 2}']:.3f} 0.75kNy{nc}={st[f'ratio_at_075kny{nc}']:.3f} "
                  f"0.9kNy{nc}={st[f'ratio_at_09kny{nc}']:.3f} | r@0.9kNy{nc}={st[f'r_at_09kny{nc}']:.3f} | "
                  f"band min/max P/P={min(st['Pratio']):.3f}/{max(st['Pratio']):.3f} | 1%/5% to k={st['within1pc']:.2f}/{st['within5pc']:.2f}",
                  flush=True)
        with open(os.path.join(run, f"vs_ref_{args.tag}.json" if args.tag else "vs_ref.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
