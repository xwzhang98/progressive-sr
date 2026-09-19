#!/usr/bin/env python3
"""Precompute the IC octave of a level once per set (2026-09-19, for the memory-mapped 256->512 training): exactly what
Batcher.make(eta="true") uses as the source's octave, lin = scaffold.band(ic_f * growth, "high") (physical units, kpc/h, the
fine grid of the level), saved as float32 (3, Nf, Nf, Nf) to data/psc/octave/dmo-{Nf}/set{k}.npy for tiling.CropSource.
  python runs/precompute_octave.py --nc 256 --sets 0 1 ... 15 [--device cpu]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402

L, GR, OFF = 100000.0, 76.7439, 0.5
IC = "data/psc/dmo-{N}/set{seed}/IC/disp.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, default=256)
    ap.add_argument("--sets", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    Nc, Nf = args.nc, 2 * args.nc
    sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, torch.device(args.device), window="cube")
    outdir = f"data/psc/octave/dmo-{Nf}"
    os.makedirs(outdir, exist_ok=True)
    for k in args.sets:
        path = f"{outdir}/set{k}.npy"
        if os.path.exists(path):
            continue
        t0 = time.time()
        ic = p0.load(IC.replace("{seed}", str(k)), Nf)
        lin = sc.band(torch.from_numpy(np.ascontiguousarray(ic, dtype=np.float32)[None]) * GR, "high")[0]
        np.save(path + ".tmp.npy", lin.cpu().numpy().astype(np.float32))
        os.replace(path + ".tmp.npy", path)
        print(f"set {k}: octave of the {Nf}^3 IC saved ({time.time() - t0:.0f} s, rms {float(lin.std()):.3f} kpc/h)", flush=True)


if __name__ == "__main__":
    main()
