#!/usr/bin/env python3
"""Design probe for patch-cropped training at 128->256 (owner approved patch cropping 2026-09-18).

(1) Halo: the network uses circular 'same' padding, correct on the periodic box and wrong at a crop's edges. Run the
    existing shared operator (M48 base; M48 flow at t = 0.5) on the FULL 256^3 box and on periodic crops of size C taken
    from the same full-box inputs, and measure the relative rms difference of the network output as a function of the
    Chebyshev distance d from the crop boundary. The halo is the d beyond which the crop output equals the full-box
    output to the chosen tolerance.
(2) Memory: peak GPU memory of one training step (forward + backward, width 48, batch 1) on crops of size C.

  python runs/crop_probe.py --set 13 --crops 128 160 192 --out runs/crop_probe
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import octave_flow_toy as oft      # noqa: E402

_spec = importlib.util.spec_from_file_location("chain_shared", os.path.join(ROOT, "runs", "chain_shared.py"))
_cs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cs)
L, GR, OFF, DIS, IC = _cs.L, _cs.GR, _cs.OFF, _cs.DIS, _cs.IC


def crop(x, o, C):
    """periodic crop (B, ch, N, N, N) -> (B, ch, C, C, C) starting at o = (i, j, k)."""
    x = torch.roll(x, shifts=(-o[0], -o[1], -o[2]), dims=(2, 3, 4))
    return x[:, :, :C, :C, :C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="runs/M48_reg_psc/model_ema.pt")
    ap.add_argument("--flow-ckpt", default="runs/M48_resflow_psc/model_ema.pt")
    ap.add_argument("--set", type=int, default=13)
    ap.add_argument("--nc", type=int, default=128)
    ap.add_argument("--s", type=float, default=1.2151)
    ap.add_argument("--crops", type=int, nargs="+", default=[128, 160, 192])
    ap.add_argument("--out", default="runs/crop_probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    baseM, _ = _cs.load_unet(args.base_ckpt, dev)
    flowM, _ = _cs.load_unet(args.flow_ckpt, dev)
    Nc, Nf = args.nc, 2 * args.nc
    sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, dev, window="cube")
    te = oft.load_real(DIS, IC, Nc, Nf, [args.set])[0]
    b = oft.Batcher(sc, [te], np.random.default_rng(0), augment_on=False, growth=GR, octave_transverse=True)
    x0, x1, Pc, D = b.make([te], eta="true")
    s = torch.full((1,), args.s, device=dev); z = torch.zeros(1, device=dev); th = torch.full((1,), 0.5, device=dev)
    with torch.no_grad():
        inp = oft.net_input(x0, Pc, D)
        vb_full = baseM(inp, z, s)
        xb = x0 + vb_full
        xt = 0.5 * xb + 0.5 * x1
        vf_full = flowM(oft.net_input(xt, Pc, D), th, s)
    res = {"args": vars(args), "halo": {}, "memory_GB": {}}
    rng = np.random.default_rng(0)
    for C in args.crops:
        prof = {"base": np.zeros(C // 2 + 1), "flow": np.zeros(C // 2 + 1)}
        norm = {"base": np.zeros(C // 2 + 1), "flow": np.zeros(C // 2 + 1)}
        ax = np.arange(C)
        dist1 = np.minimum(ax, C - 1 - ax)
        dist = np.minimum(np.minimum(dist1[:, None, None], dist1[None, :, None]), dist1[None, None, :])   # Chebyshev distance to the crop boundary
        dist_t = torch.from_numpy(dist).to(dev)
        for rep in range(3):
            o = tuple(int(v) for v in rng.integers(0, Nf, size=3))
            with torch.no_grad():
                vb_c = baseM(crop(inp, o, C), z, s)
                vf_c = flowM(crop(oft.net_input(xt, Pc, D), o, C), th, s)
            for nm, vc, vfull in (("base", vb_c, vb_full), ("flow", vf_c, vf_full)):
                ref = crop(vfull, o, C)
                d2 = ((vc - ref) ** 2).sum(1)[0]; r2 = (ref ** 2).sum(1)[0]
                prof[nm] += torch.bincount(dist_t.ravel(), weights=d2.ravel().double(), minlength=C // 2 + 1).cpu().numpy()
                norm[nm] += torch.bincount(dist_t.ravel(), weights=r2.ravel().double(), minlength=C // 2 + 1).cpu().numpy()
        out = {}
        for nm in ("base", "flow"):
            rel = np.sqrt(prof[nm] / np.maximum(norm[nm], 1e-30))
            out[nm] = rel.tolist()
            first = {tol: int(np.argmax(rel < tol)) if (rel < tol).any() else None for tol in (0.1, 0.03, 0.01)}
            print(f"crop {C}: {nm:4s} rel. rms diff (crop vs full box) at d = 0,4,8,16,24,32,48: "
                  + " ".join(f"{rel[d]:.3f}" for d in (0, 4, 8, 16, 24, 32, 48) if d < len(rel))
                  + f" | first d below 10%/3%/1%: {first[0.1]}/{first[0.03]}/{first[0.01]}", flush=True)
        res["halo"][str(C)] = out
    del inp, vb_full, xb, xt, vf_full
    torch.cuda.empty_cache()
    # ---- training-step memory on crops (width 48, batch 1, random inputs of the right shape) ----
    model = oft.UNet3D(cin=12, cout=3, base=48).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for C in args.crops:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        try:
            xin = torch.randn(1, 12, C, C, C, device=dev); tgt = torch.randn(1, 3, C, C, C, device=dev)
            v = model(xin, torch.rand(1, device=dev), s)
            loss = ((v - tgt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"crop {C}: training step peak memory {peak:.1f} GB", flush=True)
            res["memory_GB"][str(C)] = peak
        except torch.cuda.OutOfMemoryError:
            print(f"crop {C}: training step OOM", flush=True)
            res["memory_GB"][str(C)] = None
        del xin, tgt
        torch.cuda.empty_cache()
    with open(os.path.join(args.out, "crop_probe.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}/crop_probe.json", flush=True)


if __name__ == "__main__":
    main()
