#!/usr/bin/env python3
"""Style-scalar scan at a ZERO-SHOT level (2026-09-18): is the shared operator's power deficit at 128->256
(octave P/P ~0.65 with the extrapolated s = ln sigma_c = 1.215, vs ~0.98 at the trained levels) a mis-calibrated
level conditioning, or does no value of s make the operator produce the right detail power?
Emulator mode (true IC octave), direct step from the true coarse run, one calibration set (not a test set), for a
list of s values; prints the training scripts' summary for the base and the flow at each s and saves the fields.

  python runs/s_scan.py --base-ckpt runs/M48_reg_psc/model_ema.pt --flow-ckpt runs/M48_resflow_psc/model_ema.pt \
      --nc 128 --set 13 --s 0.655 0.958 1.215 1.5 --out runs/s_scan_M48_set13
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
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402

_spec = importlib.util.spec_from_file_location("chain_shared", os.path.join(ROOT, "runs", "chain_shared.py"))
_cs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cs)
L, GR, OFF, DIS, IC = _cs.L, _cs.GR, _cs.OFF, _cs.DIS, _cs.IC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--nc", type=int, default=128)
    ap.add_argument("--set", type=int, default=13)
    ap.add_argument("--s", type=float, nargs="+", required=True)
    ap.add_argument("--nsteps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    baseM, _ = _cs.load_unet(args.base_ckpt, dev)
    flowM, ckf = _cs.load_unet(args.flow_ckpt, dev)
    Nc, Nf = args.nc, 2 * args.nc
    sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, dev, window="cube")
    te = oft.load_real(DIS, IC, Nc, Nf, [args.set])[0]
    ms = oft.multistream_mask(sc, te["dis_c"])
    b = oft.Batcher(sc, [te], np.random.default_rng(0), augment_on=False, growth=GR, octave_transverse=True)
    x0, x1, Pc, D = b.make([te], eta="true")
    print(f"level {Nc}->{Nf}, set {args.set}; checkpoint s = {ckf.get('s_values')} (levels {ckf['args']['levels']})", flush=True)
    res = {"args": vars(args)}
    for s_val in args.s:
        s = torch.full((1,), float(s_val), device=dev); z = torch.zeros(1, device=dev)
        with torch.no_grad():
            xb = x0 + baseM(oft.net_input(x0, Pc, D), z, s)
            pred = oft.sample_flow(flowM, xb, Pc, D, s, nsteps=args.nsteps)
        hf = sc.hf
        fb = (xb[0] * hf).cpu().numpy().astype(np.float32); fp = (pred[0] * hf).cpu().numpy().astype(np.float32)
        rb = oft.summarize(sc, f"s={s_val:.3f} base", fb, te["dis_f"], ms)
        rp = oft.summarize(sc, f"s={s_val:.3f} base+flow", fp, te["dis_f"], ms)
        res[f"{s_val:.3f}"] = dict(base={k: v for k, v in rb.items() if not isinstance(v, list)},
                                   flow={k: v for k, v in rp.items() if not isinstance(v, list)})
        np.save(os.path.join(args.out, f"field_flow_s{s_val:.3f}.npy"), fp)
        np.save(os.path.join(args.out, f"field_base_s{s_val:.3f}.npy"), fb)
        del xb, pred
        torch.cuda.empty_cache()
    np.save(os.path.join(args.out, "field_truth.npy"), te["dis_f"].astype(np.float32))
    with open(os.path.join(args.out, "s_scan.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}/s_scan.json", flush=True)


if __name__ == "__main__":
    main()
