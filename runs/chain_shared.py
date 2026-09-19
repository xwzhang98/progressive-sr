#!/usr/bin/env python3
"""Autoregressive chain with the SHARED two-stage operator (runs/multi_resflow_train.py checkpoints: one regression
base + one residual flow for all levels, conditioned on the level through s_l = ln sigma_c), 2026-09-18.

For every start level i and every later level j the chain feeds the previous prediction as the coarse input:
    from{levels[i]}:  true Psi_{levels[i]} -> step -> ... -> step (levels[j] -> 2 levels[j])
so `from{Nc}` at level Nc is the DIRECT step (true coarse input) and `from{M}` with M < Nc is a chained prediction
with (Nc/M) earlier steps. Levels not in the checkpoint's training levels are ZERO-SHOT: their style scalar is
extrapolated, s_l = ln sigma_c measured from the level's own ICs (the multi_train.sigma_c definition).
Modes: emulator (true IC octave at every step) and generative (each step samples its own octave; independent-T
"full" sampler, the training scripts' convention).
Every prediction is scored against the native fine run with the training scripts' own summary (octave r and P/P,
coarse (= correction) band r and P_eps/P, rms split by the TRUE coarse run's multi-stream mask) and dumped as
out/field_{mode}_from{M}_{Nc}to{Nf}.npy (kpc/h, gitignored) for runs/eval_vs_ref.py (density vs native 2 Nf reference).

  python runs/chain_shared.py --base-ckpt runs/M48_reg_psc/model_ema.pt --flow-ckpt runs/M48_resflow_psc/model_ema.pt \
      --set 14 --levels 32 64 128 --out runs/chain_M48_set14
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402
import tiling                      # noqa: E402

_spec = importlib.util.spec_from_file_location("multi_train", os.path.join(ROOT, "runs", "multi_train.py"))
_mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mt)
sigma_c = _mt.sigma_c

L, GR, OFF = 100000.0, 76.7439, 0.5
DIS, IC = "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy"


def load_unet(path, dev):
    ck = torch.load(path, map_location="cpu")
    m = oft.UNet3D(cin=12, cout=3, base=int(ck.get("base", 24)))
    m.load_state_dict(ck["state_dict"]); m.eval(); m.to(dev)
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--set", type=int, default=14)
    ap.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128], help="coarse grids, each step doubles")
    ap.add_argument("--s-sets", type=int, nargs="+", default=[12, 13], help="sets whose fine ICs give sigma_c and the linear power")
    ap.add_argument("--modes", nargs="+", default=["emulator", "generative"])
    ap.add_argument("--starts", type=int, nargs="*", default=None, help="start levels of the chains (default: every level)")
    ap.add_argument("--gen-starts", type=int, nargs="*", default=None, help="start levels in generative mode (default: --starts)")
    ap.add_argument("--nsteps", type=int, default=8)
    ap.add_argument("--amp-from", type=int, default=0, help="bf16 autocast for steps with Nf >= this (0 = never)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    baseM, ckb = load_unet(args.base_ckpt, dev)
    flowM, ckf = load_unet(args.flow_ckpt, dev)
    trained = dict(zip(ckf["args"]["levels"], ckf["s_values"]))
    crops = dict(zip(ckf["args"]["levels"], ckf["args"].get("crops") or [0] * len(ckf["args"]["levels"])))
    assert (ckb["args"].get("crops") or [0] * len(ckb["args"]["levels"])) == [crops[n] for n in ckb["args"]["levels"]], \
        "base and flow were trained with different crop geometries"
    tile_core = int(ckf["args"].get("tile_core", 64))
    assert ckb.get("s_values") == ckf.get("s_values"), "base and flow were trained with different style scalars"
    print(f"shared operator: base {args.base_ckpt} ({ckb.get('base')}), flow {args.flow_ckpt} ({ckf.get('base')}), "
          f"trained levels {sorted(trained)} s = {[round(v, 4) for v in ckf['s_values']]}", flush=True)

    levels = []
    for Nc in args.levels:
        Nf = 2 * Nc
        sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, dev, window="cube")
        te = oft.load_real(DIS, IC, Nc, Nf, [args.set])[0]
        ics = [p0.load(IC.replace("{seed}", str(k)), Nf) for k in args.s_sets]
        assert all(ic is not None for ic in ics), f"missing {Nf}^3 ICs for sets {args.s_sets}"
        sc.fit_linear_power([ic * GR for ic in ics])
        s_meas = float(np.log(sigma_c(ics, GR, Nc, L)))
        s_l = trained.get(Nc, s_meas)
        zs = Nc not in trained
        print(f"level {Nc}->{Nf}: s checkpoint {trained.get(Nc)} | measured (sets {args.s_sets}) {s_meas:.4f} | used {s_l:.4f}"
              + ("  ZERO-SHOT (extrapolated)" if zs else "  trained"), flush=True)
        ms = oft.multistream_mask(sc, te["dis_c"])        # the TRUE coarse run's mask, for every input at this level
        C = crops.get(Nc, 0)
        if C:
            print(f"   {Nc}->{Nf} was trained on {C}^3 crops: tiled inference, core {tile_core}", flush=True)
        levels.append(dict(Nc=Nc, Nf=Nf, sc=sc, te=te, s=s_l, s_meas=s_meas, zero_shot=zs, ms=ms,
                           netB=tiling.TiledNet(baseM, C, tile_core) if C else baseM,
                           netF=tiling.TiledNet(flowM, C, tile_core) if C else flowM))

    def step(lv, dis_c, mode, gen):
        sc = lv["sc"]
        it = dict(lv["te"]); it["dis_c"] = dis_c
        b = oft.Batcher(sc, [it], np.random.default_rng(0), augment_on=False, growth=GR, octave_transverse=True)
        x0, x1, Pc, D = b.make([it], eta="true" if mode == "emulator" else "sample", gen=gen)
        s = torch.full((1,), lv["s"], device=dev); z = torch.zeros(1, device=dev)
        amp = bool(args.amp_from) and lv["Nf"] >= args.amp_from
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            xb = x0 + lv["netB"](oft.net_input(x0, Pc, D), z, s)
            pred = oft.sample_flow(lv["netF"], xb.float(), Pc, D, s, nsteps=args.nsteps)
        hf = sc.hf
        out = (pred[0].float() * hf).cpu().numpy().astype(np.float32), (xb[0].float() * hf).cpu().numpy().astype(np.float32)
        del x0, x1, Pc, D, xb, pred
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out

    res = {"args": vars(args), "levels": [dict(Nc=lv["Nc"], s=lv["s"], s_meas=lv["s_meas"], zero_shot=lv["zero_shot"]) for lv in levels]}
    t0 = time.time()
    for mode in args.modes:
        starts = args.starts if (mode == "emulator" or args.gen_starts is None) else args.gen_starts
        for i, lv0 in enumerate(levels):
            if starts is not None and lv0["Nc"] not in starts:
                continue
            coarse = lv0["te"]["dis_c"]                      # true coarse run at the start level
            for j in range(i, len(levels)):
                lv = levels[j]
                gen = torch.Generator(device=lv["sc"].fdev).manual_seed(1000 + 10 * i + j)
                pred, base = step(lv, coarse, mode, gen)
                tag = f"{mode}_from{lv0['Nc']}_{lv['Nc']}to{lv['Nf']}"
                name = f"{mode} from {lv0['Nc']}^3 ({j - i} earlier steps)" + (" [0-shot]" if lv["zero_shot"] else "")
                s_ = oft.summarize(lv["sc"], name, pred, lv["te"]["dis_f"], lv["ms"])
                res[tag] = {k: v for k, v in s_.items() if not isinstance(v, list)}
                np.save(os.path.join(args.out, f"field_{tag}.npy"), pred)
                if j == i and mode == "emulator":
                    np.save(os.path.join(args.out, f"field_base_from{lv0['Nc']}_{lv['Nc']}to{lv['Nf']}.npy"), base)
                coarse = pred                                    # feed forward
                print(f"      ({time.time() - t0:.0f} s)", flush=True)
    for lv in levels:                                            # native fine runs, for eval_vs_ref's 'truth' row
        np.save(os.path.join(args.out, f"field_truth_{lv['Nc']}to{lv['Nf']}.npy"), lv["te"]["dis_f"].astype(np.float32))
    with open(os.path.join(args.out, "chain.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}/chain.json", flush=True)


if __name__ == "__main__":
    main()
