#!/usr/bin/env python3
"""Rollout-aware fine-tuning of the SHARED three-level residual flow (owner 2026-09-19: "先1后2"), i.e. the RFT2 protocol
(owner's spec 2026-09-15, runs/rollout_ft2.py) generalised to the weight-shared operator M48c:

  * frozen: the shared base (runs/M48c_reg_psc) and the upstream predictions, cached ONCE from M48c (base + flow, emulator
    mode, true IC octave): pred64 = 32->64 from the true 32; pred128_64 = 64->128 from the true 64; pred128_32 = 64->128 from
    pred64 (two steps);
  * tuned: ONLY the shared residual flow, warm-started from runs/M48c_resflow_psc; lambda per level REUSED from its checkpoint;
  * per training step the level's coarse input is the native run with probability 1 - mix, otherwise a cached prediction:
    64->128 -> pred64; 128->256 -> pred128_64 or pred128_32 with equal odds; 32->64 is always native (no upstream);
  * the base is re-evaluated on the CHOSEN coarse and the velocity target is X - B; PAIRED augmentation of (chosen coarse,
    target, IC): numpy before make() on the full-box levels, the equivalent GPU crop augmentation at the crop level
    (tiling.augment_crops, checks/tiling_check.py 3);
  * short low-LR schedule: --steps per level (default 800) at --lr 5e-5, levels in rotation, EMA 0.99 as in RFT2;
  * the crop level keeps C3's geometry (crop, halo, crop-tapered Q_J); the checkpoint carries the same args schema, so
    runs/chain_shared.py evaluates it unchanged (tiled inference at the crop level).
Rows for the report: no-FT (M48c) / mix0 (control: same schedule, native inputs only) / mix50.

  python runs/rollout_shared.py --mix 0.5 --out runs/RS_mix50
  python runs/rollout_shared.py --mix 0.0 --out runs/RS_mix0
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import octave_flow_toy as oft      # noqa: E402
import eulerian_metric as em       # noqa: E402
import tiling                      # noqa: E402

L, GR, OFF = 100000.0, 76.7439, 0.5
DIS, IC = "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy"


def load_unet(path, dev):
    ck = torch.load(path, map_location="cpu")
    m = oft.UNet3D(cin=12, cout=3, base=int(ck.get("base", 24)))
    m.load_state_dict(ck["state_dict"]); m.to(dev)
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="runs/M48c_reg_psc/model_ema.pt")
    ap.add_argument("--flow-ckpt", default="runs/M48c_resflow_psc/model_ema.pt")
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=800, help="steps PER LEVEL")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(14)))
    ap.add_argument("--cache", default="runs/rollout_cache_M48c")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    baseM, ckb = load_unet(args.base_ckpt, dev); baseM.eval()
    for p_ in baseM.parameters():
        p_.requires_grad_(False)
    model, ckf = load_unet(args.flow_ckpt, dev)
    a0 = ckf["args"]
    lvls, crops, H, tile_core = a0["levels"], a0.get("crops") or [0] * len(a0["levels"]), a0.get("halo", 16), a0.get("tile_core", 64)
    s_vals, lams = ckf["s_values"], ckf["lams"]
    jp, je = a0.get("jac_p", 2.0), a0.get("jac_eps", 0.1)
    print(f"warm start {args.flow_ckpt}: levels {lvls}, crops {crops}, s {[round(v, 4) for v in s_vals]}, "
          f"lambda (reused) {[round(v, 5) for v in lams]}; mix {args.mix}", flush=True)

    levels = []
    for Nc, C, s_l, lam in zip(lvls, crops, s_vals, lams):
        Nf = 2 * Nc
        sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, dev, window="cube")
        tr = oft.load_real(DIS, IC, Nc, Nf, args.train_seeds)
        sc.fit_linear_power([it["ic_f"] * GR for it in tr])
        levels.append(dict(Nc=Nc, Nf=Nf, sc=sc, tr=tr, s=s_l, lam=lam, crop=C,
                           taper=tiling.taper(C, H, sc.fdev) if C else None,
                           mk=oft.Batcher(sc, [], rng, augment_on=False, growth=GR, octave_transverse=True)))
        print(f"level {Nc}->{Nf}: {len(tr)} train boxes, s {s_l:.4f}, lambda {lam:.5g}" + (f", crop {C} halo {H}" if C else ""), flush=True)
    by_nc = {lv["Nc"]: lv for lv in levels}

    # ---- upstream predictions of the FROZEN M48c pair, cached once (full-box levels only: 32->64 and 64->128) ----------
    def predict(lv, dis_c, item):
        it = dict(item); it["dis_c"] = dis_c
        x0, x1, Pc, D = lv["mk"].make([it], eta="true")
        s = torch.full((1,), lv["s"], device=dev); z = torch.zeros(1, device=dev)
        with torch.no_grad():
            xb = x0 + baseM(oft.net_input(x0, Pc, D), z, s)
            pred = oft.sample_flow(flowM0, xb, Pc, D, s, nsteps=8)
        return (pred[0] * lv["sc"].hf).cpu().numpy().astype(np.float32)

    paths = {s: {k: f"{args.cache}/{k}_s{s}.npy" for k in ("pred64", "pred128_64", "pred128_32")} for s in args.train_seeds}
    if not all(os.path.exists(p) for d in paths.values() for p in d.values()):
        flowM0, _ = load_unet(args.flow_ckpt, dev); flowM0.eval()
        for i, s in enumerate(args.train_seeds):
            p64 = predict(by_nc[32], by_nc[32]["tr"][i]["dis_c"], by_nc[32]["tr"][i])
            np.save(paths[s]["pred64"], p64)
            np.save(paths[s]["pred128_64"], predict(by_nc[64], by_nc[64]["tr"][i]["dis_c"], by_nc[64]["tr"][i]))
            np.save(paths[s]["pred128_32"], predict(by_nc[64], p64, by_nc[64]["tr"][i]))
            print(f"upstream predictions cached for set {s}", flush=True)
        del flowM0
        torch.cuda.empty_cache()
    cache = {s: {k: np.load(p) for k, p in d.items()} for s, d in paths.items()}

    # ---- fine-tuning --------------------------------------------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    log, counts = [], {lv["Nc"]: {"native": 0, "pred": 0} for lv in levels}
    total = len(levels) * args.steps
    t0 = time.time()
    for g in range(1, total + 1):
        lv = levels[(g - 1) % len(levels)]
        sc, C = lv["sc"], lv["crop"]
        i = int(rng.integers(len(args.train_seeds))); s_id = args.train_seeds[i]
        item = lv["tr"][i]
        dis_c = item["dis_c"]
        if lv["Nc"] > lvls[0] and rng.random() < args.mix:
            if lv["Nc"] == 64:
                dis_c = cache[s_id]["pred64"]
            else:
                dis_c = cache[s_id]["pred128_64"] if rng.random() < 0.5 else cache[s_id]["pred128_32"]
            counts[lv["Nc"]]["pred"] += 1
        else:
            counts[lv["Nc"]]["native"] += 1
        if C:
            x0, x1, Pc, D = lv["mk"].make([dict(dis_c=dis_c, dis_f=item["dis_f"], ic_f=item["ic_f"])], eta="true")
            with torch.no_grad():
                Jf, adjf = em.jacobian_and_adjugate(Pc, sc.fdev)
            o = [int(v_) for v_ in rng.integers(0, lv["Nf"], size=3)]
            x0, x1, Pc, D = (tiling.crop_periodic(a_, o, C) for a_ in (x0, x1, Pc, D))
            Nn = adjf.shape[1]
            Jc = tiling.crop_periodic(Jf[:, None], o, C)[:, 0]
            adjc = tiling.crop_periodic(adjf.reshape(1, Nn, Nn, Nn, 9).permute(0, 4, 1, 2, 3), o, C)
            del Jf, adjf
            (x0, x1, Pc), (D33, adj33), (Jc,) = tiling.augment_crops(
                rng, vecs=(x0, x1, Pc), tens=(D.reshape(1, 3, 3, C, C, C), adjc.reshape(1, 3, 3, C, C, C)), scals=(Jc,))
            D = D33.reshape(1, 9, C, C, C); adjc = adj33.permute(0, 3, 4, 5, 1, 2)
        else:
            dc, df, icf = oft.augment([dis_c, item["dis_f"], item["ic_f"]], rng, OFF)       # PAIRED augmentation
            x0, x1, Pc, D = lv["mk"].make([dict(dis_c=dc, dis_f=df, ic_f=icf)], eta="true")
        s = torch.full((1,), lv["s"], device=dev); z = torch.zeros(1, device=dev)
        with torch.no_grad():
            xb = x0 + baseM(oft.net_input(x0, Pc, D), z, s)                                     # B for the CHOSEN coarse
        t = torch.rand(1, device=dev)
        xt = (1 - t)[:, None, None, None, None] * xb + t[:, None, None, None, None] * x1
        v = model(oft.net_input(xt, Pc, D), t, s)
        target = x1 - xb                                                                        # target X - B
        e = v - target
        if C:
            loss = F.mse_loss(tiling.interior(v, H), tiling.interior(target, H))
            dE = em.spectral_gradient(e * lv["taper"], sc.fdev).permute(0, 3, 4, 5, 1, 2)
            dJ = (adjc * dE.transpose(-1, -2)).sum(dim=(-1, -2))
            term = tiling.interior(dJ ** 2 * (Jc ** 2 + je ** 2) ** (-jp / 2), H).mean()
        else:
            loss = F.mse_loss(v, target)
            term = em.jacobian_form(e, Pc.detach(), p=jp, eps=je, fft_device=sc.fdev)
        loss = loss + lv["lam"] * term
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, p_ in model.state_dict().items():
                ema[k].mul_(0.99).add_(p_.detach(), alpha=0.01)
        log.append(float(loss.detach()))
        if g % max(1, total // 24) == 0 or g == 1:
            print(f"gstep {g:5d} [{lv['Nc']}->{lv['Nf']}] loss {np.mean(log[-36:]):.4e} ({(time.time() - t0) / g:.2f}s/step)", flush=True)
    model.load_state_dict(ema)
    out_args = dict(a0); out_args.update(rollout=dict(vars(args)), mode="resflow")
    torch.save({"state_dict": model.state_dict(), "base": ckf.get("base", 48), "args": out_args, "mode": "resflow",
                "lams": lams, "s_values": s_vals}, os.path.join(args.out, "model_ema.pt"))
    print(f"done: input counts per level {counts}; checkpoint {args.out}/model_ema.pt "
          f"(evaluate with runs/chain_shared.py --base-ckpt {args.base_ckpt} --flow-ckpt {args.out}/model_ema.pt)", flush=True)


if __name__ == "__main__":
    main()
