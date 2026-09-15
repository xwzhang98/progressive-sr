#!/usr/bin/env python3
"""Rollout pilot, round 2, on the resflow pair (owner's spec, 2026-09-15).

Frozen: the whole upstream (base32 = R_reg_phys_3k + flow32 = RF_resflow) and the downstream
base128 = F_reg_128_3k. Tuned: ONLY the downstream residual flow, warm-started from
RF_resflow_128.

Fixes relative to round 1, all owner-specified:
  * the velocity target is recomputed for the CHOSEN coarse field: whichever coarse enters
    the batch (true or predicted), the base is re-evaluated on it and the target is X - B —
    round 1 reused the physical-source pipeline and must not be repeated;
  * PAIRED augmentation: the chosen coarse is augmented together with the true target and
    its IC (the same cubic-group element), restoring what round 1 dropped;
  * lambda is NOT re-derived: it is RF_resflow_128's value, for comparability;
  * short, low-LR schedule (default 800 steps at 5e-5) — the round-1 regime damage came
    from 1500 steps at 1e-4 on 8 boxes.

Rows for the report: no-FT (= RF_resflow_128 as is) / mix0 control / mix50.

  python runs/rollout_ft2.py --mix 0.5 --out runs/RFT2_mix50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402
import eulerian_metric as em       # noqa: E402

L, OFF, GR = 100000.0, 0.0, 76.7439
DIS, IC = "data/selfsim/s{seed}/dis_{N}.npy", "data/selfsim/s{seed}/ic_dis_{N}.npy"
TRAIN = list(range(8))


def load_net(path):
    ck = torch.load(path, map_location="cpu")
    m = oft.UNet3D(cin=12, cout=3, base=int(ck.get("base", 24)))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    # ---- frozen upstream chain -> predicted 64^3 per training box (cached) -------------
    cache = os.path.join(os.path.dirname(args.out), "RFT2_pred64")
    os.makedirs(cache, exist_ok=True)
    need = [s for s in TRAIN if not os.path.exists(f"{cache}/s{s}.npy")]
    if need:
        b32, _ = load_net("runs/R_reg_phys_3k/model_ema.pt")
        f32, _ = load_net("runs/RF_resflow/model_ema.pt")
        scA = oft.Scaffold(32, 64, L, OFF, 1.0, torch.device("cpu"), window="cube")
        trA = oft.load_real(DIS, IC, 32, 64, TRAIN)
        scA.fit_linear_power([it["ic_f"] * GR for it in trA])
        bA = oft.Batcher(scA, trA, np.random.default_rng(0), augment_on=False, growth=GR,
                         octave_transverse=True)
        z = torch.zeros(1)
        for i, s in enumerate(TRAIN):
            if s not in need:
                continue
            x0, x1, Pc, D = bA.make([trA[i]], eta="true")
            with torch.no_grad():
                xb = x0 + b32(oft.net_input(x0, Pc, D), z, z)
            pred = oft.sample_flow(f32, xb, Pc, D, z, nsteps=8)
            np.save(f"{cache}/s{s}.npy", (pred[0] * scA.hf).numpy().astype(np.float32))
            print(f"upstream (resflow) prediction cached for seed {s}", flush=True)

    # ---- downstream: frozen base128, tuned flow128 --------------------------------------
    scB = oft.Scaffold(64, 128, L, OFF, 1.0, dev, window="cube")
    trB = oft.load_real(DIS, IC, 64, 128, TRAIN)
    scB.fit_linear_power([it["ic_f"] * GR for it in trB])
    pred64 = {s: np.load(f"{cache}/s{s}.npy") for s in TRAIN}
    baseM, _ = load_net("runs/F_reg_128_3k/model_ema.pt"); baseM.to(dev)
    for p_ in baseM.parameters():
        p_.requires_grad_(False)
    ckf = torch.load("runs/RF_resflow_128/model_ema.pt", map_location="cpu")
    lam = float(ckf["lam"])                      # NOT re-derived
    model = oft.UNet3D(cin=12, cout=3, base=24).to(dev)
    model.load_state_dict(ckf["state_dict"])
    print(f"warm-started from RF_resflow_128; lambda (reused) = {lam:.5g}; mix = {args.mix}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log = []; step0 = 0
    state_path = os.path.join(args.out, "train_state.pt")
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ema = {k: v.to(dev) for k, v in st["ema"].items()}
        step0, log = st["step"], st["log"]
        torch.manual_seed(args.seed + step0)
        print(f"resumed from step {step0}")
    rng = np.random.default_rng(args.seed + step0)
    bB = oft.Batcher(scB, trB, rng, augment_on=False, growth=GR, octave_transverse=True)
    zb = torch.zeros(1, device=dev)
    t0 = time.time()
    for step in range(step0 + 1, args.steps + 1):
        i = int(rng.integers(len(TRAIN)))
        it = trB[i]
        dis_c = pred64[TRAIN[i]] if rng.random() < args.mix else it["dis_c"]
        # PAIRED augmentation: one cubic-group element applied to the chosen coarse, the
        # true target and its IC together
        dc, df, icf = oft.augment([dis_c, it["dis_f"], it["ic_f"]], rng, OFF)
        x0, x1, Pc, D = bB.make([dict(dis_c=dc, dis_f=df, ic_f=icf)], eta="true")
        with torch.no_grad():
            xb = x0 + baseM(oft.net_input(x0, Pc, D), zb, zb)   # B for the CHOSEN coarse
        t = torch.rand(1, device=dev)
        xt = (1 - t)[:, None, None, None, None] * xb + t[:, None, None, None, None] * x1
        v = model(oft.net_input(xt, Pc, D), t, zb)
        loss = F.mse_loss(v, x1 - xb)                            # target is X - B
        e = v - (x1 - xb)
        loss = loss + lam * em.jacobian_form(e, Pc.detach(), p=2.0, eps=0.1, fft_device=scB.fdev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, p_ in model.state_dict().items():
                ema[k].mul_(0.99).add_(p_.detach(), alpha=0.01)
        log.append(float(loss.detach()))
        if step % max(1, args.steps // 16) == 0 or step == step0 + 1:
            print(f"step {step:5d} loss {np.mean(log[-32:]):.4e} ({(time.time()-t0)/(step-step0):.2f}s/step)",
                  flush=True)
        if args.max_seconds is not None and time.time() - t0 > args.max_seconds and step < args.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema,
                            step=step, log=log), state_path)
            print(f"time budget reached at step {step}/{args.steps}; state saved")
            return
    model.load_state_dict(ema); model.eval()
    torch.save({"state_dict": model.state_dict(), "base": 24, "lam": lam, "args": vars(args)},
               os.path.join(args.out, "model_ema.pt"))
    print("done; evaluate with runs/eval_chain.py --pair resflow --flow-b <this checkpoint>")


if __name__ == "__main__":
    main()
