#!/usr/bin/env python3
"""Rollout fine-tuning of the shared operator (owner's protocol, 2026-09-13).

The chained 64->128 step conditions on the 32->64 model's OUTPUT, whose statistics differ
from the true coarse fields the model was trained on -- the distribution-shift problem of
notes Sec. 2.3/9. Measured cost per level: Lagrangian octave power nearly unchanged
(0.856 -> 0.850) while the density tail halves (0.579 -> 0.282 at 2 k_Ny,c), so this is a
coarse-error/detail-coupling problem, not a displacement-power problem.

Protocol (exactly as specified):
  1. FREEZE the upstream model (M_qj_multi at the 32->64 level) and generate the predicted
     64^3 coarse field for every TRAINING box (emulator mode, true octave -- same mode as
     the chained evaluation).
  2. Fine-tune the downstream (64->128) level of a COPY of the shared model: each batch uses
     the predicted coarse with probability --mix (default 0.5), else the true coarse; the
     128^3 target and its IC octave are always the TRUE ones of the same realization.
  3. Evaluate single-level (true coarse) AND chained (predicted coarse), seeds 8 and 9,
     Lagrangian + Eulerian probes -- did the 0.50/0.28 tail recover, and did the direct
     column pay for it?

Only the 64->128 level is fine-tuned (the upstream stays frozen), so any change in the
direct column is a real cost of the mixing, not upstream drift.

  python runs/rollout_ft.py --steps 1500 --mix 0.5 --out runs/RFT_mix50
"""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--mix", type=float, default=0.5, help="P(batch uses the PREDICTED coarse)")
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tuning LR (below the 3e-4 of training)")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    ck = torch.load("runs/M_qj_multi/model_ema.pt", map_location="cpu")
    sA, sB = ck["s_values"]
    lamB = ck["lams"][1]
    base = int(ck["base"])

    # ---- step 1: frozen upstream predictions for the TRAINING boxes (cached on disk) ----
    cache = os.path.join(args.out, "pred64")
    os.makedirs(cache, exist_ok=True)
    need = [s for s in TRAIN if not os.path.exists(f"{cache}/s{s}.npy")]
    if need:
        up = oft.UNet3D(cin=12, cout=3, base=base); up.load_state_dict(ck["state_dict"]); up.eval()
        scA = oft.Scaffold(32, 64, L, OFF, 1.0, torch.device("cpu"), window="cube")
        trA = oft.load_real(DIS, IC, 32, 64, TRAIN)
        scA.fit_linear_power([it["ic_f"] * GR for it in trA])
        bA = oft.Batcher(scA, trA, np.random.default_rng(0), augment_on=False, growth=GR,
                         octave_transverse=True)
        for i, s in enumerate(TRAIN):
            if s not in need:
                continue
            x0, x1, Pc, D = bA.make([trA[i]], eta="true")
            pred = oft.sample_flow(up, x0, Pc, D, torch.full((1,), sA), nsteps=8)
            np.save(f"{cache}/s{s}.npy", (pred[0] * scA.hf).numpy().astype(np.float32))
            print(f"upstream prediction cached for seed {s}", flush=True)

    # ---- downstream data: true 64/128 pairs + the predicted-64 alternatives -------------
    scB = oft.Scaffold(64, 128, L, OFF, 1.0, dev, window="cube")
    trB = oft.load_real(DIS, IC, 64, 128, TRAIN)
    scB.fit_linear_power([it["ic_f"] * GR for it in trB])
    pred64 = {s: np.load(f"{cache}/s{s}.npy") for s in TRAIN}

    model = oft.UNet3D(cin=12, cout=3, base=base).to(dev)
    model.load_state_dict(ck["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log = []
    step0 = 0
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
    # augmentation off: the predicted-coarse copy must stay aligned with the true target,
    # and augmenting the pair consistently would need the predicted field inside the
    # augment() call -- left out of this minimal version and noted in the RUNLOG.

    s_t = torch.full((1,), sB, device=dev)
    t0 = time.time()
    for step in range(step0 + 1, args.steps + 1):
        i = int(rng.integers(len(TRAIN)))
        it = dict(trB[i])
        use_pred = rng.random() < args.mix
        if use_pred:
            it["dis_c"] = pred64[TRAIN[i]]
        x0, x1, Pc, D = bB.make([it], eta="true")
        t = torch.rand(1, device=dev)
        xt = (1 - t)[:, None, None, None, None] * x0 + t[:, None, None, None, None] * x1
        v = model(oft.net_input(xt, Pc, D), t, s_t)
        loss = F.mse_loss(v, x1 - x0)
        e = v - (x1 - x0)
        loss = loss + lamB * em.jacobian_form(e, Pc.detach(), p=2.0, eps=0.1, fft_device=scB.fdev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, p_ in model.state_dict().items():
                ema[k].mul_(0.99).add_(p_.detach(), alpha=0.01)
        log.append(float(loss.detach()))
        if step % max(1, args.steps // 20) == 0 or step == step0 + 1:
            print(f"step {step:5d} loss {np.mean(log[-40:]):.4e} ({(time.time()-t0)/(step-step0):.2f}s/step)",
                  flush=True)
        if args.max_seconds is not None and time.time() - t0 > args.max_seconds and step < args.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema,
                            step=step, log=log), state_path)
            print(f"time budget reached at step {step}/{args.steps}; state saved")
            return
    # save: only the 64->128 level was tuned; the 32->64 half of the operator is the
    # ORIGINAL upstream (frozen), so store the tuned weights with the original metadata
    model.load_state_dict(ema); model.eval()
    torch.save({"state_dict": model.state_dict(), "base": base, "args": vars(args),
                "lams": ck["lams"], "s_values": ck["s_values"]},
               os.path.join(args.out, "model_ema.pt"))
    print("fine-tuning done; evaluate with runs/eval_chain.py using this checkpoint for step 2")


if __name__ == "__main__":
    main()
