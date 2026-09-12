#!/usr/bin/env python3
"""Multi-transition weight-shared training: ONE operator for 32->64 AND 64->128.

Owner-approved 2026-09-11 ("可以两个一起训练，32-64和64-128") -- the ask-first item
"multi-transition (style scalar) training" in CLAUDE.md. This is Phase 1 of notes Sec. 11:
a single weight-shared 2x operator conditioned on the level through the style scalar s_l.
The matched-sigma test (RUNLOG 2026-09-09) grounds it: second moments collapse to <1% across
the two levels once sigma is matched, so sigma is the natural conditioning variable.

Design (all reusing octave_flow_toy's classes; the training script itself is not touched):
  * one UNet3D(cin=12), one Adam, one EMA;
  * two (Scaffold, Batcher) pairs: 32->64 at batch 2, 64->128 at batch 1 (memory: batch 2
    at 128^3 thrashes, RUNLOG 2026-09-09);
  * steps alternate between the levels (even -> A, odd -> B), `--steps` counts PER LEVEL;
  * s_l = ln(sigma_c) with sigma_c the z=0-scaled linear rms delta at the coarse cell scale,
    measured from the training ICs per level -- physical, and extrapolates to future levels
    (a plain level tag would not);
  * loss = Lagrangian MSE + lambda_l * Q_J (the Stage-7 winner), lambda equalised at each
    level's own first step and PINNED (persisted in the state file; the Stage-7 drift bug).

Evaluation at the end, per level: the standard Lagrangian summary plus Eulerian density
ratios at (1, 1.5, 2) k_Ny,c for emulator and generative (independent-T sampler, to be
comparable with the single-level runs' recorded numbers; the oracle prior is a separate
evaluation). Written to <out>/results_<Nc>to<Nf>.json.

  python runs/multi_train.py --steps 3000 --max-seconds 900 --resume --out runs/M_qj_multi

Chunked/resumable exactly like the main script; honours the same PAUSE-driver pattern.
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


def sigma_c(ic_list, growth, Nc, L):
    """z=0-scaled linear rms delta smoothed at the coarse cell scale (|k| < pi/h_c)."""
    g = p0.Grid(ic_list[0].shape[-1], L)
    acc = 0.0
    for ic in ic_list:
        Fi = p0.rfftn((ic * growth).astype(np.float32))
        delta = 1j * (g.kx * Fi[0] + g.ky * Fi[1] + g.kz * Fi[2])
        W = (g.kmag < np.pi / (L / Nc))
        acc += float((np.abs(p0.irfftn((delta * W)[None], s=ic.shape[-3:])[0]) ** 2).mean())
    return np.sqrt(acc / len(ic_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000, help="steps PER LEVEL")
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--jac-p", type=float, default=2.0)
    ap.add_argument("--jac-eps", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    L, off, gr = 100000.0, 0.0, 76.7439
    DIS, IC = "data/selfsim/s{seed}/dis_{N}.npy", "data/selfsim/s{seed}/ic_dis_{N}.npy"
    TRAIN, TEST = list(range(8)), [8]

    levels = []
    for (Nc, Nf, batch) in [(32, 64, 2), (64, 128, 1)]:
        sc = oft.Scaffold(Nc, Nf, L, off, 1.0, dev, window="cube")
        tr = oft.load_real(DIS, IC, Nc, Nf, TRAIN)
        te = oft.load_real(DIS, IC, Nc, Nf, TEST)
        sc.fit_linear_power([it["ic_f"] * gr for it in tr])
        rng = np.random.default_rng(args.seed + Nc)
        ba = oft.Batcher(sc, tr, rng, augment_on=True, growth=gr, octave_transverse=True)
        s_l = float(np.log(sigma_c([it["ic_f"] for it in tr], gr, Nc, L)))
        levels.append(dict(Nc=Nc, Nf=Nf, batch=batch, sc=sc, tr=tr, te=te, ba=ba,
                           s=s_l, lam=None))
        print(f"level {Nc}->{Nf}: s_l = ln(sigma_c) = {s_l:.4f}")

    model = oft.UNet3D(cin=12, cout=3, base=args.base).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M (base={args.base})")
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
        for lv, lam in zip(levels, st["lams"]):
            lv["lam"] = lam
        torch.manual_seed(args.seed + step0)
        for lv in levels:
            lv["ba"].rng = np.random.default_rng(args.seed + lv["Nc"] + step0)
        print(f"resumed from global step {step0}")

    def save_state(step):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema, step=step,
                        log=log, base=args.base, lams=[lv["lam"] for lv in levels]), state_path)

    total = 2 * args.steps
    t0 = time.time()
    for gstep in range(step0 + 1, total + 1):
        lv = levels[(gstep - 1) % 2]
        sc, ba = lv["sc"], lv["ba"]
        x0, x1, Pc, D = ba.sample(lv["batch"])
        t = torch.rand(lv["batch"], device=dev)
        xt = (1 - t)[:, None, None, None, None] * x0 + t[:, None, None, None, None] * x1
        s = torch.full((lv["batch"],), lv["s"], device=dev)
        v = model(oft.net_input(xt, Pc, D), t, s)
        loss = F.mse_loss(v, x1 - x0)
        e = v - (x1 - x0)
        term = em.jacobian_form(e, Pc.detach(), p=args.jac_p, eps=args.jac_eps, fft_device=sc.fdev)
        if lv["lam"] is None:
            lv["lam"] = float(loss.detach() / term.detach().clamp_min(1e-30))
            print(f"lambda[{lv['Nc']}->{lv['Nf']}] = {lv['lam']:.5g} "
                  f"(L_mse={float(loss):.4e}, L_qj={float(term):.4e})")
        loss = loss + lv["lam"] * term
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, p_ in model.state_dict().items():
                ema[k].mul_(0.99).add_(p_.detach(), alpha=0.01)
        log.append(float(loss.detach()))
        if gstep % max(1, total // 40) == 0 or gstep == step0 + 1:
            print(f"gstep {gstep:5d} [{lv['Nc']}->{lv['Nf']}] loss {np.mean(log[-40:]):.4e} "
                  f"({(time.time()-t0)/(gstep-step0):.2f}s/step)", flush=True)
        if args.max_seconds is not None and time.time() - t0 > args.max_seconds and gstep < total:
            save_state(gstep)
            print(f"time budget reached at global step {gstep}/{total}; state saved")
            return
    save_state(total)
    model.load_state_dict(ema); model.eval()
    torch.save({"state_dict": model.state_dict(), "base": args.base, "args": vars(args),
                "lams": [lv["lam"] for lv in levels], "s_values": [lv["s"] for lv in levels]},
               os.path.join(args.out, "model_ema.pt"))

    # ---- evaluation, per level ------------------------------------------------
    for lv in levels:
        sc = lv["sc"]
        teb = oft.Batcher(sc, lv["te"], np.random.default_rng(0), augment_on=False,
                          growth=gr, octave_transverse=True)
        x0, x1, Pc, D = teb.make(lv["te"], eta="true")
        s = torch.full((1,), lv["s"], device=dev)
        ms = None
        res = {"s_l": lv["s"], "lambda": lv["lam"], "args": vars(args)}
        pred = oft.sample_flow(model, x0, Pc, D, s, nsteps=8)
        gen = torch.Generator(device=sc.fdev).manual_seed(1)
        x0g, _, _, _ = teb.make(lv["te"], eta="sample", gen=gen)
        predg = oft.sample_flow(model, x0g, Pc, D, s, nsteps=8)
        # Lagrangian octave-band numbers (same definitions as summarize())
        g = sc.gf
        W = sc.W.cpu().numpy()
        knyc = sc.knyc
        def lag(a, b):
            Fa, Fb = p0.rfftn(a.astype(np.float32)), p0.rfftn(b.astype(np.float32))
            hi_a, hi_b = Fa * (1 - W), Fb * (1 - W)
            s_ = p0.spectra_k(g, hi_a, hi_b, kmin=knyc)
            return (float(np.average(s_["r"])),
                    float(np.average(s_["Paa"] / np.maximum(s_["Pbb"], 1e-30))))
        hf = sc.hf
        tru = (x1[0] * hf).cpu().numpy()
        r_e, pp_e = lag((pred[0] * hf).cpu().numpy(), tru)
        r_g, pp_g = lag((predg[0] * hf).cpu().numpy(), tru)
        rms = float((((pred - x1) ** 2).mean()) ** 0.5)
        res["emulator"] = dict(r_high=r_e, ratio_P_high=pp_e, rms_err_over_h=rms)
        res["generative"] = dict(r_high=r_g, ratio_P_high=pp_g)
        # Eulerian probes
        def dens(f):
            pos = em.eulerian_positions(f.cpu(), off)
            ones = torch.ones((1, 1) + f.shape[-3:])
            return em.cic_deposit(ones, pos, f.shape[-1])[0, 0]
        rt = dens(x1); k0, P0, _ = None, None, None
        def spec(r_):
            d = (r_ / r_.mean() - 1).numpy().astype(np.float32)
            Fk = p0.rfftn(d[None]); P = g.shell_avg(np.abs(Fk[0]) ** 2)
            m = (g.shell_norm > 0) & (g.kshell > 0)
            return g.kshell[m], P[m]
        kT, PT = spec(rt)
        for nm, f in (("emulator", pred), ("generative", predg)):
            k, P = spec(dens(f))
            res[f"Pdelta_ratio_{nm}"] = [float(P[np.argmin(abs(k - q))] / PT[np.argmin(abs(k - q))])
                                         for q in (knyc, 1.5 * knyc, 2 * knyc)]
        tag = f"{lv['Nc']}to{lv['Nf']}"
        with open(os.path.join(args.out, f"results_{tag}.json"), "w") as fjs:
            json.dump(res, fjs, indent=1)
        print(f"[{tag}] emu r={r_e:.4f} P/P={pp_e:.4f} rms={rms:.3f} | gen P/P={pp_g:.4f} | "
              f"Eul emu {res['Pdelta_ratio_emulator']} gen {res['Pdelta_ratio_generative']}")


if __name__ == "__main__":
    main()
