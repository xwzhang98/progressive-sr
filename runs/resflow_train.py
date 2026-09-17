#!/usr/bin/env python3
"""Learned base + residual flow (reviewer's recommended model comparison; owner-approved).

Base  = the FROZEN trained regression (runs/R_reg_phys_3k): one Euler step from the physical
        source, B(coarse, octave) -- the best available conditional-mean predictor.
Flow  = a fresh network trained by flow matching from x0' = B to x1 = truth, i.e. it carries
        only the RESIDUAL the base cannot express (rms |x1 - B| = 0.31 h_f against the
        physical source's 0.58). qj loss, lambda auto-equalised, --octave-sampler full
        conventions throughout, so the comparison against J_flow_qj is one-factor.

Emulator mode:   B(true coarse, true octave)   -> integrate the residual flow.
Generative mode: B(true coarse, sampled octave) -> integrate. The stochasticity enters through
the octave inside the base's input, exactly as in the single-stage flow.

This is the local version of the RFMSR/PixelIR deterministic-base + stochastic-residual
pattern, with the perceptual losses replaced by our physical metrics.

  python runs/resflow_train.py --steps 3000 --out runs/RF_resflow
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

L, GR = 100000.0, 76.7439
DATASETS = {   # name -> (dis pattern, ic pattern, grid offset)
    "selfsim": ("data/selfsim/s{seed}/dis_{N}.npy", "data/selfsim/s{seed}/ic_dis_{N}.npy", 0.0),
    "psc": ("data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy", 0.5),
}


def cic_term(x_pred, x_true, offset, factor, compress):
    """MSE between compressed CIC densities (mean 1) of q + x_pred and q + x_true, fields in units of h_f,
    deposited on a (factor N)^3 periodic mesh with eulerian_metric.cic_deposit (differentiable in positions)."""
    def rho(f):
        n = f.shape[-1] * factor
        pos = em.eulerian_positions(f, offset) * factor
        return em.cic_deposit(f.new_ones((f.shape[0], 1) + tuple(f.shape[-3:])), pos, n)[:, 0] * factor ** 3
    a, b = rho(x_pred), rho(x_true)
    if compress == "log1p":
        a, b = torch.log1p(a), torch.log1p(b)
    elif compress == "sqrt":
        a, b = torch.sqrt(a + 1e-6), torch.sqrt(b + 1e-6)
    return F.mse_loss(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--nc", type=int, default=32); ap.add_argument("--nf", type=int, default=64)
    ap.add_argument("--base-ckpt", type=str, default="runs/R_reg_phys_3k/model_ema.pt",
                    help="frozen deterministic base (a --regression checkpoint for this level)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str, default="selfsim", choices=sorted(DATASETS),
                    help="selfsim (laptop, offset 0) or psc (cluster series, offset 0.5)")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[8])
    ap.add_argument("--oracle-coarse", action="store_true", help="DIAGNOSTIC: condition on R Psi_f, train and test")
    ap.add_argument("--cic-weight", type=float, default=0.0,
                    help="approved NONLINEAR Eulerian loss on the x-prediction (CIC density of q + x1_pred vs truth); a controlled "
                         "bias for the flow -- read GENERATIVE mode too")
    ap.add_argument("--cic-compress", type=str, default="log1p", choices=["log1p", "sqrt", "lin"],
                    help="compression of the density before the MSE: log1p (toy default, void-weighted), sqrt, lin (peak-weighted)")
    ap.add_argument("--cic-factor", type=int, default=1, help="deposit mesh = factor x fine grid (2 moves the CIC kernel ceiling to 2 k_Ny,f)")
    ap.add_argument("--jac-weight", type=float, default=0.0,
                    help="approved NONLINEAR loss on the x-prediction (oft.jac_loss, asinh J): a controlled bias for the flow "
                         "(notes 5.5/5.7d) -- read its effect in GENERATIVE mode too")
    args = ap.parse_args()
    DIS, IC, OFF = DATASETS[args.data]
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    sc = oft.Scaffold(args.nc, args.nf, L, OFF, 1.0, dev, window="cube")
    tr = oft.load_real(DIS, IC, args.nc, args.nf, args.train_seeds, box=L, offset=OFF, oracle_coarse=args.oracle_coarse)
    te = oft.load_real(DIS, IC, args.nc, args.nf, args.test_seeds[:1],   # residual target = NATIVE fine run
                       box=L, offset=OFF, oracle_coarse=args.oracle_coarse)
    sc.fit_linear_power([it["ic_f"] * GR for it in tr])
    rng = np.random.default_rng(args.seed)
    ba = oft.Batcher(sc, tr, rng, augment_on=True, growth=GR, octave_transverse=True)

    ckb = torch.load(args.base_ckpt, map_location="cpu")
    baseM = oft.UNet3D(cin=12, cout=3, base=int(ckb.get("base", 24)))
    baseM.load_state_dict(ckb["state_dict"]); baseM.eval(); baseM.to(dev)
    for p_ in baseM.parameters():
        p_.requires_grad_(False)

    @torch.no_grad()
    def base_pred(x0, Pc, D):
        z = torch.zeros(x0.shape[0], device=dev)
        return x0 + baseM(oft.net_input(x0, Pc, D), z, z)     # one Euler step at t=0

    model = oft.UNet3D(cin=12, cout=3, base=24).to(dev)
    print(f"residual-flow params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M; base frozen")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log = []; step0 = 0; lam = None
    state_path = os.path.join(args.out, "train_state.pt")
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ema = {k: v.to(dev) for k, v in st["ema"].items()}
        step0, log, lam = st["step"], st["log"], st["lam"]
        torch.manual_seed(args.seed + step0); ba.rng = np.random.default_rng(args.seed + step0)
        print(f"resumed from step {step0}")

    s_t = torch.zeros(args.batch, device=dev)
    t0 = time.time()
    for step in range(step0 + 1, args.steps + 1):
        x0, x1, Pc, D = ba.sample(args.batch)
        xb = base_pred(x0, Pc, D)                              # the learned base
        t = torch.rand(args.batch, device=dev)
        xt = (1 - t)[:, None, None, None, None] * xb + t[:, None, None, None, None] * x1
        v = model(oft.net_input(xt, Pc, D), t, s_t)
        loss = F.mse_loss(v, x1 - xb)
        e = v - (x1 - xb)
        term = em.jacobian_form(e, Pc.detach(), p=2.0, eps=0.1, fft_device=sc.fdev)
        if lam is None:
            lam = float(loss.detach() / term.detach().clamp_min(1e-30))
            print(f"lambda auto = {lam:.5g} (L_mse={float(loss):.4e}, L_qj={float(term):.4e})")
        loss = loss + lam * term
        if args.cic_weight > 0:
            x1c = xt + (1 - t)[:, None, None, None, None] * v          # the velocity's own endpoint prediction
            lcic = cic_term(x1c, x1, OFF, args.cic_factor, args.cic_compress)
            if step % max(1, args.steps // 20) == 0 or step == step0 + 1:
                print(f"        [cic] L_flow={float(loss):.4e} cic_weight*L_cic={float(args.cic_weight * lcic):.4e}", flush=True)
            loss = loss + args.cic_weight * lcic
        if args.jac_weight > 0:
            x1p = xt + (1 - t)[:, None, None, None, None] * v          # the velocity's own endpoint prediction
            ljac = oft.jac_loss(sc, x1p, x1)
            if step % max(1, args.steps // 20) == 0 or step == step0 + 1:
                print(f"        [jac] L_flow={float(loss):.4e} jac_weight*L_jac={float(args.jac_weight * ljac):.4e}", flush=True)
            loss = loss + args.jac_weight * ljac
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
                            step=step, log=log, lam=lam), state_path)
            print(f"time budget reached at step {step}/{args.steps}; state saved")
            return

    model.load_state_dict(ema); model.eval()
    torch.save({"state_dict": model.state_dict(), "base": 24, "lam": lam, "args": vars(args)},
               os.path.join(args.out, "model_ema.pt"))

    # ---- evaluation: emulator and generative, Lagrangian + Eulerian probes -------------
    bt = oft.Batcher(sc, te, np.random.default_rng(0), augment_on=False, growth=GR,
                     octave_transverse=True)
    x0, x1, Pc, D = bt.make(te, eta="true")
    xb = base_pred(x0, Pc, D)
    s1 = torch.zeros(1, device=dev)
    pred = oft.sample_flow(model, xb, Pc, D, s1, nsteps=8)
    gen = torch.Generator(device=sc.fdev).manual_seed(1)
    x0g, _, _, _ = bt.make(te, eta="sample", gen=gen)
    predg = oft.sample_flow(model, base_pred(x0g, Pc, D), Pc, D, s1, nsteps=8)
    g = sc.gf; W = sc.W.cpu().numpy(); knyc = sc.knyc; hf = sc.hf
    res = {"lam": lam}
    def lag(a, b):
        s_ = p0.spectra_k(g, p0.rfftn(a.astype(np.float32)) * (1 - W),
                          p0.rfftn(b.astype(np.float32)) * (1 - W), kmin=knyc)
        return (float(np.average(s_["r"])),
                float(np.average(s_["Paa"] / np.maximum(s_["Pbb"], 1e-30))))
    tru = (x1[0] * hf).cpu().numpy()
    def dens(f):
        pos = em.eulerian_positions(f.cpu(), OFF)
        return em.cic_deposit(torch.ones((1, 1) + f.shape[-3:]), pos, f.shape[-1])[0, 0]
    def spec(r_):
        d = (r_ / r_.mean() - 1).numpy().astype(np.float32)
        Fk = p0.rfftn(d[None]); P = g.shell_avg(np.abs(Fk[0]) ** 2)
        m = (g.shell_norm > 0) & (g.kshell > 0)
        return g.kshell[m], P[m]
    kT, PT = spec(dens(x1))
    for nm, f in (("base", xb), ("emulator", pred), ("generative", predg)):
        r_, pp = lag((f[0] * hf).cpu().numpy(), tru)
        k, P = spec(dens(f))
        eul = [float(P[np.argmin(abs(k - q))] / PT[np.argmin(abs(k - q))]) for q in (knyc, 1.5 * knyc, 2 * knyc)]
        rms = float((((f - x1) ** 2).mean()) ** 0.5)
        res[nm] = dict(r_high=r_, ratio_P_high=pp, rms=rms, eul=eul)
        print(f"{nm:11s} r={r_:.4f} P/P={pp:.4f} rms={rms:.3f} | Eul {[round(x,3) for x in eul]}")
    res["args"] = vars(args)
    with open(os.path.join(args.out, "results_resflow.json"), "w") as fjs:
        json.dump(res, fjs, indent=1)
    # test-box fields in box units (kpc/h here), for the density-vs-reference evaluation (gitignored .npy)
    for nm, f in (("truth", x1), ("base", xb), ("emulator", pred), ("generative", predg)):
        np.save(os.path.join(args.out, f"field_{nm}.npy"), (f[0] * hf).cpu().numpy().astype(np.float32))


if __name__ == "__main__":
    main()
