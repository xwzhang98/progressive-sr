#!/usr/bin/env python3
"""C2: ONE weight-shared two-stage operator for 32->64 AND 64->128 (HANDOFF §2 C2 / §6; owner
2026-09-16 "三个都做"; two-transition training pre-approved 2026-09-11).

Three modes, each ONE UNet3D(cin=12) shared across the levels and conditioned on the level through
the style scalar s_l = ln(sigma_c) (multi_train.py convention: z=0-scaled linear rms delta at the
coarse cell scale, measured from the training ICs per level):
  --mode base    shared REGRESSION base: t = 0 only, target x1 - x0, one Euler step at evaluation
                 (the two-level version of runs/R_reg_psc* / F_reg_128_psc).
  --mode resflow shared RESIDUAL FLOW on a frozen shared base (--base-ckpt): x0' = B(x0), flow
                 matching from x0' to x1 with the Q_J form, lambda equalised at each level's own
                 first step and pinned (resflow_train.py / multi_train.py conventions).
  --mode reg2    the owner's pending control (HANDOFF §6): the SAME second network trained as a
                 one-step regression on the base's residual (t = 0, target x1 - B(x0), same
                 MSE + lambda Q_J loss) -- separates the two-stage structure from the flow.
Steps alternate between the levels (even -> A, odd -> B); `--steps` counts PER LEVEL; batch 2 at
32->64 and 1 at 64->128. Evaluation per level on the first test set: Lagrangian octave numbers,
Eulerian probes vs the native truth, and the test-box field dumps
(field_{truth,base,emulator,generative}_<Nc>to<Nf>.npy) for runs/eval_vs_ref.py.

  python runs/multi_resflow_train.py --mode base    --data psc --out runs/M_reg_psc
  python runs/multi_resflow_train.py --mode resflow --data psc --base-ckpt runs/M_reg_psc/model_ema.pt --out runs/M_resflow_psc
  python runs/multi_resflow_train.py --mode reg2    --data psc --base-ckpt runs/M_reg_psc/model_ema.pt --out runs/M_reg2_psc
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402
import eulerian_metric as em       # noqa: E402
import tiling                      # noqa: E402

_spec = importlib.util.spec_from_file_location("multi_train", os.path.join(ROOT, "runs", "multi_train.py"))
_mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mt)
sigma_c = _mt.sigma_c

L, GR = 100000.0, 76.7439
DATASETS = {   # name -> (dis pattern, ic pattern, grid offset)
    "selfsim": ("data/selfsim/s{seed}/dis_{N}.npy", "data/selfsim/s{seed}/ic_dis_{N}.npy", 0.0),
    "psc": ("data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy", 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "resflow", "reg2"], required=True)
    ap.add_argument("--base-ckpt", type=str, default=None, help="frozen shared base (--mode base checkpoint)")
    ap.add_argument("--steps", type=int, default=3000, help="steps PER LEVEL")
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--jac-p", type=float, default=2.0)
    ap.add_argument("--jac-eps", type=float, default=0.1)
    ap.add_argument("--jac-weight", type=float, default=0.0, help="approved asinh(J) loss on the endpoint prediction (resflow mode; controlled bias)")
    ap.add_argument("--levels", type=int, nargs="+", default=[32, 64], help="coarse grids; fine = 2x")
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--crops", type=int, nargs="+", default=None,
                    help="per level: fine-grid crop side for training (0 = full box); owner approved patch cropping 2026-09-18")
    ap.add_argument("--halo", type=int, default=16, help="crop cells excluded from the loss on every face")
    ap.add_argument("--tile-core", type=int, default=64, help="inference at crop levels: tiles of the crop side, central core kept")
    ap.add_argument("--stream-levels", type=int, nargs="*", default=[],
                    help="coarse grids whose training data are STREAMED (tiling.CropSource: coarse run in RAM, fine target and the "
                         "precomputed IC octave memory-mapped, only crops read; owner 2026-09-19 '内存映射读取'); must be crop levels")
    ap.add_argument("--stream-ic-sets", type=int, nargs="*", default=None,
                    help="sets whose ICs give the linear power and sigma_c at stream levels (default: the first two training sets)")
    ap.add_argument("--octave-dir", default="data/psc/octave/dmo-{N}", help="precomputed IC octaves (runs/precompute_octave.py)")
    ap.add_argument("--eval-skip-levels", type=int, nargs="*", default=None,
                    help="coarse grids skipped by the end-of-training evaluation (default: the stream levels; use runs/chain_shared.py)")
    ap.add_argument("--data", type=str, default="psc", choices=sorted(DATASETS))
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(14)))
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[14])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", type=str, default=None,
                    help="checkpoint of THIS mode: skip training, evaluate on --test-seeds (second-box checks)")
    args = ap.parse_args()
    if args.mode != "base":
        assert args.base_ckpt, "--mode resflow/reg2 need --base-ckpt"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    DIS, IC, OFF = DATASETS[args.data]

    crops = args.crops if args.crops is not None else [0] * len(args.levels)
    assert len(crops) == len(args.levels) == len(args.batches), "--levels, --batches and --crops must align"
    assert not (args.jac_weight > 0 and any(crops)), "--jac-weight is not implemented on cropped levels"
    levels = []
    for Nc, batch, C in zip(args.levels, args.batches, crops):
        Nf = 2 * Nc
        sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, dev, window="cube")
        tr = oft.load_real(DIS, IC, Nc, Nf, args.train_seeds)          # p0.load memory-maps: nothing is read yet
        te = oft.load_real(DIS, IC, Nc, Nf, args.test_seeds[:1])
        rng = np.random.default_rng(args.seed + Nc)
        src = None
        if Nc in args.stream_levels:
            assert C and batch == 1, "stream levels must be crop levels with batch 1"
            for it in tr:
                it["dis_c"] = np.array(it["dis_c"], dtype=np.float32)      # coarse runs in RAM (200 MB each at 256^3)
            lin = [np.load(os.path.join(args.octave_dir.format(N=Nf), f"set{k}.npy"), mmap_mode="r") for k in args.train_seeds]
            icsets = args.stream_ic_sets or args.train_seeds[:2]
            ics = [p0.load(IC.replace("{seed}", str(k)), Nf) for k in icsets]
            sc.fit_linear_power([ic * GR for ic in ics])
            s_l = float(np.log(sigma_c(ics, GR, Nc, L)))
            del ics
            src = tiling.CropSource(sc, [it["dis_c"] for it in tr], [it["dis_f"] for it in tr], lin)
            ba = oft.Batcher(sc, [], rng, augment_on=False, growth=GR, octave_transverse=True)   # only its rng is used
        else:
            sc.fit_linear_power([it["ic_f"] * GR for it in tr])
            # crop levels: no numpy augmentation of full boxes (CPU-bound at 256^3); the crops are augmented on the GPU instead
            ba = oft.Batcher(sc, tr, rng, augment_on=(C == 0), growth=GR, octave_transverse=True)
            s_l = float(np.log(sigma_c([it["ic_f"] for it in tr], GR, Nc, L)))
        assert C == 0 or (C < Nf and C % 4 == 0 and C > 2 * args.halo and (C - args.tile_core) // 2 >= args.halo), f"bad crop {C}"
        levels.append(dict(Nc=Nc, Nf=Nf, batch=batch, sc=sc, tr=tr, te=te, ba=ba, s=s_l, lam=None, crop=C, src=src,
                           taper=tiling.taper(C, args.halo, sc.fdev) if C else None))
        print(f"level {Nc}->{Nf}: {len(tr)} train / {len(te)} test boxes, s_l = ln(sigma_c) = {s_l:.4f}"
              + (f", CROPPED training {C}^3 (halo {args.halo}), tiled inference core {args.tile_core}" if C else ", full box")
              + (f", STREAMED (mmap crops; linear power and s from sets {icsets})" if src is not None else ""), flush=True)

    baseM = None
    if args.mode != "base":
        ckb = torch.load(args.base_ckpt, map_location="cpu")
        baseM = oft.UNet3D(cin=12, cout=3, base=int(ckb.get("base", args.base)))
        baseM.load_state_dict(ckb["state_dict"]); baseM.eval(); baseM.to(dev)
        for p_ in baseM.parameters():
            p_.requires_grad_(False)
        print(f"frozen shared base from {args.base_ckpt} (s_values {ckb.get('s_values')})", flush=True)

    @torch.no_grad()
    def base_pred(x0, Pc, D, s):
        z = torch.zeros(x0.shape[0], device=dev)
        return x0 + baseM(oft.net_input(x0, Pc, D), z, s)   # one Euler step at t = 0

    model = oft.UNet3D(cin=12, cout=3, base=args.base).to(dev)
    print(f"[{args.mode}] shared params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M (base={args.base})"
          + (f"; frozen base {sum(p.numel() for p in baseM.parameters())/1e6:.2f}M" if baseM else ""), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log = []; step0 = 0
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
        print(f"resumed from global step {step0}", flush=True)

    def save_state(step):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema, step=step, log=log,
                        base=args.base, lams=[lv["lam"] for lv in levels]), state_path)

    if args.eval_only:
        cke = torch.load(args.eval_only, map_location=dev)
        assert cke.get("mode", args.mode) == args.mode, "checkpoint mode differs from --mode"
        model.load_state_dict(cke["state_dict"]); model.eval()
        for lv, lam in zip(levels, cke.get("lams") or [None] * len(levels)):
            lv["lam"] = lam
        print(f"eval-only: {args.eval_only} on test set {args.test_seeds[0]}", flush=True)
    else:
        nlev = len(levels); total = nlev * args.steps
        t0 = time.time()
        for gstep in range(step0 + 1, total + 1):
            lv = levels[(gstep - 1) % nlev]
            sc, ba, B = lv["sc"], lv["ba"], lv["batch"]
            C, H = lv["crop"], args.halo
            if lv["src"] is not None:
                # streamed level: prolongation and D_ij on the full box on the device, target and IC octave read as crops
                i_ = int(ba.rng.integers(len(lv["tr"])))
                o = [int(v_) for v_ in ba.rng.integers(0, lv["Nf"], size=3)]
                x0, x1, Pc, D = lv["src"].sample(i_, o, C)
                if args.mode != "base":
                    Jc, adjc = tiling.jac_adj_from_D(D)            # pointwise: identical to cropping the full-box J/adj
            else:
                x0, x1, Pc, D = ba.sample(B)
            if C and lv["src"] is None:
                # all spectral operations were done on the full box inside ba.sample; the Q_J state (J, adj) too
                if args.mode != "base":
                    with torch.no_grad():
                        Jf, adjf = em.jacobian_and_adjugate(Pc.detach(), sc.fdev)
                o = [int(v_) for v_ in ba.rng.integers(0, lv["Nf"], size=3)]
                x0, x1, Pc, D = (tiling.crop_periodic(a_, o, C) for a_ in (x0, x1, Pc, D))
                if args.mode != "base":
                    Jc = tiling.crop_periodic(Jf[:, None], o, C)[:, 0]
                    Bn = adjf.shape[0]; Nn = adjf.shape[1]
                    adjc = tiling.crop_periodic(adjf.reshape(Bn, Nn, Nn, Nn, 9).permute(0, 4, 1, 2, 3), o, C)
                    adjc = adjc.permute(0, 2, 3, 4, 1).reshape(Bn, C, C, C, 3, 3)
                    del Jf, adjf
            if C:
                # cube-group augmentation of the crops (equivalent to augmenting the raw fields, checks/tiling_check.py 3)
                Bn = x0.shape[0]
                if args.mode != "base":
                    (x0, x1, Pc), (D33, adj33), (Jc,) = tiling.augment_crops(ba.rng, vecs=(x0, x1, Pc),
                        tens=(D.reshape(Bn, 3, 3, C, C, C), adjc.permute(0, 4, 5, 1, 2, 3)), scals=(Jc,))
                    adjc = adj33.permute(0, 3, 4, 5, 1, 2)
                else:
                    (x0, x1, Pc), (D33,), _ = tiling.augment_crops(ba.rng, vecs=(x0, x1, Pc), tens=(D.reshape(Bn, 3, 3, C, C, C),))
                D = D33.reshape(Bn, 9, C, C, C)
            s = torch.full((B,), lv["s"], device=dev)
            if args.mode == "base":
                src = x0
            else:
                src = base_pred(x0, Pc, D, s)
            if args.mode == "resflow":
                t = torch.rand(B, device=dev)
                xt = (1 - t)[:, None, None, None, None] * src + t[:, None, None, None, None] * x1
            else:
                t = torch.zeros(B, device=dev); xt = src
            v = model(oft.net_input(xt, Pc, D), t, s)
            target = x1 - src
            loss = F.mse_loss(tiling.interior(v, H), tiling.interior(target, H)) if C else F.mse_loss(v, target)
            if args.mode != "base":
                e = v - target
                if C:
                    # Q_J on a crop: taper e to ~0 across the halo (smooth periodic continuation), exact spectral gradient
                    # in the interior, form averaged over the interior only; J/adj from the full-box state
                    dE = em.spectral_gradient(e * lv["taper"], sc.fdev).permute(0, 3, 4, 5, 1, 2)
                    dJ = (adjc * dE.transpose(-1, -2)).sum(dim=(-1, -2))
                    wq = (Jc ** 2 + args.jac_eps ** 2) ** (-args.jac_p / 2)
                    term = tiling.interior(dJ ** 2 * wq, H).mean()
                else:
                    term = em.jacobian_form(e, Pc.detach(), p=args.jac_p, eps=args.jac_eps, fft_device=sc.fdev)
                if lv["lam"] is None:
                    lv["lam"] = float(loss.detach() / term.detach().clamp_min(1e-30))
                    print(f"lambda[{lv['Nc']}->{lv['Nf']}] = {lv['lam']:.5g} (L_mse={float(loss):.4e}, L_qj={float(term):.4e})", flush=True)
                loss = loss + lv["lam"] * term
                if args.jac_weight > 0 and args.mode == "resflow":
                    x1p = xt + (1 - t)[:, None, None, None, None] * v
                    loss = loss + args.jac_weight * oft.jac_loss(sc, x1p, x1)
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
                print(f"time budget reached at global step {gstep}/{total}; state saved", flush=True)
                return
        save_state(total)
        model.load_state_dict(ema); model.eval()
        torch.save({"state_dict": model.state_dict(), "base": args.base, "args": vars(args), "mode": args.mode,
                    "lams": [lv["lam"] for lv in levels], "s_values": [lv["s"] for lv in levels]},
                   os.path.join(args.out, "model_ema.pt"))

    # ---- evaluation, per level (crop levels: tiled inference with the training crop geometry) --------------------
    skip = args.eval_skip_levels if args.eval_skip_levels is not None else args.stream_levels
    for lv in levels:
        if lv["Nc"] in skip:
            print(f"[{lv['Nc']}to{lv['Nf']}] end-of-training evaluation skipped (evaluate with runs/chain_shared.py)", flush=True)
            continue
        sc = lv["sc"]; tag = f"{lv['Nc']}to{lv['Nf']}"
        netM = tiling.TiledNet(model, lv["crop"], args.tile_core) if lv["crop"] else model
        netB = (tiling.TiledNet(baseM, lv["crop"], args.tile_core) if lv["crop"] else baseM) if baseM is not None else None
        def base_pred(x0, Pc, D, s, netB=netB):
            z_ = torch.zeros(x0.shape[0], device=dev)
            return x0 + netB(oft.net_input(x0, Pc, D), z_, s)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        teb = oft.Batcher(sc, lv["te"], np.random.default_rng(0), augment_on=False, growth=GR, octave_transverse=True)
        x0, x1, Pc, D = teb.make(lv["te"], eta="true")
        s = torch.full((1,), lv["s"], device=dev)
        gen = torch.Generator(device=sc.fdev).manual_seed(1)
        x0g, _, _, _ = teb.make(lv["te"], eta="sample", gen=gen)
        z = torch.zeros(1, device=dev)
        with torch.no_grad():
            if args.mode == "base":
                xb = x0; xbg = x0g
                pred = x0 + netM(oft.net_input(x0, Pc, D), z, s)
                predg = x0g + netM(oft.net_input(x0g, Pc, D), z, s)
            else:
                xb = base_pred(x0, Pc, D, s); xbg = base_pred(x0g, Pc, D, s)
                if args.mode == "resflow":
                    pred = oft.sample_flow(netM, xb, Pc, D, s, nsteps=8)
                    predg = oft.sample_flow(netM, xbg, Pc, D, s, nsteps=8)
                else:
                    pred = xb + netM(oft.net_input(xb, Pc, D), z, s)
                    predg = xbg + netM(oft.net_input(xbg, Pc, D), z, s)
        g = sc.gf; W = sc.W.cpu().numpy(); knyc = sc.knyc; hf = sc.hf
        def lag(a, b):
            s_ = p0.spectra_k(g, p0.rfftn(a.astype(np.float32)) * (1 - W), p0.rfftn(b.astype(np.float32)) * (1 - W), kmin=knyc)
            return (float(np.average(s_["r"])), float(np.average(s_["Paa"] / np.maximum(s_["Pbb"], 1e-30))))
        def dens(f):
            pos = em.eulerian_positions(f.cpu(), OFF)
            return em.cic_deposit(torch.ones((1, 1) + f.shape[-3:]), pos, f.shape[-1])[0, 0]
        def spec(r_):
            d = (r_ / r_.mean() - 1).numpy().astype(np.float32)
            Fk = p0.rfftn(d[None]); P = g.shell_avg(np.abs(Fk[0]) ** 2)
            m = (g.shell_norm > 0) & (g.kshell > 0)
            return g.kshell[m], P[m]
        tru = (x1[0] * hf).cpu().numpy()
        kT, PT = spec(dens(x1))
        res = {"s_l": lv["s"], "lambda": lv["lam"], "mode": args.mode, "args": vars(args)}
        for nm, f in (("base", xb), ("emulator", pred), ("generative", predg)):
            r_, pp = lag((f[0] * hf).cpu().numpy(), tru)
            k, P = spec(dens(f))
            eul = [float(P[np.argmin(abs(k - q))] / PT[np.argmin(abs(k - q))]) for q in (knyc, 1.5 * knyc, 2 * knyc)]
            rms = float((((f - x1) ** 2).mean()) ** 0.5)
            res[nm] = dict(r_high=r_, ratio_P_high=pp, rms=rms, eul=eul)
            print(f"[{tag}] {nm:10s} r={r_:.4f} P/P={pp:.4f} rms={rms:.3f} | Eul {[round(x, 3) for x in eul]}", flush=True)
        for nm, f in (("truth", x1), ("base", xb), ("emulator", pred), ("generative", predg)):
            np.save(os.path.join(args.out, f"field_{nm}_{tag}.npy"), (f[0] * hf).cpu().numpy().astype(np.float32))
        with open(os.path.join(args.out, f"results_{tag}.json"), "w") as fjs:
            json.dump(res, fjs, indent=1)


if __name__ == "__main__":
    main()
