#!/usr/bin/env python3
"""The chained 32->64->128 evaluation: feed the 32->64 model's OUTPUT to the 64->128 step.

Fills the right-hand column of the four-cell table (owner's review, 2026-09-11):

                      | true coarse per level | chained (prev level's output)
    specialist models |  J_flow_qj, J_flow_128|  this script, --pair specialist
    shared model      |  M_qj_multi (per lvl) |  this script, --pair shared

Emulator mode at BOTH levels (true IC octaves), so the only thing that changes between the
columns is the coarse pathway: the chained run's 64->128 step conditions on the GENERATED
64 field (P Psi_c, D_ij, and the source's coarse part all rebuilt from it). This isolates
the error-accumulation question from the generative-sampler question.

Outputs Lagrangian octave metrics and Eulerian density probes at 128 against the true 128
field, for the direct step (true coarse) and the chained step, same test seed.

  python runs/eval_chain.py --pair specialist --test-seed 8
  python runs/eval_chain.py --pair shared     --test-seed 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402
import eulerian_metric as em       # noqa: E402

L, OFF, GR = 100000.0, 0.0, 76.7439
DIS, IC = "data/selfsim/s{seed}/dis_{N}.npy", "data/selfsim/s{seed}/ic_dis_{N}.npy"
TRAIN = list(range(8))


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    m = oft.UNet3D(cin=12, cout=3, base=int(ck.get("base", 24)))
    m.load_state_dict(sd); m.eval()
    return m, ck


def build_level(Nc, Nf, test_seed):
    sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, torch.device("cpu"), window="cube")
    tr = oft.load_real(DIS, IC, Nc, Nf, TRAIN)
    te = oft.load_real(DIS, IC, Nc, Nf, [test_seed])
    sc.fit_linear_power([it["ic_f"] * GR for it in tr])
    return sc, te


def predict(model, sc, items, s_val):
    b = oft.Batcher(sc, items, np.random.default_rng(0), augment_on=False, growth=GR,
                    octave_transverse=True)
    x0, x1, Pc, D = b.make(items, eta="true")
    s = torch.full((1,), s_val)
    if isinstance(model, tuple):                      # (frozen base, residual flow)
        baseM, flowM = model
        with torch.no_grad():
            z = torch.zeros(1)
            xb = x0 + baseM(oft.net_input(x0, Pc, D), z, z)
        return oft.sample_flow(flowM, xb, Pc, D, s, nsteps=8), x1, sc
    return oft.sample_flow(model, x0, Pc, D, s, nsteps=8), x1, sc


def metrics(sc, pred, x1, tag):
    g = sc.gf; W = sc.W.cpu().numpy(); knyc = sc.knyc; hf = sc.hf
    a = (pred[0] * hf).numpy(); b = (x1[0] * hf).numpy()
    Fa, Fb = p0.rfftn(a.astype(np.float32)) * (1 - W), p0.rfftn(b.astype(np.float32)) * (1 - W)
    s_ = p0.spectra_k(g, Fa, Fb, kmin=knyc)
    r = float(np.average(s_["r"])); pp = float(np.average(s_["Paa"] / np.maximum(s_["Pbb"], 1e-30)))
    rms = float((((pred - x1) ** 2).mean()) ** 0.5)
    def dens(f):
        pos = em.eulerian_positions(f, OFF)
        return em.cic_deposit(torch.ones((1, 1) + f.shape[-3:]), pos, f.shape[-1])[0, 0]
    def spec(r_):
        d = (r_ / r_.mean() - 1).numpy().astype(np.float32)
        Fk = p0.rfftn(d[None]); P = g.shell_avg(np.abs(Fk[0]) ** 2)
        m = (g.shell_norm > 0) & (g.kshell > 0)
        return g.kshell[m], P[m]
    rt = dens(x1); rp = dens(pred)
    kT, PT = spec(rt); k, P = spec(rp)
    eul = [float(P[np.argmin(abs(k - q))] / PT[np.argmin(abs(k - q))]) for q in (knyc, 1.5 * knyc, 2 * knyc)]
    Ft = p0.rfftn((rt / rt.mean() - 1).numpy().astype(np.float32)[None])
    Fp = p0.rfftn((rp / rp.mean() - 1).numpy().astype(np.float32)[None])
    num = g.shell_avg((Fp[0] * np.conj(Ft[0])).real)
    den = np.sqrt(g.shell_avg(np.abs(Fp[0]) ** 2) * g.shell_avg(np.abs(Ft[0]) ** 2))
    mm = (g.shell_norm > 0) & (g.kshell > 0)
    rdel = (num / np.maximum(den, 1e-30))[mm]
    rdl = [float(rdel[np.argmin(abs(k - q))]) for q in (knyc, 1.5 * knyc, 2 * knyc)]
    coh = [round(e * rr * rr, 3) for e, rr in zip(eul, rdl)]
    print(f"  {tag:28s} r={r:.4f} P/P={pp:.4f} rms/h_f={rms:.3f} | Eul P_d/P = {[round(x,3) for x in eul]} "
          f"| r_d = {[round(x,3) for x in rdl]} | coherent = {coh}")
    return dict(r_high=r, ratio_P_high=pp, rms=rms, eul=eul, r_delta=rdl, coherent=coh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["specialist", "shared", "resflow"], required=True)
    ap.add_argument("--test-seed", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--flow-b", default=None,
                    help="resflow pair only: override the downstream residual-flow checkpoint")
    ap.add_argument("--ckpt-b", default=None,
                    help="override the step-2 (64->128) checkpoint, e.g. a rollout-fine-tuned "
                         "copy of the shared model; the upstream stays the pair's default")
    args = ap.parse_args()

    if args.pair == "resflow":
        # two-stage at each level: frozen regression base + residual flow
        bA, _ = load_model("runs/R_reg_phys_3k/model_ema.pt")
        fA, _ = load_model("runs/RF_resflow/model_ema.pt")
        bB, _ = load_model("runs/F_reg_128_3k/model_ema.pt")
        fB, _ = load_model("runs/RF_resflow_128/model_ema.pt")
        sA = sB = 0.0
        if args.flow_b:
            fB, _ = load_model(args.flow_b)
        mA, mB = (bA, fA), (bB, fB)
    elif args.pair == "specialist":
        mA, _ = load_model("runs/J_flow_qj/model_ema.pt")
        mB, _ = load_model("runs/J_flow_128/model_ema.pt")
        sA = sB = 0.0
    else:
        mA, ckA = load_model("runs/M_qj_multi/model_ema.pt")
        mB = mA
        sA, sB = ckA["s_values"]
    if args.ckpt_b:
        mB, _ = load_model(args.ckpt_b)

    scA, teA = build_level(32, 64, args.test_seed)
    scB, teB = build_level(64, 128, args.test_seed)
    print(f"=== chain {args.pair}, test seed {args.test_seed} ===")

    # step 1: 32->64 with the true coarse field
    predA, x1A, _ = predict(mA, scA, teA, sA)
    metrics(scA, predA, x1A, "step1 32->64 (true coarse)")

    # step 2 direct: 64->128 with the TRUE 64 field (reference column)
    predB, x1B, _ = predict(mB, scB, teB, sB)
    res_direct = metrics(scB, predB, x1B, "step2 direct (true coarse)")

    # step 2 chained: replace the coarse input with step 1's OUTPUT
    teB_chain = [dict(teB[0])]
    teB_chain[0]["dis_c"] = (predA[0] * scA.hf).numpy().astype(np.float32)
    predC, _, _ = predict(mB, scB, teB_chain, sB)
    res_chain = metrics(scB, predC, x1B, "step2 CHAINED (generated coarse)")

    out = args.out or f"runs/chain_{args.pair}_s{args.test_seed}.json"
    with open(out, "w") as f:
        json.dump(dict(pair=args.pair, test_seed=args.test_seed,
                       direct=res_direct, chained=res_chain), f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
