#!/usr/bin/env python3
"""§4 of the owner's course<->research notes (2026-09-16): is the autoregressive chain error a
martingale difference? Chain 32->64->128 with the PSC resflow pairs on the production split
(runs/R_reg_psc14 + RF_resflow_psc14, runs/F_reg_128_psc + RF_resflow_128_psc; test set 14),
emulator mode at both steps (true IC octaves), exactly as runs/eval_chain.py, then in the
64 band of the 128 grid (the CORRECTION band, the only band where the two steps' errors overlap;
the new octave of step 2 is orthogonal to step 1's error in k-space by construction) measure
per k-shell the regression slopes

   beta(k)  = <(R C - R D) . e1> / <|e1|^2>      pass-through of the input error e1 = A - T64
                                                (1 = carried through unchanged, >1 amplified,
                                                 <1 partly repaired)
   gamma(k) = <eps2 . e1> / <|e1|^2>            eps2 = (R C - R T128) - e1, the innovation of
                                                step 2 beyond passing e1 through
   kappa(k) = <eps2 . A> / <|A|^2>              innovation vs the step-1 OUTPUT (F_1-measurable)
   gamma_D(k) = <(R D - R T128) . T64>/<|T64|^2> the direct step's own error vs its own input

with A = step-1 output, D = direct 64->128 from the true 64, C = chained from A, T = truth, R =
cube restriction 128->64. Martingale-difference hypothesis E[eps_n | F_n] = 0 => gamma, kappa,
gamma_D ~ 0 at every k, and E|e_chain|^2 = E|e1|^2 + E|eps2|^2 (orthogonality; Doob then bounds
the running maximum by 4x). Also reports the octave-band (k > k_Ny,64) error power of C vs D and
dumps the 128 fields for runs/eval_vs_ref.py (vs native 256).

  python runs/chain_martingale.py --set 14 --out runs/chain_psc14        (CPU, ~20 min)
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

L, OFF, GR = 100000.0, 0.5, 76.7439
DIS, IC = "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy"


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = oft.UNet3D(cin=12, cout=3, base=int(ck.get("base", 24)))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def build_level(Nc, Nf, train, test):
    sc = oft.Scaffold(Nc, Nf, L, OFF, 1.0, torch.device("cpu"), window="cube")
    tr = oft.load_real(DIS, IC, Nc, Nf, train)
    te = oft.load_real(DIS, IC, Nc, Nf, [test])
    sc.fit_linear_power([it["ic_f"] * GR for it in tr])
    return sc, te


def predict(pair, sc, items):
    baseM, flowM = pair
    b = oft.Batcher(sc, items, np.random.default_rng(0), augment_on=False, growth=GR, octave_transverse=True)
    x0, x1, Pc, D = b.make(items, eta="true")
    with torch.no_grad():
        z = torch.zeros(1)
        xb = x0 + baseM(oft.net_input(x0, Pc, D), z, z)
        pred = oft.sample_flow(flowM, xb, Pc, D, z, nsteps=8)
    hf = sc.hf
    return (pred[0] * hf).numpy().astype(np.float32), (x1[0] * hf).numpy().astype(np.float32)


def shell_slope(g, num_field, den_field):
    """per-shell <a.b>/<|b|^2> summed over the 3 components (real-space regression slope per k-shell)."""
    Fa = p0.rfftn(num_field.astype(np.float32)); Fb = p0.rfftn(den_field.astype(np.float32))
    num = sum(g.shell_avg((Fa[c] * np.conj(Fb[c])).real) for c in range(3))
    den = sum(g.shell_avg(np.abs(Fb[c]) ** 2) for c in range(3))
    m = (g.shell_norm > 0) & (g.kshell > 0)
    return g.kshell[m], (num / np.maximum(den, 1e-30))[m], den[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=14)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(14)))
    ap.add_argument("--out", default="runs/chain_psc14")
    ap.add_argument("--models", nargs=4, default=["runs/R_reg_psc14", "runs/RF_resflow_psc14",
                                                   "runs/F_reg_128_psc", "runs/RF_resflow_128_psc"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    pA = (load_model(f"{args.models[0]}/model_ema.pt"), load_model(f"{args.models[1]}/model_ema.pt"))
    pB = (load_model(f"{args.models[2]}/model_ema.pt"), load_model(f"{args.models[3]}/model_ema.pt"))
    scA, teA = build_level(32, 64, args.train_seeds, args.set)
    scB, teB = build_level(64, 128, args.train_seeds, args.set)
    print(f"=== chain resflow (PSC split {args.train_seeds[0]}-{args.train_seeds[-1]} / {args.set}) ===", flush=True)
    A, T64 = predict(pA, scA, teA)                       # step 1 (kpc/h)
    print("step 1 done", flush=True)
    D, T128 = predict(pB, scB, teB)                       # step 2 direct
    print("step 2 direct done", flush=True)
    teC = [dict(teB[0])]; teC[0]["dis_c"] = A
    C, _ = predict(pB, scB, teC)                          # step 2 chained
    print("step 2 chained done", flush=True)
    for nm, f in (("step1", A), ("truth64", T64), ("direct128", D), ("chained128", C), ("truth128", T128)):
        np.save(os.path.join(args.out, f"field_{nm}.npy"), f)

    g64, g128 = p0.Grid(64, L), p0.Grid(128, L)
    W = p0.cube_window(g128, g64.kny)
    R = lambda f: p0.restrict_spectral(f, g128, g64, W, OFF)
    RC, RD, RT = R(C), R(D), R(T128)
    e1 = A - T64
    eps2 = (RC - RT) - e1
    eD = RD - RT
    out = {"set": args.set, "models": args.models}
    k, beta, Pe1 = shell_slope(g64, RC - RD, e1)
    _, gamma, _ = shell_slope(g64, eps2, e1)
    _, kappa, PA = shell_slope(g64, eps2, A)
    _, gammaD, _ = shell_slope(g64, eD, T64)
    _, rho, _ = shell_slope(g64, eps2, eD)                 # innovation vs the direct step's own error
    def Pk(f):
        F = p0.rfftn(f.astype(np.float32)); m = (g64.shell_norm > 0) & (g64.kshell > 0)
        return sum(g64.shell_avg(np.abs(F[c]) ** 2) for c in range(3))[m]
    P_e1, P_eps2, P_echain, P_eD = Pk(e1), Pk(eps2), Pk(RC - RT), Pk(eD)
    kk = k / g64.kny
    out["coarse_band"] = dict(k_over_kny64=kk.tolist(), beta=beta.tolist(), gamma=gamma.tolist(), kappa=kappa.tolist(),
                              gamma_direct=gammaD.tolist(), rho_eps2_vs_eD=rho.tolist(),
                              P_e1=P_e1.tolist(), P_eps2=P_eps2.tolist(), P_echain=P_eD.tolist(), P_eD=P_eD.tolist(),
                              P_echain_actual=P_echain.tolist())
    print("coarse band of the 128 grid (k/kNy,64):  beta  gamma  kappa  gamma_D  rho | P_e1  P_eps2  P_e1+P_eps2  P_echain  P_eD")
    for q in (0.25, 0.5, 0.75, 0.9, 0.97):
        j = int(np.argmin(np.abs(kk - q)))
        print(f"   {kk[j]:4.2f}: {beta[j]:6.3f} {gamma[j]:6.3f} {kappa[j]:7.4f} {gammaD[j]:7.4f} {rho[j]:6.3f} | "
              f"{P_e1[j]:.3e} {P_eps2[j]:.3e} {P_e1[j]+P_eps2[j]:.3e} {P_echain[j]:.3e} {P_eD[j]:.3e}", flush=True)
    band = kk < 0.97
    tot = lambda P: float(np.sum(P[band] * g64.shell_norm[(g64.shell_norm > 0) & (g64.kshell > 0)][band]))
    out["coarse_band_totals"] = dict(E_e1=tot(P_e1), E_eps2=tot(P_eps2), E_sum=tot(P_e1) + tot(P_eps2), E_echain=tot(P_echain), E_eD=tot(P_eD),
                                     cross_2e1eps2=tot(P_echain) - tot(P_e1) - tot(P_eps2))
    t = out["coarse_band_totals"]
    print(f"band totals: E|e1|^2={t['E_e1']:.4g}  E|eps2|^2={t['E_eps2']:.4g}  sum={t['E_sum']:.4g}  "
          f"E|e_chain|^2={t['E_echain']:.4g}  (2<e1,eps2>={t['cross_2e1eps2']:.4g}; direct step E|e_D|^2={t['E_eD']:.4g})", flush=True)
    # octave band of the 128 grid: error power of chained vs direct
    def Phi(f):
        F = p0.rfftn(f.astype(np.float32)) * (1 - W); m = (g128.shell_norm > 0) & (g128.kshell > g64.kny)
        return g128.kshell[m] / g64.kny, sum(g128.shell_avg(np.abs(F[c]) ** 2) for c in range(3))[m]
    kh, PhiD = Phi(D - T128); _, PhiC = Phi(C - T128); _, PhiT = Phi(T128)
    out["octave_band"] = dict(k_over_kny64=kh.tolist(), err_direct_over_P=(PhiD / PhiT).tolist(), err_chain_over_P=(PhiC / PhiT).tolist())
    for q in (1.1, 1.5, 1.9):
        j = int(np.argmin(np.abs(kh - q)))
        print(f"octave band k/kNy,64={kh[j]:4.2f}: err power / P_true  direct {PhiD[j]/PhiT[j]:.3f}  chained {PhiC[j]/PhiT[j]:.3f}", flush=True)
    with open(os.path.join(args.out, "chain_martingale.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}/chain_martingale.json")


if __name__ == "__main__":
    main()
