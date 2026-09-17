#!/usr/bin/env python3
"""§1 analysis: Var(fine | F_n) from TRUE conditional samples (runs/resample_octave_ic.py + MP-Gadget),
and what the true conditional says about the flow's generative mode.

Inputs: the z=0 128^3 displacement fields of the resampled-octave runs (converted with the validated
convention into <sim>/<sample>/PART_000/disp.npy), the real 128 run of the same set, and the test-box
field dumps of a trained 64->128 model (runs/RF_resflow_128_psc: truth/base/emulator/generative).
Per k-shell on the 128 grid (complete shells, cube-window bands marked at k_Ny,64):
  * P_real(k); conditional mean M = <sample>; Var(fine|F_n)(k) = <|sample - M|^2> (n/(n-1));
    unpredictable fraction Var/P_real; r(M, real), r(M, learned base), r(base, real);
    r(sample, real) and r(sample_i, sample_j) (the r a TRUE conditional sample has -- the reference
    for the flow's generative r), P_sample/P_real;
  * controls: ctrl_exact (pipeline reproducibility) and ctrl_zel (Zel'dovich/Nyquist-plane effect).
Eulerian (verified estimator, common 256^3 mesh, offset 0.5), band k < k_Ny,128 vs the real 128 run:
P_delta/P_true and r_delta at (k_Ny,64, 0.75, 0.9 k_Ny,128) for every true sample (mean +- spread), the
controls, and the model's emulator / generative fields -- the theorem "a sampler from the true conditional
has P_delta/P_true = 1 in expectation at r < 1" tested against the simulator's own conditional samples.

  python runs/condvar_analysis.py --set 14 --sim <sim root>/dmo-128-resample/set14 --model-run runs/RF_resflow_128_psc
Writes runs/condvar_set<S>.json and runs/condvar_set<S>.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0                          # noqa: E402
from compute_spectra import Shells, density_coeff    # noqa: E402

L_KPC, L_MPC = 100000.0, 100.0
PSC = "data/psc"


def deposit(psi_kpc, n, offset=0.5):
    return density_coeff(psi_kpc.astype(np.float64) / 1000.0 + offset * (L_MPC / n))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=14)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--samples", nargs="+", default=[f"oct{j}" for j in range(8)])
    ap.add_argument("--controls", nargs="+", default=["ctrl_exact", "ctrl_zel"])
    ap.add_argument("--model-run", default="runs/RF_resflow_128_psc")
    ap.add_argument("--snap", default="PART_000")
    args = ap.parse_args()
    N = args.N; Nc = N // 2
    g = p0.Grid(N, L_KPC); gc = p0.Grid(Nc, L_KPC)
    real = np.load(f"{PSC}/dmo-{N}/set{args.set}/{args.snap.replace('PART_000', 'PART_009')}/disp.npy").astype(np.float32)
    samples = {s: np.load(f"{args.sim}/{s}/{args.snap}/disp.npy").astype(np.float32) for s in args.samples}
    controls = {s: np.load(f"{args.sim}/{s}/{args.snap}/disp.npy").astype(np.float32) for s in args.controls}
    model = {}
    for nm in ("truth", "base", "emulator", "generative"):
        p = f"{args.model_run}/field_{nm}.npy"
        if os.path.exists(p):
            model[nm] = np.load(p).astype(np.float32)
    if "truth" in model:
        assert np.allclose(model["truth"], real, atol=1e-2), "model test box is not this set"
    msk = (g.shell_norm > 0) & (g.kshell > 0)
    kk = g.kshell[msk] / gc.kny                      # k / k_Ny,64 (octave band = 1..2)
    F = {}
    def ft(f):
        return p0.rfftn(f.astype(np.float32))
    def P(Fa):
        return sum(g.shell_avg(np.abs(Fa[c]) ** 2) for c in range(3))[msk]
    def X(Fa, Fb):
        return sum(g.shell_avg((Fa[c] * np.conj(Fb[c])).real) for c in range(3))[msk]
    def r(Fa, Fb):
        return X(Fa, Fb) / np.sqrt(np.maximum(P(Fa) * P(Fb), 1e-30))
    Fr = ft(real); Pr = P(Fr)
    Fs = [ft(f) for f in samples.values()]
    n = len(Fs)
    FM = sum(Fs) / n
    PM = P(FM)
    Var = sum(P(Fj - FM) for Fj in Fs) / (n - 1)          # unbiased conditional variance per shell
    out = {"set": args.set, "N": N, "n_samples": n, "k_over_kny64": kk.tolist(),
           "P_real": Pr.tolist(), "P_condmean": PM.tolist(), "Var_cond": Var.tolist(),
           "unpredictable_fraction": (Var / Pr).tolist(),
           "r_condmean_real": r(FM, Fr).tolist(),
           "r_sample_real_mean": (sum(r(Fj, Fr) for Fj in Fs) / n).tolist(),
           "P_sample_over_real_mean": (sum(P(Fj) for Fj in Fs) / n / Pr).tolist(),
           "r_sample_sample_mean": (sum(r(Fs[i], Fs[j]) for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)).tolist()}
    for s, f in controls.items():
        out[f"r_{s}_real"] = r(ft(f), Fr).tolist(); out[f"P_{s}_over_real"] = (P(ft(f)) / Pr).tolist()
    if "base" in model:
        Fb = ft(model["base"]); out["r_condmean_base"] = r(FM, Fb).tolist(); out["r_base_real"] = r(Fb, Fr).tolist()
        out["P_base_over_condmean"] = (P(Fb) / np.maximum(PM, 1e-30)).tolist()
    for nm in ("emulator", "generative"):
        if nm in model:
            out[f"r_{nm}_real"] = r(ft(model[nm]), Fr).tolist()
    # ---- printed Lagrangian table --------------------------------------------------------------
    def at(key, q):
        v = np.asarray(out[key]); return float(v[np.argmin(np.abs(kk - q))])
    qs = (0.5, 0.9, 1.1, 1.25, 1.5, 1.75, 1.95)
    print(f"[set{args.set}] {n} true conditional samples; Lagrangian, per shell (k/kNy,64):")
    print("   " + "".join(f"{q:>8.2f}" for q in qs))
    for key in ("unpredictable_fraction", "r_condmean_real", "r_sample_real_mean", "r_sample_sample_mean", "P_sample_over_real_mean",
                "r_base_real", "r_condmean_base", "P_base_over_condmean", "r_emulator_real", "r_generative_real",
                "r_ctrl_exact_real", "r_ctrl_zel_real"):
        if key in out:
            print(f"   {key:26s}" + "".join(f"{at(key, q):8.3f}" for q in qs))
    # ---- Eulerian: every field vs the real 128 run ---------------------------------------------
    sh = Shells(256); kny128 = np.pi * N / L_MPC; kny64 = np.pi * Nc / L_MPC
    probes = {"kNy64": kny64, "075kNy128": 0.75 * kny128, "09kNy128": 0.9 * kny128}
    cref = deposit(real, N)
    def eul(f):
        pw = sh.powers(deposit(f, N), cref); k = pw["k"]; band = (k > 0) & (k < kny128)
        ratio = pw["P"][band] / np.maximum(pw["Pb"][band], 1e-30); rr = pw["r"][band]; kb = k[band]
        res = {"k": kb.tolist(), "Pratio": ratio.tolist(), "r": rr.tolist()}
        for nm, q in probes.items():
            j = int(np.argmin(np.abs(kb - q))); res[f"P_{nm}"] = float(ratio[j]); res[f"r_{nm}"] = float(rr[j])
        return res
    E = {}
    for s, f in samples.items():
        E[s] = eul(f); print(f"   eul {s:10s} P/P {E[s]['P_kNy64']:.3f}/{E[s]['P_075kNy128']:.3f}/{E[s]['P_09kNy128']:.3f}  r {E[s]['r_kNy64']:.3f}/{E[s]['r_075kNy128']:.3f}/{E[s]['r_09kNy128']:.3f}", flush=True)
    for s, f in controls.items():
        E[s] = eul(f); print(f"   eul {s:10s} P/P {E[s]['P_kNy64']:.3f}/{E[s]['P_075kNy128']:.3f}/{E[s]['P_09kNy128']:.3f}  r {E[s]['r_kNy64']:.3f}/{E[s]['r_075kNy128']:.3f}/{E[s]['r_09kNy128']:.3f}", flush=True)
    for nm in ("base", "emulator", "generative"):
        if nm in model:
            E[f"model_{nm}"] = eul(model[nm]); e = E[f"model_{nm}"]
            print(f"   eul model_{nm:10s} P/P {e['P_kNy64']:.3f}/{e['P_075kNy128']:.3f}/{e['P_09kNy128']:.3f}  r {e['r_kNy64']:.3f}/{e['r_075kNy128']:.3f}/{e['r_09kNy128']:.3f}", flush=True)
    summ = {}
    for key in ("P_kNy64", "P_075kNy128", "P_09kNy128", "r_kNy64", "r_075kNy128", "r_09kNy128"):
        v = np.array([E[s][key] for s in samples]); summ[key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
    print("   TRUE conditional samples, mean +- std over samples: " + "  ".join(f"{k} {v['mean']:.3f}+-{v['std']:.3f}" for k, v in summ.items()))
    out["eulerian"] = E; out["eulerian_true_samples_summary"] = summ
    with open(f"runs/condvar_set{args.set}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    # ---- figure ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(kk, out["unpredictable_fraction"], label="Var(fine|F_n)/P_real"); ax[0].plot(kk, 1 - np.asarray(out["r_condmean_real"]) ** 2, "--", label="1 - r^2(cond. mean, real)")
    if "r_base_real" in out:
        ax[0].plot(kk, 1 - np.asarray(out["r_base_real"]) ** 2, ":", label="1 - r^2(learned base, real)")
    ax[0].axvline(1, color="0.7"); ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("k / k_Ny,64"); ax[0].legend(fontsize=7); ax[0].set_title("unpredictable fraction per shell")
    for key, lab in (("r_sample_real_mean", "true sample vs real"), ("r_generative_real", "flow generative vs real"), ("r_emulator_real", "flow emulator vs real"), ("r_ctrl_exact_real", "ctrl_exact vs real"), ("r_ctrl_zel_real", "ctrl_zel vs real")):
        if key in out:
            ax[1].plot(kk, out[key], label=lab)
    ax[1].axvline(1, color="0.7"); ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("k / k_Ny,64"); ax[1].legend(fontsize=7); ax[1].set_title("Lagrangian r(k) vs the real run")
    for s in list(samples)[:8]:
        ax[2].plot(np.asarray(E[s]["k"]) / kny128, E[s]["Pratio"], color="0.6", lw=0.7)
    for nm, c in (("model_emulator", "C1"), ("model_generative", "C2"), ("model_base", "C3"), ("ctrl_exact", "k")):
        if nm in E:
            ax[2].plot(np.asarray(E[nm]["k"]) / kny128, E[nm]["Pratio"], color=c, label=nm)
    ax[2].axhline(1, color="0.7"); ax[2].set_ylim(0.4, 1.6); ax[2].set_xlabel("k / k_Ny,128"); ax[2].legend(fontsize=7); ax[2].set_title("P_delta / P_delta,real (grey: true conditional samples)")
    fig.tight_layout(); fig.savefig(f"runs/condvar_set{args.set}.png", dpi=120)
    print(f"wrote runs/condvar_set{args.set}.json and .png")


if __name__ == "__main__":
    main()
