#!/usr/bin/env python3
"""WHAT in the octave detail carries the density? (2026-09-17, follow-up of RUNLOG 2026-09-17 00:00: true
conditional samples have P_delta/P_true = 1.01 at r_delta 0.94 while the residual flow has 0.70 at 0.91.)

Training-free hybrid fields on the fine grid: coarse band (cube window W of the coarse level, inside the
fine grid) from one field + octave detail (1 - W) from another, deposited with the verified estimator and
compared with the real fine run T in k < k_Ny,f:
  T                      sanity (must be 1)
  lowT + 0               no detail at all
  lowT + high(E)         the MODEL's detail (emulator; G = generative) on the TRUE coarse band
  low(E) + high(T)       the TRUE detail on the MODEL's coarse band
  lowT + PR[high(T)]     the true detail PHASE-RANDOMISED: same per-mode amplitude, random phases
                         (a Gaussian-like detail with exactly the right power spectrum)
  lowT + MOD[high(T)]    the phase-randomised detail re-MODULATED so that its local power envelope
                         (sum_c d_c^2 smoothed with a Gaussian of sigma = h_c) equals the true detail's,
                         then re-projected on (1 - W) and rescaled to the true detail power: right power
                         spectrum AND right spatial distribution of the detail power, random phases
  lowT + high(S_j)       the detail of ANOTHER true conditional sample (different octave realisation)
  S_j                    the true conditional samples themselves (reference, RUNLOG 00:00)
Hypotheses: H1 PR reproduces the flow's deficit (the deficit = Gaussianity of the detail, not its power);
H2 MOD recovers most of it (what matters is WHERE the detail power sits, i.e. the conditional variance
field, not the exact phases); H3 low(E)+high(T) ~ 1 (the model's coarse band is not the problem).

  python runs/hybrid_detail_test.py --set 14 --model-run runs/RF_resflow_128_psc \
      --sim <sim root>/dmo-128-resample/set14        -> runs/hybrid_set14.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0                          # noqa: E402
from compute_spectra import Shells, density_coeff    # noqa: E402

L_KPC, L_MPC, OFF = 100000.0, 100.0, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=14)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--model-run", default="runs/RF_resflow_128_psc")
    ap.add_argument("--field-suffix", default="", help="e.g. _64to128 for multi_resflow_train dumps")
    ap.add_argument("--sim", default=None, help="resampled-octave sim root (optional)")
    ap.add_argument("--nsamples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    N, Nc = args.N, args.N // 2
    g, gc = p0.Grid(N, L_KPC), p0.Grid(Nc, L_KPC)
    W = p0.cube_window(g, gc.kny)
    T = np.load(f"data/psc/dmo-{N}/set{args.set}/PART_009/disp.npy").astype(np.float32)
    fld = {nm: np.load(f"{args.model_run}/field_{nm}{args.field_suffix}.npy").astype(np.float32)
           for nm in ("emulator", "generative", "base") if os.path.exists(f"{args.model_run}/field_{nm}{args.field_suffix}.npy")}
    rng = np.random.default_rng(args.seed)

    def split(f):
        F = p0.rfftn(f.astype(np.float32))
        lo = p0.irfftn(F * W, s=(N,) * 3).astype(np.float32)
        return lo, (f - lo).astype(np.float32)

    def project_high(d):
        F = p0.rfftn(d.astype(np.float32))
        return p0.irfftn(F * (1 - W), s=(N,) * 3).astype(np.float32)

    def phase_randomise(d):
        """same |F_k| per mode and component, phases of a real white field (Hermitian by construction);
        ONE phase field for the three components would keep their relative phases, so use three."""
        out = []
        for c in range(3):
            Fd = np.fft.rfftn(d[c].astype(np.float64))
            Wn = np.fft.rfftn(rng.standard_normal((N, N, N)))
            ph = np.where(np.abs(Wn) > 0, Wn / np.abs(Wn), 1.0)
            out.append(np.fft.irfftn(np.abs(Fd) * ph, s=(N, N, N)))
        return np.stack(out).astype(np.float32)

    def smooth(a, sigma):
        k2 = (g.kx ** 2 + g.ky ** 2 + g.kz ** 2)
        return np.fft.irfftn(np.fft.rfftn(a.astype(np.float64)) * np.exp(-0.5 * k2 * sigma ** 2), s=(N, N, N))

    def modulate(gd, d, sigma):
        envT = np.maximum(smooth((d ** 2).sum(0), sigma), 0); envG = np.maximum(smooth((gd ** 2).sum(0), sigma), 1e-30)
        m = project_high(gd * np.sqrt(envT / envG)[None].astype(np.float32))
        return m * np.sqrt((d ** 2).mean() / (m ** 2).mean())

    loT, hiT = split(T)
    sh = Shells(256); knyf = np.pi * N / L_MPC; knyc = np.pi * Nc / L_MPC
    def dep(f):
        return density_coeff(f.astype(np.float64) / 1000.0 + OFF * (L_MPC / N))[0]
    cref = dep(T)
    probes = (knyc, 0.75 * knyf, 0.9 * knyf)
    out = {"set": args.set, "N": N, "model_run": args.model_run, "probes_k_over_knyf": [p / knyf for p in probes]}

    def report(name, f):
        pw = sh.powers(dep(f), cref); k = pw["k"]; band = (k > 0) & (k < knyf)
        ratio = pw["P"][band] / np.maximum(pw["Pb"][band], 1e-30); rr = pw["r"][band]; kb = k[band]
        P = [float(ratio[np.argmin(np.abs(kb - q))]) for q in probes]; R = [float(rr[np.argmin(np.abs(kb - q))]) for q in probes]
        # Lagrangian octave-band r and power of the detail w.r.t. the true detail
        Fa, Fb = p0.rfftn(f.astype(np.float32)) * (1 - W), p0.rfftn(T) * (1 - W)
        s_ = p0.spectra_k(g, Fa, Fb, kmin=gc.kny)
        rl = float(np.average(s_["r"])) if np.isfinite(s_["r"]).all() else float("nan")
        pl = float(np.average(s_["Paa"] / np.maximum(s_["Pbb"], 1e-30)))
        out[name] = dict(P=P, r=R, k=kb.tolist(), Pratio=ratio.tolist(), rcurve=rr.tolist(), r_lag_detail=rl, P_lag_detail=pl)
        print(f"  {name:30s} P_delta/P {P[0]:.3f}/{P[1]:.3f}/{P[2]:.3f}  r_delta {R[0]:.3f}/{R[1]:.3f}/{R[2]:.3f}  | detail r_lag {rl:.3f} P_lag {pl:.3f}", flush=True)

    print(f"[set{args.set} {Nc}->{N}] hybrids vs the real {N} run; probes k_Ny,{Nc} / 0.75 / 0.9 k_Ny,{N}")
    report("T (sanity)", T)
    report("lowT + 0", loT)
    for nm, f in fld.items():
        lo, hi = split(f)
        report(f"{nm} (as is)", f)
        report(f"lowT + high({nm})", loT + hi)
        report(f"low({nm}) + high(T)", lo + hiT)
    for j in range(2):
        gd = phase_randomise(hiT)
        report(f"lowT + PR[high(T)] #{j}", loT + gd)
        for sig_cells in (1.0, 2.0):
            report(f"lowT + MOD[sigma={sig_cells:g} h_c] #{j}", loT + modulate(gd, hiT, sig_cells * L_KPC / Nc))
    if args.sim:
        for j in range(args.nsamples):
            p = f"{args.sim}/oct{j}/PART_000/disp.npy"
            if os.path.exists(p):
                S = np.load(p).astype(np.float32); loS, hiS = split(S)
                report(f"oct{j} (true cond. sample)", S)
                report(f"lowT + high(oct{j})", loT + hiS)
                report(f"low(oct{j}) + high(T)", loS + hiT)
    outp = args.out or f"runs/hybrid_set{args.set}.json"
    with open(outp, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
