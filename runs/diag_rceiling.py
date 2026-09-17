#!/usr/bin/env python3
"""Where does the r_delta ceiling of the residual flow sit? (owner 2026-09-16, option 3 after the
Phase B verdict: both residual flows end at r_delta = 0.92 at 0.9 k_Ny,64 against the converged
native 128, while the native 64 run has 0.974.)

For the test-box fields dumped by runs/resflow_train.py (truth = native c^3 run, base, emulator,
generative; box units kpc/h, PSC offset 0.5) this script measures, against the native 2c reference:
  1. the full r_delta(k) and P/P(k) curves (verified estimator: interlaced CIC on the common 256^3
     mesh, complete shells) up to k_Ny,c, not just the three probes;
  2. the same for SUBSET deposits: only particles inside / outside the multi-stream mask
     (coarse det(I + D) < 0 of the coarse run, upsampled x2 to the c grid and x4 to the 2c grid --
     the same Lagrangian mask for every field, eval_eulerian T4 convention), i.e. the density field
     sourced by the multi-stream Lagrangian volume vs by the single-stream volume;
  3. Lagrangian displacement r(k) per octave region split by the same mask is not a Fourier
     quantity, so instead: rms displacement error in/out of the mask and the negative-parity
     fraction and quantiles of J = det(I + dPsi/dq) on the c grid for every field (eval_eulerian
     conventions), plus the delta > 100 mass fraction from the fine-grid CIC.
  python runs/diag_rceiling.py runs/RF_resflow_psc --set 8 [--nc 64]
Writes <run>/diag_rceiling.json and <run>/diag_rceiling.png.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import phase0_octaves as p0                          # noqa: E402
import eulerian_metric as em                         # noqa: E402
from compute_spectra import Shells, density_coeff    # noqa: E402  (verified estimator, reused)

L_MPC, L_KPC = 100.0, 100000.0
DATASETS = {"psc": ("data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", 0.5),
            "selfsim": ("data/selfsim/s{seed}/dis_{N}.npy", 0.0)}
QUANTS = (1, 5, 25, 50, 75, 95, 99)


def deposit(psi_kpc, n, offset, mask=None, mesh=256):
    """density coefficients on the common mesh; particle i at (i + offset) h; mask selects particles
    (masked-out particles are dropped, so the subset field's mean is the subset's mass)."""
    f = psi_kpc.astype(np.float64) / 1000.0 + offset * (L_MPC / n)
    if mask is None:
        return density_coeff(f, mesh=mesh)[0]
    # density_coeff deposits the whole cube; emulate a subset by pushing the dropped particles far
    # outside the box is not possible (periodic), so deposit twice: all minus complement is linear.
    # Simpler and exact: deposit the subset directly with the same interlaced-CIC code path.
    return subset_density_coeff(f, mask, mesh)


def subset_density_coeff(f, mask, mesh):
    """Same estimator as compute_spectra.density_coeff restricted to particles with mask=True
    (two half-cell interlaced CIC grids, window deconvolved, normalised by the FULL particle count
    so that subset fields add up to the full field)."""
    from scipy import fft
    n = f.shape[-1]
    q = np.arange(n, dtype=np.float64) * (L_MPC / n)
    pos = f.reshape(3, -1).copy()
    pos[0] += np.broadcast_to(q[:, None, None], (n, n, n)).ravel()
    pos[1] += np.broadcast_to(q[None, :, None], (n, n, n)).ravel()
    pos[2] += np.broadcast_to(q[None, None, :], (n, n, n)).ravel()
    pos = pos[:, mask.ravel()]
    u = (pos % L_MPC) * (mesh / L_MPC)
    m = fft.fftfreq(mesh).astype(np.float32) * mesh
    mz = fft.rfftfreq(mesh).astype(np.float32) * mesh
    modes = (m[:, None, None], m[None, :, None], mz[None, None, :])
    accum = None
    for shift in (0., .5):
        shifted = u - shift
        lower = np.floor(shifted).astype(np.int32)
        frac = shifted - lower
        rho = np.zeros((mesh, mesh, mesh), dtype=np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = (((lower[0] + dx) % mesh) * mesh + (lower[1] + dy) % mesh) * mesh + (lower[2] + dz) % mesh
                    w = (frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1]) * (frac[2] if dz else 1 - frac[2])
                    rho += np.bincount(idx, weights=w, minlength=mesh ** 3).reshape(rho.shape)
        F = fft.rfftn(rho.astype(np.float32), workers=2) / n ** 3
        F[0, 0, 0] = 0.
        if shift:
            F *= np.exp((-1j * np.pi / mesh) * sum(modes)).astype(np.complex64)
            accum += F
        else:
            accum = F
    accum *= .5
    for k in modes:
        accum /= np.sinc(k / mesh) ** 2
    return accum


def up(mask, factor):
    m = mask
    for ax in (0, 1, 2):
        m = np.repeat(m, factor, axis=ax)
    return m


def curves(sh, c_model, c_ref, kmax):
    pw = sh.powers(c_model, c_ref)
    k = pw["k"]; band = (k > 0) & (k < kmax)
    return dict(k=k[band].tolist(), Pratio=(pw["P"][band] / np.maximum(pw["Pb"][band], 1e-30)).tolist(),
                r=pw["r"][band].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", type=int, required=True)
    ap.add_argument("--nc", type=int, default=64)
    ap.add_argument("--data", default="psc", choices=sorted(DATASETS))
    ap.add_argument("--fields", nargs="+", default=["truth", "base", "emulator", "generative"])
    args = ap.parse_args()
    pat, off = DATASETS[args.data]
    nc, nf, ncc = args.nc, 2 * args.nc, args.nc // 2
    kny = np.pi * nc / L_MPC
    sh = Shells(256)
    ref_field = np.load(pat.replace("{N}", str(nf)).replace("{seed}", str(args.set)))
    coarse = np.load(pat.replace("{N}", str(ncc)).replace("{seed}", str(args.set)))
    # multi-stream mask of the COARSE run (what the model is conditioned on), coarse grid -> c grid
    ms_c = p0.coarse_invariants(coarse, p0.Grid(ncc, L_KPC))["detJ"] < 0
    ms = up(ms_c, 2); ms_ref = up(ms_c, 4)
    out = {"set": args.set, "nc": nc, "multistream_fraction": float(ms.mean()), "offset": off}
    print(f"[{args.run} set{args.set}] multi-stream fraction (coarse {ncc}^3 run, upsampled) = {ms.mean():.3f}")
    ref = {"all": deposit(ref_field, nf, off), "ms": deposit(ref_field, nf, off, ms_ref), "ss": deposit(ref_field, nf, off, ~ms_ref)}
    fields = {}
    for nm in args.fields:
        path = os.path.join(args.run, f"field_{nm}.npy")
        if os.path.exists(path):
            fields[nm] = np.load(path)
    tru = fields.get("truth")
    g = p0.Grid(nc, L_KPC)
    for nm, f in fields.items():
        res = {}
        for sub, msk in (("all", None), ("ms", ms), ("ss", ~ms)):
            res[sub] = curves(sh, deposit(f, nc, off, msk), ref[sub], kny)
        # Lagrangian error split, J statistics, delta>100 mass fraction (fine-grid CIC, eval_eulerian conventions)
        hf = L_KPC / nc
        if tru is not None and nm != "truth":
            e = (f - tru) / hf
            res["rms_err_over_h"] = dict(all=float(np.sqrt((e ** 2).mean())), ms=float(np.sqrt((e[:, ms] ** 2).mean())),
                                         ss=float(np.sqrt((e[:, ~ms] ** 2).mean())))
        fh = torch.from_numpy((f / hf).astype(np.float32))[None]
        J, _ = em.jacobian_and_adjugate(fh)
        Jn = J[0].numpy()
        res["J"] = dict(neg_frac=float((Jn < 0).mean()), neg_frac_ms=float((Jn[ms] < 0).mean()), neg_frac_ss=float((Jn[~ms] < 0).mean()),
                        quantiles={str(q): float(np.percentile(Jn, q)) for q in QUANTS})
        pos = em.eulerian_positions(fh, off)
        d = em.cic_deposit(torch.ones((1, 1, nc, nc, nc)), pos, nc)[0, 0].numpy()
        res["mass_d100"] = float(d[d > 100].sum() / d.sum()); res["mass_d10"] = float(d[d > 10].sum() / d.sum())
        out[nm] = res
        ra = np.asarray(res["all"]["r"]); ka = np.asarray(res["all"]["k"])
        def at(sub, q, key="r"):
            kk = np.asarray(res[sub]["k"]); v = np.asarray(res[sub][key]); return float(v[np.argmin(np.abs(kk - q * kny))])
        print(f"  {nm:10s} r@0.5/0.75/0.9 kNy: all {at('all',.5):.3f}/{at('all',.75):.3f}/{at('all',.9):.3f} | "
              f"MS {at('ms',.5):.3f}/{at('ms',.75):.3f}/{at('ms',.9):.3f} | SS {at('ss',.5):.3f}/{at('ss',.75):.3f}/{at('ss',.9):.3f} || "
              f"P/P@0.9: all {at('all',.9,'Pratio'):.3f} MS {at('ms',.9,'Pratio'):.3f} SS {at('ss',.9,'Pratio'):.3f} | "
              f"J<0 {res['J']['neg_frac']:.3f} (MS {res['J']['neg_frac_ms']:.3f}/SS {res['J']['neg_frac_ss']:.3f}) | m(d>100) {res['mass_d100']:.3f}"
              + (f" | rms err all/MS/SS {res['rms_err_over_h']['all']:.3f}/{res['rms_err_over_h']['ms']:.3f}/{res['rms_err_over_h']['ss']:.3f}" if "rms_err_over_h" in res else ""),
              flush=True)
    with open(os.path.join(args.run, "diag_rceiling.json"), "w") as fh_:
        json.dump(out, fh_, indent=1)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, sub in enumerate(("all", "ms", "ss")):
        for nm in fields:
            k = np.asarray(out[nm][sub]["k"]) / kny
            ax[0, j].plot(k, out[nm][sub]["r"], label=nm); ax[1, j].plot(k, out[nm][sub]["Pratio"], label=nm)
        ax[0, j].set_title(f"r_delta(k) vs native {nf}, {sub}"); ax[1, j].set_title(f"P/P vs native {nf}, {sub}")
        ax[0, j].set_ylim(0.5, 1.02); ax[1, j].axhline(1, color="0.6", lw=0.8); ax[1, j].set_ylim(0.5, 1.6)
        for a in ax[:, j]:
            a.set_xlabel("k / k_Ny,c"); a.legend(fontsize=7)
    fig.suptitle(f"{os.path.basename(args.run.rstrip('/'))} set{args.set}: multi-stream fraction {ms.mean():.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(args.run, "diag_rceiling.png"), dpi=120)
    print(f"wrote {args.run}/diag_rceiling.json and .png")


if __name__ == "__main__":
    main()
