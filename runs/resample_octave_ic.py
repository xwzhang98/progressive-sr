#!/usr/bin/env python3
"""§1 of the owner's course<->research notes (2026-09-16): TRUE conditional samples of the fine
run given the coarse band, from the simulator itself.

Take the real 128^3 IC of one set, keep every mode inside the 64-cube (|k_i| < k_Ny,64: the
F_n-measurable part, sigma-algebra of the coarse modes), and REPLACE the octave (the 128-cube minus
the 64-cube, Nyquist planes zeroed, the cube-window convention of the project) by fresh modes with
the SAME per-mode amplitude A(k) (GenIC's UnitaryAmplitude = 1: |delta_k| = A(k) exactly; A is
measured shell by shell from the real IC) and new random phases. Zel'dovich at z = 99, particles at
q = (i + 1/2) h (the series' offset 0.5), IDs 1-based C order, header copied from the real IC, so
hpc/convert_snapshot.py reads the outputs with the validated convention.
Controls: `ctrl_exact` = the real displacement written through this writer (pipeline test: the run
must reproduce the real 128 run), `ctrl_zel` = the real modes re-derived through delta -> Zel'dovich
(tests the Zel/2LPT-and-Nyquist-plane difference on their own).
After the MP-Gadget runs (hpc/slurm_resample128.sh) the z=0 fields give, per k-shell,
Var(fine | F_n) (spread of the samples), the empirical conditional mean (their average) to compare
with the learned regression base, and the density P_delta/P_true and r_delta of each TRUE conditional
sample against the real run -- the reference point for the flow's generative-mode deficit.

  python runs/resample_octave_ic.py --set 14 --nsamples 8 --out <sim root>/dmo-128-resample/set14
"""
import argparse
import json
import os
import sys

import numpy as np
import bigfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0  # noqa: E402

L = 100000.0
RAW = "/hildafs/home/xzhangn/xzhangn/sim_output/dmo-100MPC/int_redshift_same_cosmology"
PSC = "data/psc"


def write_bigfile(path, psi, N, offset, header_src):
    q = (np.arange(N) + offset) * L / N
    Q = np.stack(np.meshgrid(q, q, q, indexing="ij"))
    pos = np.mod(Q + psi, L).reshape(3, -1).T.astype(np.float64)
    vel = (C_VEL * psi).reshape(3, -1).T.astype(np.float32)
    ids = np.arange(1, N ** 3 + 1, dtype=np.uint64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = bigfile.File(path, create=True)
    h = f.create("Header")
    th = bigfile.File(header_src)["Header"]
    for k in th.attrs.keys():
        h.attrs[k] = th.attrs[k]
    b = f.create("1/Position", dtype=("f8", (3,)), size=N ** 3, Nfile=1); b.write(0, pos)
    b = f.create("1/Velocity", dtype=("f4", (3,)), size=N ** 3, Nfile=1); b.write(0, vel)
    b = f.create("1/ID", dtype="u8", size=N ** 3, Nfile=1); b.write(0, ids)


def main():
    global C_VEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=14)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    N, Nc = args.N, args.N // 2
    psi = np.load(f"{PSC}/dmo-{N}/set{args.set}/IC/disp.npy").astype(np.float64)       # kpc/h, offset 0.5 convention
    # velocity convention measured on the same set's 64 IC (disp AND vel converted there)
    d64 = np.load(f"{PSC}/dmo-64/set{args.set}/IC/disp.npy"); v64 = np.load(f"{PSC}/dmo-64/set{args.set}/IC/vel.npy")
    ratio = (v64 * d64).sum() / (d64 * d64).sum(); resid = np.sqrt(((v64 - ratio * d64) ** 2).mean()) / v64.std()
    C_VEL = float(ratio)
    print(f"velocity convention from the 64 IC: v = {C_VEL:.8f} * Psi, residual {resid:.2e} (pure Zel'dovich if << 1)")
    g, gc = p0.Grid(N, L), p0.Grid(Nc, L)
    F = p0.rfftn(psi.astype(np.float32)).astype(np.complex128)
    delta = -1j * (g.kx * F[0] + g.ky * F[1] + g.kz * F[2])
    Wc = p0.cube_window(g, gc.kny)                # 1 inside the 64 cube (its Nyquist planes zeroed)
    Wf = p0.cube_window(g, g.kny)                 # 1 inside the 128 cube (its Nyquist planes zeroed)
    octave = (Wf > 0) & (Wc == 0)
    # per-mode amplitude in the octave: check GenIC's unitary convention, then take the shell mean
    amp = np.abs(delta)
    Ps = g.shell_avg(amp ** 2); A = np.interp(g.kmag.ravel(), g.kshell, np.sqrt(np.maximum(Ps, 0))).reshape(g.kmag.shape)
    spread = np.std(amp[octave] / np.maximum(A[octave], 1e-30))
    print(f"octave: {int(octave.sum())} modes, |delta|/A(k) spread over modes = {spread:.3f} (0 = unitary amplitude)")
    k2 = np.maximum(g.kmag ** 2, 1e-30)
    def zel(dk):
        return np.stack([np.fft.irfftn(1j * k * dk / k2, s=(N,) * 3) for k in (g.kx, g.ky, g.kz)])
    hdr = f"{RAW}/dmo-{N}/set{args.set}/output/IC"
    manifest = {"set": args.set, "N": N, "C_VEL": C_VEL, "offset": args.offset, "octave_modes": int(octave.sum()),
                "unitary_spread": float(spread), "samples": {}}
    # controls
    write_bigfile(f"{args.out}/ctrl_exact/IC", psi, N, args.offset, hdr); manifest["samples"]["ctrl_exact"] = "real Psi through this writer"
    psi_z = zel(delta * Wf)
    print(f"ctrl_zel: |Psi_zel - Psi_real| / rms = {np.sqrt(((psi_z - psi) ** 2).mean()) / psi.std():.3e} (2LPT part + Nyquist planes)")
    write_bigfile(f"{args.out}/ctrl_zel/IC", psi_z, N, args.offset, hdr); manifest["samples"]["ctrl_zel"] = "real modes, Zel'dovich, Nyquist planes zeroed"
    # resampled octaves
    for j in range(args.nsamples):
        rng = np.random.default_rng(args.seed0 + j)
        Wn = np.fft.rfftn(rng.standard_normal((N, N, N)))
        ph = np.where(np.abs(Wn) > 0, Wn / np.abs(Wn), 1.0)
        dj = delta * Wc + (A * ph) * octave
        psi_j = zel(dj)
        Rj = p0.restrict_spectral(psi_j.astype(np.float32), g, gc, Wc, 0.0)
        Rz = p0.restrict_spectral(psi_z.astype(np.float32), g, gc, Wc, 0.0)
        same = np.sqrt(((Rj - Rz) ** 2).mean()) / Rz.std()
        write_bigfile(f"{args.out}/oct{j}/IC", psi_j, N, args.offset, hdr)
        manifest["samples"][f"oct{j}"] = dict(seed=args.seed0 + j, coarse_band_identity=float(same), rms_psi=float(psi_j.std()))
        print(f"oct{j}: rms Psi = {psi_j.std():.3f} kpc/h (real {psi.std():.3f}); coarse band identical to ctrl_zel to {same:.1e}", flush=True)
    with open(f"{args.out}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {args.nsamples} resampled + 2 control ICs under {args.out}")


if __name__ == "__main__":
    main()
