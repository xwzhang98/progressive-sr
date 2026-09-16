#!/usr/bin/env python3
"""Strictly nested initial conditions from one complex Fourier mother field.

Why this exists (audit, RUNLOG 2026-09-15): MP-GenIC's default Nmesh = 2*Ngrid puts folded
super-Nyquist content into every level (cross-level rel-RMS 1.46e-1 on shared modes), and
even at Nmesh = Ngrid the levels agree only to 1.4-3.1e-2 (shell amplitudes match to 1e-4 —
measured transfer T = 1.0000 — so the residual is a phase-level boundary effect of the
mesh-dependent mode seeding). Strict nesting therefore needs an independent generator.

Construction:
  * ONE Hermitian mother field delta(k) on the top grid (default 128^3 rfft layout):
    unitary amplitudes |delta| = A(k) exactly (MP-GenIC's UnitaryAmplitude=1 convention),
    phases from a seeded white real field (Hermitian symmetry by construction).
  * A(k) is the shell spectrum measured from a GenIC template at Nmesh = Ngrid = top
    (same cosmology / power file / sigma8 pipeline; the shape is GenIC's own).
  * Level N: copy the mother modes with all |k_i| < pi/h_N into the N-grid rfft layout
    (the same index mapping as the prolongation in phase0_octaves), Nyquist planes zeroed —
    the cube-window convention used everywhere in this project. Shared bands are therefore
    IDENTICAL between levels by construction, to float precision.
  * Zel'dovich on the band-limited field, sampled at the N lattice (exact, no aliasing):
    Psi = i k delta / k^2, x = q + Psi, v = C_VEL * Psi with C_VEL measured from the GenIC
    template (v = 0.52776101 Psi, residual 2.5e-8 -> pure Zel'dovich at z_init = 99).
  * BigFile output with the Header attributes copied verbatim from the same-Ngrid GenIC
    template (masses, units, redshift untouched). Blocks: 1/Position (f8), 1/Velocity (f4),
    1/ID (uint64, 1-based, C order) — the same conventions hpc/convert_snapshot.py expects.

2LPT is deliberately NOT included: the mother field is Zel'dovich-only, which keeps the
truncation exact (2LPT products would re-introduce band coupling). The templates' 2LPT
correction at z=99 is ~1e-2 of the displacement; this is a documented physical difference
from the production pipeline, not an accident.

  python nested_ic.py --seed 5000 --levels 32 64 128 --out data/nestedic/s5000
"""
import argparse
import os
import sys

import numpy as np
import bigfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_octaves as p0  # noqa: E402

L = 100000.0
C_VEL = 0.52776101          # km/s per kpc/h at z_init = 99, measured from the GenIC template
TOP = 128
TMPL = "data/nestedic/tmpl_{N}/IC"


def mother_amplitude(top):
    """A(k) on the top rfft grid: unitary per-mode amplitude with GenIC's measured shell P."""
    import bigfile as bf
    f = bf.File(TMPL.format(N=top))
    ids = f["1/ID"][:]; o = np.argsort(ids)
    x = f["1/Position"][:][o].reshape(top, top, top, 3)
    q = (np.arange(top)) * L / top
    Q = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1)
    psi = np.moveaxis(((x - Q + L / 2) % L - L / 2), -1, 0).astype(np.float32)
    g = p0.Grid(top, L)
    F = p0.rfftn(psi)
    delta = -1j * (g.kx * F[0] + g.ky * F[1] + g.kz * F[2])
    P = g.shell_avg(np.abs(delta) ** 2)               # per-mode |delta|^2, shell-averaged
    A = np.interp(g.kmag.ravel(), g.kshell, np.sqrt(np.maximum(P, 0))).reshape(g.kmag.shape)
    A[g.kmag == 0] = 0.0
    return A, g


def mother_field(seed, top):
    """Hermitian delta(k) with |delta| = A(k) exactly and seeded random phases."""
    A, g = mother_amplitude(top)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((top, top, top)).astype(np.float64)
    W = np.fft.rfftn(w)
    ph = np.where(np.abs(W) > 0, W / np.abs(W), 1.0)   # unit-modulus, Hermitian by construction
    return (A * ph).astype(np.complex128), g


def level_from_mother(dm, gtop, N):
    """Copy the strictly-sub-Nyquist modes of the mother into the N-grid rfft layout."""
    gN = p0.Grid(N, L)
    # _coarse_slices(Nf, Nc): ax indexes the FINE (mother) array, c_ax the COARSE (level) one
    ax, c_ax, az = p0._coarse_slices(gtop.N, N)
    sub = dm[ax][:, ax][:, :, az].copy()
    dN = np.zeros((N, N, N // 2 + 1), dtype=np.complex128)
    dN[np.ix_(c_ax, c_ax, az)] = sub
    # zero the +/- Nyquist planes of the N grid (the cube-window convention)
    kny_idx = N // 2
    dN[kny_idx, :, :] = 0; dN[:, kny_idx, :] = 0; dN[:, :, kny_idx] = 0
    # scale: mother modes are per-mode amplitudes of the top FFT; the N-grid FFT of the same
    # continuum field carries a factor (N/top)^3 in this normalisation
    dN *= (N / gtop.N) ** 3
    return dN, gN


def zeldovich(dN, gN):
    k2 = np.maximum(gN.kmag ** 2, 1e-30)
    psi = np.stack([np.fft.irfftn(1j * k * dN / k2, s=(gN.N,) * 3)
                    for k in (gN.kx, gN.ky, gN.kz)])
    return psi.astype(np.float64)


def write_bigfile(path, psi, N):
    tmpl = bigfile.File(TMPL.format(N=N))
    q = (np.arange(N)) * L / N
    Q = np.stack(np.meshgrid(q, q, q, indexing="ij"))
    pos = np.mod(Q + psi, L).reshape(3, -1).T.astype(np.float64)
    vel = (C_VEL * psi).reshape(3, -1).T.astype(np.float32)
    ids = np.arange(1, N ** 3 + 1, dtype=np.uint64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = bigfile.File(path, create=True)
    h = f.create("Header")
    th = tmpl["Header"]
    for k in th.attrs.keys():
        h.attrs[k] = th.attrs[k]
    b = f.create("1/Position", dtype=("f8", (3,)), size=N ** 3, Nfile=1); b.write(0, pos)
    b = f.create("1/Velocity", dtype=("f4", (3,)), size=N ** 3, Nfile=1); b.write(0, vel)
    b = f.create("1/ID", dtype="u8", size=N ** 3, Nfile=1); b.write(0, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--check", action="store_true", help="acceptance checks after writing")
    args = ap.parse_args()

    dm, gtop = mother_field(args.seed, args.top)
    fields = {}
    for N in args.levels:
        dN, gN = level_from_mother(dm, gtop, N)
        psi = zeldovich(dN, gN)
        fields[N] = psi
        write_bigfile(os.path.join(args.out, f"N{N}", "IC"), psi, N)
        np.save(os.path.join(args.out, f"ic_dis_{N}.npy"), psi.astype(np.float32))
        print(f"level {N}: rms(Psi) = {psi.std():.3f} kpc/h  -> {args.out}/N{N}/IC")

    if args.check:
        print("acceptance checks (strict nesting, float precision expected):")
        Ls = sorted(args.levels)
        for a, b in zip(Ls[:-1], Ls[1:]):
            ga, gb = p0.Grid(a, L), p0.Grid(b, L)
            R = p0.restrict_spectral(fields[b].astype(np.float32), gb, ga,
                                     p0.cube_window(gb, ga.kny), 0.0)
            rel = np.sqrt(((R - fields[a]) ** 2).mean()) / fields[a].std()
            print(f"  |R Psi_{b} - Psi_{a}| / rms = {rel:.3e}")
        # shell spectra vs the GenIC template at the top level
        gt = p0.Grid(args.top, L)
        F = p0.rfftn(fields[args.top].astype(np.float32))
        d = -1j * (gt.kx * F[0] + gt.ky * F[1] + gt.kz * F[2])
        P = gt.shell_avg(np.abs(d) ** 2)
        # compare shells against the GenIC template's own spectrum (A^2 shell average)
        A, _ = mother_amplitude(args.top)
        Ptmpl = gt.shell_avg(A.astype(np.float32) ** 2)
        sel = (gt.shell_norm > 0) & (gt.kshell > 0) & (gt.kshell < 0.9 * gt.kny)
        ratio = P[sel] / np.maximum(Ptmpl[sel], 1e-30)
        print(f"  top-level shell P vs template: median ratio {np.median(ratio):.4f}, "
              f"max |ratio-1| = {np.abs(ratio-1).max():.2e}")
        # velocity consistency is by construction (v = C_VEL * Psi exactly)


if __name__ == "__main__":
    main()
