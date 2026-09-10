#!/usr/bin/env python3
"""Figures for a trained octave-flow model: full-range spectra and real-space slices.

Complements the per-run `summary.png` that `octave_flow_toy.py` writes (which only shows
the octave band).  Nothing here is imported by the two main scripts; it only reads a
checkpoint and re-synthesises the same test box.

  fig_spectra_<tag>.png   P_theta(k) of the displacement divergence theta = -div Psi over
                          the whole k range for P Psi_c / baseline x0 / emulator /
                          generative / truth, plus P/P_true and r(k) vs truth.
  fig_slices_<tag>.png    the same five fields in real space through one slab: the
                          Lagrangian theta field, and the Eulerian CIC density.

The Eulerian row is a plain CIC deposit of the particles at q + Psi(q); it is only a
visualisation.  Quantitative Eulerian statistics (the CIC loss, density spectra) are
deliberately not part of the toy -- see the "deliberately left out" list in README.md.

Runs on the CPU by default so it can be used while a training job has the GPU.

  python runs/make_figures.py --ckpt runs/A_flow_phys/model_ema.pt --window cube --tag cube
  python runs/make_figures.py --ckpt runs/toy32_main/model_ema.pt --window sphere --tag sphere

Keep --nc/--nf/--rms-delta/--test-seeds/--window the same as the training run, otherwise
the test box (and the band split) will not be the one the model was trained for.
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0        # noqa: E402
import octave_flow_toy as oft      # noqa: E402


def cic(psi, Nf, L, offset):
    """Eulerian density from Lagrangian displacements: deposit q + Psi(q) with CIC."""
    h = L / Nf
    q = (np.arange(Nf) + offset) * h
    Q = np.stack(np.meshgrid(q, q, q, indexing="ij"))
    x = ((Q + psi) / h) % Nf
    i0 = np.floor(x).astype(np.int64)
    f = x - i0
    rho = np.zeros((Nf, Nf, Nf), dtype=np.float64)
    for dx in (0, 1):
        wx = (1 - f[0]) if dx == 0 else f[0]
        for dy in (0, 1):
            wy = (1 - f[1]) if dy == 0 else f[1]
            for dz in (0, 1):
                wz = (1 - f[2]) if dz == 0 else f[2]
                idx = np.unravel_index(((((i0[0] + dx) % Nf) * Nf + ((i0[1] + dy) % Nf)) * Nf
                                        + ((i0[2] + dz) % Nf)).ravel(), (Nf, Nf, Nf))
                np.add.at(rho, idx, (wx * wy * wz).ravel())
    assert abs(rho.sum() - Nf**3) < 1e-3 * Nf**3, "CIC lost mass"
    return rho


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="path to a model_ema.pt")
    ap.add_argument("--window", default="cube", choices=["cube", "sphere"])
    ap.add_argument("--tag", default=None, help="suffix for the file names (default: --window)")
    ap.add_argument("--nc", type=int, default=32)
    ap.add_argument("--nf", type=int, default=64)
    ap.add_argument("--box", type=float, default=100.0)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rms-delta", type=float, default=2.0)
    ap.add_argument("--n-index", type=float, default=-1.5)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--nsteps-sample", type=int, default=8)
    ap.add_argument("--dis", default=None, help="real-data pattern with {N} and {seed}; without it, synthetic 2LPT")
    ap.add_argument("--ic", default=None, help="real-data IC pattern with {N} and {seed}")
    ap.add_argument("--growth", type=float, default=1.0, help="D(z_out)/D(z_ic) for real ICs")
    ap.add_argument("--regression", action="store_true",
                    help="the checkpoint was trained with --regression: evaluate with one Euler step")
    ap.add_argument("--slab", type=int, nargs=2, default=(24, 32), help="slab of fine cells to show")
    ap.add_argument("--device", default="cpu", help="cpu by default: leaves the GPU to a training job")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    tag = args.tag or args.window
    Nc, Nf, L, off = args.nc, args.nf, args.box, args.offset
    dev = torch.device(args.device)
    torch.manual_seed(0)

    sc = oft.Scaffold(Nc, Nf, L, off, args.alpha, dev, window=args.window)
    if args.dis:
        train = oft.load_real(args.dis, args.ic, Nc, Nf, args.train_seeds)
        test = oft.load_real(args.dis, args.ic, Nc, Nf, args.test_seeds)
    else:
        train = oft.make_synthetic(Nc, Nf, L, off, args.train_seeds, args.rms_delta, args.n_index, False)
        test = oft.make_synthetic(Nc, Nf, L, off, args.test_seeds, args.rms_delta, args.n_index, False)
    sc.fit_linear_power([it["ic_f"] * args.growth for it in train])
    b = oft.Batcher(sc, test, np.random.default_rng(0), augment_on=False, growth=args.growth)
    x0, x1, Pc, D = b.make(test, eta="true")

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    base = int(ck["base"]) if isinstance(ck, dict) and "base" in ck else 24
    model = oft.UNet3D(cin=12, cout=3, base=base)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args.ckpt} (base={base}), window={args.window}, device={dev}")

    s = torch.zeros(1, device=dev)
    nst = 1 if args.regression else args.nsteps_sample
    meth = "euler" if args.regression else "heun"
    xe = oft.sample_flow(model, x0, Pc, D, s, nsteps=nst, method=meth)
    gen = torch.Generator().manual_seed(1)
    x0g, _, _, _ = b.make(test, eta="sample", gen=gen)
    xg = oft.sample_flow(model, x0g, Pc, D, s, nsteps=nst, method=meth)

    hf = sc.hf
    C, B0, T = r"coarse  $P\Psi_c$", r"baseline $x_0$", r"truth $\Psi_f$"
    F = {C: (Pc[0] * hf).cpu().numpy(), B0: (x0[0] * hf).cpu().numpy(),
         "emulator": (xe[0] * hf).cpu().numpy(), "generative": (xg[0] * hf).cpu().numpy(),
         T: (x1[0] * hf).cpu().numpy()}
    order = [C, B0, "emulator", "generative", T]
    col = {C: "0.55", B0: "C0", "emulator": "C2", "generative": "C3", T: "k"}
    gf = p0.Grid(Nf, L)

    # ---- spectra ------------------------------------------------------------
    sp = {lab: p0.spectra(gf, f, divergence=True) for lab, f in F.items()}
    Pt = sp[T]["Paa"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab in order:
        d = sp[lab]
        ax[0].loglog(d["k"], d["Paa"], color=col[lab], lw=2 if lab == T else 1.3,
                     ls="--" if lab == T else "-", label=lab)
    for lab in order:
        if lab == T:
            continue
        ax[1].semilogx(sp[T]["k"], sp[lab]["Paa"] / np.maximum(Pt, 1e-30), color=col[lab], lw=1.4, label=lab)
        c = p0.spectra(gf, F[lab], F[T], divergence=True)
        ax[2].semilogx(c["k"], c["r"], color=col[lab], lw=1.4, label=lab)
    for a in ax:
        a.axvline(sc.knyc, color="0.3", ls=":", lw=1)
        a.set_xlabel(r"$k$  [$h/$Mpc]")
    ax[0].set_ylabel(r"$P_\vartheta(k)$")
    ax[0].set_title(r"displacement divergence $\vartheta=-\nabla\cdot\Psi$")
    ax[1].axhline(1, color="0.6", lw=.8); ax[1].set_ylim(0, 1.35); ax[1].set_title(r"$P/P_{\rm true}$")
    ax[2].axhline(1, color="0.6", lw=.8); ax[2].set_ylim(-0.05, 1.05); ax[2].set_title(r"$r(k)$ vs truth")
    for a, loc in zip(ax, ["lower left", "lower left", "lower left"]):
        a.legend(fontsize=7, loc=loc)
    ax[0].text(sc.knyc * 1.05, ax[0].get_ylim()[1] * 0.3, r"$k_{\rm Ny,c}$", fontsize=8, color="0.3")
    fig.suptitle(f"{Nc}->{Nf}, {tag}, {os.path.dirname(args.ckpt)}, "
                 f"rms_delta={args.rms_delta}, test seed {args.test_seeds}", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(args.out, f"fig_spectra_{tag}.png")
    fig.savefig(p1, dpi=140); print("wrote", p1)

    # ---- real space ---------------------------------------------------------
    def theta(f):
        Ff = p0.rfftn(f.astype(np.float32))
        th = -1j * (gf.kx * Ff[0] + gf.ky * Ff[1] + gf.kz * Ff[2])
        return p0.irfftn(th[None], s=(Nf,) * 3)[0]

    sl = slice(*args.slab)
    TH = {lab: theta(F[lab]) for lab in order}
    RH = {lab: cic(F[lab], Nf, L, off) for lab in order}
    vt = np.percentile(np.abs(TH[T][sl].mean(0)), 99)
    fig, ax = plt.subplots(2, 5, figsize=(19, 7.6))
    for j, lab in enumerate(order):
        ax[0, j].imshow(TH[lab][sl].mean(0), cmap="RdBu_r", vmin=-vt, vmax=vt)
        ax[0, j].set_title(lab, fontsize=11)
        d = RH[lab][sl].sum(0) / RH[lab][sl].sum(0).mean()
        ax[1, j].imshow(np.log10(np.maximum(d, 1e-2)), cmap="magma", vmin=-0.8, vmax=1.1)
        for a in (ax[0, j], ax[1, j]):
            a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel(r"Lagrangian  $\vartheta=-\nabla\cdot\Psi$", fontsize=10)
    ax[1, 0].set_ylabel(r"Eulerian  $\log_{10}(1+\delta)$  (CIC)", fontsize=10)
    fig.suptitle(f"{Nc}->{Nf} in a {L:g} Mpc/h box, {tag} — cells {args.slab[0]}-{args.slab[1]} "
                 f"of the same test box", fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(args.out, f"fig_slices_{tag}.png")
    fig.savefig(p2, dpi=130); print("wrote", p2)


if __name__ == "__main__":
    main()
