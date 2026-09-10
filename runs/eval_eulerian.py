#!/usr/bin/env python3
"""Eulerian evaluation of a trained octave-flow run (Stage 7, CPU only).

For each run directory: rebuild the test box from the args stored in results.json, produce
the emulator (8 Heun; 1 Euler for --regression checkpoints), 1-step and generative
predictions plus baseline x0, coarse P Psi_c and truth, and write <run>/eulerian.json and
<run>/eulerian.png with

  * P_delta(k)/P_delta,true(k) and r_delta(k) from CIC on the fine grid, tabulated at
    k_Ny,c / 1.5 k_Ny,c / 2 k_Ny,c;
  * mass fractions at delta > 10 and > 100, max density;
  * J = det(I + dPsi/dq) quantiles (1/5/25/50/75/95/99%) and the J < 0 fraction,
    prediction and truth (spectral gradients, same convention as jac_loss);
  * kernel fractions of the residual eps = Psi_pred - Psi_true at the TRUE Eulerian
    positions: [Q_E(eps)/<eps^2>] and [Q_J(eps)/<eps^2>] (p=2, eps=0.1, state = truth),
    each divided by the same ratio for a Gaussian field with the residual's spectrum
    (3 realisations) -- ~1 means the residual is in general position w.r.t. the form;
  * T4: the same P_delta ratio from a deposit of single-stream particles only (coarse
    det(I + D^L) < 0 mask, upsampled x2; the same q-mask for every field).

Uses only eulerian_metric + the frozen training script's classes; edits nothing.

  python runs/eval_eulerian.py runs/F_flow_128_3k runs/F_reg_128_3k ...
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_octaves as p0          # noqa: E402
import octave_flow_toy as oft        # noqa: E402
import eulerian_metric as em         # noqa: E402

QUANTS = (1, 5, 25, 50, 75, 95, 99)


def density(psi_h, offset):
    """CIC density (mean 1) of a displacement field in units of h, via eulerian_metric."""
    pos = em.eulerian_positions(psi_h, offset)
    ones = psi_h.new_ones((psi_h.shape[0], 1) + psi_h.shape[-3:])
    return em.cic_deposit(ones, pos, psi_h.shape[-1])


def dspec(g, delta):
    F = p0.rfftn(delta.astype(np.float32)[None])
    P = g.shell_avg(np.abs(F[0]) ** 2)
    m = (g.shell_norm > 0) & (g.kshell > 0)
    return g.kshell[m], P[m], F


def rdelta(g, Fa, Fb):
    num = g.shell_avg((Fa[0] * np.conj(Fb[0])).real)
    den = np.sqrt(g.shell_avg(np.abs(Fa[0]) ** 2) * g.shell_avg(np.abs(Fb[0]) ** 2))
    m = (g.shell_norm > 0) & (g.kshell > 0)
    return (num / np.maximum(den, 1e-30))[m]


def gauss_like(eps, g, rng):
    """Gaussian field with eps's isotropic per-component shell spectrum."""
    out = []
    for c in range(3):
        F = p0.rfftn(eps[c][None].numpy())
        P = g.shell_avg(np.abs(F[0]) ** 2)
        W = np.fft.rfftn(rng.standard_normal(eps.shape[-3:]).astype(np.float32))
        Pm = np.interp(g.kmag, g.kshell, P)
        # white noise has E|W|^2 = N^3 per mode
        out.append(np.fft.irfftn(W * np.sqrt(np.maximum(Pm, 0) / eps.shape[-1] ** 3),
                                 s=eps.shape[-3:]))
    return torch.from_numpy(np.stack(out).astype(np.float32))


def main():
    runs = [a for a in sys.argv[1:] if not a.startswith("-")]
    for rd in runs:
        args = json.load(open(os.path.join(rd, "results.json")))["args"]
        Nc, Nf, L, off, gr = args["nc"], args["nf"], args["box"], args["offset"], args["growth"]
        reg = args.get("regression", False)
        transv = args.get("octave_sampler", "longitudinal") == "full"
        dev = torch.device("cpu")
        torch.manual_seed(args.get("seed", 0))
        print(f"=== {rd}  ({Nc}->{Nf}, {'regression' if reg else 'flow'}, sampler "
              f"{'full' if transv else 'longitudinal'})")
        sc = oft.Scaffold(Nc, Nf, L, off, args.get("alpha", 1.0), dev, window=args.get("window", "cube"))
        if args.get("dis"):
            train = oft.load_real(args["dis"], args["ic"], Nc, Nf, args["train_seeds"])
            test = oft.load_real(args["dis"], args["ic"], Nc, Nf, args["test_seeds"])
        else:
            train = oft.make_synthetic(Nc, Nf, L, off, args["train_seeds"], args["rms_delta"], args["n_index"], args.get("dealias", False))
            test = oft.make_synthetic(Nc, Nf, L, off, args["test_seeds"], args["rms_delta"], args["n_index"], args.get("dealias", False))
        sc.fit_linear_power([it["ic_f"] * gr for it in train])
        b = oft.Batcher(sc, test, np.random.default_rng(args.get("seed", 0)), augment_on=False,
                        growth=gr, octave_transverse=transv)
        x0, x1, Pc, D = b.make(test, eta="true")

        ck = torch.load(os.path.join(rd, "model_ema.pt"), map_location="cpu")
        sd = ck["state_dict"] if "state_dict" in ck else ck
        model = oft.UNet3D(cin=12, cout=3, base=int(ck.get("base", 24)))
        model.load_state_dict(sd); model.eval()
        s = torch.zeros(1)
        nst, meth = (1, "euler") if reg else (8, "heun")
        pred = oft.sample_flow(model, x0, Pc, D, s, nsteps=nst, method=meth)
        one = oft.sample_flow(model, x0, Pc, D, s, nsteps=1, method="euler")
        gen = torch.Generator().manual_seed(1)
        x0g, _, _, _ = b.make(test, eta="sample", gen=gen)
        predg = oft.sample_flow(model, x0g, Pc, D, s, nsteps=nst, method=meth)

        fields = {"coarse": Pc, "baseline": x0, "emulator": pred, "onestep": one,
                  "generative": predg, "truth": x1}
        g = p0.Grid(Nf, L)
        knyc = np.pi / (L / Nc)
        kprobe = [knyc, 1.5 * knyc, 2.0 * knyc]

        # densities and spectra ------------------------------------------------
        rho = {n: density(f, off)[0, 0] for n, f in fields.items()}
        spec, Fd = {}, {}
        for n, r in rho.items():
            k, P, F = dspec(g, (r / r.mean() - 1).numpy())
            spec[n] = (k, P); Fd[n] = F
        kt, Pt = spec["truth"]
        out = {"knyc": knyc, "kprobe": [float(x) for x in kprobe]}
        for n in fields:
            if n == "truth":
                continue
            k, P = spec[n]
            rr = rdelta(g, Fd[n], Fd["truth"])
            out[f"Pdelta_ratio_{n}"] = [float(P[np.argmin(abs(k - q))] / Pt[np.argmin(abs(k - q))]) for q in kprobe]
            out[f"rdelta_{n}"] = [float(rr[np.argmin(abs(k - q))]) for q in kprobe]
        for n, r in rho.items():
            d = (r / r.mean()).numpy()
            out[f"dens_{n}"] = dict(max=float(d.max()),
                                    mass_d10=float(d[d > 10].sum() / d.sum()),
                                    mass_d100=float(d[d > 100].sum() / d.sum()))

        # J quantiles ----------------------------------------------------------
        for n in ("emulator", "generative", "truth"):
            J, _ = em.jacobian_and_adjugate(fields[n])
            Jn = J[0].numpy()
            out[f"J_{n}"] = dict(quantiles={str(q): float(np.percentile(Jn, q)) for q in QUANTS},
                                 neg_frac=float((Jn < 0).mean()))

        # kernel fractions of the emulator residual -----------------------------
        eps = (pred - x1)
        pos_true = em.eulerian_positions(x1, off)
        e2 = float((eps ** 2).mean())
        qe = float(em.eulerian_form(eps, pos_true)) / e2
        qj = float(em.jacobian_form(eps, x1, p=2.0, eps=0.1)) / e2
        rngG = np.random.default_rng(7)
        qeg, qjg = [], []
        for _ in range(3):
            gau = gauss_like(eps[0], g, rngG)[None]
            g2 = float((gau ** 2).mean())
            qeg.append(float(em.eulerian_form(gau, pos_true)) / g2)
            qjg.append(float(em.jacobian_form(gau, x1, p=2.0, eps=0.1)) / g2)
        out["kernel_fraction_E"] = qe / float(np.mean(qeg))
        out["kernel_fraction_J"] = qj / float(np.mean(qjg))
        out["kernel_raw"] = dict(QE_over_e2=qe, QJ_over_e2=qj,
                                 QE_gauss=float(np.mean(qeg)), QJ_gauss=float(np.mean(qjg)))

        # T4: single-stream-only deposit ----------------------------------------
        ms = p0._up2(p0.coarse_invariants(test[0]["dis_c"], p0.Grid(Nc, L))["detJ"] < 0)
        keep = torch.from_numpy(np.broadcast_to(~ms, (1, 1) + ms.shape).copy().astype(np.float32))
        t4 = {}
        for n, f in fields.items():
            pos = em.eulerian_positions(f, off)
            r = em.cic_deposit(keep, pos, Nf)[0, 0]
            k, P, F = dspec(g, (r / r.mean() - 1).numpy())
            t4[n] = (k, P)
        kT, PTt = t4["truth"]
        for n in fields:
            if n == "truth":
                continue
            k, P = t4[n]
            out[f"T4_Pdelta_ratio_{n}"] = [float(P[np.argmin(abs(k - q))] / PTt[np.argmin(abs(k - q))]) for q in kprobe[:2]]

        with open(os.path.join(rd, "eulerian.json"), "w") as f:
            json.dump(out, f, indent=1)

        # print the headline row -------------------------------------------------
        print(f"  P_delta/P_true @ (1, 1.5, 2) k_Ny,c: emulator {out['Pdelta_ratio_emulator']}, "
              f"generative {out['Pdelta_ratio_generative']}")
        print(f"  r_delta @ k_Ny,c: emulator {out['rdelta_emulator'][0]:.3f} | "
              f"mass>100: emu {out['dens_emulator']['mass_d100']:.4f} truth {out['dens_truth']['mass_d100']:.4f} | "
              f"kernel fractions E {out['kernel_fraction_E']:.3f} J {out['kernel_fraction_J']:.3f}")
        print(f"  T4 single-stream P/P @ k_Ny,c: emulator {out['T4_Pdelta_ratio_emulator'][0]:.3f}, "
              f"generative {out['T4_Pdelta_ratio_generative'][0]:.3f}")

        # figure ------------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        cols = {"coarse": "0.55", "baseline": "C0", "emulator": "C2", "onestep": "C5",
                "generative": "C4", "truth": "k"}
        for n in fields:
            if n == "truth":
                continue
            k, P = spec[n]
            ax[0].semilogx(k * 1000, P / np.maximum(Pt, 1e-30), color=cols[n], lw=1.5, label=n)
            ax[1].semilogx(k * 1000, rdelta(g, Fd[n], Fd["truth"]), color=cols[n], lw=1.5, label=n)
        for a in ax[:2]:
            a.axvline(knyc * 1000, color="0.3", ls=":", lw=1); a.grid(alpha=.2)
            a.set_xlabel(r"$k$ [$h$/Mpc]")
        ax[0].axhline(1, color="0.6", lw=.8); ax[0].set_ylim(0, 2.6); ax[0].set_title(r"$P_\delta/P_{\delta,\rm true}$")
        ax[0].legend(fontsize=7)
        ax[1].axhline(1, color="0.6", lw=.8); ax[1].set_ylim(0, 1.05); ax[1].set_title(r"$r_\delta(k)$")
        qs = np.array(QUANTS, float)
        for n, c in (("truth", "k"), ("emulator", "C2"), ("generative", "C4")):
            v = [out[f"J_{n}"]["quantiles"][str(q)] for q in QUANTS]
            ax[2].plot(qs, v, "o-", color=c, label=f"{n} (J<0: {out[f'J_{n}']['neg_frac']:.2f})")
        ax[2].set_xlabel("percentile"); ax[2].set_title(r"quantiles of $J=\det(I+\partial\Psi/\partial q)$")
        ax[2].axhline(0, color="0.6", lw=.8); ax[2].legend(fontsize=7); ax[2].grid(alpha=.2)
        fig.suptitle(f"{rd}  ({Nc}->{Nf})", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(rd, "eulerian.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
