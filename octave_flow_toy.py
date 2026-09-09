#!/usr/bin/env python3
"""
Octave flow — a laptop-sized end-to-end prototype of the recommended method:

    one 2x Lagrangian step, trained by conditional flow matching whose SOURCE is
    "prolonged coarse field + linear-theory octave" and whose training coupling
    is the PHYSICAL one (true initial-condition octave  ->  true fine field).

        x_0 = P Psi_c + Psi_lin[eta]         (fine grid, units of h_f)
        x_1 = Psi_f
        loss = E || v_theta(x_t - P Psi_c, t | D_ij(P Psi_c)) - (x_1 - x_0) ||^2

    inference:  eta = true IC octave      -> deterministic emulator (Paper-IV mode)
                eta ~ Gaussian(P_lin)     -> generative model

Everything the network sees is translation invariant (residuals and gradients).
The low-k part of the residual is the coarse correction eps, the octave band is the
detail: one output, two Fourier bands.  The restriction R is the cube by default
(--window cube): nothing of the coarse run is discarded and "the new randomness" is
exactly the octave of the initial conditions.  --window sphere is the isotropic
ablation; it discards the coarse corners and re-samples them, which is only an
approximation (see notes/octave_flow_derivation.pdf, Remark 2.4).

Data: nested 2LPT synthesised on the fly (needs phase0_octaves.py next to this
file), or real map2map fields via --dis/--ic patterns with {N} and {seed}.

Examples
--------
  python octave_flow_toy.py --nc 16 --nf 32 --steps 200            # smoke test, ~1 min CPU
  python octave_flow_toy.py --nc 32 --nf 64 --steps 1500 --device mps   # Mac
  python octave_flow_toy.py --nc 64 --nf 128 --dis "/d/{seed}/{N}/dis.npy" \
        --ic "/d/{seed}/{N}/ic.npy" --train-seeds 0 1 2 3 4 5 --test-seeds 6 --device cuda
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_octaves as p0  # noqa: E402


# ----------------------------------------------------------------------------
# spectral scaffold in torch (fine grid)
# ----------------------------------------------------------------------------
class Scaffold:
    """k-vectors, sphere window W, prolongation P (coarse->fine), gradients, linear-octave sampler."""

    def __init__(self, Nc, Nf, L, offset, alpha, device, fft_device=None, window="cube"):
        self.Nc, self.Nf, self.L, self.offset, self.alpha = Nc, Nf, L, offset, alpha
        self.window = window
        self.hf, self.hc = L / Nf, L / Nc
        self.knyc = math.pi / self.hc
        self.device = device
        # torch.fft with complex tensors is not available on MPS: do spectral ops on fft_device
        self.fdev = torch.device(fft_device) if fft_device is not None else \
            (torch.device("cpu") if torch.device(device).type == "mps" else torch.device(device))
        device = self.fdev
        gf = p0.Grid(Nf, L)
        self.gf = gf
        # cube (default): the state keeps every coarse mode, the octave is exactly the new IC modes.
        # sphere: isotropic, but discards the coarse corners and re-samples them (an approximation).
        W = p0.cube_window(gf, self.knyc) if window == "cube" else p0.sphere_window(gf, self.knyc, alpha)
        self.W = torch.from_numpy(W).to(device)
        self.k = [torch.from_numpy(np.broadcast_to(a, gf.kmag.shape).copy()).to(device)
                  for a in (gf.kx, gf.ky, gf.kz)]
        self.kmag = torch.from_numpy(gf.kmag).to(device)
        self.k2 = torch.clamp(self.kmag**2, min=1e-30)
        ph = gf.phase(offset)
        self.phase = None if isinstance(ph, float) else torch.from_numpy(ph).to(device)
        ax, c_ax, az = p0._coarse_slices(Nf, Nc)
        self.ax, self.c_ax, self.az = (torch.from_numpy(a).to(device) for a in (ax, c_ax, az))
        self.kf_shell = None  # filled by fit_linear_power
        self.filters_fitted = False

    # -- P: zero padding + phase (batched, (B,3,Nc,Nc,Nc) -> (B,3,Nf,Nf,Nf)) --------
    def prolong(self, xc):
        xc = xc.to(self.fdev)
        B, C = xc.shape[:2]
        Fc = torch.fft.rfftn(xc, dim=(-3, -2, -1))
        Fc[..., self.Nc // 2] = 0
        Ff = torch.zeros((B, C, self.Nf, self.Nf, self.Nf // 2 + 1), dtype=Fc.dtype, device=xc.device)
        sub = Fc[:, :, self.c_ax][:, :, :, self.c_ax][:, :, :, :, self.az]
        idx = torch.meshgrid(self.ax, self.ax, self.az, indexing="ij")
        Ff[:, :, idx[0], idx[1], idx[2]] = sub
        Ff = Ff * self.W  # also kills the coarse Nyquist planes
        if self.phase is not None:
            Ff = Ff * self.phase
        Ff = Ff * (self.Nf / self.Nc) ** 3
        return torch.fft.irfftn(Ff, s=(self.Nf,) * 3, dim=(-3, -2, -1)).to(self.device)

    def band(self, x, which):
        """project a fine field onto the coarse band ('low', W) or the octave ('high', 1-W)."""
        x = x.to(self.fdev)
        Fx = torch.fft.rfftn(x, dim=(-3, -2, -1))
        Fx = Fx * (self.W if which == "low" else (1 - self.W))
        return torch.fft.irfftn(Fx, s=x.shape[-3:], dim=(-3, -2, -1)).to(self.device)

    def gradients(self, x):
        """D_ij = d_i x_j for a physical-unit field (B,3,N,N,N) -> (B,9,N,N,N)."""
        x = x.to(self.fdev)
        Fx = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out = []
        for i in range(3):
            for j in range(3):
                out.append(torch.fft.irfftn(1j * self.k[i] * Fx[:, j], s=x.shape[-3:], dim=(-3, -2, -1)))
        return torch.stack(out, dim=1).to(self.device)

    # -- linear octave prior: P_delta,lin(k) measured from training ICs ------------
    def fit_linear_power(self, ic_list):
        """isotropic P_delta(k) of the linear field (divergence of IC displacement), on the fine grid."""
        g = self.gf
        acc = np.zeros(g.nshell)
        for ic in ic_list:
            Fi = p0.rfftn(ic.astype(np.float32))
            div = 1j * (g.kx * Fi[0] + g.ky * Fi[1] + g.kz * Fi[2])
            acc += g.shell_avg(np.abs(div) ** 2) * (g.L / g.N**2) ** 3
        P = acc / len(ic_list)
        # per-mode amplitude A(k) so that white noise -> field with power P(k)
        Pm = np.interp(g.kmag.ravel(), g.kshell, P).reshape(g.kmag.shape)
        A = np.sqrt(np.maximum(Pm, 0) / ((g.L / g.N**2) ** 3 * g.N**3))
        self.A_delta = torch.from_numpy(A.astype(np.float32)).to(self.fdev)

    def fit_source_filters(self, train, growth=1.0):
        """Data-driven linear source (the best linear prediction of Psi_f, mode by mode):
             coarse band : Tc(k) = P_{c x f}/P_{c c}   applied to P Psi_c   (Wiener estimate of R Psi_f)
             octave band : G(k)  = P_{d x lin}/P_{lin} applied to the linear octave (its propagator)
           Both are isotropic multipliers measured on the training pairs; they leave the
           information content of the state unchanged and only shorten x1 - x0.  On real N-body
           data at z=0 the raw linear octave overshoots the true detail power (G < 1) and the
           coarse run is wrong near its Nyquist (Tc < 1)."""
        g = self.gf; W = self.W.cpu().numpy()
        Pcc = np.zeros(g.nshell); Pcf = np.zeros(g.nshell); Pll = np.zeros(g.nshell); Pdl = np.zeros(g.nshell)
        for it in train:
            Fc = p0.rfftn(self.prolong(torch.from_numpy(it["dis_c"][None]))[0].cpu().numpy())
            Ff = p0.rfftn(it["dis_f"].astype(np.float32))
            Fl = p0.rfftn((it["ic_f"] * growth).astype(np.float32)) * (1 - W)
            Fd = Ff * (1 - W)
            Pcc += g.shell_avg((np.abs(Fc) ** 2).sum(0)); Pcf += g.shell_avg((Fc * np.conj(Ff)).real.sum(0))
            Pll += g.shell_avg((np.abs(Fl) ** 2).sum(0)); Pdl += g.shell_avg((Fd * np.conj(Fl)).real.sum(0))
        Tc = np.where(Pcc > 0, Pcf / np.maximum(Pcc, 1e-30), 1.0)
        G = np.where(Pll > 0, Pdl / np.maximum(Pll, 1e-30), 1.0)
        # shell estimates are noisy where shells hold few modes: smooth over 5 shells and pin the
        # coarse-band filter to 1 below 0.3 k_Ny,c (the coarse run is exact there up to the white floor)
        def smooth(x):
            xs = x.copy(); n = len(x)
            for i in range(n):
                lo, hi = max(0, i - 2), min(n, i + 3); xs[i] = np.mean(x[lo:hi])
            return xs
        Tc, G = smooth(Tc), smooth(G)
        Tc[g.kshell < 0.3 * self.knyc] = 1.0
        Tc = np.clip(Tc, 0.0, 1.5); G = np.clip(G, 0.0, 1.5)
        Tm = np.interp(g.kmag.ravel(), g.kshell, Tc).reshape(g.kmag.shape)
        Gm = np.interp(g.kmag.ravel(), g.kshell, G).reshape(g.kmag.shape)
        self.Tc = torch.from_numpy((Tm * W).astype(np.float32)).to(self.fdev)       # acts inside W only
        self.G = torch.from_numpy((Gm * (1 - W)).astype(np.float32)).to(self.fdev)  # acts outside W only
        sel = (g.shell_norm > 0) & (g.kshell > 0)
        kk = g.kshell[sel] / self.knyc
        def at(x, q):
            i = np.argmin(np.abs(kk - q)); return float(x[sel][i])
        print("source filters: Tc(k/kNy,c) = " + ", ".join(f"{q:.2f}:{at(Tc, q):.3f}" for q in (0.25, 0.5, 0.75, 0.95))
              + " | G(k/kNy,c) = " + ", ".join(f"{q:.2f}:{at(G, q):.3f}" for q in (1.05, 1.3, 1.6, 1.9)))
        self.filters_fitted = True

    def apply_filter(self, x, which):
        """multiply a fine field by Tc (which='low') or G (which='high') in Fourier space."""
        x = x.to(self.fdev)
        Fx = torch.fft.rfftn(x, dim=(-3, -2, -1)) * (self.Tc if which == "low" else self.G)
        return torch.fft.irfftn(Fx, s=x.shape[-3:], dim=(-3, -2, -1)).to(self.device)

    def sample_linear_octave(self, B, gen=None):
        """Gaussian linear displacement restricted to the octave band, curl-free: Psi = i k delta / k^2."""
        w = torch.randn((B, self.Nf, self.Nf, self.Nf), generator=gen, device=self.fdev)
        Fw = torch.fft.rfftn(w, dim=(-3, -2, -1)) * self.A_delta * (1 - self.W)
        comps = [torch.fft.irfftn(1j * self.k[i] * Fw / self.k2, s=(self.Nf,) * 3, dim=(-3, -2, -1)) for i in range(3)]
        return torch.stack(comps, dim=1).to(self.device)


# ----------------------------------------------------------------------------
# cubic-group augmentation of vector fields on a periodic grid
# ----------------------------------------------------------------------------
def augment(fields, rng, offset):
    """Random axis permutation + flips applied consistently to a list of (3,N,N,N) arrays."""
    perm = rng.permutation(3)
    flips = rng.integers(0, 2, size=3).astype(bool)
    out = []
    for f in fields:
        g = f[perm]                                   # component permutation
        g = np.transpose(g, (0,) + tuple(1 + perm))   # axis permutation (same perm)
        for a in range(3):
            if flips[a]:
                g = np.flip(g, axis=1 + a)
                if offset == 0:                       # position i -> -i mod N
                    g = np.roll(g, 1, axis=1 + a)
                g = g.copy()
                g[a] = -g[a]
        out.append(np.ascontiguousarray(g))
    return out


# ----------------------------------------------------------------------------
# network: small 3D U-Net with FiLM(t) conditioning, periodic convolutions
# ----------------------------------------------------------------------------
def tembed(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None] * 1000.0
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cdim):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1, padding_mode="circular")
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, padding_mode="circular")
        self.film = nn.Linear(cdim, 2 * cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.film(c)[:, :, None, None, None].chunk(2, dim=1)
        h = F.silu(self.n2(h) * (1 + s) + b)
        return self.skip(x) + self.c2(h)


class UNet3D(nn.Module):
    def __init__(self, cin=12, cout=3, base=24, mult=(1, 2, 2), cdim=128):
        super().__init__()
        self.cdim = cdim
        self.cmlp = nn.Sequential(nn.Linear(64 + 1, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.inc = nn.Conv3d(cin, base, 3, padding=1, padding_mode="circular")
        chs = [base * m for m in mult]
        self.down = nn.ModuleList()
        c = base
        for i, co in enumerate(chs):
            self.down.append(nn.ModuleList([ResBlock(c, co, cdim), ResBlock(co, co, cdim)]))
            c = co
        self.mid = ResBlock(c, c, cdim)
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(chs))):
            self.up.append(nn.ModuleList([ResBlock(c + co, co, cdim), ResBlock(co, co, cdim)]))
            c = co
        self.outc = nn.Conv3d(c, cout, 3, padding=1, padding_mode="circular")
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)

    def forward(self, x, t, s):
        c = self.cmlp(torch.cat([tembed(t), s[:, None]], dim=1))
        h = self.inc(x)
        skips = []
        for i, (b1, b2) in enumerate(self.down):
            h = b2(b1(h, c), c)
            skips.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool3d(h, 2)
        h = self.mid(h, c)
        for i, (b1, b2) in enumerate(self.up):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = torch.cat([h, skips.pop()], dim=1)
            h = b2(b1(h, c), c)
        return self.outc(h)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def make_synthetic(Nc, Nf, L, offset, seeds, rms_delta, n_index, dealias):
    data = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        d = p0.make_selftest([Nc, Nf], L, offset, rng, n_index=n_index, rms_delta=rms_delta, dealias=dealias)
        data.append(dict(dis_c=d[Nc][0], dis_f=d[Nf][0], ic_f=d[Nf][1]))
    return data


def load_real(dis_pat, ic_pat, Nc, Nf, seeds):
    data = []
    for s in seeds:
        dc = p0.load(dis_pat.replace("{seed}", str(s)), Nc)
        df = p0.load(dis_pat.replace("{seed}", str(s)), Nf)
        icf = p0.load(ic_pat.replace("{seed}", str(s)), Nf)
        assert dc is not None and df is not None and icf is not None, f"missing files for seed {s}"
        data.append(dict(dis_c=dc, dis_f=df, ic_f=icf))
    return data


class Batcher:
    """Builds (x0, x1, cond) on the device from raw numpy fields, with augmentation."""

    def __init__(self, sc, data, rng, augment_on=True, growth=1.0, coupling="physical", source_filter="none"):
        self.sc, self.data, self.rng, self.aug, self.growth = sc, data, rng, augment_on, growth
        self.coupling = coupling   # physical: x0 uses the true IC octave of x1; independent: a fresh sample
        self.source_filter = source_filter

    def make(self, items, eta="true", gen=None):
        sc = self.sc
        dc = torch.from_numpy(np.stack([it["dis_c"] for it in items]))
        df = torch.from_numpy(np.stack([it["dis_f"] for it in items])).to(sc.device)
        icf = torch.from_numpy(np.stack([it["ic_f"] for it in items]))
        Pc = sc.prolong(dc)                                        # physical units, fine grid
        if eta == "true":
            lin = sc.band(icf * self.growth, "high")              # true linear octave
        else:
            # NB: no * self.growth here -- fit_linear_power() is given ic_f * growth, so
            # A_delta (and therefore sample_linear_octave) already carries the growth factor.
            # Applying it twice made the sampled octave growth^2 = 5890x too powerful at
            # growth = 76.7 (generative P/P_true = 7397 on the first real-data run); it was
            # invisible on the 2LPT toy because every toy run uses the default growth = 1.
            lin = sc.sample_linear_octave(dc.shape[0], gen)
        if self.source_filter == "wiener":
            Pc_src, lin = sc.apply_filter(Pc, "low"), sc.apply_filter(lin, "high")
        else:
            Pc_src = Pc
        x0 = (Pc_src + lin) / sc.hf
        x1 = df / sc.hf
        D = sc.gradients(Pc)                                       # dimensionless, from the unfiltered coarse run
        return x0, x1, Pc_src / sc.hf, D

    def sample(self, B):
        items = []
        for _ in range(B):
            it = self.data[self.rng.integers(len(self.data))]
            if self.aug:
                dc, df, icf = augment([it["dis_c"], it["dis_f"], it["ic_f"]], self.rng, self.sc.offset)
                it = dict(dis_c=dc, dis_f=df, ic_f=icf)
            items.append(it)
        return self.make(items, eta="true" if self.coupling == "physical" else "sample")


def git_commit():
    """Short git hash of the repository containing this file, or None."""
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def net_input(x_t, Pc, D):
    return torch.cat([x_t - Pc, D], dim=1)


@torch.no_grad()
def sample_flow(model, x0, Pc, D, s, nsteps=8, method="heun"):
    x = x0.clone()
    ts = torch.linspace(0, 1, nsteps + 1, device=x.device)
    for i in range(nsteps):
        t0, t1 = ts[i], ts[i + 1]
        tt = torch.full((x.shape[0],), float(t0), device=x.device)
        v0 = model(net_input(x, Pc, D), tt, s)
        if method == "euler" or i == nsteps - 1:
            x = x + (t1 - t0) * v0
        else:
            xe = x + (t1 - t0) * v0
            v1 = model(net_input(xe, Pc, D), torch.full_like(tt, float(t1)), s)
            x = x + 0.5 * (t1 - t0) * (v0 + v1)
    return x


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def band_spectra(sc, a, b):
    """cross spectra of two fine fields (physical units) split into coarse band and octave."""
    g = sc.gf
    Fa, Fb = p0.rfftn(a), p0.rfftn(b)
    W = sc.W.cpu().numpy()
    lo = p0.spectra_k(g, Fa * W, Fb * W)
    hi = p0.spectra_k(g, Fa * (1 - W), Fb * (1 - W), kmin=(sc.alpha if sc.window == "sphere" else 1.0) * sc.knyc)
    return lo, hi


_MS_MASK = {}


def multistream_mask(sc, dis_c):
    """fine-grid boolean mask of the coarse run's multi-stream Lagrangian patches, det(I + D^L) < 0."""
    key = id(dis_c)
    if key not in _MS_MASK:
        gc = p0.Grid(sc.Nc, sc.L)
        inv = p0.coarse_invariants(np.asarray(dis_c, dtype=np.float32), gc)
        _MS_MASK[key] = p0._up2(inv["detJ"] < 0)
    return _MS_MASK[key]


def summarize(sc, name, pred, true, ms=None):
    lo, hi = band_spectra(sc, pred, true)
    # restrict the low band to k < 0.9 kny,c to avoid the shell straddling the sphere edge
    m = lo["k"] < 0.9 * sc.knyc
    out = dict(
        r_low=float(np.average(lo["r"][m])), r_high=float(np.average(hi["r"])),
        ratio_P_high=float(np.average(hi["Paa"] / np.maximum(hi["Pbb"], 1e-30))),
        ratio_P_low=float(np.average(lo["Paa"][m] / np.maximum(lo["Pbb"][m], 1e-30))),
        eps_rel_low=float(np.average((lo["Paa"][m] + lo["Pbb"][m] - 2 * lo["Pab"][m]) / np.maximum(lo["Pbb"][m], 1e-30))),
        rms_err_over_h=float(np.sqrt(((pred - true) ** 2).mean()) / sc.hf),
        k_high=hi["k"].tolist(), r_high_k=hi["r"].tolist(),
        Pratio_high_k=(hi["Paa"] / np.maximum(hi["Pbb"], 1e-30)).tolist(),
    )
    extra = ""
    if ms is not None and ms.any() and (~ms).any():
        e2 = ((pred - true) ** 2).mean(0)      # per-component, like rms_err_over_h
        out["rms_err_over_h_multistream"] = float(np.sqrt(e2[ms].mean()) / sc.hf)
        out["rms_err_over_h_singlestream"] = float(np.sqrt(e2[~ms].mean()) / sc.hf)
        out["multistream_fraction"] = float(ms.mean())
        extra = f" (multi/single-stream: {out['rms_err_over_h_multistream']:.3f}/{out['rms_err_over_h_singlestream']:.3f})"
    print(f"  {name:28s} octave: r={out['r_high']:.3f}  P/P_true={out['ratio_P_high']:.3f} | "
          f"coarse band: r={out['r_low']:.4f}  P_eps/P={out['eps_rel_low']:.2e} | rms err/h_f={out['rms_err_over_h']:.3f}{extra}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", type=int, default=32); ap.add_argument("--nf", type=int, default=64)
    ap.add_argument("--box", type=float, default=100.0); ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0, help="sphere radius in units of k_Ny,c (only with --window sphere)")
    ap.add_argument("--window", type=str, default="cube", choices=["cube", "sphere"],
                    help="restriction R: cube (default, exact nested-IC split) or sphere (isotropic ablation)")
    ap.add_argument("--dis", type=str, default=None, help="real data pattern with {N} and {seed}")
    ap.add_argument("--ic", type=str, default=None, help="real IC pattern with {N} and {seed}")
    ap.add_argument("--growth", type=float, default=1.0, help="D(z)/D(z_init) if the IC is stored at z_init")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--rms-delta", type=float, default=1.5, help="synthetic nonlinearity (rms delta_lin at fine level)")
    ap.add_argument("--n-index", type=float, default=-1.5)
    ap.add_argument("--dealias", action="store_true", help="synthetic coarse levels without aliasing")
    ap.add_argument("--base", type=int, default=24); ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nsteps-sample", type=int, default=8)
    ap.add_argument("--eval-only", type=str, default=None,
                    help="path to a saved model_ema.pt: skip training, only evaluate (e.g. sampler-step sweeps)")
    ap.add_argument("--source-filter", type=str, default="none", choices=["none", "wiener"],
                    help="wiener: x0 = P[Tc Psi_c] + G * linear octave with Tc, G measured on the training set "
                         "(best linear prediction; recommended on real N-body data)")
    ap.add_argument("--coupling", type=str, default="physical", choices=["physical", "independent"],
                    help="training coupling: physical (x0 built from the true octave of x1) or independent "
                         "(x0 built from a fresh sampled octave; same marginals, uninformative source)")
    ap.add_argument("--regression", action="store_true",
                    help="direct one-step regression baseline: same network and inputs, trained at t=0 only "
                         "(predict x1 - x0 from x0), evaluated with a single Euler step")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/train_state.pt if it exists (model, EMA, optimizer, step, loss log)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop training gracefully after this wall time, saving train_state.pt (for chunked runs)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--fft-device", type=str, default=None,
                    help="device for the spectral ops (default: cpu when --device mps, else same as --device)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", type=str, default="octave_flow_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    print(f"device={dev}  Nc={args.nc} Nf={args.nf} L={args.box}")

    # ---- data -----------------------------------------------------------------
    t0 = time.time()
    if args.dis:
        train = load_real(args.dis, args.ic, args.nc, args.nf, args.train_seeds)
        test = load_real(args.dis, args.ic, args.nc, args.nf, args.test_seeds)
    else:
        train = make_synthetic(args.nc, args.nf, args.box, args.offset, args.train_seeds, args.rms_delta, args.n_index, args.dealias)
        test = make_synthetic(args.nc, args.nf, args.box, args.offset, args.test_seeds, args.rms_delta, args.n_index, args.dealias)
    print(f"data ready ({len(train)} train / {len(test)} test boxes) in {time.time()-t0:.1f}s")

    sc = Scaffold(args.nc, args.nf, args.box, args.offset, args.alpha, dev, fft_device=args.fft_device, window=args.window)
    print(f"spectral ops on {sc.fdev}, network on {dev}, window={args.window}")
    sc.fit_linear_power([it["ic_f"] * args.growth for it in train])
    if args.source_filter == "wiener":
        sc.fit_source_filters(train, growth=args.growth)
    batcher = Batcher(sc, train, rng, augment_on=not args.no_augment, growth=args.growth, coupling=args.coupling,
                      source_filter=args.source_filter)
    print(f"training coupling: {args.coupling}" + ("  (regression baseline, t=0 only)" if args.regression else ""))
    s_level = torch.zeros(args.batch, device=dev)  # scale/style scalar: constant for a single transition

    # ---- baseline: x0 alone (linear octave, no learning) ----------------------
    print("baseline (x0 = P Psi_c + linear octave, no network" + (", wiener-filtered source)" if args.source_filter == "wiener" else "):"))
    x0_te, x1_te, Pc_te, D_te = batcher.make(test, eta="true")
    ms = multistream_mask(sc, test[0]["dis_c"])
    print(f"test box: multi-stream fraction of the coarse run = {ms.mean():.3f}")
    base = summarize(sc, "x0 (true octave)", (x0_te[0] * sc.hf).cpu().numpy(), (x1_te[0] * sc.hf).cpu().numpy(), ms)

    # ---- train ------------------------------------------------------------------
    ckpt = None
    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=dev)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            args.base = int(ckpt.get("base", args.base)); ckpt = ckpt["state_dict"]
    model = UNet3D(cin=12, cout=3, base=args.base).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e6:.2f}M (base={args.base})")
    log = []
    if args.eval_only:
        ema = ckpt
        print(f"loaded weights from {args.eval_only}; skipping training")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        step0 = 0
        state_path = os.path.join(args.out, "train_state.pt")
        if args.resume and os.path.exists(state_path):
            st = torch.load(state_path, map_location=dev)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            ema = {k: v.to(dev) for k, v in st["ema"].items()}
            step0, log = st["step"], st["log"]
            rng = np.random.default_rng(args.seed + step0); batcher.rng = rng
            torch.manual_seed(args.seed + step0)
            print(f"resumed from step {step0}")

        def save_state(step):
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema, step=step, log=log,
                            base=args.base), state_path)

        t0 = time.time()
        step = step0
        for step in range(step0 + 1, args.steps + 1):
            x0, x1, Pc, D = batcher.sample(args.batch)
            t = torch.zeros(args.batch, device=dev) if args.regression else torch.rand(args.batch, device=dev)
            xt = (1 - t)[:, None, None, None, None] * x0 + t[:, None, None, None, None] * x1
            v = model(net_input(xt, Pc, D), t, s_level[: args.batch])
            loss = F.mse_loss(v, x1 - x0)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            with torch.no_grad():
                for k, p in model.state_dict().items():
                    ema[k].mul_(0.99).add_(p.detach(), alpha=0.01)
            log.append(float(loss))
            if step % max(1, args.steps // 20) == 0 or step == step0 + 1:
                print(f"step {step:5d}  loss {np.mean(log[-50:]):.4e}  ({(time.time()-t0)/(step-step0):.2f}s/step)")
            if args.max_seconds is not None and time.time() - t0 > args.max_seconds and step < args.steps:
                save_state(step)
                print(f"time budget reached at step {step}/{args.steps}; state saved to {state_path} "
                      f"(rerun with --resume to continue)")
                return
        save_state(step)
    model.load_state_dict(ema); model.eval()

    # ---- evaluate ---------------------------------------------------------------
    s_te = torch.zeros(len(test), device=dev)
    true = (x1_te * sc.hf).cpu().numpy()
    if args.regression:
        args.nsteps_sample = 1
        print("regression baseline: evaluating with a single Euler step")
    print("emulator mode (true octave as source):")
    xe = sample_flow(model, x0_te, Pc_te, D_te, s_te, nsteps=args.nsteps_sample)
    emu = summarize(sc, f"flow, {args.nsteps_sample} Heun steps", (xe[0] * sc.hf).cpu().numpy(), true[0], ms)
    x1e = sample_flow(model, x0_te, Pc_te, D_te, s_te, nsteps=1, method="euler")
    emu1 = summarize(sc, "flow, 1 Euler step (=regression)", (x1e[0] * sc.hf).cpu().numpy(), true[0], ms)
    print("generative mode (sampled Gaussian octave):")
    gen = torch.Generator(device=sc.fdev).manual_seed(1)
    x0_g, _, _, _ = batcher.make(test, eta="sample", gen=gen)
    xg = sample_flow(model, x0_g, Pc_te, D_te, s_te, nsteps=args.nsteps_sample)
    gen1 = summarize(sc, "flow, sampled octave A", (xg[0] * sc.hf).cpu().numpy(), true[0], ms)
    x0_g2, _, _, _ = batcher.make(test, eta="sample", gen=gen)
    xg2 = sample_flow(model, x0_g2, Pc_te, D_te, s_te, nsteps=args.nsteps_sample)
    gen2 = summarize(sc, "sample A vs sample B", (xg[0] * sc.hf).cpu().numpy(), (xg2[0] * sc.hf).cpu().numpy())
    res = dict(args=vars(args), argv=sys.argv, git_commit=git_commit(), loss=log, baseline=base, emulator=emu,
               emulator_1step=emu1, generative=gen1, sample_vs_sample=gen2, nparam=nparam)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f)
    if not args.eval_only:
        torch.save({"state_dict": model.state_dict(), "base": args.base, "args": vars(args)},
                   os.path.join(args.out, "model_ema.pt"))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        if len(log) > 20:
            ax[0].semilogy(np.convolve(log, np.ones(20) / 20, mode="valid"))
        ax[0].set_title("flow-matching loss"); ax[0].set_xlabel("step")
        k = np.array(base["k_high"]) / sc.knyc
        for lab, d in (("x0 baseline", base), ("emulator (true octave)", emu), ("1-step", emu1), ("sampled octave", gen1)):
            ax[1].plot(k, d["r_high_k"], label=lab)
        ax[1].set_ylim(0, 1.05); ax[1].set_xlabel(r"$k/k_{\rm Ny,c}$"); ax[1].set_title("r(k) vs truth, octave band"); ax[1].legend(fontsize=7)
        for lab, d in (("x0 baseline", base), ("emulator", emu), ("sampled octave", gen1)):
            ax[2].plot(k, d["Pratio_high_k"], label=lab)
        ax[2].axhline(1, color="0.6", lw=0.8); ax[2].set_xlabel(r"$k/k_{\rm Ny,c}$"); ax[2].set_title("P(k)/P_true, octave band"); ax[2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "summary.png"), dpi=130)
        print("wrote", os.path.join(args.out, "summary.png"))
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
