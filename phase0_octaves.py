#!/usr/bin/env python3
"""
Phase-0 measurements for Lagrangian progressive 2x super-resolution.

Works on map2map-style Lagrangian displacement fields: float arrays of shape
(3, N, N, N) in Mpc/h, one per resolution level, all from the SAME nested
initial-condition seed. For a nested series N_0 < N_1 < ... (each 2x) it
measures, for every transition l -> l+1:

  (a) correction   eps_l = R Psi_{l+1} - Psi_l      for several restriction
      operators R (isotropic spectral window, sharp cube, Haar block mean),
      its power spectrum relative to the coarse field, the low-k slope
      (theory: displacement P_eps ~ k^2, divergence ~ k^4), the coarse/fine
      cross-correlation coefficient, and the Wiener transfer function
      T_l(k) = P_{l x l+1} / P_{l+1}  ("the coarse-graining that a real LR
      simulation looks most like").
  (b) detail       d_l = (I - P R) Psi_{l+1} vs. the LINEAR octave built from
      the initial-condition displacement: cross-correlation r(k) and the
      propagator T(k) = P_{d x lin}/P_{lin}.  1 - r^2 is the stochasticity a
      generative model must supply; r ~ 1 means the step is deterministic.
      Also vs. the coarse-only 2LPT harmonics (I - P_W) T2 S[delta_c]: the part of
      the detail that is deterministic given the coarse ICs (needs a coarse IC).
  (c) conditional statistics of d_l/h_l binned by local invariants of the
      coarse deformation tensor D_ij = d_i Psi_j (trace ~ -delta_L, traceless
      magnitude, sign of det(I + D) as a multi-stream proxy): variance and
      excess kurtosis per bin.  Tests the "conditionally near-Gaussian +
      local" hypothesis before any training.

Conventions
-----------
* Lagrangian grid positions q = (i + offset) * h, offset in {0, 0.5}. The
  spectral R/P operators carry the phase that makes RP = I exactly for any
  offset (checked in --selftest).
* Restriction R = a 0/1 Fourier window W inside the coarse cube, P = zero
  padding.  W in {0,1} so R P = I and P R = W exactly.  Panel (a) compares the
  sharp cube, the sharp sphere |k| < alpha*k_Ny,c and Haar.  The detail
  diagnostics (b),(c) use --window (default cube: the exact nested-IC split;
  sphere: isotropic, but it discards the coarse corners, which the retained
  field is still informative about - an approximation).  A smooth taper would
  break R P = I; use one only inside a model's source distribution, never as R.

Usage
-----
  python phase0_octaves.py --selftest --levels 32 64 128 --out st/            # aliased 2LPT levels
  python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128   # ideal continuum levels
  python phase0_octaves.py --levels 64 128 256 512 --box 100 \
      --dis "/data/{N}/dis.npy" --ic "/data/{N}/ic_dis.npy" --offset 0.5 --alpha 1.0 \
      --out phase0_out/

--ic may be a per-level pattern (best: also enables the nestedness check
R IC_{l+1} == IC_l) or a single top-level file (then lower-level ICs are
built by sharp cube truncation).  --vel is optional and analysed like --dis.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import scipy.fft as sfft

    def rfftn(x):
        return sfft.rfftn(x, axes=(-3, -2, -1), workers=-1)

    def irfftn(x, s):
        return sfft.irfftn(x, s=s, axes=(-3, -2, -1), workers=-1)
except Exception:  # pragma: no cover
    def rfftn(x):
        return np.fft.rfftn(x, axes=(-3, -2, -1))

    def irfftn(x, s):
        return np.fft.irfftn(x, s=s, axes=(-3, -2, -1))


# ----------------------------------------------------------------------------
# spectral scaffolding: k-grids, windows, R, P, Haar
# ----------------------------------------------------------------------------
class Grid:
    """Fourier bookkeeping for an N^3 periodic box of side L (rfft layout)."""

    def __init__(self, N, L):
        self.N, self.L = int(N), float(L)
        self.h = self.L / self.N
        self.kf = 2 * np.pi / self.L
        self.kny = np.pi / self.h
        k1 = np.fft.fftfreq(N, d=self.h) * 2 * np.pi
        kz = np.fft.rfftfreq(N, d=self.h) * 2 * np.pi
        self.kx = k1[:, None, None].astype(np.float32)
        self.ky = k1[None, :, None].astype(np.float32)
        self.kz = kz[None, None, :].astype(np.float32)
        self.kmag = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        # multiplicity of rfft modes for shell averages
        mult = np.full(self.kmag.shape, 2.0, dtype=np.float32)
        mult[..., 0] = 1.0
        if N % 2 == 0:
            mult[..., -1] = 1.0
        self.mult = mult
        self.ishell = np.rint(self.kmag / self.kf).astype(np.int64)
        self.nshell = int(self.ishell.max()) + 1
        self.shell_norm = np.bincount(self.ishell.ravel(), weights=mult.ravel(),
                                      minlength=self.nshell)
        self.kshell = np.bincount(self.ishell.ravel(),
                                  weights=(mult * self.kmag).ravel(),
                                  minlength=self.nshell) / np.maximum(self.shell_norm, 1)

    def shell_avg(self, w):
        """Shell average of a real (Nx,Ny,Nz/2+1) array."""
        s = np.bincount(self.ishell.ravel(), weights=(self.mult * w).ravel(),
                        minlength=self.nshell)
        return s / np.maximum(self.shell_norm, 1)

    def phase(self, offset_cells):
        """exp(-i k . (offset_cells * h)) per axis, broadcastable."""
        if offset_cells == 0:
            return 1.0
        s = offset_cells * self.h
        return np.exp(-1j * (self.kx + self.ky + self.kz) * s).astype(np.complex64)


def sphere_window(grid_fine, kc, alpha=1.0):
    """Isotropic sharp projector: 1 for |k| < alpha*kc (kc = coarse Nyquist), 0 otherwise
    (and 0 on/after the coarse Nyquist planes).  W in {0,1} => P R is a projector,
    R P = I exactly, and the detail subspace is (1 - W)."""
    W = (grid_fine.kmag < alpha * kc).astype(np.float32)
    for kk in (grid_fine.kx, grid_fine.ky, grid_fine.kz):
        W = np.where(np.abs(kk) >= kc - 1e-6 * kc, 0.0, W).astype(np.float32)
    return W


def cube_window(grid_fine, kc):
    """Sharp cube truncation |k_i| < kc (coarse Nyquist planes zeroed)."""
    W = np.ones(grid_fine.kmag.shape, dtype=np.float32)
    for kk in (grid_fine.kx, grid_fine.ky, grid_fine.kz):
        W = np.where(np.abs(kk) >= kc - 1e-6 * kc, 0.0, W)
    return W


def _coarse_slices(Nf, Nc):
    """Index arrays mapping fine rfft layout -> coarse rfft layout."""
    half = Nc // 2
    ax = np.concatenate([np.arange(0, half), np.arange(Nf - half + 1, Nf)])
    # coarse ordering: 0..half-1, [Nyquist slot -> dropped], half+1..Nc-1
    c_ax = np.concatenate([np.arange(0, half), np.arange(half + 1, Nc)])
    return ax, c_ax, np.arange(0, half + 1)  # last axis: 0..half (half = Nyquist, zeroed)


def restrict_spectral(field_fine, gf, gc, W, offset):
    """R: fine (C,Nf,Nf,Nf) -> coarse (C,Nc,Nc,Nc) with window W (fine layout)."""
    Nf, Nc = gf.N, gc.N
    F = rfftn(field_fine.astype(np.float32)) * W
    if offset:
        # true coefficients are F * exp(-i k o h_f); re-referencing to the
        # coarse grid positions (i + o) h_c = (2i + 2o) h_f needs exp(+i k o h_f)
        F = F * np.conj(gf.phase(offset))
    ax, c_ax, az = _coarse_slices(Nf, Nc)
    Fc = np.zeros((F.shape[0], Nc, Nc, Nc // 2 + 1), dtype=np.complex64)
    sub = F[:, ax][:, :, ax][:, :, :, az]
    Fc[:, c_ax[:, None, None], c_ax[None, :, None], az[None, None, :]] = sub
    Fc[..., Nc // 2] = 0.0
    Fc *= (Nc / Nf) ** 3
    return irfftn(Fc, s=(Nc, Nc, Nc)).astype(np.float32)


def prolong_spectral(field_coarse, gc, gf, offset):
    """P: coarse -> fine by zero padding (band-limited interpolation), RP = I."""
    Nf, Nc = gf.N, gc.N
    Fc = rfftn(field_coarse.astype(np.float32))
    Fc[..., Nc // 2] = 0.0
    ax, c_ax, az = _coarse_slices(Nf, Nc)
    F = np.zeros((Fc.shape[0], Nf, Nf, Nf // 2 + 1), dtype=np.complex64)
    F[:, ax[:, None, None], ax[None, :, None], az[None, None, :]] = \
        Fc[:, c_ax[:, None, None], c_ax[None, :, None], az[None, None, :]]
    # zero the coarse-Nyquist planes that landed in the fine array
    for i, kk in enumerate((gf.kx, gf.ky, gf.kz)):
        F = np.where(np.abs(kk) >= gc.kny - 1e-6 * gc.kny, 0.0, F)
    if offset:
        F = F * gf.phase(offset)
    F *= (Nf / Nc) ** 3
    return irfftn(F, s=(Nf, Nf, Nf)).astype(np.float32)


def restrict_haar(field_fine):
    C, N = field_fine.shape[0], field_fine.shape[1]
    return field_fine.reshape(C, N // 2, 2, N // 2, 2, N // 2, 2).mean(axis=(2, 4, 6))


def apply_window_fine(field_fine, W):
    F = rfftn(field_fine.astype(np.float32)) * W
    return irfftn(F, s=field_fine.shape[1:]).astype(np.float32)


# ----------------------------------------------------------------------------
# spectra
# ----------------------------------------------------------------------------
def spectra_k(g, FA, FB=None, divergence=False, kmin=0.0):
    """Shell-averaged auto/cross power from rfft coefficient arrays (C,Nx,Ny,Nz/2+1).
    Returns dict with k, Paa, Pbb, Pab, r (only shells with kmin <= k <= k_Ny)."""
    if FB is None:
        FB = FA
    if divergence:
        FA = 1j * (g.kx * FA[0] + g.ky * FA[1] + g.kz * FA[2])[None]
        FB = FA if FB is FA else 1j * (g.kx * FB[0] + g.ky * FB[1] + g.kz * FB[2])[None]
    norm = (g.L / g.N**2) ** 3  # -> (Mpc/h)^3 units
    Paa = g.shell_avg((np.abs(FA) ** 2).sum(0)) * norm
    Pbb = Paa if FB is FA else g.shell_avg((np.abs(FB) ** 2).sum(0)) * norm
    Pab = Paa if FB is FA else g.shell_avg((FA * np.conj(FB)).real.sum(0)) * norm
    r = Pab / np.sqrt(np.maximum(Paa * Pbb, 1e-30))
    sel = (g.shell_norm > 0) & (g.kshell >= max(kmin, 0.5 * g.kf)) & (g.kshell <= g.kny)
    return dict(k=g.kshell[sel], Paa=Paa[sel], Pbb=Pbb[sel], Pab=Pab[sel], r=r[sel])


def spectra(g, A, B=None, divergence=False):
    FA = rfftn(A.astype(np.float32))
    FB = None if B is None else rfftn(B.astype(np.float32))
    return spectra_k(g, FA, FB, divergence=divergence)


def lowk_slope(k, P, kny_c, lo=0.1, hi=0.5, min_shells=6):
    """log-log slope of P over lo*k_Ny,c < k < hi*k_Ny,c (NaN if too few shells)."""
    sel = (k > lo * kny_c) & (k < hi * kny_c) & (P > 0)
    if sel.sum() < min_shells:
        return np.nan
    return float(np.polyfit(np.log(k[sel]), np.log(P[sel]), 1)[0])


# ----------------------------------------------------------------------------
# deformation tensor & conditional statistics
# ----------------------------------------------------------------------------
def deformation_tensor(field, g):
    """D_ij = d_i Psi_j on the field's own grid (spectral derivative). Returns (3,3,N,N,N)."""
    F = rfftn(field.astype(np.float32))
    ks = (g.kx, g.ky, g.kz)
    D = np.empty((3, 3) + field.shape[1:], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            D[i, j] = irfftn(1j * ks[i] * F[j], s=field.shape[1:]).astype(np.float32)
    return D


def excess_kurtosis(x):
    x = x - x.mean()
    m2 = (x**2).mean()
    return (x**4).mean() / max(m2**2, 1e-30) - 3.0


def coarse_invariants(field_c, gc):
    """Scalar invariants of D_ij on the coarse grid: delta_L = -tr D, traceless |S|, det(I + D)."""
    D = deformation_tensor(field_c, gc)
    trD = D[0, 0] + D[1, 1] + D[2, 2]
    S = 0.5 * (D + D.transpose(1, 0, 2, 3, 4))
    for i in range(3):
        S[i, i] -= trD / 3
    tidal = np.sqrt((S**2).sum(axis=(0, 1)))
    del S
    J = D.copy()
    for i in range(3):
        J[i, i] += 1.0
    detJ = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
            - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
            + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
    del J, D
    return dict(delta_L=-trD, tidal=tidal, detJ=detJ)


def _up2(x):
    return np.repeat(np.repeat(np.repeat(x, 2, axis=0), 2, axis=1), 2, axis=2)


def conditional_stats(d_over_h, inv, nbins=10):
    """Bin fine-grid points by coarse invariants (nearest coarse cell); var / kurtosis of details."""
    out = {}
    for name in ("delta_L", "tidal"):
        key = _up2(inv[name])
        edges = np.quantile(key, np.linspace(0, 1, nbins + 1))
        rows = []
        for b in range(nbins):
            m = (key >= edges[b]) & ((key < edges[b + 1]) if b < nbins - 1 else (key <= edges[b + 1]))
            vals = d_over_h[:, m].ravel()
            rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(m.sum()),
                             var=float(vals.var()), kurt=float(excess_kurtosis(vals))))
        out[name] = rows
        del key
    ms = _up2(inv["detJ"] < 0)
    out["multistream"] = dict(frac=float(ms.mean()),
                              var_in=float(d_over_h[:, ms].var()) if ms.any() else None,
                              kurt_in=float(excess_kurtosis(d_over_h[:, ms].ravel())) if ms.any() else None,
                              var_out=float(d_over_h[:, ~ms].var()),
                              kurt_out=float(excess_kurtosis(d_over_h[:, ~ms].ravel())))
    out["global"] = dict(var=float(d_over_h.var()), kurt=float(excess_kurtosis(d_over_h.ravel())))
    return out


# ----------------------------------------------------------------------------
# 2LPT self-test data
# ----------------------------------------------------------------------------
def gaussian_delta(g, n_index, amp, rng):
    """Gaussian delta_lin with P(k) = amp * (k/k_ny)^n on grid g. Returns rfft coefficients."""
    N = g.N
    white = rng.standard_normal((N, N, N)).astype(np.float32)
    F = rfftn(white[None])[0]
    with np.errstate(divide="ignore"):
        P = np.where(g.kmag > 0, amp * (g.kmag / g.kny) ** n_index, 0.0)
    F *= np.sqrt(P / (g.L / g.N**2) ** 3).astype(np.float32)
    return F


def lpt_displacement(delta_k, g, order=2):
    """Zel'dovich (+2LPT) displacement from delta_lin coefficients (Bouchet+95 conventions)."""
    k2 = np.maximum(g.kmag**2, 1e-30)
    phi = np.where(g.kmag > 0, -delta_k / k2, 0)  # laplacian phi = delta
    ks = (g.kx, g.ky, g.kz)
    N = g.N
    Psi = np.stack([irfftn((-1j * ks[i] * phi)[None], s=(N, N, N))[0] for i in range(3)]).astype(np.float32)
    if order < 2:
        return Psi
    # phi_ij and the 2LPT source sum_{i<j} phi_ii phi_jj - phi_ij^2
    phid = {}
    for i in range(3):
        for j in range(i, 3):
            phid[(i, j)] = irfftn((-ks[i] * ks[j] * phi)[None], s=(N, N, N))[0]
    S = (phid[(0, 0)] * phid[(1, 1)] - phid[(0, 1)] ** 2
         + phid[(0, 0)] * phid[(2, 2)] - phid[(0, 2)] ** 2
         + phid[(1, 1)] * phid[(2, 2)] - phid[(1, 2)] ** 2)
    Sk = rfftn(S.astype(np.float32)[None])[0]
    phi2 = np.where(g.kmag > 0, -Sk / k2, 0)
    Psi2 = np.stack([irfftn((1j * ks[i] * phi2)[None], s=(N, N, N))[0] for i in range(3)]).astype(np.float32)
    return Psi + (-3.0 / 7.0) * Psi2


def truncate_delta(delta_k_fine, gf, gc):
    """Nested IC: coarse delta = sharp cube truncation of the fine coefficients."""
    Nf, Nc = gf.N, gc.N
    ax, c_ax, az = _coarse_slices(Nf, Nc)
    Fc = np.zeros((Nc, Nc, Nc // 2 + 1), dtype=np.complex64)
    Fc[c_ax[:, None, None], c_ax[None, :, None], az[None, None, :]] = \
        delta_k_fine[ax][:, ax][:, :, az]
    Fc[..., Nc // 2] = 0.0
    Fc *= (Nc / Nf) ** 3
    return Fc


def shift_field(field, g, offset):
    """Resample a band-limited field from positions i*h to (i+offset)*h."""
    if offset == 0:
        return field
    F = rfftn(field.astype(np.float32)) * np.conj(g.phase(offset))
    return irfftn(F, s=field.shape[1:]).astype(np.float32)


def make_selftest(levels, L, offset, rng, n_index=-1.5, rms_delta=0.8, dealias=False):
    """dict N -> (dis, ic_dis) from nested 2LPT (same delta_lin, cube-truncated per level),
    sampled at Lagrangian positions (i + offset) h_N.

    dealias=False: each level's 2LPT is computed on its own grid, so the quadratic
      products alias like a real low-resolution run does (discreteness error, not
      k^2-suppressed).  This mimics the (A)-type physical conditional.
    dealias=True: every level (top included) is computed on a 2x finer grid from
      its truncated delta and then restricted -> ideal continuum runs.  Then
      eps = R[2K(dL,eta) + K(eta,eta)] exactly and P_eps ~ k^2 at low k."""
    Ntop = max(levels)
    Ng = 2 * Ntop if dealias else Ntop
    gg = Grid(Ng, L)
    dk = gaussian_delta(gg, n_index, 1.0, rng)
    if dealias:  # keep the top level nested: only modes inside the top cube
        gt = Grid(Ntop, L)
        dk = dk * cube_window(gg, gt.kny)
    var = (np.abs(dk) ** 2 * gg.mult).sum() / gg.N**6      # rms of delta_lin at top level
    dk *= (rms_delta / np.sqrt(var)).astype(np.float32)
    data = {}
    for N in levels:
        g = Grid(N, L)
        if dealias:
            dk_lvl = dk * cube_window(gg, g.kny)              # truncated delta, still on the big grid
            Wc = cube_window(gg, g.kny)
            dis = restrict_spectral(lpt_displacement(dk_lvl, gg, order=2), gg, g, Wc, 0)
            ic = restrict_spectral(lpt_displacement(dk_lvl, gg, order=1), gg, g, Wc, 0)
        else:
            dkl = dk if N == Ntop else truncate_delta(dk, gg, g)
            dis = lpt_displacement(dkl, g, order=2)
            ic = lpt_displacement(dkl, g, order=1)            # "IC" = Zel'dovich, growth ratio 1
        data[N] = (shift_field(dis, g, offset), shift_field(ic, g, offset))
    return data


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------
def coarse_harmonics(ic_c, gc, gf, offset, W_model, growth=1.0):
    """Second-order (2LPT) displacement generated by the COARSE linear modes alone, on the fine
    grid, projected outside W: the deterministic 'harmonics' part of the detail, d2_harm =
    (I - P_W) T2 S[delta_c].  ic_c is the coarse linear displacement (scaled by growth to the
    output time); products of two coarse modes reach at most 2 k_Ny,c = k_Ny,f, so the fine
    grid holds them without aliasing."""
    Pic = prolong_spectral(ic_c * growth, gc, gf, offset)           # linear coarse displacement on fine grid
    F = rfftn(Pic)
    delta_k = -1j * (gf.kx * F[0] + gf.ky * F[1] + gf.kz * F[2])     # delta = -div Psi^(1)
    # delta_k came from samples at (i+o)h; multipliers and pointwise products keep that
    # labelling, so the result is already sampled at the data's offset positions.
    psi2 = lpt_displacement(delta_k, gf, order=2) - lpt_displacement(delta_k, gf, order=1)
    return rfftn(psi2) * (1.0 - W_model)


def analyse_transition(Nc, Nf, L, dis_c, dis_f, ic_f, ic_c, offset, alpha, out, tag, nbins, window="cube", growth=1.0):
    gc, gf = Grid(Nc, L), Grid(Nf, L)
    res = dict(Nc=Nc, Nf=Nf, L=L, h_c=gc.h, h_f=gf.h, kny_c=gc.kny, kny_f=gf.kny, alpha=alpha)
    W_iso = sphere_window(gf, gc.kny, alpha)
    W_cube = cube_window(gf, gc.kny)
    W_model = W_cube if window == "cube" else W_iso       # split used for the detail diagnostics (b), (c)

    # --- (a) corrections for each R --------------------------------------
    corr = {}
    for name, R in (("sphere", lambda f: restrict_spectral(f, gf, gc, W_iso, offset)),
                    ("cube", lambda f: restrict_spectral(f, gf, gc, W_cube, offset)),
                    ("haar", restrict_haar)):
        Rf = R(dis_f)
        eps = Rf - dis_c
        s_eps = spectra(gc, eps)
        s_cx = spectra(gc, dis_c, Rf)
        s_div = spectra(gc, eps, divergence=True)
        s_divc = spectra(gc, dis_c, divergence=True)
        corr[name] = dict(
            k=s_eps["k"].tolist(),
            P_eps=s_eps["Paa"].tolist(), P_coarse=s_cx["Paa"].tolist(), P_Rfine=s_cx["Pbb"].tolist(),
            r_coarse_Rfine=s_cx["r"].tolist(),
            Pdiv_eps=s_div["Paa"].tolist(), Pdiv_coarse=s_divc["Paa"].tolist(),
            slope_lowk_P_eps=float(lowk_slope(s_eps["k"], s_eps["Paa"], gc.kny)),
            slope_lowk_Pdiv_eps=float(lowk_slope(s_div["k"], s_div["Paa"], gc.kny)),
            rms_eps_over_h_c=float(np.sqrt((eps**2).mean()) / gc.h),
        )
        print(f"[{tag}] R={name:6s}: rms(eps)/h_c={corr[name]['rms_eps_over_h_c']:.4f}  "
              f"slope[0.1-0.5 kNy,c] P_eps={corr[name]['slope_lowk_P_eps']:+.2f} (theory 2 for spectral R)  "
              f"P_div,eps={corr[name]['slope_lowk_Pdiv_eps']:+.2f} (theory 4)")
    # Wiener transfer function on the fine grid (no window): T = P_cf / P_ff
    Pc_f = prolong_spectral(dis_c, gc, gf, offset)
    s_w = spectra(gf, Pc_f, dis_f)
    res["wiener"] = dict(k=s_w["k"].tolist(), T=(s_w["Pab"] / np.maximum(s_w["Pbb"], 1e-30)).tolist(),
                         r=s_w["r"].tolist())
    res["correction"] = corr

    # --- (b) details vs linear octave (all in Fourier space: no cancellation noise) ---
    Ff = rfftn(dis_f.astype(np.float32))
    Fd = Ff * (1.0 - W_model)                                 # (I - PR) Psi_f
    del Ff
    Flin = rfftn(ic_f.astype(np.float32)) * (1.0 - W_model)   # linear octave from the IC
    s_b = spectra_k(gf, Fd, Flin, kmin=(alpha if window == "sphere" else 1.0) * gc.kny)
    res["detail_vs_linear"] = dict(k=s_b["k"].tolist(), P_d=s_b["Paa"].tolist(), P_lin=s_b["Pbb"].tolist(),
                                   r=s_b["r"].tolist(),
                                   T=(s_b["Pab"] / np.maximum(s_b["Pbb"], 1e-30)).tolist())
    # coarse-only harmonics: how much of the detail is deterministic given the coarse ICs
    if ic_c is not None:
        Fh = coarse_harmonics(ic_c, gc, gf, offset, W_model, growth)
        kmin_b = (alpha if window == "sphere" else 1.0) * gc.kny
        s_h = spectra_k(gf, Fd, Fh, kmin=kmin_b)
        Fnl = Fd - Flin * growth                                   # detail minus its linear octave
        s_h2 = spectra_k(gf, Fnl, Fh, kmin=kmin_b)
        frac = float(np.sum(s_h["Pbb"]) / max(np.sum(s_h["Paa"]), 1e-30))
        frac_nl = float(np.sum(s_h2["Pbb"]) / max(np.sum(s_h2["Paa"]), 1e-30))
        res["detail_vs_coarse_harmonics"] = dict(k=s_h["k"].tolist(), P_d=s_h["Paa"].tolist(),
                                                 P_harm=s_h["Pbb"].tolist(), r=s_h["r"].tolist(),
                                                 P_d_nonlinear=s_h2["Paa"].tolist(), r_nonlinear=s_h2["r"].tolist(),
                                                 power_fraction=frac, power_fraction_of_nonlinear=frac_nl)
        print(f"[{tag}] coarse-only 2LPT harmonics: P_harm/P_d={frac:.3f}, r(d,harm)={float(np.mean(s_h['r'])):.3f} | "
              f"vs nonlinear detail (d - linear octave): P_harm/P_nl={frac_nl:.3f}, r={float(np.mean(s_h2['r'])):.3f}")
        del Fh, Fnl
    del Flin
    # nestedness / convention check if a coarse IC is available
    if ic_c is not None:
        Ric = restrict_spectral(ic_f, gf, gc, W_cube, offset)
        num = np.sqrt(((Ric - ic_c) ** 2).mean())
        den = np.sqrt((ic_c**2).mean())
        res["ic_nestedness_rel_rms"] = float(num / den)
        print(f"[{tag}] nestedness/convention check |R_cube IC_f - IC_c| / |IC_c| = {num/den:.3e} "
              f"(~1e-6 if the ICs are nested AND --offset matches the IC generator)")
        del Ric

    # --- (c) conditional statistics ---------------------------------------
    d = irfftn(Fd, s=dis_f.shape[1:]).astype(np.float32)
    del Fd
    inv = coarse_invariants(dis_c, gc)                        # from the coarse input itself
    res["conditional"] = conditional_stats(d / gf.h, inv, nbins=nbins)
    del d, inv
    g = res["conditional"]["global"]
    ms = res["conditional"]["multistream"]
    print(f"[{tag}] details/h_f: var={g['var']:.4f} kurt={g['kurt']:+.2f} | multistream frac={ms['frac']:.3f} "
          f"kurt in/out={ms['kurt_in']}/{ms['kurt_out']:+.2f}")

    np.savez_compressed(os.path.join(out, f"transition_{tag}.npz"),
                        **{f"corr_{n}_{k}": np.array(v) for n, dd in corr.items() for k, v in dd.items()
                           if isinstance(v, list)})
    return res


def plot_all(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    for tag in tags:
        r = results[tag]
        c0 = r["correction"]["sphere"]
        ax.loglog(np.array(c0["k"]) / r["kny_c"], c0["P_coarse"], color="0.75", lw=0.8)
        for name, ls in (("sphere", "-"), ("cube", "--"), ("haar", ":")):
            c = r["correction"][name]
            ax.loglog(np.array(c["k"]) / r["kny_c"], c["P_eps"], ls, label=f"{tag} R={name}")
        k = np.array(c0["k"]); P = np.array(c0["P_eps"])
        i0 = np.argmin(np.abs(k - 0.3 * r["kny_c"]))
        kk = np.array([0.05, 0.6]) * r["kny_c"]
        ax.loglog(kk / r["kny_c"], P[i0] * (kk / k[i0]) ** 2, color="0.4", lw=0.8)
    ax.text(0.06, ax.get_ylim()[0] * 3, r"grey: $P_{\Psi_\ell}$;  thin line: $\propto k^2$", fontsize=8, color="0.3")
    ax.set_xlabel(r"$k/k_{\rm Ny,c}$"); ax.set_ylabel(r"$P_\varepsilon(k)$  [(Mpc/h)$^3$ per component sum]")
    ax.set_title("(a) correction power  (theory: $k^2$ at low $k$ for spectral $R$)"); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for tag in tags:
        r = results[tag]
        w = r["wiener"]; k = np.array(w["k"])
        ax.semilogx(k / r["kny_c"], w["T"], label=f"{tag} T(k)")
        ax.semilogx(k / r["kny_c"], w["r"], "--", label=f"{tag} r(k)")
    ax.axvline(1, color="0.7", lw=0.8); ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel(r"$k/k_{\rm Ny,c}$"); ax.set_title(r"(a') Wiener $T_\ell=P_{c\times f}/P_f$ and $r$ (coarse vs fine)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for tag in tags:
        r = results[tag]; b = r["detail_vs_linear"]; k = np.array(b["k"])
        ax.plot(k / r["kny_c"], np.array(b["r"]), label=f"{tag} r(d, lin octave)")
        ax.plot(k / r["kny_c"], 1 - np.array(b["r"]) ** 2, "--", label=f"{tag} 1-r$^2$")
    ax.axvline(1, color="0.7", lw=0.8); ax.axvline(2, color="0.7", lw=0.8); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$k/k_{\rm Ny,c}$"); ax.set_title("(b) detail vs linear octave: determinism"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    for tag in tags:
        rows = results[tag]["conditional"]["delta_L"]
        x = [0.5 * (r_["lo"] + r_["hi"]) for r_ in rows]
        ax.plot(x, [r_["var"] for r_ in rows], "o-", label=f"{tag} var(d/h)")
        ax.plot(x, [max(r_["kurt"], 1e-3) for r_ in rows], "s--", label=f"{tag} excess kurt")
    ax.set_yscale("log"); ax.set_xlabel(r"coarse $\delta_L=-{\rm tr}\,D$ (bin centre)")
    ax.set_title("(c) conditional var / kurtosis of details"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "phase0_summary.png"), dpi=140)
    print("wrote", os.path.join(out, "phase0_summary.png"))


def check_operators(N, L, offset, alpha, rng):
    """RP = I and PR idempotent, with a random band-limited field."""
    gc, gf = Grid(N // 2, L), Grid(N, L)
    W = sphere_window(gf, gc.kny, alpha)
    x = rng.standard_normal((3, N // 2, N // 2, N // 2)).astype(np.float32)
    # make x band-limited w.r.t. the window (so R P x = x exactly)
    xf = prolong_spectral(x, gc, gf, offset)
    xf = apply_window_fine(xf, W)
    x = restrict_spectral(xf, gf, gc, W, offset)
    err1 = np.abs(restrict_spectral(prolong_spectral(x, gc, gf, offset), gf, gc, W, offset) - x).max() / np.abs(x).max()
    y = rng.standard_normal((3, N, N, N)).astype(np.float32)
    PRy = prolong_spectral(restrict_spectral(y, gf, gc, W, offset), gc, gf, offset)
    Wy = apply_window_fine(y, W)
    err2 = np.abs(PRy - Wy).max() / np.abs(Wy).max()
    print(f"operator check (N={N}, offset={offset}): |RP x - x|/|x| = {err1:.2e}, |PR y - W y|/|W y| = {err2:.2e}")
    return err1, err2


def load(pattern, N):
    if pattern is None:
        return None
    path = pattern.format(N=N)
    if not os.path.exists(path):
        return None
    a = np.load(path, mmap_mode="r")
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4 and a.shape[-1] == 3 and a.shape[0] != 3:
        a = np.moveaxis(a, -1, 0)
    assert a.shape == (3, N, N, N), f"{path}: expected (3,{N},{N},{N}), got {a.shape}"
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[64, 128, 256, 512],
                    help="successive 2x levels (for --selftest e.g. 32 64 128)")
    ap.add_argument("--box", type=float, default=100.0, help="box size in Mpc/h")
    ap.add_argument("--dis", type=str, default=None, help="pattern with {N}, e.g. /data/{N}/dis.npy")
    ap.add_argument("--vel", type=str, default=None, help="optional pattern for velocity fields")
    ap.add_argument("--ic", type=str, default=None,
                    help="IC displacement: pattern with {N} (per level) or a single top-level file")
    ap.add_argument("--offset", type=float, default=0.5, help="Lagrangian grid offset in cells (0 or 0.5)")
    ap.add_argument("--alpha", type=float, default=1.0, help="sphere R keeps |k| < alpha*k_Ny,c (<=1); the (a) panel always compares sphere, cube and Haar")
    ap.add_argument("--window", type=str, default="cube", choices=["cube", "sphere"],
                    help="split used for the detail diagnostics (b),(c): cube (default) or sphere")
    ap.add_argument("--growth", type=float, default=1.0,
                    help="D(z_out)/D(z_ic) to scale the IC displacement to the output time (harmonics diagnostic)")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--out", type=str, default="phase0_out")
    ap.add_argument("--selftest", action="store_true", help="run on nested 2LPT synthetic data")
    ap.add_argument("--selftest-dealias", action="store_true",
                    help="selftest with de-aliased (ideal continuum) coarse levels: P_eps ~ k^2 must hold")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    levels = sorted(args.levels)
    t0 = time.time()

    check_operators(min(64, levels[-1]), args.box, args.offset, args.alpha, rng)

    if args.selftest:
        data = make_selftest(levels, args.box, args.offset, rng, dealias=args.selftest_dealias)
        get_dis = lambda N: data[N][0]
        get_ic = lambda N: data[N][1]
    else:
        if args.dis is None:
            sys.exit("--dis is required unless --selftest")
        get_dis = lambda N: load(args.dis, N)
        top = levels[-1]
        if args.ic is not None and "{N}" not in args.ic:
            ic_top = np.asarray(np.load(args.ic, mmap_mode="r"), dtype=np.float32)
            if ic_top.shape[0] != 3:
                ic_top = np.moveaxis(ic_top, -1, 0)
            assert ic_top.shape == (3, top, top, top), f"--ic must be the {top}^3 IC displacement"
            gt = Grid(top, args.box)
            cache = {top: ic_top}

            def get_ic(N):
                if N not in cache:
                    g = Grid(N, args.box)
                    cache[N] = restrict_spectral(ic_top, gt, g, cube_window(gt, g.kny), args.offset)
                return cache[N]
        else:
            get_ic = lambda N: load(args.ic, N)

    results = {}
    for Nc, Nf in zip(levels[:-1], levels[1:]):
        assert Nf == 2 * Nc, "levels must be successive 2x refinements"
        tag = f"{Nc}to{Nf}"
        print(f"\n=== transition {tag} ===")
        dis_c, dis_f = get_dis(Nc), get_dis(Nf)
        ic_f, ic_c = get_ic(Nf), get_ic(Nc)
        if ic_f is None:
            print("no IC for the fine level; skipping (b) -> using the fine field itself as a placeholder")
            ic_f = dis_f
        results[tag] = analyse_transition(Nc, Nf, args.box, dis_c, dis_f, ic_f, ic_c,
                                          args.offset, args.alpha, args.out, tag, args.nbins, window=args.window,
                                          growth=args.growth)
        if args.vel and not args.selftest:
            vc, vf = load(args.vel, Nc), load(args.vel, Nf)
            if vc is not None and vf is not None:
                results[tag + "_vel"] = analyse_transition(Nc, Nf, args.box, vc, vf, vf, None,
                                                           args.offset, args.alpha, args.out, tag + "_vel", args.nbins, window=args.window)
    try:
        import subprocess
        results["_git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        results["_git_commit"] = None
    results["_argv"] = sys.argv
    with open(os.path.join(args.out, "phase0_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    plot_all({k: v for k, v in results.items()
              if not k.endswith("_vel") and not k.startswith("_")}, args.out)
    print(f"done in {time.time()-t0:.1f}s -> {args.out}/phase0_results.json")


if __name__ == "__main__":
    main()
