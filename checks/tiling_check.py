#!/usr/bin/env python3
"""Checks for tiling.py and the crop-aware Q_J of runs/multi_resflow_train.py (2026-09-18).

1. Geometry: with a LOCAL translation-equivariant network (one circular 3x3x3 conv), TiledNet must reproduce the full-box
   output exactly (the tile margin exceeds the receptive field), and crop_periodic must commute with it in the interior.
2. Q_J on a crop: the tapered spectral gradient on a periodic crop must reproduce the full-box Q_J integrand in the
   crop interior (J/adj from the full-box state, as in training).
   python checks/tiling_check.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eulerian_metric as em   # noqa: E402
import tiling                  # noqa: E402


class LocalNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c = nn.Conv3d(12, 3, 3, padding=1, padding_mode="circular")

    def forward(self, x, t, s):
        return self.c(x)


def smooth_field(N, ch, rng, slope=-1.5):
    k1 = 2 * np.pi * np.fft.fftfreq(N); kr = 2 * np.pi * np.fft.rfftfreq(N)
    kx, ky, kz = np.meshgrid(k1, k1, kr, indexing="ij"); kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    out = []
    for _ in range(ch):
        W = np.fft.rfftn(rng.standard_normal((N, N, N))) * np.where(kk > 0, kk ** (slope / 2), 0)
        out.append(np.fft.irfftn(W, s=(N, N, N)))
    a = np.stack(out)
    return torch.tensor(a / a.std(), dtype=torch.float64)[None]


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(1)
    N, C, S, H = 64, 32, 16, 6
    # ---- 1. geometry ----
    net = LocalNet().double()
    x = smooth_field(N, 12, rng)
    t = torch.zeros(1); s = torch.zeros(1)
    with torch.no_grad():
        full = net(x, t, s)
        tiled = tiling.TiledNet(net, C, S)(x, t, s)
        o = (50, 7, 33)
        cr = net(tiling.crop_periodic(x, o, C), t, s)
        ref = tiling.crop_periodic(full, o, C)
    e1 = float((tiled - full).abs().max() / full.abs().max())
    e2 = float((tiling.interior(cr, 1) - tiling.interior(ref, 1)).abs().max() / ref.abs().max())
    e3 = float((cr - ref).abs().max() / ref.abs().max())
    print(f"1a TiledNet vs full box (local net):            max rel diff {e1:.2e}  (expect ~1e-16)")
    print(f"1b crop output vs full-box crop, interior d>=1: max rel diff {e2:.2e}  (expect ~1e-16)")
    print(f"1c same including the crop faces:               max rel diff {e3:.2e}  (expect O(1): wrap at the faces)")
    # ---- 2. Q_J on a crop ----
    psi = smooth_field(N, 3, rng) * 0.6            # state, units of h: mild multi-streaming
    e = smooth_field(N, 3, rng, slope=-0.5) * 0.2  # a rough error field
    Q, J, dJ = em.jacobian_form(e, psi, p=2.0, eps=0.1, return_parts=True)
    integ = dJ ** 2 * (J ** 2 + 0.01) ** -1.0      # (B, N, N, N) full-box integrand
    Jf, adjf = em.jacobian_and_adjugate(psi)
    worst = 0.0
    for trial in range(4):
        o = tuple(int(v) for v in rng.integers(0, N, 3))
        ec = tiling.crop_periodic(e, o, C)
        Jc = tiling.crop_periodic(Jf[:, None], o, C)[:, 0]
        adjc = tiling.crop_periodic(adjf.reshape(1, N, N, N, 9).permute(0, 4, 1, 2, 3), o, C).permute(0, 2, 3, 4, 1).reshape(1, C, C, C, 3, 3)
        dE = em.spectral_gradient(ec * tiling.taper(C, H, ec.device, ec.dtype)).permute(0, 3, 4, 5, 1, 2)
        dJc = (adjc * dE.transpose(-1, -2)).sum(dim=(-1, -2))
        integ_c = tiling.interior(dJc ** 2 * (Jc ** 2 + 0.01) ** -1.0, H)
        integ_ref = tiling.interior(tiling.crop_periodic(integ[:, None], o, C)[:, 0], H)
        rel_mean = float(abs(integ_c.mean() - integ_ref.mean()) / integ_ref.mean())
        rel_pt = float(((integ_c - integ_ref) ** 2).mean().sqrt() / (integ_ref ** 2).mean().sqrt())
        worst = max(worst, rel_mean)
        print(f"2  crop at {o}: interior Q_J mean rel diff {rel_mean:.2e}, pointwise rel rms {rel_pt:.2e}")
    print(f"   worst interior-mean deviation {worst:.2e} (small = crop Q_J reproduces the full-box form)")


if __name__ == "__main__":
    main()
