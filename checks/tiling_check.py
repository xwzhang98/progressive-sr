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


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def check_augment_equivalence():
    """3. GPU/crop augmentation == numpy augmentation of the raw fields before make() (64->128 level of PSC set 0, CPU)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import octave_flow_toy as oft
    torch.set_default_dtype(torch.float32)
    DIS, IC = "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy"
    sc = oft.Scaffold(64, 128, 100000.0, 0.5, 1.0, torch.device("cpu"), window="cube")
    it = oft.load_real(DIS, IC, 64, 128, [0])[0]
    worst = 0.0
    for seed in range(6):
        rA = np.random.default_rng(seed); rB = np.random.default_rng(seed)
        dc, df, icf = oft.augment([it["dis_c"], it["dis_f"], it["ic_f"]], rA, 0.5)
        bA = oft.Batcher(sc, [], rA, augment_on=False, growth=76.7439, octave_transverse=True)
        xA = bA.make([dict(dis_c=dc, dis_f=df, ic_f=icf)], eta="true")
        JA, adjA = em.jacobian_and_adjugate(xA[2])
        xR = bA.make([it], eta="true")
        JR, adjR = em.jacobian_and_adjugate(xR[2])
        (x0, x1, Pc), (D, adj), (J,) = tiling.augment_crops(rB, vecs=xR[:3], tens=(xR[3].reshape(1, 3, 3, 128, 128, 128),
                                                                                   adjR.permute(0, 4, 5, 1, 2, 3)), scals=(JR,))
        pairs = (("x0", x0, xA[0]), ("x1", x1, xA[1]), ("Pc", Pc, xA[2]), ("D", D.reshape(1, 9, 128, 128, 128), xA[3]),
                 ("J", J, JA), ("adj", adj.permute(0, 3, 4, 5, 1, 2), adjA))
        errs = {nm: float((a - b).abs().max() / b.abs().max()) for nm, a, b in pairs}
        worst = max(worst, max(errs.values()))
        print(f"3  seed {seed}: " + "  ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    print(f"   worst max rel diff {worst:.1e} (float32 FFT round-off expected)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "augment":
    check_augment_equivalence()


def check_cropsource():
    """4. tiling.CropSource == full-box Batcher.make(eta='true') + crop_periodic (128->256 level of set 14, CPU)."""
    import octave_flow_toy as oft
    torch.set_default_dtype(torch.float32)
    DIS, IC = "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy"
    sc = oft.Scaffold(128, 256, 100000.0, 0.5, 1.0, torch.device("cpu"), window="cube")
    it = oft.load_real(DIS, IC, 128, 256, [14])[0]
    b = oft.Batcher(sc, [], np.random.default_rng(0), augment_on=False, growth=76.7439, octave_transverse=True)
    full = b.make([it], eta="true")
    lin = sc.band(torch.from_numpy(np.asarray(it["ic_f"], dtype=np.float32)[None]) * 76.7439, "high")[0].numpy()
    src = tiling.CropSource(sc, [it["dis_c"]], [it["dis_f"]], [lin])
    for o in ((0, 0, 0), (200, 17, 250), (130, 255, 64)):
        got = src.sample(0, o, 128)
        errs = [float((g - tiling.crop_periodic(f, o, 128)).abs().max() / f.abs().max()) for g, f in zip(got, full)]
        print(f"4  crop at {o}: max rel diff x0 {errs[0]:.1e}  x1 {errs[1]:.1e}  Pc {errs[2]:.1e}  D {errs[3]:.1e}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cropsource":
    check_cropsource()


def check_jac():
    """5. tiling.jac_adj_from_D(scaffold D channels) == eulerian_metric.jacobian_and_adjugate(Pc / h) (64->128, set 14, CPU)."""
    import octave_flow_toy as oft
    torch.set_default_dtype(torch.float32)
    DIS, IC = "data/psc/dmo-{N}/set{seed}/PART_009/disp.npy", "data/psc/dmo-{N}/set{seed}/IC/disp.npy"
    sc = oft.Scaffold(64, 128, 100000.0, 0.5, 1.0, torch.device("cpu"), window="cube")
    it = oft.load_real(DIS, IC, 64, 128, [14])[0]
    x0, x1, Pc, D = oft.Batcher(sc, [], np.random.default_rng(0), augment_on=False, growth=76.7439, octave_transverse=True).make([it], eta="true")
    J1, adj1 = em.jacobian_and_adjugate(Pc)
    J2, adj2 = tiling.jac_adj_from_D(D)
    print(f"5  J max rel diff {float((J2 - J1).abs().max() / J1.abs().max()):.1e}, adj {float((adj2 - adj1).abs().max() / adj1.abs().max()):.1e}")
    net = LocalNet().float()
    xin = torch.randn(1, 12, 128, 128, 128)
    with torch.no_grad():
        e = float((tiling.TiledNet(net, 64, 32)(xin, None, None) - net(xin, None, None)).abs().max())
    print(f"5  TiledNet (index_select tiles) vs full box, local net: max abs diff {e:.1e}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "jac":
    check_jac()
