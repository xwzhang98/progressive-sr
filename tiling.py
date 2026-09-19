"""Overlap-tile helpers for levels that do not fit as full boxes (owner approved patch cropping, 2026-09-18).

The network (octave_flow_toy.UNet3D) uses circular 'same' padding and GroupNorm over the whole input volume, so its output
on a crop differs from its full-box output EVERYWHERE, not only near the crop faces (runs/crop_probe.py: 30-40% at the crop
centre, RUNLOG 2026-09-18 13:00). A level trained on crops must therefore be evaluated with the same crop geometry:

  training : random periodic C^3 crops of the full-box tensors (all spectral operations done on the full box first),
             loss on the interior (Chebyshev distance >= H from the crop faces);
  inference: TiledNet applies the network on C^3 tiles of the circularly padded full field and keeps each tile's
             central S^3 block, i.e. only voxels >= (C - S)/2 from the tile faces (choose (C - S)/2 >= H).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_periodic(x, o, C):
    """(B, ch, N, N, N) -> (B, ch, C, C, C): the periodic block of the box starting at o = (i, j, k)."""
    x = torch.roll(x, shifts=(-int(o[0]), -int(o[1]), -int(o[2])), dims=(2, 3, 4))
    return x[:, :, :C, :C, :C]


def interior(x, H):
    """drop H cells on every face of the last three dimensions."""
    return x[..., H:-H, H:-H, H:-H] if H > 0 else x


def taper(C, H, device, dtype=torch.float32):
    """(C, C, C) window: ~0 on the crop faces, cos^2 ramp across the H halo cells, exactly 1 in the interior. Multiplying a
    non-periodic crop field by it makes its periodic continuation smooth, so spectral derivatives are exact in the interior
    up to the (small) leakage of the smooth ramp."""
    i = torch.arange(C, device=device, dtype=dtype)
    d = torch.minimum(i, C - 1 - i)
    w = torch.where(d >= H, torch.ones_like(d), torch.sin(0.5 * torch.pi * (d + 0.5) / max(H, 1)) ** 2)
    return w[:, None, None] * w[None, :, None] * w[None, None, :]


class TiledNet(nn.Module):
    """Apply `net(x, t, s)` tile by tile on a periodic field whose side N exceeds C: circularly pad by m = (C - S)/2, run
    the network on each C^3 tile, keep the central S^3 block. N must be a multiple of S. Fields with N <= C go through
    unchanged (the full-box levels keep their full-box behaviour)."""

    def __init__(self, net, C=128, S=64):
        super().__init__()
        assert (C - S) % 2 == 0 and S > 0
        self.net, self.C, self.S = net, C, S

    def forward(self, x, t, s):
        N = x.shape[-1]
        if N <= self.C:
            return self.net(x, t, s)
        C, S = self.C, self.S
        m = (C - S) // 2
        assert N % S == 0, f"box side {N} is not a multiple of the tile core {S}"
        # tiles are gathered with wrap-around indices (no padded copy of the full field: 9 GB at 512^3 with 12 channels)
        idx = {a: (torch.arange(C, device=x.device) + a - m) % N for a in range(0, N, S)}
        out = None
        for a in range(0, N, S):
            xa = x.index_select(2, idx[a])
            for b in range(0, N, S):
                xab = xa.index_select(3, idx[b])
                for c in range(0, N, S):
                    y = self.net(xab.index_select(4, idx[c]), t, s)
                    if out is None:
                        out = x.new_zeros((x.shape[0], y.shape[1], N, N, N), dtype=y.dtype)
                    out[:, :, a:a + S, b:b + S, c:c + S] = y[:, :, m:m + S, m:m + S, m:m + S]
        return out


def augment_crops(rng, vecs=(), tens=(), scals=()):
    """Cubic-group augmentation of crops ON THE DEVICE, the same group element and convention as octave_flow_toy.augment
    (offset != 0: a flip i -> N-1-i is the reflection x -> L - x). Because every spectral operation of the scaffold commutes
    with the cube group, augmenting the outputs (and crops) is equivalent to augmenting the raw fields before make() — it just
    avoids the numpy copies of full 256^3 boxes (checks/tiling_check.py verifies the equivalence).
      vecs : (B, 3, C, C, C) vector fields (displacements)          v'_c(x') = s_c v_{perm c}(x)
      tens : (B, 3, 3, C, C, C) rank-2 tensors (D_ij = d_i Psi_j, adj) T'_ij = s_i s_j T_{perm i, perm j}
      scals: (B, C, C, C) scalars (J)
    Draws the group element exactly like octave_flow_toy.augment: rng.permutation(3), then rng.integers(0, 2, 3)."""
    perm = [int(v) for v in rng.permutation(3)]
    flips = rng.integers(0, 2, size=3).astype(bool)

    def spatial(x, lead):
        x = x.permute(*range(lead), *[lead + a for a in perm])
        dims = [lead + a for a in range(3) if flips[a]]
        return torch.flip(x, dims) if dims else x

    def sgn(like):
        return torch.tensor([-1.0 if f else 1.0 for f in flips], device=like.device, dtype=like.dtype)

    out_v = [(spatial(v[:, perm], 2) * sgn(v).view(1, 3, 1, 1, 1)).contiguous() for v in vecs]
    out_t = [(spatial(T[:, perm][:, :, perm], 3) * (sgn(T).view(1, 3, 1, 1, 1, 1) * sgn(T).view(1, 1, 3, 1, 1, 1))).contiguous()
             for T in tens]
    out_s = [spatial(s_, 1).contiguous() for s_ in scals]
    return out_v, out_t, out_s


def crop_memmap(a, o, C):
    """Periodic C^3 crop starting at o of a (ch, N, N, N) array (typically an np.memmap), read with at most 8 basic-slice blocks
    so that only the needed rows are touched on disk. Returns a new float32 ndarray (ch, C, C, C)."""
    N = a.shape[-1]
    segs = []
    for oa in o:
        oa = int(oa) % N
        segs.append([(oa, oa + C)] if oa + C <= N else [(oa, N), (0, oa + C - N)])
    xs = []
    for (x0, x1) in segs[0]:
        ys = []
        for (y0, y1) in segs[1]:
            zs = [np.asarray(a[:, x0:x1, y0:y1, z0:z1], dtype=np.float32) for (z0, z1) in segs[2]]
            ys.append(np.concatenate(zs, axis=3))
        xs.append(np.concatenate(ys, axis=2))
    return np.concatenate(xs, axis=1)


class CropSource:
    """Training crops at a level whose full boxes are too large to prepare every step (256->512; owner 2026-09-19: memory-
    mapped reading). Equivalent to scaffold Batcher.make(eta="true") on the full box followed by crop_periodic (checked in
    checks/tiling_check.py crop), but: the coarse run is prolonged and differentiated on the full box ON THE DEVICE and each
    of the 9 D_ij components is cropped as soon as it is formed; the fine target and the IC octave (precomputed once per set
    by runs/precompute_octave.py, = scaffold.band(ic_f * growth, "high")) are memory-mapped and only the crop is read."""

    def __init__(self, sc, dis_c, dis_f, lin):
        self.sc, self.dis_c, self.dis_f, self.lin = sc, dis_c, dis_f, lin     # lists per training box

    def sample(self, i, o, C):
        sc = self.sc
        dev = sc.fdev
        dc = torch.from_numpy(np.ascontiguousarray(self.dis_c[i], dtype=np.float32)[None]).to(dev)
        Pc_full = sc.prolong(dc).to(dev)                                    # (1, 3, N, N, N), physical units
        Fx = torch.fft.rfftn(Pc_full, dim=(-3, -2, -1))
        Pc = crop_periodic(Pc_full, o, C)
        del Pc_full
        Ds = []
        for a in range(3):
            for b in range(3):
                g = torch.fft.irfftn(1j * sc.k[a] * Fx[:, b], s=(sc.Nf,) * 3, dim=(-3, -2, -1))
                Ds.append(crop_periodic(g[:, None], o, C)[:, 0])
                del g
        del Fx
        D = torch.stack(Ds, dim=1).to(sc.device)
        lin = torch.from_numpy(crop_memmap(self.lin[i], o, C)[None]).to(sc.device)
        df = torch.from_numpy(crop_memmap(self.dis_f[i], o, C)[None]).to(sc.device)
        Pc = Pc.to(sc.device)
        return (Pc + lin) / sc.hf, df / sc.hf, Pc / sc.hf, D


def jac_adj_from_D(D9):
    """J = det(A) and adj(A) with A = I + D, D9 (B, 9, ...) the scaffold's gradient channels (index 3 i + j = d_i Psi_j,
    dimensionless) -- the same A, J and cofactor adjugate as eulerian_metric.jacobian_and_adjugate, but pointwise, so it can be
    evaluated on a crop of full-box spectral gradients (checks/tiling_check.py jac). Returns J (B, ...) and adj (B, ..., 3, 3)."""
    B = D9.shape[0]
    a = D9.reshape(B, 3, 3, *D9.shape[2:]).movedim(1, -1).movedim(1, -1)          # (B, ..., 3, 3), a[..., i, j] = d_i Psi_j
    a = a + torch.eye(3, device=D9.device, dtype=D9.dtype)
    J = torch.linalg.det(a)
    cof = torch.stack([
        torch.stack([a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1],
                     a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2],
                     a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]], dim=-1),
        torch.stack([a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2],
                     a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0],
                     a[..., 0, 1] * a[..., 2, 0] - a[..., 0, 0] * a[..., 2, 1]], dim=-1),
        torch.stack([a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1],
                     a[..., 0, 2] * a[..., 1, 0] - a[..., 0, 0] * a[..., 1, 2],
                     a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]], dim=-1)], dim=-2)
    return J, cof.transpose(-1, -2)
