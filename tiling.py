"""Overlap-tile helpers for levels that do not fit as full boxes (owner approved patch cropping, 2026-09-18).

The network (octave_flow_toy.UNet3D) uses circular 'same' padding and GroupNorm over the whole input volume, so its output
on a crop differs from its full-box output EVERYWHERE, not only near the crop faces (runs/crop_probe.py: 30-40% at the crop
centre, RUNLOG 2026-09-18 13:00). A level trained on crops must therefore be evaluated with the same crop geometry:

  training : random periodic C^3 crops of the full-box tensors (all spectral operations done on the full box first),
             loss on the interior (Chebyshev distance >= H from the crop faces);
  inference: TiledNet applies the network on C^3 tiles of the circularly padded full field and keeps each tile's
             central S^3 block, i.e. only voxels >= (C - S)/2 from the tile faces (choose (C - S)/2 >= H).
"""
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
        xp = F.pad(x, (m, m, m, m, m, m), mode="circular")
        out = None
        for a in range(0, N, S):
            for b in range(0, N, S):
                for c in range(0, N, S):
                    y = self.net(xp[:, :, a:a + C, b:b + C, c:c + C], t, s)
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
