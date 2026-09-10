"""Eulerian-aware loss metric and inputs for the octave flow (notes: eulerian_power.pdf).

Everything here is a function of the *current state* (the coarse field, or the interpolant
x_t) and never of the target x_1, so using it as the quadratic form of the flow-matching loss
leaves the exact minimiser -- the marginal velocity u_t = E[x_1 - x_0 | x_t] -- unchanged
(eulerian_power.pdf, Sec. 5.1).  It changes how a finite network spends its capacity: the
first-order density error of a displacement error e is  -div[ rho e ]  evaluated at the
Eulerian positions, so errors that do not move mass between Eulerian cells (relabelling of
particles that sit on top of each other, solenoidal momentum) are in the kernel of this
form and are not penalised.

Conventions match octave_flow_toy.py: fields are (B, 3, N, N, N) displacements in units of
the fine grid spacing h_f, Lagrangian positions q = (i + offset) h_f, periodic box.

    cic_deposit(values, pos, N)      -> trilinear (CIC) deposit of `values` at `pos` (grid units)
    pullback(field, pos)             -> trilinear interpolation of an Eulerian grid field at `pos`
    eulerian_positions(psi, offset)  -> q + psi in grid units, wrapped
    cic_gradient_deposit(e, pos, N)  -> exact first-order CIC density change under e (linear in e)
    eulerian_form(e, pos, ...)       -> mean_k W(k) |h_e(k)|^2, the quadratic form
    eulerian_inputs(psi_c_over_h, offset) -> log(1 + delta_c) of the coarse field, pulled back to q
    jacobian_form(e, psi_state, p, eps)   -> Lagrangian form mean |tr[adj(A) de]|^2/(J^2+eps^2)^{p/2}, A = I + dPsi_state/dq

Self-test:  python eulerian_metric.py
"""
import numpy as np
import torch


# --------------------------------------------------------------------------- CIC (periodic)
def _corners(pos, N):
    """pos: (B, 3, ...) in grid units. Returns (i0, frac) with i0 int64 wrapped, frac in [0, 1)."""
    x = torch.remainder(pos, N)
    i0 = torch.floor(x).to(torch.int64)
    f = x - i0.to(x.dtype)
    return i0, f


def cic_deposit(values, pos, N):
    """Deposit values (B, C, N, N, N) at Eulerian positions pos (B, 3, N, N, N) [grid units]
    onto a periodic N^3 grid with trilinear (CIC) weights. Differentiable in `values` and `pos`.
    Depositing values = 1 gives the CIC number density (mean 1 per cell)."""
    B, C = values.shape[:2]
    i0, f = _corners(pos, N)
    out = values.new_zeros((B, C, N * N * N))
    vals = values.reshape(B, C, -1)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        ix = torch.remainder(i0[:, 0] + dx, N)
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            iy = torch.remainder(i0[:, 1] + dy, N)
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                iz = torch.remainder(i0[:, 2] + dz, N)
                w = (wx * wy * wz).reshape(B, 1, -1)
                idx = ((ix * N + iy) * N + iz).reshape(B, 1, -1).expand(B, C, -1)
                out = out.scatter_add(2, idx, vals * w)
    return out.reshape(B, C, N, N, N)


def cic_gradient_deposit(e, pos, N):
    """Exact first-order change of the CIC number density under a displacement e (grid units):
        h(x_g) = d/d eps  rho[pos + eps e](x_g) |_{eps=0}  =  sum_p e_p . grad_{x_p} W_cic(x_g - x_p)
    Linear in e (so |h|^2 is a quadratic form), and exact for the CIC estimator up to the set of
    measure zero where a particle sits on a cell face.  Returns (B, 1, N, N, N) with mean 0."""
    B = e.shape[0]
    i0, f = _corners(pos, N)
    out = e.new_zeros((B, N * N * N))
    ev = e.reshape(B, 3, -1)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]; gx = 1.0 if dx else -1.0
        ix = torch.remainder(i0[:, 0] + dx, N)
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]; gy = 1.0 if dy else -1.0
            iy = torch.remainder(i0[:, 1] + dy, N)
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]; gz = 1.0 if dz else -1.0
                iz = torch.remainder(i0[:, 2] + dz, N)
                # d(wx wy wz)/dx_p = gx wy wz  etc.  (weights are linear in the fractional position)
                val = (ev[:, 0] * (gx * wy * wz).reshape(B, -1) + ev[:, 1] * (gy * wx * wz).reshape(B, -1)
                       + ev[:, 2] * (gz * wx * wy).reshape(B, -1))
                idx = ((ix * N + iy) * N + iz).reshape(B, -1)
                out = out.scatter_add(1, idx, val)
    return out.reshape(B, 1, N, N, N)


def pullback(field, pos):
    """Trilinear interpolation of an Eulerian grid field (B, C, N, N, N) at positions pos
    (B, 3, N, N, N) [grid units, periodic]. Returns (B, C, N, N, N) on the Lagrangian grid."""
    B, C, N = field.shape[0], field.shape[1], field.shape[-1]
    i0, f = _corners(pos, N)
    flat = field.reshape(B, C, -1)
    out = torch.zeros_like(flat)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        ix = torch.remainder(i0[:, 0] + dx, N)
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            iy = torch.remainder(i0[:, 1] + dy, N)
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                iz = torch.remainder(i0[:, 2] + dz, N)
                w = (wx * wy * wz).reshape(B, 1, -1)
                idx = ((ix * N + iy) * N + iz).reshape(B, 1, -1).expand(B, C, -1)
                out = out + torch.gather(flat, 2, idx) * w
    return out.reshape(B, C, N, N, N)


def eulerian_positions(psi_over_h, offset):
    """x = q + Psi in grid units for a displacement field (B, 3, N, N, N) given in units of h."""
    N = psi_over_h.shape[-1]
    q = torch.arange(N, device=psi_over_h.device, dtype=psi_over_h.dtype) + offset
    Q = torch.stack(torch.meshgrid(q, q, q, indexing="ij"))[None]
    return Q + psi_over_h


# --------------------------------------------------------------------------- the quadratic form
def _kvec(N, device, dtype):
    k1 = 2 * np.pi * np.fft.fftfreq(N)            # cycles per cell * 2pi  (h = 1)
    kr = 2 * np.pi * np.fft.rfftfreq(N)
    kx, ky, kz = np.meshgrid(k1, k1, kr, indexing="ij")
    return [torch.as_tensor(a, device=device, dtype=dtype) for a in (kx, ky, kz)]


def eulerian_form(e, pos, weight_k=None, fft_device=None):
    """The Eulerian quadratic form of a displacement error field:
        Q_E(e) = mean_k W(k) |h_e(k)|^2,   h_e = first-order CIC density change under e at `pos`
    (= -div[rho e] in the continuum).  e: (B, 3, N, N, N) in units of h; pos: (B, 3, N, N, N)
    Eulerian positions in grid units taken from the coarse field or the interpolant -- NEVER
    from the target, otherwise the flow-matching minimiser changes (notes, Sec. 5.1).
    Normalised so that Q_E(e) = mean over cells of h_e^2 (Parseval) when weight_k is None."""
    N = e.shape[-1]
    h = cic_gradient_deposit(e, pos, N) / 1.0            # number-density change per cell (mean density 1)
    if weight_k is None:
        return (h ** 2).mean()
    dev = fft_device or h.device
    H = torch.fft.rfftn(h.to(dev)[:, 0], dim=(-3, -2, -1))
    kx, ky, kz = _kvec(N, dev, H.real.dtype)
    w = weight_k(torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2))
    # rfft half-plane: weight the kz=0 (and Nyquist) planes once, the rest twice, then Parseval
    mult = torch.full_like(w, 2.0); mult[..., 0] = 1.0
    if N % 2 == 0: mult[..., -1] = 1.0
    return ((H.real ** 2 + H.imag ** 2) * w * mult).sum(dim=(-3, -2, -1)).mean() / float(N ** 3) ** 2


def eulerian_inputs(psi_c_over_h, offset):
    """Extra conditioning channel: log(1 + delta_c) of the *coarse* field, evaluated at the
    Eulerian position of each Lagrangian grid point (a function of the coarse state only)."""
    N = psi_c_over_h.shape[-1]
    pos = eulerian_positions(psi_c_over_h, offset)
    ones = psi_c_over_h.new_ones((psi_c_over_h.shape[0], 1, N, N, N))
    rho = cic_deposit(ones, pos, N)
    return pullback(torch.log1p(rho.clamp_min(0.0)), pos)


# --------------------------------------------------------------------------- Lagrangian (Jacobian) form
def spectral_gradient(field, fft_device=None):
    """d field_j / d q_i on the periodic grid (grid units, h = 1) by FFT. field: (B, C, N, N, N)
    -> (B, 3, C, N, N, N) with [.., i, j] = d_i field_j."""
    N = field.shape[-1]
    dev = fft_device or field.device
    F = torch.fft.rfftn(field.to(dev), dim=(-3, -2, -1))
    kx, ky, kz = _kvec(N, dev, F.real.dtype)
    out = []
    for ki in (kx, ky, kz):
        out.append(torch.fft.irfftn(1j * ki * F, s=(N, N, N), dim=(-3, -2, -1)))
    return torch.stack(out, dim=1).to(field.device)


def jacobian_and_adjugate(psi_state_over_h, fft_device=None):
    """A = I + dPsi/dq (B, N, N, N, 3, 3), J = det A, adj(A) = J A^{-1} (finite at J = 0)."""
    D = spectral_gradient(psi_state_over_h, fft_device)                     # (B, 3, 3, N, N, N), D[i, j] = d_i Psi_j
    A = D.permute(0, 3, 4, 5, 1, 2) + torch.eye(3, device=D.device, dtype=D.dtype)
    J = torch.linalg.det(A)
    # adjugate by cofactors (no division): adj(A)_{ij} = cofactor_{ji}
    a = A
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
    adj = cof.transpose(-1, -2)
    return J, adj


def jacobian_form(e, psi_state_over_h, p=2.0, eps=0.1, fft_device=None, return_parts=False):
    """The Lagrangian (Jacobian) quadratic form of a displacement error e, notes Sec. 5.7:
        dJ = tr[ adj(A) de/dq ]   (Jacobi's formula; first-order change of J = det(I + dPsi/dq)),
        dJ / J = tr[ A^{-1} de/dq ] = div_x e   (Eulerian divergence of the error, Lagrangian frame),
        Q_J(e) = mean_q  |dJ|^2 / (J^2 + eps^2)^{p/2}.
    p = 0: absolute Jacobian error; p = 2: log|J| (relative density) error; p = 3: Eulerian-like
    (density change per Eulerian volume).  A and J come from the STATE (coarse field or interpolant),
    never from the target, so this is admissible for flow matching (Sec. 5.1).  Local, no deposit."""
    J, adj = jacobian_and_adjugate(psi_state_over_h, fft_device)
    dE = spectral_gradient(e, fft_device).permute(0, 3, 4, 5, 1, 2)         # (B, N, N, N, 3, 3), [i, j] = d_i e_j
    dJ = (adj * dE.transpose(-1, -2)).sum(dim=(-1, -2))                       # tr[adj(A) dE]
    w = (J ** 2 + eps ** 2) ** (-p / 2)
    Q = (dJ ** 2 * w).mean()
    return (Q, J, dJ) if return_parts else Q


# --------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    N, B = 32, 1
    # a strongly nonlinear Zel'dovich-like field: multi-stream, so the kernel test is meaningful
    k1 = 2 * np.pi * np.fft.fftfreq(N); kr = 2 * np.pi * np.fft.rfftfreq(N)
    kx, ky, kz = np.meshgrid(k1, k1, kr, indexing="ij"); kk = np.sqrt(kx**2 + ky**2 + kz**2)
    rng = np.random.default_rng(0)
    W = np.fft.rfftn(rng.standard_normal((N, N, N))) * np.where(kk > 0, (kk / (np.pi / 2)) ** -0.75, 0)
    k2 = np.where(kk > 0, kk ** 2, 1.0)
    psi = np.stack([np.fft.irfftn(1j * ki * W / k2, s=(N, N, N)) for ki in (kx, ky, kz)])
    psi = torch.tensor(psi / psi.std() * 2.0, dtype=torch.float64)[None]      # rms 2 h -> multi-stream
    pos = eulerian_positions(psi, 0.0)
    ones = torch.ones((B, 1, N, N, N), dtype=torch.float64)
    rho = cic_deposit(ones, pos, N)
    print(f"mass conservation: sum rho / N^3 - 1 = {float(rho.sum() / N**3 - 1):.2e}, max rho = {float(rho.max()):.1f}")

    # pullback of a smooth Eulerian field at the Lagrangian positions equals that field (pos = q)
    fld = torch.cos(2 * np.pi * torch.arange(N, dtype=torch.float64) / N)[None, None, :, None, None].expand(B, 1, N, N, N)
    q = eulerian_positions(torch.zeros_like(psi), 0.0)
    print(f"pullback identity at pos=q: max |err| = {float((pullback(fld, q) - fld).abs().max()):.2e}")

    # first-order identity: (rho[x + eps e] - rho[x]) / eps  ==  cic_gradient_deposit(e)  (exact for CIC)
    e = torch.randn_like(psi)
    eps = 1e-4
    rho1 = cic_deposit(ones, pos + eps * e, N)
    fd = (rho1 - rho)[:, 0] / eps
    h = cic_gradient_deposit(e, pos, N)[:, 0]
    print(f"first-order identity (finite difference vs gradient deposit): rel. err = {float((fd - h).norm() / h.norm()):.2e}")

    # kernel: exchanging two particles that (nearly) coincide in Eulerian space changes the density
    # only at second order in their separation, so Q_E(e_swap)/|e_swap|^2 -> 0 as the pairs get closer
    Q0 = eulerian_form(e, pos); L0 = (e ** 2).mean()
    cell = torch.floor(torch.remainder(pos, N)).to(torch.int64)
    flat = ((cell[:, 0] * N + cell[:, 1]) * N + cell[:, 2]).reshape(-1)
    order = torch.argsort(flat); fs = flat[order]
    same = torch.nonzero(fs[1:] == fs[:-1]).reshape(-1)
    ia, ib = order[same], order[same + 1]
    P = pos.reshape(3, -1)
    sep = (P[:, ia] - P[:, ib]).norm(dim=0)
    for smax in (0.5, 0.2, 0.1):
        m = sep < smax
        e_swap = torch.zeros((3, N ** 3), dtype=psi.dtype)
        e_swap[:, ia[m]] = P[:, ib[m]] - P[:, ia[m]]; e_swap[:, ib[m]] = P[:, ia[m]] - P[:, ib[m]]
        e_swap = e_swap.reshape_as(psi)
        Qs = eulerian_form(e_swap, pos); Ls = (e_swap ** 2).mean()
        print(f"kernel test, pairs closer than {smax} cell ({int(m.sum())} pairs): "
              f"[Q_E/|e|^2](swap) / [Q_E/|e|^2](random) = {float((Qs / Ls) / (Q0 / L0)):.3f}")


    # Jacobi identity: J(psi + eps e) - J(psi)  ==  eps * tr[adj(A) de]   (exact to O(eps^2))
    Q, J, dJ = jacobian_form(e, psi, p=0.0, return_parts=True)
    J1, _ = jacobian_and_adjugate(psi + 1e-4 * e)
    fd = (J1 - J) / 1e-4
    print(f"Jacobi identity (finite difference vs tr[adj(A) de]): rel. err = {float((fd - dJ).norm() / dJ.norm()):.2e};"
          f"  multi-stream fraction of the test field = {float((J < 0).double().mean()):.3f}")
    # shrinkage: inside a patch where x = x_h + a u, J scales as a^3 -> dJ/da at a=1 is 3J for e = psi (whole field)
    Qs, Js, dJs = jacobian_form(psi, psi, p=0.0, return_parts=True)
    # d/da det(I + a D) at a=1 = tr[adj(A) D]; compare with 3J - (terms from I): just report the ratio to 3J-2*tr(adj)... skip
    print(f"jacobian_form values for a random error: p=0 {float(jacobian_form(e, psi, p=0.0)):.3e}, p=2 {float(jacobian_form(e, psi, p=2.0)):.3e}, p=3 {float(jacobian_form(e, psi, p=3.0)):.3e}")

    # inputs channel
    inp = eulerian_inputs(psi, 0.0)
    print(f"eulerian_inputs: log(1+delta_c) pulled back, min/mean/max = {float(inp.min()):.2f}/{float(inp.mean()):.2f}/{float(inp.max()):.2f}")


if __name__ == "__main__":
    _selftest()
