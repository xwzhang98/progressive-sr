"""Check of the smearing formula on a strongly multi-stream Zel'dovich field.

P_delta[Psi + n](k) = e^{-k^2 s^2} P_delta[Psi](k) + (1 - e^{-k^2 s^2}) h^3     (white Gaussian n, per-component var s^2)

and the band-limited (octave) version of n with the same s^2, which damps less at k < k_n.
"""
import numpy as np, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy.fft import rfftn, irfftn

N = 128; L = 100.0; h = L / N
rng = np.random.default_rng(1)

def kgrid(N, L):
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kr = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k1, k1, kr, indexing="ij")
    return kx, ky, kz, np.sqrt(kx**2 + ky**2 + kz**2)

kx, ky, kz, kk = kgrid(N, L)
kny_c = np.pi / (2 * h)   # coarse Nyquist for a 64^3 "coarse" grid
kny_f = np.pi / h

# Gaussian delta_lin with P ~ k^-1.5, amplitude set by rms displacement
white = rng.standard_normal((N, N, N))
W = rfftn(white)
Pk = np.where(kk > 0, (kk / kny_c) ** -1.5, 0.0)
dk = W * np.sqrt(Pk)
k2 = np.where(kk > 0, kk**2, 1.0)
psi = np.stack([irfftn(1j * ki * dk / k2, s=(N, N, N)) for ki in (kx, ky, kz)])
target_rms = 1.4 * 2 * h    # rms displacement per component = 1.4 h_c -> strongly multi-stream
psi *= target_rms / psi.std()
# multistream fraction from det(I + dPsi/dq) (spectral gradient)
D = np.zeros((3, 3, N, N, N))
for i, ki in enumerate((kx, ky, kz)):
    for j in range(3):
        D[i, j] = irfftn(1j * ki * rfftn(psi[j]), s=(N, N, N))
J = np.eye(3)[:, :, None, None, None] + D
detJ = np.linalg.det(np.moveaxis(J, (0, 1), (-2, -1)))
print(f"rms Psi/h_c = {psi.std()/(2*h):.2f}, multistream fraction = {(detJ<0).mean():.3f}")

def cic(psi):
    q = np.arange(N) * h
    Q = np.stack(np.meshgrid(q, q, q, indexing="ij"))
    x = ((Q + psi) / h) % N
    i0 = np.floor(x).astype(np.int64); f = x - i0
    rho = np.zeros((N, N, N))
    for dx in (0, 1):
        wx = (1 - f[0]) if dx == 0 else f[0]
        for dy in (0, 1):
            wy = (1 - f[1]) if dy == 0 else f[1]
            for dz in (0, 1):
                wz = (1 - f[2]) if dz == 0 else f[2]
                flat = ((((i0[0] + dx) % N) * N + ((i0[1] + dy) % N)) * N + ((i0[2] + dz) % N)).ravel()
                rho += np.bincount(flat, weights=(wx * wy * wz).ravel(), minlength=N**3).reshape(N, N, N)
    return rho

edges = np.linspace(0, kny_f * np.sqrt(3), 60)
ibin = np.digitize(kk.ravel(), edges) - 1
def pk(field):
    F = rfftn(field - field.mean()) * (L / N) ** 3 / L**3   # |delta_k|^2 / V
    P = (np.abs(F) ** 2).ravel() * L**3
    num = np.bincount(ibin, weights=P, minlength=len(edges)); den = np.bincount(ibin, minlength=len(edges))
    return num[:len(edges)-1] / np.maximum(den[:len(edges)-1], 1)
kc = 0.5 * (edges[1:] + edges[:-1])

t0 = time.time(); rho0 = cic(psi); P0 = pk(rho0); print(f"cic+pk {time.time()-t0:.1f}s")

def octave_noise(s):
    """Gaussian, isotropic per component, cube-octave band [k_Ny,c, k_Ny,f] in each axis, per-component variance s^2 (units of h)."""
    n = np.stack([irfftn(rfftn(rng.standard_normal((N, N, N))) * (1 - ((np.abs(kx) <= kny_c) & (np.abs(ky) <= kny_c) & (np.abs(kz) <= kny_c))), s=(N, N, N)) for _ in range(3)])
    return n * (s * h) / n.std()

def white_noise(s):
    return rng.standard_normal((3, N, N, N)) * s * h

out = {"k": kc.tolist(), "kny_c": kny_c, "P0": P0.tolist(), "h3": h**3, "cases": {}}
for s in (0.3, 0.42, 0.6):
    for kind, gen in (("white", white_noise), ("octave", octave_noise)):
        n = gen(s)
        P = pk(cic(psi + n))
        pred = np.exp(-(kc * s * h) ** 2) * P0 + (1 - np.exp(-(kc * s * h) ** 2)) * h**3
        out["cases"][f"{kind}_s{s}"] = {"ratio": (P / P0).tolist(), "pred_white": (pred / P0).tolist()}
        sel = [np.argmin(np.abs(kc - kny_c * f)) for f in (0.5, 1.0, 1.5, 2.0)]
        print(f"{kind:6s} s={s:.2f} h_f: measured P/P0 at k/kNy,c=0.5,1,1.5,2: ", np.round((P / P0)[sel], 3),
              " white-formula:", np.round((pred / P0)[sel], 3))
json.dump(out, open(__file__.replace(".py", ".json"), "w"))
