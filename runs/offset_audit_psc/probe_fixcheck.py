"""After the multi-octave phase fix: (1) direct R[512->64] == chained R[512->256->128->64] at offset 0.5;
(2) RP = I for m = 4, 8 at offset 0.5; (3) z-axis cross-phase of R[Nf->64] vs the native 64 field
for Nf = 128/256/512 (z=0) and 512 (IC) -- should now be ~0 like the one-octave case."""
import sys, numpy as np
sys.path.insert(0, "/hildafs/projects/phy200018p/xzhangn/progressive_sr/progressive-sr")
from phase0_octaves import Grid, restrict_spectral, prolong_spectral, cube_window
D = "/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology"
BOX = 100000.0; s = 0; OFF = 0.5
def load(N, snap): return np.asarray(np.load(f"{D}/dmo-{N}/set{s}/{snap}/disp.npy", mmap_mode="r"), dtype=np.float32)
G = {N: Grid(N, BOX) for N in (64, 128, 256, 512)}
def R(f, Nf, Nc): return restrict_spectral(f, G[Nf], G[Nc], cube_window(G[Nf], G[Nc].kny), OFF)
N = 64; kf = 2*np.pi/BOX; kx = np.fft.fftfreq(N, 1/N)*kf; kz = np.fft.rfftfreq(N, 1/N)*kf; kny = np.pi*N/BOX
KX = kx[:,None,None]+0*kx[None,:,None]+0*kz[None,None,:]; KY = 0*KX + kx[None,:,None]; KZ = 0*KX + kz[None,None,:]
def zphase(a, b):
    A = np.fft.rfftn(a, axes=(1,2,3)); B = np.fft.rfftn(b, axes=(1,2,3)); edges = np.linspace(0, kny, 9); out = []
    for i in range(8):
        m = (KX==0)&(KY==0)&(KZ > edges[i])&(KZ <= edges[i+1]); c = (A[:, m]*np.conj(B[:, m])).sum()
        sh = (np.sqrt(KX**2+KY**2+KZ**2) > edges[i]) & (np.sqrt(KX**2+KY**2+KZ**2) <= edges[i+1])
        cs = (A[:, sh]*np.conj(B[:, sh])).sum(); r = cs.real/np.sqrt((np.abs(A[:, sh])**2).sum()*(np.abs(B[:, sh])**2).sum())
        out.append(f"{np.degrees(np.angle(c)):6.1f}/{r:.3f}")
    return "  ".join(out)
d64 = load(64, "PART_009"); d512 = load(512, "PART_009")
direct = R(d512, 512, 64); chained = R(R(R(d512, 512, 256), 256, 128), 128, 64)
print(f"(1) |direct - chained| / |chained| = {np.sqrt(np.mean((direct-chained)**2))/np.sqrt(np.mean(chained**2)):.3e}")
for Nc in (128, 64):
    x = R(d512, 512, Nc); y = restrict_spectral(prolong_spectral(x, G[Nc], G[512], OFF), G[512], G[Nc], cube_window(G[512], G[Nc].kny), OFF)
    print(f"(2) RP = I at m={512//Nc}, offset {OFF}: |RPx - x|/|x| = {np.sqrt(np.mean((y-x)**2))/np.sqrt(np.mean(x**2)):.3e}")
print("(3) z-axis phase[deg]/shell r(k) per k-bin (0..kNy,64 in 8 bins), offset 0.5, vs native 64:")
print(f"    z=0 R[512->64]: {zphase(direct, d64)}")
del d512
for Nf in (128, 256):
    print(f"    z=0 R[{Nf}->64]: {zphase(R(load(Nf, 'PART_009'), Nf, 64), d64)}")
print(f"    IC  R[512->64]: {zphase(R(load(512, 'IC'), 512, 64), load(64, 'IC'))}")
