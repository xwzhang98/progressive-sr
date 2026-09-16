"""Shell-wise offset diagnostic: r(k) and the mean cross-spectrum phase between a restricted
fine field and the native coarse field, for offset 0 and 0.5. Wrong offset -> phase grows ~ k h_f/2
per axis and r(k) falls toward k_Ny,c; right offset -> phase ~0 and r(k) limited only by GenIC folding."""
import sys, numpy as np
sys.path.insert(0, "/hildafs/projects/phy200018p/xzhangn/progressive_sr/progressive-sr")
from phase0_octaves import Grid, restrict_spectral, cube_window
D = "/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology"
BOX = 100000.0; s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
def load(N, snap): return np.asarray(np.load(f"{D}/dmo-{N}/set{s}/{snap}/disp.npy", mmap_mode="r"), dtype=np.float32)
def shells(a, b, N):
    kf = 2*np.pi/BOX; kx = np.fft.fftfreq(N, 1/N)*kf; kz = np.fft.rfftfreq(N, 1/N)*kf
    K = np.sqrt(kx[:,None,None]**2 + kx[None,:,None]**2 + kz[None,None,:]**2)
    kny = np.pi*N/BOX; edges = np.linspace(0, kny, 9)
    A = np.fft.rfftn(a, axes=(1,2,3)); B = np.fft.rfftn(b, axes=(1,2,3))
    idx = np.digitize(K, edges) - 1
    out = []
    for i in range(8):
        m = (idx == i)
        if m.sum() == 0: continue
        cab = (A[:, m]*np.conj(B[:, m])).sum(); pa = (np.abs(A[:, m])**2).sum(); pb = (np.abs(B[:, m])**2).sum()
        # axis-mode phase: modes with only kx != 0 in this shell
        ax = m & (kx[None,:,None]==0) & (kz[None,None,:]==0)
        cax = (A[:, ax]*np.conj(B[:, ax])).sum() if ax.sum() else 0
        out.append((0.5*(edges[i]+edges[i+1])/kny, cab.real/np.sqrt(pa*pb), np.degrees(np.angle(cab)), np.degrees(np.angle(cax)) if ax.sum() else np.nan, pa/pb))
    return out
def report(tag, a, b, N):
    print(f"  {tag}: k/kNy,c   r(k)   phase_shell[deg]  phase_x-axis[deg]  P_R/P_c")
    for k, r, ph, phx, pr in shells(a, b, N): print(f"        {k:5.2f}  {r:6.4f}  {ph:8.2f}  {phx:8.2f}  {pr:6.3f}")
gc = Grid(64, BOX)
ic64 = load(64, "IC"); ic512 = load(512, "IC"); gt = Grid(512, BOX)
print(f"set{s} (a) IC: R_cube[512->64] vs native 64 IC")
for off in (0.0, 0.5):
    report(f"offset {off}", restrict_spectral(ic512, gt, gc, cube_window(gt, gc.kny), off), ic64, 64)
del ic512
d64 = load(64, "PART_009"); d128 = load(128, "PART_009"); g128 = Grid(128, BOX)
print(f"set{s} (b) z=0: R_cube[128->64] vs native 64 run")
for off in (0.0, 0.5):
    report(f"offset {off}", restrict_spectral(d128, g128, gc, cube_window(g128, gc.kny), off), d64, 64)
