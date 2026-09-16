import sys, numpy as np
sys.path.insert(0, "/hildafs/projects/phy200018p/xzhangn/progressive_sr/progressive-sr")
from phase0_octaves import Grid, restrict_spectral, cube_window
D = "/hildafs/home/xzhangn/xzhangn/cosmo_sr/2-data/train/int_redshift_same_cosmology"
BOX = 100000.0; s = 0
def load(N, snap): return np.asarray(np.load(f"{D}/dmo-{N}/set{s}/{snap}/disp.npy", mmap_mode="r"), dtype=np.float32)
N = 64; kf = 2*np.pi/BOX; kx = np.fft.fftfreq(N, 1/N)*kf; kz = np.fft.rfftfreq(N, 1/N)*kf; kny = np.pi*N/BOX
KX = kx[:,None,None]+0*kx[None,:,None]+0*kz[None,None,:]; KY = 0*KX + kx[None,:,None]; KZ = 0*KX + kz[None,None,:]
def axis_phase(a, b, tag):
    A = np.fft.rfftn(a, axes=(1,2,3)); B = np.fft.rfftn(b, axes=(1,2,3))
    print(f"  {tag}:  phase [deg] of <A B*> for modes along one axis, per |k|/kNy bin, and the half-cell prediction")
    print("        k/kNy   x-only    y-only    z-only   |  diag(x=y=z)  | pred 1-axis half-cell(h64)  half-cell(h512)")
    edges = np.linspace(0, kny, 9)
    for i in range(8):
        row = []
        for K, sel in ((KX, (KY==0)&(KZ==0)), (KY, (KX==0)&(KZ==0)), (KZ, (KX==0)&(KY==0))):
            m = sel & (np.abs(K) > edges[i]) & (np.abs(K) <= edges[i+1])
            c = (A[:, m]*np.conj(B[:, m])).sum()
            row.append(np.degrees(np.angle(c)))
        m = (np.abs(KX)==np.abs(KY)) & (np.abs(KY)==np.abs(KZ)) & (np.abs(KX) > edges[i]/np.sqrt(3)) & (np.abs(KX) <= edges[i+1]/np.sqrt(3))
        c = (A[:, m]*np.conj(B[:, m])).sum(); dg = np.degrees(np.angle(c))
        kmid = 0.5*(edges[i]+edges[i+1])
        print(f"        {kmid/kny:5.2f}  {row[0]:8.2f}  {row[1]:8.2f}  {row[2]:8.2f}  |  {dg:8.2f}  |  {np.degrees(kmid*BOX/64/2):7.2f}  {np.degrees(kmid*BOX/512/2):7.2f}")
gc = Grid(64, BOX)
ic64 = load(64, "IC")
for Nf in (512,):
    icf = load(Nf, "IC"); gf = Grid(Nf, BOX)
    axis_phase(restrict_spectral(icf, gf, gc, cube_window(gf, gc.kny), 0.5), ic64, f"(a) IC R[{Nf}->64] offset 0.5 vs IC64")
    del icf
d64 = load(64, "PART_009")
for Nf in (128, 256, 512):
    df = load(Nf, "PART_009"); gf = Grid(Nf, BOX)
    axis_phase(restrict_spectral(df, gf, gc, cube_window(gf, gc.kny), 0.5), d64, f"(b) z=0 R[{Nf}->64] offset 0.5 vs dis64")
    del df
