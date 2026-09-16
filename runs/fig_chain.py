import os
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import phase0_octaves as p0, octave_flow_toy as oft, eulerian_metric as em
from eval_chain import load_model, build_level, predict, L, OFF, GR
mA, ckA = load_model("runs/M_qj_multi/model_ema.pt"); sA,sB = ckA["s_values"]
scA, teA = build_level(32, 64, 8); scB, teB = build_level(64, 128, 8)
predA, x1A, _ = predict(mA, scA, teA, sA)
teBc=[dict(teB[0])]; teBc[0]["dis_c"]=(predA[0]*scA.hf).numpy().astype(np.float32)
predC, x1B, _ = predict(mA, scB, teBc, sB)
def dens(f,N):
    pos=em.eulerian_positions(f,OFF); return em.cic_deposit(torch.ones((1,1,N,N,N)),pos,N)[0,0].numpy()
d32 = dens(torch.from_numpy(teA[0]["dis_c"][None]/(L/32)).float(),32)
d64g= dens(predA,64); d64t=dens(x1A,64)
d128g=dens(predC,128); d128t=dens(x1B,128)
fig,ax=plt.subplots(1,5,figsize=(21,4.6))
panels=[(d32,32,"input: TRUE 32$^3$ coarse run"),(d64g,64,"step 1 OUTPUT: generated 64$^3$"),
        (d64t,64,"(truth 64$^3$)"),(d128g,128,"step 2 OUTPUT: generated 128$^3$\n(from the generated 64$^3$)"),
        (d128t,128,"(truth 128$^3$)")]
for a,(d,N,t) in zip(ax,panels):
    sl=d[: max(4,N//16)].sum(0); sl=sl/sl.mean()
    a.imshow(np.log10(np.maximum(sl,3e-2)),cmap="magma",vmin=-1.0,vmax=1.4)
    a.set_title(t,fontsize=10); a.set_xticks([]); a.set_yticks([])
fig.suptitle("ONE shared operator applied twice: 32$^3$ -> 64$^3$ -> 128$^3$ (emulator octaves; density, same slab)",fontsize=12)
fig.tight_layout(); fig.savefig("runs/fig9_chain.png",dpi=135); print("wrote runs/fig9_chain.png")
