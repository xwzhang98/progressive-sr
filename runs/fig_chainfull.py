import os
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import phase0_octaves as p0, octave_flow_toy as oft, eulerian_metric as em
from eval_chain import load_model, build_level, predict, L, OFF, GR
torch.set_num_threads(4)
mS,ckS=load_model("runs/M_qj_multi/model_ema.pt"); sA,sB=ckS["s_values"]
mA_sp,_=load_model("runs/J_flow_qj/model_ema.pt"); mB_sp,_=load_model("runs/J_flow_128/model_ema.pt")
scA,teA=build_level(32,64,8); scB,teB=build_level(64,128,8)
# shared pipeline
pA_sh,_,_=predict(mS,scA,teA,sA)
pB_sh_dir,x1B,_=predict(mS,scB,teB,sB)
teC=[dict(teB[0])]; teC[0]["dis_c"]=(pA_sh[0]*scA.hf).numpy().astype(np.float32)
pB_sh_ch,_,_=predict(mS,scB,teC,sB)
# specialist chained (for the spectra)
pA_sp,_,_=predict(mA_sp,scA,teA,0.0)
teC2=[dict(teB[0])]; teC2[0]["dis_c"]=(pA_sp[0]*scA.hf).numpy().astype(np.float32)
pB_sp_ch,_,_=predict(mB_sp,scB,teC2,0.0)
g=scB.gf; knyc=scB.knyc
def dens(f):
    pos=em.eulerian_positions(f,OFF); return em.cic_deposit(torch.ones((1,1)+f.shape[-3:]),pos,f.shape[-1])[0,0]
def spec(r_):
    d=(r_/r_.mean()-1).numpy().astype(np.float32); Fk=p0.rfftn(d[None]); P=g.shell_avg(np.abs(Fk[0])**2)
    m=(g.shell_norm>0)&(g.kshell>0); return g.kshell[m],P[m],Fk
RH={"truth":dens(x1B)}
kT,PT,FT=spec(RH["truth"])
fields={"coarse $P\\Psi_c$ (64 run)":teB[0]["dis_c"],None:None}
Pc=oft.Scaffold(64,128,L,OFF,1.0,torch.device("cpu"),window="cube").prolong(torch.from_numpy(teB[0]["dis_c"][None]))
curves={}
for nm,f in [("coarse 64 run (prolonged)",Pc/ (L/128)),("shared, direct",pB_sh_dir),
             ("shared, CHAINED",pB_sh_ch),("specialist, CHAINED",pB_sp_ch)]:
    r_=dens(f.float()); RH[nm]=r_
    k,P,Fk=spec(r_)
    num=g.shell_avg((Fk[0]*np.conj(FT[0])).real)
    den=np.sqrt(g.shell_avg(np.abs(Fk[0])**2)*g.shell_avg(np.abs(FT[0])**2))
    m=(g.shell_norm>0)&(g.kshell>0)
    curves[nm]=(k,P/np.maximum(PT,1e-30),(num/np.maximum(den,1e-30))[m])
cols={"coarse 64 run (prolonged)":"0.55","shared, direct":"C1","shared, CHAINED":"C3","specialist, CHAINED":"C0"}
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
for nm,(k,R,rr) in curves.items():
    ax[0].semilogx(k*1000,R,color=cols[nm],lw=1.9,label=nm)
    ax[1].semilogx(k*1000,rr,color=cols[nm],lw=1.9,label=nm)
for a in ax:
    a.axvline(knyc*1000,color="0.4",ls=":",lw=1.1); a.grid(alpha=.2); a.set_xlabel(r"$k$ [$h$/Mpc]")
ax[0].axhline(1,color="0.6",lw=.9); ax[0].set_ylim(0,1.3); ax[0].set_title(r"$P_\delta/P_{\delta,\rm true}$ at $128^3$")
ax[0].text(knyc*1000*1.04,0.08,r"$k_{\rm Ny,c}$",color="0.35",fontsize=9); ax[0].legend(fontsize=8,loc="lower left")
ax[1].axhline(1,color="0.6",lw=.9); ax[1].set_ylim(0.3,1.02); ax[1].set_title(r"$r_\delta(k)$ vs truth")
fig.suptitle("the chain problem, quantified: direct vs chained at 128 (emulator octaves, seed 8)",fontsize=11)
fig.tight_layout(); fig.savefig("runs/fig10_chain_spectra.png",dpi=140)
# visualization: truth / shared direct / shared chained, zoom
sl=slice(56,72); z=slice(20,84)
fig,ax=plt.subplots(1,4,figsize=(18,4.9))
show=[("coarse 64 run (prolonged)","coarse 64$^3$ run"),("shared, direct","shared, TRUE coarse (direct)"),
      ("shared, CHAINED","shared, GENERATED coarse (chained)"),("truth","TRUTH 128$^3$")]
for a,(key,t) in zip(ax,show):
    d=RH[key][sl].sum(0).numpy(); d=d/d.mean()
    a.imshow(np.log10(np.maximum(d[z,z],3e-2)),cmap="magma",vmin=-1.0,vmax=1.4)
    a.set_title(t,fontsize=11); a.set_xticks([]); a.set_yticks([])
fig.suptitle("what the chained tail deficit looks like: 50 Mpc/h zoom, same colour scale",fontsize=12)
fig.tight_layout(); fig.savefig("runs/fig10_chain_zoom.png",dpi=135)
print("wrote runs/fig10_chain_spectra.png and runs/fig10_chain_zoom.png")
