import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, phase0_octaves as p0, octave_flow_toy as oft, eulerian_metric as em
L,OFF,GR=100000.0,0.0,76.7439
DIS,IC="data/selfsim/s{seed}/dis_{N}.npy","data/selfsim/s{seed}/ic_dis_{N}.npy"
sc=oft.Scaffold(64,128,L,OFF,1.0,torch.device("cpu"),window="cube")
tr=oft.load_real(DIS,IC,64,128,list(range(8))); te=oft.load_real(DIS,IC,64,128,[9])
sc.fit_linear_power([it["ic_f"]*GR for it in tr])
ckb=torch.load("runs/F_reg_128_3k/model_ema.pt",map_location="cpu")
baseM=oft.UNet3D(cin=12,cout=3,base=24); baseM.load_state_dict(ckb["state_dict"]); baseM.eval()
ckf=torch.load("runs/RF_resflow_128/model_ema.pt",map_location="cpu")
flowM=oft.UNet3D(cin=12,cout=3,base=24); flowM.load_state_dict(ckf["state_dict"]); flowM.eval()
b=oft.Batcher(sc,te,np.random.default_rng(0),augment_on=False,growth=GR,octave_transverse=True)
x0,x1,Pc,D=b.make(te,eta="true")
z=torch.zeros(1)
with torch.no_grad(): xb=x0+baseM(oft.net_input(x0,Pc,D),z,z)
pred=oft.sample_flow(flowM,xb,Pc,D,z,nsteps=8)
gen=torch.Generator().manual_seed(1); x0g,_,_,_=b.make(te,eta="sample",gen=gen)
with torch.no_grad(): xbg=x0g+baseM(oft.net_input(x0g,Pc,D),z,z)
predg=oft.sample_flow(flowM,xbg,Pc,D,z,nsteps=8)
g=sc.gf; W=sc.W.cpu().numpy(); knyc=sc.knyc; hf=sc.hf
def lag(a,bb):
    s_=p0.spectra_k(g,p0.rfftn(a.astype(np.float32))*(1-W),p0.rfftn(bb.astype(np.float32))*(1-W),kmin=knyc)
    return float(np.average(s_["r"])), float(np.average(s_["Paa"]/np.maximum(s_["Pbb"],1e-30)))
def dens(f):
    pos=em.eulerian_positions(f,OFF); return em.cic_deposit(torch.ones((1,1)+f.shape[-3:]),pos,f.shape[-1])[0,0]
def spec(r_):
    d=(r_/r_.mean()-1).numpy().astype(np.float32); Fk=p0.rfftn(d[None]); P=g.shell_avg(np.abs(Fk[0])**2)
    m=(g.shell_norm>0)&(g.kshell>0); return g.kshell[m],P[m]
kT,PT=spec(dens(x1)); tru=(x1[0]*hf).numpy()
for nm,f in (("base",xb),("emulator",pred),("generative",predg)):
    r_,pp=lag((f[0]*hf).numpy(),tru); k,P=spec(dens(f))
    eul=[float(P[np.argmin(abs(k-q))]/PT[np.argmin(abs(k-q))]) for q in (knyc,1.5*knyc,2*knyc)]
    print(f"seed9 {nm:11s} r={r_:.4f} P/P={pp:.4f} | Eul {[round(x,3) for x in eul]}")
