"""Directive step 3: existing SR64 models into the SAME 128^3 density reference."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, phase0_octaves as p0, octave_flow_toy as oft
from compute_spectra import Shells, density_coeff
L,OFF,GR=100000.0,0.0,76.7439
DIS,IC="data/selfsim/s{seed}/dis_{N}.npy","data/selfsim/s{seed}/ic_dis_{N}.npy"
sc=oft.Scaffold(32,64,L,OFF,1.0,torch.device("cpu"),window="cube")
tr=oft.load_real(DIS,IC,32,64,list(range(8)))
sc.fit_linear_power([it["ic_f"]*GR for it in tr])
def net(path):
    ck=torch.load(path,map_location="cpu"); m=oft.UNet3D(cin=12,cout=3,base=24)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m
reg=net("runs/R_reg_phys_3k/model_ema.pt"); rfl=net("runs/RF_resflow/model_ema.pt")
sh=Shells(256); kny64=np.pi*64/100.0; kny32=np.pi*32/100.0
out={}
for seed in (8,9):
    te=oft.load_real(DIS,IC,32,64,[seed])
    b=oft.Batcher(sc,te,np.random.default_rng(0),augment_on=False,growth=GR,octave_transverse=True)
    x0,x1,Pc,D=b.make(te,eta="true"); z=torch.zeros(1)
    with torch.no_grad(): xb=x0+reg(oft.net_input(x0,Pc,D),z,z)
    pred_rf=oft.sample_flow(rfl,xb,Pc,D,z,nsteps=8)
    p128=np.load(f"data/selfsim/s{seed}/dis_128.npy")
    fields={"native64":te[0]["dis_f"]/1000.0,
            "regression":(xb[0]*sc.hf).numpy()/1000.0,
            "resflow":(pred_rf[0]*sc.hf).numpy()/1000.0,
            "Y64":p0.restrict_spectral(p128,p0.Grid(128,L),p0.Grid(64,L),
                                       p0.cube_window(p0.Grid(128,L),p0.Grid(64,L).kny),0.0)/1000.0,
            "ref128":p128/1000.0}
    C={n:density_coeff(f.astype(np.float64))[0] for n,f in fields.items()}
    res={}
    for n in fields:
        if n=="ref128": continue
        pw=sh.powers(C[n],C["ref128"]); k=pw["k"]; band=(k>0)&(k<kny64)
        ratio=pw["P"][band]/np.maximum(pw["Pb"][band],1e-30); r=pw["r"][band]; kk=k[band]
        at=lambda q,arr: float(arr[np.argmin(np.abs(kk-q))])
        res[n]=dict(P_kny32=at(kny32,ratio),P_075=at(0.75*kny64,ratio),P_09=at(0.9*kny64,ratio),
                    r_09=at(0.9*kny64,r))
        print(f"s{seed} {n:11s} P/P128 @(kNy32,0.75,0.9 kNy64) = {at(kny32,ratio):.3f}/{at(0.75*kny64,ratio):.3f}/{at(0.9*kny64,ratio):.3f}  r@0.9={at(0.9*kny64,r):.3f}")
    out[str(seed)]=res
json.dump(out,open("runs/y64_models_selfsim.json","w"),indent=1)
print("wrote runs/y64_models_selfsim.json")
