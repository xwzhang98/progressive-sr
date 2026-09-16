import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, phase0_octaves as p0, octave_flow_toy as oft, eulerian_metric as em
L,OFF,GR=100000.0,0.0,76.7439
DIS,IC="data/selfsim/s{seed}/dis_{N}.npy","data/selfsim/s{seed}/ic_dis_{N}.npy"
sc=oft.Scaffold(32,64,L,OFF,1.0,torch.device("cpu"),window="cube")
tr=oft.load_real(DIS,IC,32,64,list(range(8))); te=oft.load_real(DIS,IC,32,64,[8])
sc.fit_linear_power([it["ic_f"]*GR for it in tr])
ckb=torch.load("runs/R_reg_phys_3k/model_ema.pt",map_location="cpu")
baseM=oft.UNet3D(cin=12,cout=3,base=24); baseM.load_state_dict(ckb["state_dict"]); baseM.eval()
ckf=torch.load("runs/RF_resflow/model_ema.pt",map_location="cpu")
flowM=oft.UNet3D(cin=12,cout=3,base=24); flowM.load_state_dict(ckf["state_dict"]); flowM.eval()
b=oft.Batcher(sc,te,np.random.default_rng(0),augment_on=False,growth=GR,octave_transverse=True)
x0,x1,Pc,D=b.make(te,eta="true"); hf=sc.hf
g=sc.gf; W=p0.cube_window(g,sc.knyc); knyc=sc.knyc; z=torch.zeros(1)
def run_from(lin,label):
    x0v=((Pc[0]*hf+lin[0])/hf)[None].float()
    with torch.no_grad(): xb=x0v+baseM(oft.net_input(x0v,Pc,D),z,z)
    pred=oft.sample_flow(flowM,xb,Pc,D,z,nsteps=8)
    def dens(f):
        pos=em.eulerian_positions(f,OFF); return em.cic_deposit(torch.ones((1,1)+f.shape[-3:]),pos,f.shape[-1])[0,0]
    def spec(r_):
        d=(r_/r_.mean()-1).numpy().astype(np.float32); Fk=p0.rfftn(d[None]); P=g.shell_avg(np.abs(Fk[0])**2)
        m=(g.shell_norm>0)&(g.kshell>0); return g.kshell[m],P[m]
    kT,PT=spec(dens(x1)); k,P=spec(dens(pred))
    eul=[float(P[np.argmin(abs(k-q))]/PT[np.argmin(abs(k-q))]) for q in (knyc,1.5*knyc,2*knyc)]
    lo=p0.apply_window_fine((pred[0]*hf).numpy(),1-W.astype(np.float32) if W.dtype!=np.float32 else 1-W)
    lt=p0.apply_window_fine((x1[0]*hf).numpy(),1-W)
    print(f"{label:32s} Lag oct P/P={lo.var()/lt.var():.3f} | Eul P_d/P @(1,1.5,2)kNyc = {[round(x,3) for x in eul]}")
# independent-T (current)
gen=torch.Generator().manual_seed(1)
run_from(sc.sample_linear_octave(1,gen,transverse=True),"(C) independent-T sampler")
for s in (7001,7002):
    ic=np.load(f"/tmp/oracle_{s}.npy")*GR
    run_from(torch.from_numpy(p0.apply_window_fine(ic,1-W))[None].float(),f"(G) GenIC-oracle seed {s}")
run_from(sc.band(torch.from_numpy(te[0]["ic_f"][None])*GR,"high"),"(T) true octave (emulator ref)")
