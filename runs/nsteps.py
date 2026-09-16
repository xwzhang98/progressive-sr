import os
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, phase0_octaves as p0, octave_flow_toy as oft, eulerian_metric as em
from eval_chain import load_model, build_level, L, OFF, GR
m,_=load_model("runs/J_flow_qj/model_ema.pt")
sc,te=build_level(32,64,8)
b=oft.Batcher(sc,te,np.random.default_rng(0),augment_on=False,growth=GR,octave_transverse=True)
x0,x1,Pc,D=b.make(te,eta="true")
g=sc.gf; knyc=sc.knyc
def dens(f):
    pos=em.eulerian_positions(f,OFF); return em.cic_deposit(torch.ones((1,1)+f.shape[-3:]),pos,f.shape[-1])[0,0]
def spec(r_):
    d=(r_/r_.mean()-1).numpy().astype(np.float32); Fk=p0.rfftn(d[None]); P=g.shell_avg(np.abs(Fk[0])**2)
    mm=(g.shell_norm>0)&(g.kshell>0); return g.kshell[mm],P[mm]
kT,PT=spec(dens(x1))
print("ODE integration-step convergence, J_flow_qj checkpoint, emulator mode (seed 8):")
print(f"{'nsteps':>7} {'Lag r':>7} {'Eul@1':>7} {'@1.5':>7} {'@2':>7}")
W=sc.W.cpu().numpy()
for n in (1,2,4,8,16):
    pr=oft.sample_flow(m,x0,Pc,D,torch.zeros(1),nsteps=n,method="euler" if n==1 else "heun")
    a=(pr[0]*sc.hf).numpy(); t=(x1[0]*sc.hf).numpy()
    s_=p0.spectra_k(g,p0.rfftn(a.astype(np.float32))*(1-W),p0.rfftn(t.astype(np.float32))*(1-W),kmin=knyc)
    r=float(np.average(s_["r"]))
    k,P=spec(dens(pr))
    eu=[float(P[np.argmin(abs(k-q))]/PT[np.argmin(abs(k-q))]) for q in (knyc,1.5*knyc,2*knyc)]
    print(f"{n:>7} {r:>7.4f} {eu[0]:>7.3f} {eu[1]:>7.3f} {eu[2]:>7.3f}")
