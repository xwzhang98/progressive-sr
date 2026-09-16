import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Y=json.load(open("runs/y64_selfsim.json")); Yn=json.load(open("runs/y64_nested_s5000.json"))
M=json.load(open("runs/y64_models_selfsim.json"))
kny32,kny64=np.pi*32/100,np.pi*64/100
fig,ax=plt.subplots(1,2,figsize=(13.5,4.8))
for sd,ls in (("8","-"),("9","--")):
    for nm,c in (("native64","C0"),("Y64","C1")):
        d=Y[sd][nm]; ax[0].plot(d["k"],d["Pratio"],ls,color=c,lw=1.7,
            label=f"{nm} (s{sd})" if True else None)
        ax[1].plot(d["k"],d["r"],ls,color=c,lw=1.7,label=f"{nm} (s{sd})")
d=Yn["5000"]
ax[0].plot(d["native64"]["k"],d["native64"]["Pratio"],":",color="C0",lw=2.2,label="native64 (STRICT IC)")
ax[0].plot(d["Y64"]["k"],d["Y64"]["Pratio"],":",color="C1",lw=2.2,label="Y64 (STRICT IC)")
probes=[kny32,0.75*kny64,0.9*kny64]
for sd,mk in (("8","o"),("9","s")):
    for nm,c in (("regression","C3"),("resflow","C2")):
        v=M[sd][nm]; ax[0].plot(probes,[v["P_kny32"],v["P_075"],v["P_09"]],mk,color=c,ms=8,
            label=f"{nm} (s{sd})")
for a in ax:
    a.axvline(kny32,color="0.5",ls=":",lw=1); a.axvline(kny64,color="0.3",ls=":",lw=1.2)
    a.set_xlim(0.05,kny64*1.02); a.grid(alpha=.2); a.set_xlabel(r"$k$ [$h$/Mpc]")
ax[0].axhline(1,color="0.6",lw=.9); ax[0].set_ylim(0.6,2.0)
ax[0].set_title(r"density $P/P_{128}$ (interlaced CIC, 256$^3$ analysis mesh)")
ax[0].text(kny32*1.02,0.65,r"$k_{\rm Ny,32}$",fontsize=8,color="0.4")
ax[0].text(kny64*0.93,0.65,r"$k_{\rm Ny,64}$",fontsize=8,color="0.3")
ax[0].legend(fontsize=6.5,ncol=2,loc="upper left")
ax[1].axhline(1,color="0.6",lw=.9); ax[1].set_ylim(0.9,1.005); ax[1].set_title(r"$r_\delta(k)$ vs the 128$^3$ reference")
ax[1].legend(fontsize=7)
fig.suptitle("directive step 2/3: the ideal projected label over-concentrates; the native 64 run is the in-band benchmark",fontsize=11)
fig.tight_layout(); fig.savefig("runs/fig11_y64_verdict.png",dpi=140)
print("wrote runs/fig11_y64_verdict.png")
