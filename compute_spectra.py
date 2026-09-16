"""Verified interlaced-CIC spectra estimator (moved verbatim from the owner-reviewed
analysis of 2026-09-15, ~/Documents/Codex/.../progressive-sr-spectra-2026-09-15/; only ROOT
was made repo-relative). Read-only spectra of matched N-body resolutions; no model inference or training."""
from pathlib import Path
import json
import hashlib
import sys
import time
import numpy as np
from scipy import fft

ROOT = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import phase0_octaves as p0

L = 100.0  # Mpc/h; on-disk fields are in kpc/h
SEEDS = (8, 9)
LEVELS = (32, 64, 128)
MESH = 256
GROWTH = 76.7439
WORKERS = 2
data = {}
checks = []
inputs = []


class Shells:
    def __init__(self, n):
        self.n = n
        m = fft.fftfreq(n).astype(np.float32) * n
        mz = fft.rfftfreq(n).astype(np.float32) * n
        self.m = (m[:, None, None], m[None, :, None], mz[None, None, :])
        radius = np.sqrt(sum(x*x for x in self.m))
        self.index = np.rint(radius).astype(np.int32)
        self.weight = np.broadcast_to(np.where((mz == 0) | (mz == n/2), 1., 2.)[None, None, :], radius.shape)
        self.count = np.bincount(self.index.ravel(), weights=self.weight.ravel())
        self.k = np.bincount(self.index.ravel(), weights=(radius*self.weight).ravel()) / np.maximum(self.count, 1) * (2*np.pi/L)
        j = np.arange(len(self.count))
        # Complete isotropic shells only: the whole shell is below the axial Nyquist.
        self.valid = (j >= 1) & (j + .5 <= n/2) & (self.count > 0)

    def average(self, value):
        return np.bincount(self.index.ravel(), weights=(value*self.weight).ravel(), minlength=len(self.count)) / np.maximum(self.count, 1)

    def powers(self, a, b=None):
        # Inputs are Fourier-series coefficients FFT(field)/N^3.
        vector = (a.ndim == 4)
        aa = np.abs(a)**2
        if vector:
            aa = aa.sum(axis=0)
        result = {'k': self.k[self.valid], 'P': L**3*self.average(aa)[self.valid], 'count': self.count[self.valid]}
        if b is not None:
            bb, ab = np.abs(b)**2, (a*np.conj(b)).real
            if vector:
                bb, ab = bb.sum(axis=0), ab.sum(axis=0)
            pb, cross = L**3*self.average(bb)[self.valid], L**3*self.average(ab)[self.valid]
            result.update(Pb=pb, cross=cross, r=cross / np.sqrt(np.maximum(result['P']*pb, 1e-300)))
        return result


def read_field(seed, n, prefix='dis'):
    path = ROOT/'data/selfsim'/f's{seed}'/f'{prefix}_{n}.npy'
    a = np.load(path)
    assert a.shape == (3, n, n, n) and np.isfinite(a).all()
    meta_path = path.with_name(path.stem+'_meta.json')
    meta = json.loads(meta_path.read_text())
    assert meta['N'] == n and meta['L'] == L*1000 and meta['grid_offset'] == 0
    assert meta['Time'][0] == (1.0 if prefix == 'dis' else 0.01)
    inputs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'shape': list(a.shape), 'metadata': meta})
    return a.astype(np.float32) / 1000.


def ft(a):
    return fft.rfftn(a, axes=(-3, -2, -1), workers=WORKERS) / a.shape[-1]**3


def save(prefix, values):
    for name, value in values.items():
        data[f'{prefix}_{name}'] = np.asarray(value)


def density_coeff(psi, mesh=MESH):
    n = psi.shape[-1]
    q = np.arange(n, dtype=np.float64) * (L/n)
    pos = psi.reshape(3, -1).astype(np.float64)
    pos[0] += np.broadcast_to(q[:, None, None], (n,n,n)).ravel()
    pos[1] += np.broadcast_to(q[None, :, None], (n,n,n)).ravel()
    pos[2] += np.broadcast_to(q[None, None, :], (n,n,n)).ravel()
    u = (pos % L) * (mesh/L)
    del pos
    accum = None
    m = fft.fftfreq(mesh).astype(np.float32) * mesh
    mz = fft.rfftfreq(mesh).astype(np.float32) * mesh
    modes = (m[:, None, None], m[None, :, None], mz[None, None, :])
    for shift in (0., .5):
        shifted = u - shift
        lower = np.floor(shifted).astype(np.int32)
        frac = shifted - lower
        del shifted
        rho = np.zeros((mesh,mesh,mesh), dtype=np.float64)
        for dx in (0,1):
            for dy in (0,1):
                for dz in (0,1):
                    idx = (((lower[0]+dx) % mesh)*mesh + (lower[1]+dy) % mesh)*mesh + (lower[2]+dz) % mesh
                    weight = (frac[0] if dx else 1-frac[0]) * (frac[1] if dy else 1-frac[1]) * (frac[2] if dz else 1-frac[2])
                    rho += np.bincount(idx, weights=weight, minlength=mesh**3).reshape(rho.shape)
        mass_error = abs(rho.sum()/n**3-1.)
        assert mass_error < 1e-10
        F = fft.rfftn(rho.astype(np.float32), workers=WORKERS) / n**3
        F[0,0,0] = 0.
        del rho, lower, frac
        if shift:
            # The shifted mesh samples at x=(j+1/2)h: remove its positive sampling phase.
            F *= np.exp((-1j*np.pi/mesh)*sum(modes)).astype(np.complex64)
            accum += F
        else:
            accum = F
    accum *= .5
    for k in modes:
        accum /= np.sinc(k/mesh)**2
    return accum, mass_error


def run():
    t0 = time.time()
    shells = {n: Shells(n) for n in LEVELS}
    density_shells = Shells(MESH)
    for seed in SEEDS:
        fields, ics = {}, {}
        for n in LEVELS:
            print(f'seed {seed}, N={n}: native displacement and common-mesh density', flush=True)
            a = read_field(seed,n)
            ic = read_field(seed,n,'ic_dis')
            fields[n], ics[n] = a, ic
            F = ft(a)
            save(f'psi_s{seed}_n{n}', shells[n].powers(F))
            save(f'ic_s{seed}_n{n}', shells[n].powers(ft(ic)))
            real_variance = ((a.astype(np.float64)-a.mean(axis=(1,2,3),keepdims=True))**2).sum(axis=0).mean()
            Fvar = F.copy(); Fvar[:,0,0,0] = 0
            spectral_variance = ((np.abs(Fvar)**2).sum(axis=0)*shells[n].weight).sum()
            relative_error = abs(real_variance-spectral_variance)/real_variance
            assert relative_error < 3e-6
            checks.append({'check':'displacement_parseval','seed':seed,'N':n,'relative_error':float(relative_error)})
            D, mass_error = density_coeff(a)
            save(f'delta_s{seed}_n{n}', density_shells.powers(D))
            checks.append({'check':'CIC_mass','seed':seed,'N':n,'relative_error':float(mass_error)})
            del F, Fvar, D
        for nc,nf in ((32,64),(64,128)):
            gc,gf = p0.Grid(nc,L),p0.Grid(nf,L)
            W = p0.cube_window(gf,gc.kny)
            restricted = p0.restrict_spectral(fields[nf],gf,gc,W,0.)
            eps = restricted-fields[nc]
            result = shells[nc].powers(ft(fields[nc]),ft(restricted))
            result['Peps'] = shells[nc].powers(ft(eps))['P']
            identity_error = np.max(np.abs(result['Peps']-(result['P']+result['Pb']-2*result['cross'])) / np.maximum(result['P']+result['Pb'],1e-30))
            assert identity_error < 2e-6
            checks.append({'check':'residual_power_identity','seed':seed,'pair':[nc,nf],'maximum_relative_error':float(identity_error)})
            save(f'corr_s{seed}_{nc}to{nf}',result)
            Ric = p0.restrict_spectral(ics[nf],gf,gc,W,0.)
            initial = shells[nc].powers(ft(ics[nc]),ft(Ric))
            initial['Peps'] = shells[nc].powers(ft(Ric-ics[nc]))['P']
            save(f'ic_corr_s{seed}_{nc}to{nf}',initial)
            highfine = ft(fields[nf])*(1-W)
            highlin = ft(ics[nf])*GROWTH*(1-W)
            high = shells[nf].powers(highfine,highlin)
            highmask = high['k'] >= np.pi*nc/L
            save(f'detail_s{seed}_{nc}to{nf}',{k:v[highmask] for k,v in high.items()})
            reconstructed = p0.prolong_spectral(fields[nc]+eps,gc,gf,0.) + fft.irfftn(highfine, s=(nf,nf,nf), axes=(-3,-2,-1),workers=WORKERS)*nf**3
            rel = float(np.sqrt(np.mean((reconstructed-fields[nf])**2))/np.sqrt(np.mean(fields[nf]**2)))
            assert rel < 2e-6
            checks.append({'check':'field_decomposition_identity','seed':seed,'pair':[nc,nf],'relative_rms':rel})
        print(f'seed {seed} done; elapsed {time.time()-t0:.1f} s',flush=True)

    # A controlled Fourier mode checks the normalization and shifted-mesh phase.
    ntest, mtest = 32, 20
    test = np.zeros((3,ntest,ntest,ntest),dtype=np.float32)
    test[0] = .001*np.sin(2*np.pi*3*np.arange(ntest)[:,None,None]/ntest)
    Ftest, _ = density_coeff(test,mesh=64)
    expected = -(.001*(2*np.pi*3/L))/2
    relative = float(abs(Ftest[3,0,0].real-expected)/abs(expected))
    # CIC's finite-particle quadrature is not exact; this catches sign, scale and shift errors.
    assert relative < .04 and abs(Ftest[3,0,0].imag) < abs(expected)*.01
    checks.append({'check':'small_displacement_sinusoid','relative_amplitude_error':relative,'expected_real_coefficient':expected,'measured_real_coefficient':float(Ftest[3,0,0].real)})
    np.savez_compressed(OUT/'spectra.npz',**data)
    manifest={'box_Mpc_over_h':L,'redshift':0,'seeds':SEEDS,'levels':LEVELS,'particle_nyquist_h_over_Mpc':{n:np.pi*n/L for n in LEVELS},'density_measurement_mesh':MESH,'density_estimator':'two half-cell interlaced CIC grids; Fourier CIC window deconvolved; raw power; no shot-noise subtraction','displacement_estimator':'sum of three component powers; native Lagrangian grid; complete spherical shells only','normalization':'P = V * mean(|FFT(field)/Nmesh^3|^2); displacement units (Mpc/h)^5; density units (Mpc/h)^3','bands':'half-integer radial edges in units of fundamental mode; excludes shells crossing a native grid Nyquist','inputs':inputs,'checks':checks,'elapsed_seconds':time.time()-t0}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('saved spectra.npz and manifest.json',flush=True)


if __name__ == '__main__':
    run()
