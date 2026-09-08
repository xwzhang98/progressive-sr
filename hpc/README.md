# Moving to the cluster — checklist

Order of operations: (1) clone, environment, self-tests; (2) data conversion and the
nestedness/offset check; (3) Phase 0 on the real 64/128/256/512 series; (4) only then
training, starting with 64->128 full boxes and the regression baseline next to the flow.

## 1. Environment
* `git clone <repo>; cd progressive-sr; python -m venv .venv` (or conda), `pip install -r requirements.txt`
  with the cluster's CUDA torch wheel. Pin versions in `runs/RUNLOG.md` (torch, numpy, scipy).
* Run the two self-tests and compare with `reference_results/` before anything else.
  `--device cuda` puts the FFTs on the GPU too (`--fft-device` defaults to the model device
  except on MPS).
* Every run writes `git_commit` and the full argv into `results.json`; commit before launching
  sweeps so the hash means something.

## 2. Data (the part that decides whether the whole thing is exact)
* Fields must be Lagrangian displacements on the initial grid, `float32 (3, N, N, N)` in Mpc/h
  (map2map layout), plus the IC displacement at `z_init` for every level (or the top level).
  `hpc/convert_snapshot.py` builds them from MP-Gadget BigFile snapshots: ID -> grid index,
  minimum-image unwrapping of `x - q`, and a sanity check that at `z_init` the displacement
  is a small fraction of a cell. Check the ID convention (`--id-offset`, `--id-order`) on the IC
  snapshot first; if `|x - q|` at z_init is not << h, the mapping is wrong.
* Determine the grid offset with the nestedness check: run `phase0_octaves.py` on the 64/128
  pair with `--offset 0` and `--offset 0.5`; the right one prints ~1e-6.
* Growth factor `D(z_out)/D(z_ic)` for the cosmology of the runs (needed for the harmonics
  diagnostic and for the source amplitude in emulator mode). Compute it once and record it.
* Velocities: same layout; note the code's velocity unit convention in RUNLOG before using them.
* One seed is one realisation per level. For training beyond a first test, run more nested
  seeds (64/128/256 are cheap); for now hold out a Lagrangian sub-volume of the single box as
  the test region and never crop training patches from it.

## 3. Phase 0 on real data (no training)
* `phase0_octaves.py --levels 64 128 256 512 --box 100 --dis ... --ic ... --offset <o> --growth <g>`
  on a node with >= 64 GB (512^3 needs ~20 GB peak). Deliverables: the four panels, the
  discreteness white floor at low k in panel (a), `1 - r^2` in (b), the coarse-only harmonics
  fraction, the conditional kurtosis by delta_L bin and the multi-stream fraction in (c).
  These numbers decide the size of the correction head, the number of flow steps and whether
  the near-Gaussian-conditional hypothesis survives contact with shell crossing.

## 4. Training
* Start at 64->128 with full boxes (`--nc 64 --nf 128`, real data through `--dis/--ic/--growth`,
  `--train-seeds/--test-seeds` are directory ids in the patterns). Run the regression baseline
  (`--regression`) next to the flow every time; on smooth data the regression may win.
* 128->256 and 256->512 do not fit as full boxes: the toy has no patch cropping. Two options:
  add periodic-aware cropping (crop with margins, `padding_mode='zeros'` on padded crops, as
  map2map does) or port the octave-flow pieces (scaffold, source, loss) into map2map, which
  already has the data loader, crop/pad, distributed training and lag2eul. Porting is the
  better path once 64->128 works.
* Wall-time: `--max-seconds` a few minutes under the job limit and `--resume` in a dependency
  chain (`hpc/slurm_train.sh` does this); `train_state.pt` carries model, EMA, optimiser, step.
* One process per GPU. (Two jobs on one GPU corrupted each other on the Mac; on SLURM ask for
  the GPU exclusively.)
* Record in RUNLOG: command, commit, s/step, and the same metrics table as the toy runs.

## 5. Things that change meaning on real data
* The coarse run is a real N-body run: the correction band now contains the discreteness
  floor (not k^2-suppressed) and near-Nyquist errors of the LR run.
* The first step of the chain sees a real LR field; later steps see model output. Train with
  both when rollout starts (not yet).
* The detail in multi-stream regions is where the conditional stops being near-Gaussian; the
  2LPT toy says nothing about it. Expect the emulator-mode r to drop inside halos and the
  regression/flow comparison to change character there.
