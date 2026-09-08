# Kickoff prompt for the local Claude Code session

Stages 1–2 were already run from the Cowork session inside the desktop app's Linux VM
(see `runs/RUNLOG.md`). Start a Claude Code session in this folder
(`cd progressive-sr && claude`) and paste everything between the two rules as the
first message.

---

Read `CLAUDE.md`, `README.md` and `runs/RUNLOG.md` first, and skim `notes/progressive_sr_brainstorm.md` (§2–§6, §11). This folder is a research prototype: progressive 2x super-resolution of N-body simulations in Lagrangian space, one weight-shared 2x step trained by flow matching from a "prolonged coarse field + linear-theory octave" source with the physical coupling. The HPC cluster is down, so everything runs on this Mac.

Two stages are already done and their numbers are in `runs/RUNLOG.md`: the `phase0_octaves.py` self-tests (`runs/st_dealias`, `runs/st_alias`: RP=I to 4e-7, low-k slopes 2.19/2.18 and 4.14/4.18 for the spectral R, Haar flat) and a 16->32 smoke test of `octave_flow_toy.py` (`runs/toy16_vm`, 300 CPU steps: baseline octave r=0.917 P/P=0.834 -> emulator r=0.989 P/P=0.965; generative P/P=0.990 with r~0 vs truth). Those ran on CPU in a sandbox VM; your job is the native macOS run with MPS. Append every command and its key numbers to `runs/RUNLOG.md` (append-only, never rewrite earlier entries). You may delete `runs/toy16_probe/` and the empty `runs/toy16_vm.log`.

**Stage 0 — environment (target: 10 min).**
Create a fresh venv or conda env, `pip install -r requirements.txt`, confirm `torch.backends.mps.is_available()` is True. Then a 20-step MPS sanity run:
```
python octave_flow_toy.py --nc 16 --nf 32 --steps 20 --base 16 --rms-delta 2.0 --device mps --out runs/mps_probe
```
The script keeps the spectral ops (FFTs) on CPU automatically when the network is on MPS. If any op fails on MPS (conv3d, avg_pool3d, circular padding, GroupNorm), tell me which one and rerun with `--device cpu`; do not rewrite the model to get around it. Report the s/step you see on MPS versus the 1.43 s/step the VM got on CPU at 16->32.

**Stage 3 — the real local experiment (run each training in the background with nohup + a log file under `runs/`, poll it, don't block the session).**
1. Main run: `python octave_flow_toy.py --nc 32 --nf 64 --steps 1500 --base 24 --batch 2 --rms-delta 2.0 --device mps --out runs/toy32_main`. If it is slower than ~4 s/step, tell me before continuing (options: `--base 16`, or `--steps 800`).
2. Sampler-step sweep on the trained model without retraining: `python octave_flow_toy.py --nc 32 --nf 64 --eval-only runs/toy32_main/model_ema.pt --rms-delta 2.0 --nsteps-sample N --device mps --out runs/toy32_eval_nN` for N in 1 2 4 8 16 (same seeds and rms-delta so the test box is identical; the checkpoint carries `--base`).
3. Nonlinearity sweep: repeat step 1 with `--rms-delta 1.0` and `--rms-delta 3.0`, 800 steps each, into `runs/toy32_rms1` and `runs/toy32_rms3`.
4. Write `runs/REPORT.md`: one table with rows = runs and columns = baseline r / P-ratio (octave band), emulator r / P-ratio, 1-step P-ratio, generative P-ratio, sample-vs-sample coarse-band r, coarse-band eps_rel for emulator vs generative, s/step and wall time. Embed the three `summary.png` figures. Then 3–5 sentences on: (a) how many Heun steps are needed before the octave-band P-ratio and r saturate; (b) how the emulator-minus-baseline gap grows with `--rms-delta`; (c) whether the generative-mode coarse-band eps_rel exceeds the emulator's (theory says it should: the eta–eta backreaction is stochastic given only the coarse field — the 16->32 run already showed 1.5e-2 vs 5.5e-3). Also add one row: the direct one-step regression baseline, `python octave_flow_toy.py --nc 32 --nf 64 --steps 1500 --base 24 --batch 2 --rms-delta 2.0 --device mps --regression --out runs/toy32_regression` (same network and inputs, trained at t=0 only). Compare the flow's multi-step numbers against it; do not describe multi-step gains as "restored variance" (see CLAUDE.md).

**Stage 4 — only if local copies of the real snapshots exist.**
Ask me for the path. If I give one, run `phase0_octaves.py` on the 64/128 pair with `--offset 0` and `--offset 0.5` to find the grid convention (the nestedness check prints ~1e-6 for the right one), then the full available series with that offset, and summarise the four panels of `phase0_summary.png`.

Rules: do not change the physics conventions (0/1 spectral projector R with the cube window as default, phase conventions, units of h_f, physical coupling, no GAN, no Haar, no learned P); do not reformat the existing files; keep any code change small, additive and mentioned in RUNLOG; if something about the physics is unclear, quote the relevant paragraph of the note and ask instead of guessing. Stop and report after Stage 0 and after Stage 3.

---

## Appendix — original full prompt (only if starting from scratch on another machine)

Read `CLAUDE.md`, `README.md` and skim `notes/progressive_sr_brainstorm.md` (§2–§6, §11) first. This folder contains two verified scripts and reference results for a research prototype: progressive 2x super-resolution of N-body simulations in Lagrangian space, trained with flow matching from a "prolonged coarse field + linear-theory octave" source. Work through the stages in order, stopping to report after each; keep an append-only `runs/RUNLOG.md`.

Stage 1 — environment and self-tests: create an env from `requirements.txt`; run
```
python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128 --offset 0.5 --out runs/st_dealias
python phase0_octaves.py --selftest --levels 32 64 128 --offset 0.5 --out runs/st_alias
python phase0_octaves.py --selftest --selftest-dealias --levels 32 64 128 --offset 0 --out runs/st_dealias_off0
```
and compare with the reference: operator check `|RP x - x| ~ 3e-7`, nestedness `~3e-7`, de-aliased low-k slopes `P_eps ≈ 2.2` and `P_div,eps ≈ 4.15` for R=sphere/cube, Haar flat; the aliased run's low-k slope drops to ~0. If a number disagrees beyond noise, stop and report — do not "fix" the script.

Stage 2 — smoke test: `python octave_flow_toy.py --nc 16 --nf 32 --steps 300 --base 16 --rms-delta 2.0 --device mps --out runs/toy16_mps`; compare with `reference_results/toy_run_16to32_results.json`.

Then Stage 3 and Stage 4 as above.
