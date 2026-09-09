# Stage 5 — after the first real N-body results (paste into the Claude Code session)

---

Read `runs/RUNLOG.md` from "Stage 4" onward and the new Section "What the real N-body data changes" in `notes/octave_flow_derivation.pdf` (v3; Chinese version in `notes/octave_flow_physics.pdf`). Two script changes landed in this commit: `octave_flow_toy.py --source-filter wiener` builds the source from the best linear prediction (`Tc(k)` on the coarse band, `G(k)` on the octave band, both measured on the training pairs and printed at start-up), and every evaluation line now also prints the rms error split by the coarse run's multi-stream patches (`det(I+D^L) < 0`). Do not start anything until `runs/selfsim_train.sh` (R1/R2) has finished; the GPU must never run two trainings at once.

**R3/R4 — the data-driven source on the self-run 32->64 set.** Same settings as R1/R2 (`runs/selfsim_train.sh`), adding `--source-filter wiener`:
1. `--source-filter wiener --out runs/R_flow_wiener`
2. `--source-filter wiener --regression --out runs/R_reg_wiener`
Record the printed `Tc` and `G` values (expected: `Tc` ~1 below 0.5 k_Ny,c falling towards ~0.7 at 0.95; `G` ~0.4-0.6 across the octave band, since the linear octave overshoots by 1.49 with r=0.555). The baseline line should now show octave-band `P/P_true ≈ r^2 ≈ 0.3` instead of 1.49 and a smaller rms error. Then compare R1-R4 in one table: octave-band `r`, `P/P_true`, `r^2`, generative `P/P_true`, coarse-band `eps_rel` (emulator and generative), rms error total / multi-stream / single-stream.

**What the theory predicts, to be checked rather than assumed** (notes v3, "information-limited conditional"): in the single-stream part regression and flow should agree; in the multi-stream part (29% of this box) the regression should show a power deficit (`P/P_true` below 1, towards `r^2`) while the flow keeps `P/P_true ≈ 1` at similar `r`. If the flow's `P/P_true` is also below 1, or if regression wins everywhere as it did on 2LPT, say so plainly — that would mean the practical conditional is closer to deterministic than the chaos argument suggests at this resolution, and it changes the method's justification.

**Phase-0 panel (c) on real data.** From `runs/real_64to128_set2/phase0_results.json` and `runs/selfsim_32to64/phase0_results.json`, tabulate `conditional["delta_L"]` and `conditional["tidal"]` (variance and excess kurtosis of `d/h_f` per quantile bin) and the `multistream` in/out numbers. The question is whether the detail is close to Gaussian *outside* the multi-stream patches once binned by the local `delta_L` (kurtosis within ~1 would count as "close"); inside them it is expected to be far from Gaussian. One small figure, `runs/fig_conditional_real.png`.

**Grid offset of the production files.** RUNLOG notes the production `disp.npy` preferred offset 0.5 in the rms(eps) probe while MP-GenIC's own ICs want offset 0. Ask the owner which `q` convention the production converter used; if unknown, keep 0.5 for those files and 0 for the self-run set, and record both in every command line.

**Report.** `runs/REPORT_5.md`: the R1-R4 table with the multi/single-stream split, the panel-(c) figure, and five sentences on (a) how much the Wiener/propagator source shortens the path (baseline rms before/after), (b) whether the regression-vs-flow gap is localised in the multi-stream patches, (c) whether the coarse-band residual after `Tc` is the size predicted by `1 - r_cf^2`, (d) anything that contradicts notes v3, stated as such, (e) what the first cluster run should be (levels, seeds, which of R1-R4 configurations).

Rules unchanged: no changes to the physics conventions, additive diffs only, RUNLOG append-only, one GPU job at a time.
