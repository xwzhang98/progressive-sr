# Reference results (cloud sandbox, CPU)

| run (16->32, 600 CPU steps, cube window, rms_delta=2.0) | baseline r / P-ratio | emulator (8 Heun) r / P-ratio | one-step r / P-ratio | generative r / P-ratio | sample-vs-sample coarse r | eps_rel emulator / generative |
|---|---|---|---|---|---|---|
| flow_physical | 0.915 / 0.835 | 0.993 / 0.992 | 0.993 / 0.972 | 0.033 / 1.037 | 0.9958 | 2.83e-03 / 1.08e-02 |
| regression_physical | 0.915 / 0.835 | 0.996 / 0.990 | 0.996 / 0.990 | 0.033 / 1.029 | 0.9956 | 1.32e-03 / 1.03e-02 |
| flow_independent | 0.915 / 0.835 | 0.973 / 0.998 | 0.383 / 0.045 | 0.038 / 1.059 | 0.9972 | 1.08e-02 / 1.17e-02 |

Older sphere-window run: `toy_run_16to32_results.json` / `.png` (flow, physical coupling).
Self-test figure: `selftest_dealias_summary.png`.
