# ParallelExecution.py — characterization target

Pins the multi-GPU client orchestration helpers: detectAvailableGpus (rocm-smi /
hipInfo / default), countProblemsInConfig, createPerGpuConfig, mergeResultsCsv,
and runClientParallel (basic merge+cleanup, zero-problem skip, nonzero return
code, wall-clock timeout kill, missing-results-file preserve+warn).

Coverage: 124 stmts, 4 missed → 96.8% line (94.44% blended).

Subprocess (rocm-smi/hipInfo/client), ClientExecutionLock and globalParameters
are stubbed. Residual misses are even-split corner branches (gpuNumProblems<=0
continue) and the finally-block poll()-is-None re-kill guard.
