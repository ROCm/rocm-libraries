# Tensile.py — characterization target (helper layer)

Pins the pure / stubbable helpers of the main Tensile driver:
addCommonArguments (+ --global-parameters eval), argUpdatedGlobalParameters
(all override branches + PyTestBuildArchNames env), get_gpu_max_frequency_smi
(parse / error / exception), get_user_max_frequency (non-tty + interactive retry
loop), store_max_frequency, restore_prob_sol_map (completed / incomplete-last /
faster-winner / missing-file), TensileConfigPath/TensileTestPath, and the thin
entry-point wrappers (TensileROCBLAS*GEMM / TensileSGEMM5760 / main).

Resistance (out of scope, codegen+GPU orchestration): executeStepsInConfig,
Tensile() (the build+benchmark driver pulling BenchmarkProblems / ClientWriter /
LibraryLogic), and get_gpu_max_frequency (optional `hip` python module / pip
install path). These stay covered only by existing integration tests.
