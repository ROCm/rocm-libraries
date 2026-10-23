# Packaged-library race sweeps

This hipBLASLt client tool selects solution indices from a packaged library,
generates a GEMM for each index, and runs numerical and race checks under rocJITsu
CPU emulation. It supports both `hipblaslt-bench` and `tensilelite-client`.
The advisory CI job consumes artifacts built by the same workflow run.

The [sweep guide](sweep-guide.md) explains selection and shapes, shard-ordered
native batching, the batch-size cost model, validation and failure accounting,
local execution, and replay. It is installed alongside the runner and its tests.

The executable entry point is `rocjitsu_race_sweep.py`. Inventory and problem
selection live in `rocjitsu_sweep_plan.py`; tracked exclusions live in
`rocjitsu_known_failures.py` and the versioned policy lists. CI setup and report
publication live under `.github`.
