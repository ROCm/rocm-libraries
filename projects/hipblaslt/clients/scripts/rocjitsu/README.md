# Packaged-library race sweeps

This hipBLASLt client tool selects an original solution-index range from a packaged
library, constructs GEMMs, and runs numerical and race checks under rocJITsu CPU
emulation. It supports both `hipblaslt-bench` and `tensilelite-client` and writes
per-index results with the original IDs retained.

The [sweep guide](sweep-guide.md) explains original-index ranges, compatible hardware
profiles, native batching, the batch-size loading-cost model, numerical validation,
failure accounting and local execution. It is installed alongside the runner and
its tests.

The executable entry point is `rocjitsu_race_sweep.py`. Inventory and problem
construction live in `rocjitsu_sweep_plan.py`; tracked exclusions live in
`rocjitsu_known_failures.py` and the versioned policy lists.
