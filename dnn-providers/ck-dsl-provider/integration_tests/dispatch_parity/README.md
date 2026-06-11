# ck-dsl GEMM dispatcher C++ <-> Python selection-parity test

Confirms that the C++ runtime dispatcher
(`ck_dsl::Dispatcher` over `ck_dsl::ArtifactStore`, in
`runtime/include/ck_dsl_runtime/{dispatcher,artifact_store}.hpp`) selects the
**same kernel** as the offline Python dispatcher
(`ck_dsl.dispatch.dispatch_gemm_fp16`) for the same GEMM problem.

CPU-only: selection touches no GPU and no comgr. The C++ harness loads the REAL
shipped per-arch manifest bundle (`kernels/gfx950/`) into the REAL
`ArtifactStore` and runs the REAL `Dispatcher::select`.

## Scope

The Python dispatch corpus is limited: `ck_dsl.dispatch` implements **only fp16
RCR UniversalGemm** today (see `python/ck_dsl/dispatch/__init__.py`, "Phase 1
intentionally starts with FP16 RCR GEMM only"). The test is therefore scoped to
fp16 RCR GEMM on gfx950 (the only arch with a shipped GEMM bundle). conv / SDPA /
bf16 / fp8 / other layouts are out of scope until the Python side implements them.

## What identity is compared

The two sides share their selection *identity* via the manifest
`cache_key` (== `kernel_name`), but they mint the name from different prefixes:

- C++ shipped manifest: `ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_...`
- Python `UniversalGemmSpec.kernel_name()`: prefixed by candidate name,
  `universal_gemm_fp16_cdna_cshuffle_fp16_t128x128x32_...`

A raw-string equality test would report 0% for a purely cosmetic reason. The
load-bearing identity that both encode **identically** is the tile geometry +
traits that fully determine the HSACO:

    (block_m, block_n, block_k, pipeline, epilogue)

The harness compares on that tuple and also prints both raw kernel names.

## Run

    ./run.sh            # gfx950
    ./run.sh gfx950

Exit 0 = full parity; exit 1 = divergences (printed as JSON).

## Files

- `shapes.txt`      shared M N K corpus
- `cpp_select.cpp`  C++ harness; runs the real Dispatcher, emits JSONL picks
- `parity_check.py` runs the Python dispatcher, compares, reports match rate
- `run.sh`          build + run both sides end to end
