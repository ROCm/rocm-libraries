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

`ck_dsl.dispatch` implements GEMM (**fp16 RCR**, **bf16 RCR** UniversalGemm) plus
four operator families: **conv** (forward implicit-GEMM), **attention** (unified
FMHA path selection), **moe** (fused mega-kernel element path), and **norm**
(rmsnorm / layernorm). `run.sh` exercises every one against the REAL
`ck_dsl::Dispatcher` (gfx950):

* **fp16 GEMM**: compares against the REAL shipped `kernels/gfx950/` HSACO bundle.
* **bf16 GEMM / conv / attention / moe / norm**: no HSACO bundle ships yet, so a
  `gen_<fam>_bundle.py` synthesizes a manifest-only bundle from the Python
  candidates and the REAL `ck_dsl::Dispatcher` selects over it (CPU-only; no
  kernel is materialized). Because the manifests are minted from the same specs
  the Python side selects, any reported divergence is a genuine selection-logic
  difference, not a data-mismatch artifact.

Per-family compared identity (the tuple both sides encode identically):

| op         | identity                                              |
|------------|-------------------------------------------------------|
| gemm       | (block_m, block_n, block_k, pipeline, epilogue)       |
| conv       | (block_m, block_n, block_k, pipeline, epilogue)       |
| norm       | (block_size, vec, kind)                               |
| attention  | (path, head_size, block_size)                         |
| moe        | (path, tile_m, tile_n_inter, tile_k_gu, atom_k)       |

Deferred (documented in the family modules): the arch-tuned attention CTA
geometry (`num_warps`/`block_m_per_warp`/`tile_size`, device-query driven), the
non-fused MoE component pipeline, and fp8 / RCC / CRR GEMM layouts.

The harness defaults to **gfx950** (the only shipped GEMM HSACO bundle); running
`./run.sh gfx942` shows the fp16 GEMM step diverge because no gfx942 HSACO bundle
ships -- that is a bundle-availability gap, not a dispatcher logic divergence.

### Dtype + arch-family gates

The C++ `Dispatcher` now mirrors the Python `arch_family_supported` gate and adds
a dtype gate (`kind` suffix vs `Problem::dtype`) so a bf16 problem never selects
an fp16 kernel. `run.sh` runs both dtypes end to end.

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

- `shapes*.txt`        per-op problem corpora (`shapes.txt` = GEMM M N K)
- `cpp_select.cpp`     C++ harness; runs the real Dispatcher per `--op`, emits JSONL
- `parity_check*.py`   per-op Python dispatch + compare + match rate
- `gen_*_bundle.py`    synthesize a manifest-only bundle from Python candidates
- `run.sh`             build + run every family end to end
