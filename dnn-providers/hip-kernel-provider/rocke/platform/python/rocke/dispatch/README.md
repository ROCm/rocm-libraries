# CK DSL Dispatch

`rocke.dispatch` owns operator-to-kernel selection. It does not benchmark or
collect performance evidence; benchmark harnesses live under `rocke.benchmark`.

## Layout

```text
rocke/dispatch/
  core.py                  # operator-agnostic request/candidate/registry/result contracts
  __init__.py              # public dispatch exports
  gemm/
    common.py              # GEMM-family request/selector helpers + arch-family gate
    support.py             # GEMM config and shape support predicates
    fp16_rcr.py            # UniversalGemm FP16 RCR dispatcher case
    bf16_rcr.py            # UniversalGemm BF16 RCR dispatcher case
    tests/
      test_fp16_rcr.py
      test_bf16_rcr.py
      test_arch_family_gate.py
      test_fp16_rcr_runtime.py
      test_parallel_runtime.py
      test_registry.py
      test_support.py
  families/                # documented scaffolds for the remaining families
    conv.py attention.py moe.py norm.py
```

## Current Scope

Two GEMM cases are fully implemented: UniversalGemm FP16 RCR and BF16 RCR. The
conv / attention / moe / norm families are scaffolded under `families/` (each
defines its normalized request + registry and documents exactly which instance
builder + validator a full implementation reuses).

```python
from rocke.dispatch import GemmRequest, dispatch_gemm_fp16, dispatch_gemm_bf16

result = dispatch_gemm_fp16(GemmRequest(M=4096, N=4096, K=4096, arch="gfx950"))
result_bf16 = dispatch_gemm_bf16(
    GemmRequest(M=4096, N=4096, K=4096, arch="gfx950", dtype="bf16")
)

print(result.kernel_id.selection_key)
print(result.candidate.name)
print(result.grid, result.block)
```

### Declared coverage and the arch gate

Every GEMM candidate declares a `Capability`: the exact `gfx` targets it was
built for, its dtype and layout, and any shape bounds expressible as data. That
declaration is the arch gate. Without one, an RDNA/WMMA candidate would report
support on a CDNA arch (its spec rebuilds wave64 and a 16x16x16 MFMA atom that
also exists on CDNA), wrongly out-ranking the intended CDNA candidate. The
regression is pinned by `tests/gemm/test_arch_family_gate.py`, and the
family-wide invariants by `tests/gemm/test_capability.py`.

The gate is an explicit arch list rather than a `cdna`/`rdna` label, for two
reasons. Family does not imply wave size — gfx1250 is CDNA at wave32 — so a
family gate would admit a wave32 target into wave64 MFMA kernels. And the right
list is per candidate, not per family: bf16's cshuffle path runs on gfx90a where
fp16's does not, and the bf16 decode candidate needs a deep-K atom that exists
only on gfx950.

Capability is a prefilter, not the whole answer. Ask `candidate.admits(req)` for
the complete verdict:

```python
ok, why = candidate.admits(req)   # capability, then the residual predicate
```

`supports()` is the residual predicate alone. Since it no longer re-checks arch
or dtype, it is not a complete gate and should not be called directly.

### Identity

`KernelId` is the stable identity used by compile caches, manifests, logs, and
benchmark records. It includes the operation family, candidate, algorithm,
`spec_id`, target arch, ABI version, request hash, and spec hash, and exposes
two keys over those fields because they answer different questions:

- `compile_key` — `arch:abi_version:spec_hash`. Identifies the compiled binary.
  Problem-independent, so every request that selects the same spec shares one
  compile. This is what an HSACO cache should key on.
- `selection_key` — every field, including the request hash. Identifies the
  routing decision, which is what tuning records and dispatch logs index by.

`cache_key` is a deprecated alias for `selection_key`, kept at its pre-split
value so existing benchmark records stay comparable. Using it for a compile
cache recompiles per shape, which is why the two are now named apart.

### Lookup without dispatching

`CandidateRegistry` answers three questions that need no request:

```python
registry.get("universal_gemm_fp16_cdna_mem")   # by name
registry.resolve(kernel_id)                    # by persisted id, ABI-checked
registry.coverage()                            # JSON manifest of what is registered
```

`resolve` is the replay path: it rejects a `KernelId` whose `abi_version` no
longer matches the registry, so a tuning record from an older build fails loudly
instead of binding to a kernel whose kernarg layout has since changed.

## Run Tests

No-GPU GEMM dispatch tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m unittest discover \
  -s dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/dispatch/tests/gemm \
  -p 'test*.py'
```

The runtime tests in that directory are GPU-gated. They skip automatically when a
ROCm GPU is not visible.

Broader no-GPU regression checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python dnn-providers/hip-kernel-provider/rocke/platform/python/test/test_rocke.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m unittest \
  dnn-providers/hip-kernel-provider/rocke/platform/python/test/test_rocke_multiarch.py \
  -k TestGfx950ByteIdentical

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python:dnn-providers/hip-kernel-provider/rocke/platform/python/test \
  ~/atom-venv/bin/python -m rocke_ir_parity_harness \
  --compare dnn-providers/hip-kernel-provider/rocke/platform/python/test/golden/rocke_representative_ir_sha256.json
```

## Onboard A New Operator Family

1. Add an operator package, for example `rocke/dispatch/conv/`.
2. Keep shared operator-family utilities in `conv/common.py`.
3. Put support predicates in `conv/support.py`. Support should be split into:
   - config support: arch, dtype, CTA tile, wave shape, MMA/WMMA availability,
     LDS, block size, pipeline/epilogue constraints;
   - request support: runtime shape/layout/fusion compatibility.
4. Add one case module per stable dispatch surface, for example
   `conv/fwd_nhwc_krsc.py` or `gemm/bf16_rcr.py`.
5. Register candidates with:
   - `name`
   - `family`
   - `algorithm`
   - `spec_id`
   - `abi_version`
   - `priority`
   - support/select/build/signature/grid/block/sweep hooks
6. Return a `DispatchResult` with a `KernelId` derived from the normalized request
   and selected spec.
7. Add operator-local tests under `rocke/dispatch/tests/<operator>/`.

## Onboard A New GEMM Case

For a new GEMM case, such as BF16 RCR:

```text
rocke/dispatch/gemm/
  bf16_rcr.py
  tests/
    test_bf16_rcr.py
```

Reuse:

- `GemmRequest` from `gemm/common.py` if the request shape is compatible;
- `selector_matches` for `algorithm` / `spec_id` filtering;
- `GemmSupportQuery`, `gemm_config_supported`, and `request_shape_supported`
  from `gemm/support.py` when the support model matches UniversalGemm.

Add a case-local ABI version, for example:

```python
GEMM_BF16_RCR_ABI_VERSION = "hipkg-gemm-bf16-rcr/v1"
```

Do not put case-specific ABI constants or request fields in `dispatch/core.py`.
