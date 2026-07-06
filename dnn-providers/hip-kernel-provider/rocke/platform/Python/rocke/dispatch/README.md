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

print(result.kernel_id.cache_key)
print(result.candidate.name)
print(result.grid, result.block)
```

### Arch-family gate

Every GEMM candidate is gated to its micro-arch family (`cdna` or `rdna`) via
`arch_family_supported` (in `gemm/common.py`), which consults
`ArchTarget.family`. Without it an RDNA/WMMA candidate would report support on a
CDNA arch (its spec rebuilds wave64 and a 16x16x16 MFMA atom that also exists on
CDNA), wrongly out-ranking the intended CDNA candidate. The regression is pinned
by `tests/gemm/test_arch_family_gate.py`.

`KernelId` is the stable identity used by compile caches, manifests, logs, and
benchmark records. It includes the operation family, candidate, algorithm,
`spec_id`, target arch, ABI version, request hash, and spec hash.

## Run Tests

No-GPU GEMM dispatch tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m unittest discover \
  -s dnn-providers/hip-kernel-provider/rocKE/Python/rocke/dispatch/tests/gemm \
  -p 'test*.py'
```

The runtime tests in that directory are GPU-gated. They skip automatically when a
ROCm GPU is not visible.

Broader no-GPU regression checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python dnn-providers/hip-kernel-provider/rocKE/Python/test/test_rocke.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m unittest \
  dnn-providers/hip-kernel-provider/rocKE/Python/test/test_rocke_multiarch.py \
  -k TestGfx950ByteIdentical

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python:dnn-providers/hip-kernel-provider/rocKE/Python/test \
  ~/atom-venv/bin/python -m rocke_ir_parity_harness \
  --compare dnn-providers/hip-kernel-provider/rocKE/Python/test/golden/rocke_representative_ir_sha256.json
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
