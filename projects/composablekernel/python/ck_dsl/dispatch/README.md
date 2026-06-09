# CK DSL Dispatch

`ck_dsl.dispatch` owns operator-to-kernel selection. It does not benchmark or
collect performance evidence; benchmark harnesses live under `ck_dsl.benchmark`.

## Layout

```text
ck_dsl/dispatch/
  core.py                  # operator-agnostic request/candidate/registry/result contracts
  __init__.py              # public dispatch exports
  gemm/
    common.py              # GEMM-family request and selector helpers
    support.py             # GEMM config and shape support predicates
    fp16_rcr.py            # UniversalGemm FP16 RCR dispatcher case
    tests/
      test_fp16_rcr.py
      test_fp16_rcr_runtime.py
      test_parallel_runtime.py
      test_registry.py
      test_support.py
```

## Current Scope

The first supported case is UniversalGemm FP16 RCR:

```python
from ck_dsl.dispatch import GemmRequest, dispatch_gemm_fp16

request = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950")
result = dispatch_gemm_fp16(request)

print(result.kernel_id.cache_key)
print(result.candidate.name)
print(result.grid, result.block)
```

`KernelId` is the stable identity used by compile caches, manifests, logs, and
benchmark records. It includes the operation family, candidate, algorithm,
`spec_id`, target arch, ABI version, request hash, and spec hash.

## Run Tests

No-GPU GEMM dispatch tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=projects/composablekernel/python \
  ~/atom-venv/bin/python -m unittest discover \
  -s projects/composablekernel/python/ck_dsl/dispatch/tests/gemm \
  -p 'test*.py'
```

The runtime tests in that directory are GPU-gated. They skip automatically when a
ROCm GPU is not visible.

Broader no-GPU regression checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=projects/composablekernel/python \
  ~/atom-venv/bin/python projects/composablekernel/python/test/test_ck_dsl.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=projects/composablekernel/python \
  ~/atom-venv/bin/python -m unittest \
  projects/composablekernel/python/test/test_ck_dsl_multiarch.py \
  -k TestGfx950ByteIdentical

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=projects/composablekernel/python:projects/composablekernel/python/test \
  ~/atom-venv/bin/python -m ck_dsl_ir_parity_harness \
  --compare projects/composablekernel/python/test/golden/ck_dsl_representative_ir_sha256.json
```

## Onboard A New Operator Family

1. Add an operator package, for example `ck_dsl/dispatch/conv/`.
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
7. Add operator-local tests under `ck_dsl/dispatch/tests/<operator>/`.

## Onboard A New GEMM Case

For a new GEMM case, such as BF16 RCR:

```text
ck_dsl/dispatch/gemm/
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
