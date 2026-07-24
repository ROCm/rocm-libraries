# AGPR allocation control

rocKE can constrain AMDGPU accumulator-register allocation per kernel. This is
a code-generation control, not an architecture capability flag.

## Current interface

Kernel builders set the optional `agpr_alloc` attribute on `KernelDef`:

```python
b.kernel.attrs["agpr_alloc"] = (0, 0)
```

The Python and C++ LLVM lowerers validate either a two-element integer sequence
or a `"min,max"` string and emit:

```llvm
"amdgpu-agpr-alloc"="0,0"
```

The implementation is in:

- `python/rocke/core/lower_llvm.py::_format_agpr_alloc`;
- `cpp/core/lower_llvm/core.cpp::rocke_ll_format_agpr_alloc`;
- the kernel-attribute serializers under `cpp/core/ir/`.

When the allocation is `(0, 0)`, `python/rocke/helpers/compile.py` also passes
`-mllvm -amdgpu-mfma-vgpr-form` to COMGR. The same option can be requested with
the `mfma_vgpr_form` kernel attribute, but current product-kernel specs expose
the narrower `use_agpr_alloc_zero` control.

Invalid values fail before compilation: the minimum and maximum must be
unsigned and the minimum cannot exceed the maximum.

## Current use

The gfx942 and gfx950 tiled-attention builders use
`use_agpr_alloc_zero=True` for selected configurations whose online-softmax and
PV paths touch MFMA accumulators with VALU operations. Their spec validators
restrict the option to configurations that were explicitly implemented and
tested; it is not a blanket default for attention or GEMM.

The current product-kernel entry points are:

- `../library/kernels/gfx942/attention_tiled_2d.py`;
- `../library/kernels/gfx950/attention_tiled_2d.py`.

The C++ emitter peers mirror the same attribute under
`cpp/instances/gfx942/` and `cpp/instances/gfx950/`.

## Tradeoff

Forcing VGPR-form MFMA can remove AGPR-to-VGPR and VGPR-to-AGPR traffic around
accumulator updates. It can also increase VGPR pressure and reduce occupancy.
Consequently, new uses must be configuration-specific and supported by both
resource measurements and workload benchmarks. Pure GEMM-style kernels and
explicit inline-assembly paths should retain their existing allocator behavior
unless measurements justify a change.

Do not infer residency from the kernel attribute alone. Validate generated ISA
and compiled resource notes. In particular, check for accumulator-copy traffic
and compare VGPR, AGPR, occupancy, and performance before enabling the control
by default.

## Tests

The current contract is covered by:

- `platform/tests/test_rocke.py`, which checks LLVM attribute emission and the
  COMGR option implied by `(0, 0)`;
- `../library/tests/test_attention_builds.py`, which checks the attention
  spec, kernel name, and emitted attribute;
- parity emitters under `../library/tests/parity/`, which exercise the Python and
  C++ attention paths.

Inline-assembly helpers such as `python/rocke/helpers/asm.py` control operand
register classes directly and are a separate mechanism from `agpr_alloc`.
