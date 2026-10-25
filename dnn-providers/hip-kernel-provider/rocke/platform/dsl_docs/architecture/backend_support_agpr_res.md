# AGPR allocation control

rocKE's Python and C++ LLVM lowerers implement an engine-level
`agpr_alloc` kernel attribute. The attribute accepts an unsigned `(min, max)`
pair (or the equivalent `"min,max"` string) and emits:

```llvm
"amdgpu-agpr-alloc"="min,max"
```

Setting `agpr_alloc=(0, 0)` forbids AGPR allocation. The Python compile path
also passes `-mllvm -amdgpu-mfma-vgpr-form` for that setting, so LLVM selects
the VGPR accumulator form of MFMA instructions.

## Current code anchors

- [`python/rocke/core/lower_llvm.py`](../../python/rocke/core/lower_llvm.py)
  (`_format_agpr_alloc`) validates and emits the kernel attribute.
- [`python/rocke/helpers/compile.py`](../../python/rocke/helpers/compile.py)
  (`_is_zero_agpr_alloc`) adds the matching LLVM option for `(0, 0)`.
- [`cpp/core/lower_llvm/core.cpp`](../../cpp/core/lower_llvm/core.cpp)
  (`rocke_ll_format_agpr_alloc`) formats and emits the corresponding attribute
  in the C++ engine.
- [`tests/test_rocke.py`](../../tests/test_rocke.py) covers attribute emission
  and the zero-AGPR compile option.

The Python authoring surface currently sets this through
`kernel.attrs["agpr_alloc"]`; there is not yet a general high-level spec policy
that enables it across attention, GEMM, and MoE families. Individual C++
attention specs expose narrower `use_agpr_alloc_zero` controls.

The C++ formatter also accepts a scalar integer and uses a looser string parser
than the Python formatter. Input-validation parity is therefore not currently
guaranteed even though both engines emit the same LLVM attribute spelling.

## The other direction: forcing the AGPR form

The universal GEMM's `TraitSpec.acc_agpr` reserves exactly the accumulator registers
(`agpr_alloc = (n, n)`) and sets the kernel attribute `mfma_agpr_form`, for which the
Python compile path passes `-mllvm -amdgpu-mfma-vgpr-form=0` whenever the kernel is
lowered for an LLVM 21+ flavor (`llvm22` / `llvm23`; LLVM 20 has no such option). The
decision is a function of the kernel attributes and the lowering flavor only (the same
`ROCKE_LLVM_FLAVOR` / COMGR resolution the LLVM lowerer uses), so an offline
`lower_kernel_for_comgr` with a pinned flavor records the same options as a live compile.
Some newer LLVM builds otherwise select the VGPR MFMA form despite the reservation. Like
`mfma_vgpr_form`, the option is not part of the emitted IR, so it does not affect byte
identity; the C++ engine sets the same kernel attribute. A process-level check found no effect of the
option on kernels compiled later in the same process. `acc_agpr` is admitted only on
the targets where its bodies passed GPU full-output numerics (gfx942 and gfx950; not
gfx90a yet), and setting the `ROCKE_EXP_AGPR_ALLOC` experiment together with it is an
error.

## Tradeoff

VGPR-form MFMA can remove AGPR-to-VGPR copies in pipelines that perform VALU
work on live accumulators, such as online-softmax rescaling. It can also raise
VGPR pressure and reduce occupancy. `(0, 0)` is therefore an explicit codegen
choice, not a universal default.

## Remaining validation

Before enabling the setting broadly for a kernel family:

1. Compare ISA and confirm that the intended MFMA instructions use VGPR
   accumulators and that `v_accvgpr_read_b32` / `v_accvgpr_write_b32` traffic is
   reduced.
2. Re-run correctness and performance coverage for every affected shape.
3. Check VGPR usage and occupancy so copy removal does not hide a register
   pressure regression.

Warp-specialized inline assembly with explicit accumulator register numbers and
algorithmic reformulations that avoid accumulator rescaling remain separate
design choices.
