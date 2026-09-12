# Phase 0 — flyDSL bring-up on gfx1151 (laptop iGPU)

**Status: DONE.** A trivial flyDSL kernel compiles *and executes* end-to-end on
gfx1151, wave32, with correct numerics. Better than the planned "compile-only"
bar — this is real GPU execution on the laptop, and it needed no sudo / no
`/opt/rocm` symlink.

## What works
- `flydsl_poc_scratch/vadd.py` — vector add, high-level `@kernel` / `@jit` /
  `.launch` API. Runs on the gfx1151 iGPU and validates against torch.
- Full flyDSL lowering pipeline runs (22 stages dumped, `fly` dialect → rocdl →
  LLVM → ISA). Final ISA in `ir_dump/vadd_0/21_final_isa.s`:
  `amdgcn-amd-amdhsa--gfx1151`, `.amdhsa_wavefront_size32 1`,
  `global_load_b32` ×2 → `v_add_f32_e32` → `global_store_b32`. Legit.

## Environment (the combo that works)
- **venv:** `/home/brpepers/aot-ab-venv` — torch 2.10.0 + ROCm 7.13 wheel, py3.12,
  sees the gfx1151 GPU. flyDSL 0.1.6 pip-installed into it.
- **env:** `source /home/brpepers/rocm-sdk/env.sh` before running.
- **run:** `/home/brpepers/aot-ab-venv/bin/python vadd.py`
- **IR dump:** `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>` → per-stage `.mlir`,
  `20_llvm_ir.ll`, `21_final_isa.s`.

## Gotchas learned (save future time)
1. **torch is mandatory.** `flydsl/compiler/jit_argument.py` does a top-level
   `import torch`; only `torch.Tensor` / `torch.cuda.Stream` are registered arg
   types. You cannot drive *any* compile (even compile-only) without torch. A
   CPU/meta tensor carries enough dtype+shape metadata for compile-only if you
   ever need a GPU-less path.
2. **`gpu.thread_id("x")` returns MLIR `index` type**, but `Tensor[coord]`
   (`fly.make_int_tuple`) requires i32/i64. Cast first:
   `i = arith.index_cast(T.i32, gpu.thread_id("x"))`
   (`from flydsl.expr import arith`; `from flydsl.expr.typing import T`).
3. **The `/opt/rocm/lib` hardcode blocker did NOT bite** in this venv. `/opt/rocm`
   does not even exist here; the aot-ab-venv ROCm env resolves the runtime libs
   itself. The earlier "symlink `/opt/rocm` → rocm-sdk/current" workaround (from
   kreb/SPEC.md) is for a different environment — not needed for this laptop path.
4. **RDNA/gfx1151 support is real** for elementwise kernels. The MFMA blocker only
   hits kernels that emit `rocdl.mfma` (the aiter GEMM kernels); it does not
   affect VALU/elementwise kernels like this one.

## API shape (high-level generation)
```python
from flydsl.compiler.kernel_function import kernel   # @kernel
from flydsl.compiler.jit_function import jit          # @jit
from flydsl.expr import gpu, arith
from flydsl.expr.typing import Tensor, T

@kernel
def vadd(out: Tensor, a: Tensor, b: Tensor):
    i = arith.index_cast(T.i32, gpu.thread_id("x"))
    out[i] = a[i] + b[i]

@jit
def run(out, a, b, stream):
    vadd(out, a, b).launch(grid=(1,1,1), block=(256,1,1), stream=stream)
```
Note: all *production* aiter kernels use the low-level `flir` API instead of this
high-level one. Phase B attention will likely need the low-level path.

## Relevance to hipDNN escape hatch
The chosen delivery mechanism is **build-time flyDSL → HSACO**, executed at
runtime via the hipDNN `hsaco` UKD descriptor kind (`file` + `symbol`) →
`hipModuleLoadData` + launch. Zero runtime Python. Phase 0 confirms flyDSL can
emit a valid gfx code object; extracting the HSACO (vs the ISA `.s`) is the next
mechanical step for Phase A.
