# Extending The DSL

This page is a recipe book for the most common ways to extend `ck_dsl`:

1. Adding a new IR operation.
2. Adding a new helper that emits a CK Tile-like pattern.
3. Adding a new instance builder (a new kernel).
4. Adding a new fusion pattern.
5. Adding a new MFMA atom or a new dtype.
6. Adding a new optimization lever to the autotuner.

Each recipe is a concrete checklist. Tests in `python/test/test_ck_dsl.py` are pinned to the IR / LLVM shape; the recipes describe what to add and what to assert.

## 1. Adding A New IR Operation

Layer order matters.

### 1.1 Builder method in `core/ir.py`

Add an `IRBuilder` method. Decide:

- result type(s);
- operand SSA value(s) and any compile-time attrs;
- whether the op is pure or side-effecting.

For a new pure scalar op:

```python
def my_op(self, a: Value) -> Value:
    return self._op(
        "math.my_op",
        [a],
        [a.type],
        result_name_hint="my",
    ).result
```

For side-effecting (e.g. a new memory op), do not assign `is_pure` truthy via `attrs["pure"] = True`. The default classifier picks side-effecting names automatically; if the name is non-obvious, override with `attrs["pure"] = False` for clarity.

### 1.2 Optional textual printer in `core/ir_print.py`

If you want a clean MLIR-style print, add an entry to the op-name dispatcher. Not required for compilation; only for inspection.

### 1.3 LLVM lowering in `core/lower_llvm.py`

Two things to add:

1. Add the intrinsic declaration to `_INTRINSIC_DECLS` (or use an existing one):

   ```python
   _INTRINSIC_DECLS["my.op"] = (
       "declare <ret_type> @llvm.amdgcn.my.op(<arg_types>)"
   )
   ```

2. Add an `_op_<dotted_name>` handler in `_Lowerer`. For `math.my_op`:

   ```python
   def _op_math_my_op(self, op: Op) -> None:
       self._need("my.op")
       (a,) = op.operands
       self._current().emit(
           f"  {op.result.name} = call <ret_type> @llvm.amdgcn.my.op("
           f"<arg_type> {self._operand(a)})"
       )
   ```

The dispatch picks `_op_<op.name.replace(".", "_")>`. Missing handlers raise `NotImplementedError`.

### 1.4 Optional HIP debug lowering in `core/lower_hip.py`

Add a matching handler. Use `__builtin_amdgcn_*` if available; otherwise emit a shim in the `HIP_PROLOGUE` (`include_prologue=True`).

### 1.5 Tests

Add at least:

- A `TestCoreIR` test that builds a tiny kernel using the new op and asserts the LLVM text contains the expected intrinsic call.
- If the op is exposed through a helper, a `TestHelpers` test.

Example minimal test:

```python
def test_my_op_lowers_to_llvm(self):
    b = IRBuilder("my_op_smoke")
    b.param("A", PtrType(F32, "global"))
    v = b.const_f32(1.0)
    _ = b.my_op(v)
    ll = lower_kernel_to_llvm(b.kernel)
    self.assertIn("@llvm.amdgcn.my.op", ll)
```

## 2. Adding A New Helper

Helpers live in `helpers/`. The bar is: at least two instance builders would otherwise reimplement the same pattern. Below that bar, prefer to inline.

Template:

```python
# helpers/my_helper.py
from dataclasses import dataclass
from ..core.ir import IRBuilder, Value
from .geometry import WarpGrid

@dataclass(frozen=True)
class MyHelper:
    block_size: int
    # ... other compile-time config ...

    @classmethod
    def from_grid(cls, *, grid: WarpGrid) -> "MyHelper":
        return cls(block_size=grid.block_size)

    def emit_thing(self, b: IRBuilder, *, tid: Value) -> Value:
        # ... IR emission ...
        return result
```

Register in `helpers/__init__.py`:

```python
from .my_helper import MyHelper
__all__.append("MyHelper")
```

Re-export at the top of `ck_dsl/__init__.py` if it's part of the canonical public surface.

Add tests in `TestHelpers` that build a tiny kernel using the helper, lower it, and assert the IR shape.

## 3. Adding A New Instance Builder

This is the most common extension. The recipe is exactly what every shipped instance does.

### 3.1 Files

```text
instances/my_op.py
example/ck_tile/dsl/<NN>_my_op/gen.py          # optional, if you want CK-Tile parity tests
example/ck_tile/dsl/<NN>_my_op/expected.json   # optional, for test_ck_dsl_examples.py gate
```

### 3.2 Spec dataclass

```python
@dataclass(frozen=True)
class MyOpSpec:
    # problem fields
    n_per_block: int
    # configuration fields
    block_size: int = 256
    vec: int = 8
    dtype: str = "f16"
    name: str = "my_op"

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join
        return kernel_name_join(self.name, self.dtype,
                                f"N{self.n_per_block}",
                                f"b{self.block_size}",
                                f"v{self.vec}")
```

### 3.3 `is_valid_spec`

```python
def is_valid_spec(spec: MyOpSpec) -> Tuple[bool, str]:
    from ..helpers.spec import IOSpecRule, validate_io
    return validate_io(IOSpecRule(
        dtype=spec.dtype,
        block_size=spec.block_size,
        vec=spec.vec,
        n_per_block=spec.n_per_block,
        max_elems_per_thread=64,
    ))
```

### 3.4 Argument signature

```python
def my_op_signature(spec: MyOpSpec):
    from ..helpers.spec import SignatureBuilder
    return (SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .build())
```

### 3.5 Grid

```python
def my_op_grid(M: int, spec: MyOpSpec) -> Tuple[int, int, int]:
    from ..helpers.spec import ceil_div_grid
    return ceil_div_grid(M, 1, 1)
```

### 3.6 Builder

```python
def build_my_op(spec: MyOpSpec) -> KernelDef:
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid MyOpSpec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    X = b.param("X", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)

    # ... kernel body ...

    return b.kernel
```

### 3.7 Re-exports

Add `MyOpSpec`, `build_my_op`, `my_op_signature`, `my_op_grid` to `instances/__init__.py`.

### 3.8 Tests

Add to `TestInstances`:

```python
def test_my_op_builds(self):
    spec = MyOpSpec(n_per_block=4096)
    kernel = build_my_op(spec)
    ll = lower_kernel_to_llvm(kernel)
    self.assertIn("@my_op", ll)
```

### 3.9 Optional CK Tile parity example

For end-to-end gating, add `example/ck_tile/dsl/<NN>_my_op/gen.py`:

```python
import argparse
from pathlib import Path
from ck_dsl.helpers import compile_kernel, make_simple_op_manifest, write_artifact
from ck_dsl.instances import MyOpSpec, build_my_op, my_op_signature

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    spec = MyOpSpec(n_per_block=4096)
    artifact = compile_kernel(build_my_op(spec))
    manifest = make_simple_op_manifest(
        artifact=artifact,
        kind="my_op_fp16",
        op="my_op",
        dtype="f16",
        threads_per_block=spec.block_size,
        default_shape=[512, spec.n_per_block],
        args_signature=my_op_signature(spec),
    )
    write_artifact(artifact, args.output_dir, manifest)
```

`expected.json`:

```json
{
  "kind": "my_op_fp16",
  "shapes": [{"M": 512, "N": 4096}]
}
```

`test_ck_dsl_examples.py` will pick it up automatically.

## 4. Adding A New Fusion Pattern

`helpers/fuse.py` is the entry point. Steps:

1. If the pattern needs a new epilogue op, subclass `EpilogueOp`:

   ```python
   @dataclass(frozen=True)
   class MyActivation(EpilogueOp):
       def emit(self, b, dtype, v):
           # IR emission for v -> activation(v)
           return result
   ```

   Register it in `__all__` and the top-level helpers export.

2. Add a pattern entry to the table walked by `compile_fn`. Each entry describes:

   - the FX subgraph to match;
   - the base `UniversalGemmSpec` to use;
   - the `FusedEpilogue` op chain;
   - the `args_builder` callable that marshals the user's tensors into the kernel's arg dict.

3. Test with `explain_fn(fn)` to see what the planner picked, and with `compile_fn(fn)` to JIT-compile a CK DSL kernel.

`helpers/fusion_validation.py::FusionMatrixRunner` is the matrix test runner that exercises several fusion combinations against torch reference.

## 5. Adding A New MFMA Atom Or Dtype

### 5.1 IR Builder method

Add a new `mfma_f32_<m>x<n>x<k>_<dt>` method to `IRBuilder` and the corresponding op name. Wire it through `core/lower_llvm.py::_op_tile_mfma_f32_...` and `_INTRINSIC_DECLS`.

### 5.2 `helpers/atoms.py`

Add a new `MfmaAtom` class method and register it in `MFMA_F16_ATOMS` (or a new `MFMA_BF16_ATOMS`, etc.). Critically, fill in `lane_to_output(b, lane, i)` — this is the per-atom mapping that the epilogue depends on. Bad lane mapping is the single most common atom-extension bug.

### 5.3 Tests

Asserts:

- a small kernel using the atom lowers to the expected intrinsic call;
- `MfmaAtom.lane_to_output` for a few specific lanes returns the expected `(row, col)` SSA values when materialized (compare LLVM text against expected arithmetic).

For a new dtype:

- Add the `Type` singleton in `core/ir.py`.
- Add LLVM `_llvm_type` mapping.
- Add load / store / cast support paths.
- Add MFMA atom variants (`mfma_f32_<m>x<n>x<k>_<dt>`).
- Add manifest dtype string canonicalization (`helpers/spec.py::ptr_type_str`).
- Verify with both `lower_kernel_to_llvm` and an end-to-end `compile_kernel + KernelLauncher` test.

## 6. Adding A New Optimization Lever To The Autotuner

```python
from ck_dsl.helpers import Autotuner, AutotuneConfig
from ck_dsl.instances import UniversalGemmSpec, TileSpec, TraitSpec
from ck_dsl.helpers import gemm_args_signature

configs = []
for tile in [(128,128,32), (256,128,32), (128,256,32)]:
    for atom in [(32,32,16), (32,32,8)]:
        for wpe in [None, 2, 4]:
            spec = UniversalGemmSpec(
                name="hero",
                tile=TileSpec(*tile, warp_m=2, warp_n=2,
                              warp_tile_m=atom[0], warp_tile_n=atom[1],
                              warp_tile_k=atom[2]),
                trait=TraitSpec(pipeline="compv4", waves_per_eu=wpe),
            )
            configs.append(AutotuneConfig(
                name=f"t{tile[0]}x{tile[1]}x{tile[2]}_a{atom[0]}x{atom[1]}x{atom[2]}_wpe{wpe}",
                spec=spec,
            ))

@Autotuner(
    configs=configs,
    key_fn=lambda M, N, K, dtype: (M, N, K, dtype),
    cache_path="~/.cache/ck_dsl_autotune.json",
    build_fn=build_universal_gemm,
    signature_fn=lambda spec: gemm_args_signature(),
    prepare_args=prepare_gemm_args,
)
def launch_gemm(M, N, K, dtype, A, B, C):
    ...
```

The autotuner builds and benchmarks each config with `time_launches`, picks the median winner per `key_fn` tuple, and persists the cache to disk.

## Hygiene

Every extension PR should land with:

- a test that lowers the new path and asserts the LLVM / IR shape;
- a doc entry under `dsl_docs/` (this tree) if the new primitive is part of the public surface;
- a manifest schema update in `helpers/manifest.py` if the new kernel needs a new `kind`;
- an `expected.json` gate if you want CI to keep the kernel honest;
- a `RUNBOOK_COMPLIANCE.md` row if the extension is a new performance lever.
