# Portable IR — record · roll · replay

This package turns a **Python-authored** CK-DSL kernel into a **compact, portable
artifact** that a **pure-C runtime** can re-emit and lower to a byte-identical
HSACO — with no CPython at JIT/serve time. It is the "author in Python, ship and
run without Python" path.

The long-form rollout strategy (operator tiers, arch families, phasing) lives in
[`portable_ir_scaling_plan.md`](portable_ir_scaling_plan.md). This README is the
architecture + directory map.

## The idea in one screen

```
            author (unchanged production builder)
                         │
                         ▼
                   KernelDef (Python SSA IR)
        ┌────────────────┼─────────────────────────────┐
        │ RECORD                                        │ serialize (concrete)
        ▼                                               ▼
  recipe (ck.dsl.recipe/v1)                    portable IR (ck.dsl.ir/v1)
  concrete, per-shape                          1:1 graph, per-shape
        │ ROLL (multi-trace)                            │
        ▼                                               │
  recipe (parametric)                                   │
  one artifact covers a family (static_for / intexpr)   │
        │                                               │
        │ pack                                          │
        ▼                                               │
  bundle (ck.dsl.bundle/v1, CBOR)                        │
  many recipes keyed by (key, arch)                      │
        └───────────────┬───────────────────────────────┘
                        ▼  REPLAY (pure C, no CPython)
        ┌───────────────────────────────────────────────┐
        │ recipe VM (recipe_vm.cpp) | IR import (ir_import_json.cpp)│
        │            → ckc_lower_kernel_to_llvm → comgr → HSACO     │
        └───────────────────────────────────────────────┘
```

**Record is universal and ~free** (it just serializes the emitted op stream).
**Roll is the optional storage win** (compress repeated unrolls + spec-scaled
constants into one parametric recipe). The C engine is the *same* lowerer the
production engine uses, so output is **byte-identical**.

## Three artifacts

| Artifact | Schema | Shape | Emitter (Python) | Consumer (C) |
|---|---|---|---|---|
| **Portable IR** | `ck.dsl.ir/v1` | concrete 1:1 graph | `ck_dsl.core.ir_export` | `ir_import_json.c` (`ckc_import_kernel_from_json`) |
| **Recipe** | `ck.dsl.recipe/v1` | concrete *or* parametric | `src/recording_builder.py`, `src/roll.py` | `recipe_vm.c` (`ckc_recipe_run_from_json` / `_cbor`) |
| **Bundle** | `ck.dsl.bundle/v1` | many recipes by `(key, arch)` | `src/recipe_bundle.py` | `recipe_vm.c` (`ckc_recipe_run_from_bundle_cbor`) |

CBOR is the compact wire form (~3× smaller than JSON) and decodes into the same
DOM (`cbor_dom.c` → `jd_val_t`), so the VM runs on JSON or CBOR unchanged.

## Directory layout

```
portable_ir/
├── src/            core engine + runtime binding
│   ├── recording_builder.py   RecordingIRBuilder + record_kernel (the recorder)
│   ├── kerneldef_to_recipe.py KernelDef → concrete recipe (post-hoc walk)
│   ├── recipe_recorder.py     idiomatic parametric authoring surface
│   ├── roller.py              multi-trace structural roller
│   ├── roll.py                roll(build_at, axis, …) driver (records + verifies)
│   ├── recipe_bundle.py       CBOR codec + bundle (ck.dsl.bundle/v1)
│   └── online.py              ctypes binding to the C backend (recipe/IR → .ll)
├── utils/
│   └── recipe_expand.py       pure-Python recipe expander + recipes_equiv (oracle)
├── examples/       runnable demo kernels (--emit recipe|ll|name)
│   ├── recipe_toy.py  mini_attn.py  qk_block.py
│   ├── export_mha.py  export_gemm_cshuffle.py  recipe_multi_result.py
├── drivers/        runnable harnesses / benchmarks
│   ├── record_coverage.py         recorder coverage over the parity emitter set
│   ├── roll_coverage.py           tiered rolling coverage
│   ├── verify_recording_production.py
│   ├── roll_recipe.py             land-#2 attention rolling demo
│   ├── bench_online.py            compile-timeline benchmark
│   └── parity_matrix.py           cross-arch backend-path parity gate
├── tests/          unittest suites (recorder drift, roller, CBOR/bundle)
└── portable_ir_scaling_plan.md
```

The C++ side lives in `../../ck_dsl_c/src/portable_ir/` (C++20, part of
`libckc_core.a`; see that dir's `README.md`):
- `recipe_vm.cpp` (+ `include/ckc/recipe_vm.h`) — the recipe VM.
- `ir_import_json.cpp` (+ `ckc/ir_import.h`) — the portable-IR importer.
- `cbor_dom.cpp`, `json_dom.cpp` — DOM decoders.
- `online.cpp` (+ `ckc/online.h`) — one-call wrappers (recipe/bundle/IR → `.ll`).
- `tests/portable_ir/run_*.sh` — build + comgr demos, parity matrix, unit tests.

## Record architecture

`RecordingIRBuilder` subclasses `core.ir.IRBuilder` and intercepts the **single op
choke point** (`_emit`) plus `param` and `push_region`/`pop_region`, recording each
op into a recipe *as the kernel is built*. Because it rides `_emit` (not the public
op-builder methods), **new ops are captured automatically**.

`record_kernel(build_fn)` temporarily rebinds the `IRBuilder` name across every
imported `ck_dsl` module, runs the **unmodified** production builder, and returns
`(kernel, recipe)`. Helpers/closures/dataclass/descriptor math just execute; only
emitted ops are captured. So any `build_*` records with **zero kernel changes**.

```python
from ck_dsl.portable_ir.src.recording_builder import record_kernel
from ck_dsl.portable_ir.examples import mini_attn
kernel, recipe = record_kernel(lambda: mini_attn.build_mini_attn(0, "f32"))
```

## Replay paths

1. **Python oracle** (`utils/recipe_expand.py`): `expand_recipe(recipe, spec)` +
   `recipes_equiv` — device-free structural check that a rolled recipe expands to
   the recorded concrete recipe at sampled *and* held-out points.
2. **Engine import** (`ckc_import_kernel_from_json`): concrete portable IR → C
   builder → C lower. Byte-identical `.ll` to the Python lowerer (name hints
   survive `ir_export`).
3. **Recipe VM** (`recipe_vm.c`): concrete or parametric recipe → C build (with
   `static_for`/intexpr expansion) → C lower. Runs on JSON or CBOR; serves from a
   bundle by `(key, arch)`.
4. **Online, in-process** (`src/online.py`): ctypes into `online.c` — hand a CBOR
   recipe/bundle or IR-JSON and get `.ll` back, no subprocess, no pybind.

## Equivalence model (what "correct" means)

- **Engine path** and **recipe path** both produce `.ll` **byte-identical** to the
  native Python lowerer (with one LLVM flavor pinned on every path). For concrete
  recipes the VM names each value verbatim from its bind (empty `spec` ⇒ unique
  Python SSA names), so even the SSA text matches — not just an equivalent HSACO.
- Parametric/rolled recipes keep fresh names (binds repeat across unrolled
  iterations) and are validated by **byte-identical HSACO** (comgr canonicalises
  names) and the Python `recipes_equiv` oracle.
- HSACO byte-identity is a **same-toolchain differential** check (compare both
  paths compiled by the *same* comgr), never a stored golden across ROCm versions.

## Running things

```bash
ROOT=projects/composablekernel/python
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/ck_dsl_c/tests/portable_ir"

# cross-arch backend-path parity (engine + recipe vs Python lowerer), device-free
bash run_parity_matrix.sh llvm20 gfx942,gfx950

# byte-identical HSACO demos (need comgr; ROCM_PATH for the comgr lib)
ROCM_PATH=/opt/rocm-7.2.0 bash run_cbor_bundle_demo.sh gfx950   # CBOR + bundle
ROCM_PATH=/opt/rocm-7.2.0 bash run_recorder_demo.sh   gfx950   # recorder → VM

# unit tests + recorder coverage (pure Python)
python3 -m unittest discover -s "$ROOT/ck_dsl/portable_ir/tests"
python3 -m ck_dsl.portable_ir.drivers.record_coverage

# online in-process lowering smoke
python3 -m ck_dsl.portable_ir.src.online
```

## When does a new kernel need code here?

- **Concrete / CPython-free path:** never — it records and lowers automatically.
- **Rolling (storage win):** usually just declare the structural spec axes; a
  genuinely new structural-variation pattern needs a one-time roller extension
  (then amortized across all future kernels of that pattern).
- **Brand-new IRBuilder op:** captured generically, but the C side must know how to
  lower it (work owed to any C-JIT backend; the byte-identical oracle catches a
  missing/wrong lowering). Region-bearing ops beyond `scf.for`/`scf.if` make the
  recorder raise a loud "extend me".

See [`portable_ir_scaling_plan.md`](portable_ir_scaling_plan.md) for the full
status, caveats, and rollout plan.
