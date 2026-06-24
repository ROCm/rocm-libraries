# C++ recipe-replay tooling (`src/portable_ir/`)

The C++ side of the portable-IR path: it **replays** a serialized artifact
(portable IR, a recipe, or a recipe bundle) through the engine's lowerer to
produce a byte-identical HSACO — with **no CPython**. This is the runtime/JIT
counterpart to the Python authoring + record/roll tooling in
`ck_dsl/portable_ir/` (see that package's `README.md` for the end-to-end picture).

These are **C++20** (`.cpp`), like the rest of `ck_dsl_c`. Because they live under
`src/`, the CMake `GLOB_RECURSE src/*.cpp` folds them into `libckc_core.a`
alongside the engine — **one build, no separate step**. The public API stays
C-callable via `extern "C"` headers in `include/ckc/`, so the Python `ctypes`
binding and the C test drivers (e.g. `recipe_run.c`) link unchanged.

## C-side structure

The replay tooling (this dir) is decoupled from the C++ engine and the test
drivers; headers are the only shared surface.

```text
ck_dsl_c/
├── include/ckc/                     public headers (the shared API surface)
│   ├── json_dom.h   cbor_dom.h      DOM decoders → jd_val_t
│   ├── recipe_vm.h                  recipe VM entry points
│   ├── ir_import.h                  portable-IR importer
│   ├── online.h                     one-call recipe/bundle/IR → .ll wrappers
│   └── ir.h  arena.h  lower_llvm.h  engine API the tooling calls into
│
├── src/                             all C++20; one CMake glob -> libckc_core.a
│   ├── portable_ir/                 ◀ THIS DIR — the replay tooling
│   │   ├── json_dom.cpp             JSON  → jd_val_t
│   │   ├── cbor_dom.cpp             CBOR  → jd_val_t   (same DOM as JSON)
│   │   ├── recipe_vm.cpp            ck.dsl.recipe/v1  → ckc_kernel_def_t
│   │   ├── ir_import_json.cpp       ck.dsl.ir/v1      → ckc_kernel_def_t
│   │   ├── online.cpp               recipe/bundle/IR  → .ll (FFI for online.py)
│   │   └── README.md                (this file)
│   └── core/**/*.cpp                engine (builder, lowerer, arena, isa)
│
└── tests/portable_ir/               build + run harnesses and C drivers
    ├── test_cbor_dom.c              unit tests for the DOM decoders
    ├── run_unit_tests.sh            build core + decoders + test, run
    ├── recipe_run.c                 CLI: recipe (--cbor/--bundle) → .ll
    ├── comgr_compile_ll.c           .ll → HSACO (libamd_comgr)
    ├── run_parity_matrix.sh         both replay paths vs Python lowerer (.ll)
    └── run_*_demo.sh                core+tooling build → comgr → byte-identical HSACO
```

Dependency direction (one way):

```text
tests/portable_ir/*  ──uses──▶  src/portable_ir/*  ──calls──▶  libckc_core.a (src/core)
                       (headers in include/ckc/ are the only shared surface)
```

Internal coupling within `src/portable_ir/`:

```text
recipe_vm.cpp ─┐
               ├─▶ json_dom.cpp / cbor_dom.cpp   (parse text/CBOR → jd_val_t)
online.cpp  ───┼─▶ recipe_vm.cpp  +  ir_import.h
               └─▶ lower_llvm.h (ckc_lower_kernel_to_llvm_ex)
ir_import_json.cpp  (self-contained JSON; does not use json_dom.cpp)
```

## Files

| Source | Header | Role |
|---|---|---|
| `json_dom.cpp` | `ckc/json_dom.h` | dependency-free JSON → arena-owned tagged DOM (`jd_val_t`) |
| `cbor_dom.cpp` | `ckc/cbor_dom.h` | CBOR (RFC 8949 subset) → the **same** `jd_val_t` DOM, so consumers run on JSON or CBOR unchanged |
| `recipe_vm.cpp` | `ckc/recipe_vm.h` | the **recipe VM**: interprets `ck.dsl.recipe/v1` (concrete or parametric: `static_for`/`static_if`/intexpr/rolled lists/format names) → `ckc_kernel_def_t` |
| `ir_import_json.cpp` | `ckc/ir_import.h` | the **portable-IR importer**: `ck.dsl.ir/v1` concrete graph → `ckc_kernel_def_t` |
| `online.cpp` | `ckc/online.h` | one-call wrappers (recipe / bundle / IR-JSON → `.ll`, with phase timing); the FFI surface the Python `online.py` ctypes binding calls |

All retain `extern "C"` linkage (declared in the headers), so the C ABI is
unchanged: the `ctypes` binding and C test drivers link as before.

## Public entry points

```c
/* recipe_vm.h */
ckc_recipe_run_from_json(text, ints, n_ints, strs, n_strs, &builder, &kernel, err, cap);
ckc_recipe_run_from_cbor(data, len, ...);                 /* compact wire form  */
ckc_recipe_run_from_bundle_cbor(data, len, key, arch, ...);/* serve from a bundle */

/* ir_import.h */
ckc_import_kernel_from_json(text, opts, &builder, &kernel, err, cap);

/* online.h -- recipe/bundle/IR -> malloc'd .ll (free with ckc_online_free) */
ckc_online_recipe_cbor_to_llvm(...);  ckc_online_bundle_cbor_to_llvm(...);
ckc_online_ir_json_to_llvm(...);
```

All build a kernel into a caller-provided `ckc_ir_builder_t` (arena-owned); on
success the kernel is then lowered with `ckc_lower_kernel_to_llvm_ex`.

## Two consumers, one rule

- **Importer** (`ir_import_json.cpp`) — concrete portable IR, 1:1 with the built
  graph. Lowers **byte-identical** to the Python lowerer (the exported SSA names
  are applied verbatim).
- **Recipe VM** (`recipe_vm.cpp`) — concrete *or* parametric. For **concrete**
  recipes (empty `spec`) it names each value verbatim from its bind, so the `.ll`
  is byte-identical too; for **rolled** recipes it keeps fresh names (binds repeat
  across unrolled iterations) and parity is checked at the HSACO level.

Opcode resolution in both goes through a small portable-IR alias
(`*_opcode_from_name`) so a few dtype-less Python opcode spellings (e.g.
`tile.buffer_load_vN`) resolve to the engine registry's `*_f16` names. The engine
core / opcode registry is never modified here.

## Building

This tooling is part of the CMake `ckc_core` glob (`src/*.cpp`, recursive), so it
builds with the engine into `libckc_core.a` — no separate step. For a shared lib
(what the `ctypes` binding loads):

```bash
cmake -S ck_dsl_c -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build --target ckc_core -j
c++ -shared -fPIC -Wl,--whole-archive build/libckc_core.a -Wl,--no-whole-archive -lm -o libckc.so
```

`ck_dsl/portable_ir/src/online.py::build_lib()` and
`tests/portable_ir/run_parity_matrix.sh` do exactly this.

## Tests

- `tests/portable_ir/test_cbor_dom.c` + `run_unit_tests.sh` — C unit tests for the
  JSON/CBOR DOM decoders (scalars, strings, arrays, maps, float64, negatives,
  JSON≡CBOR equivalence, and truncation/error handling).
- `tests/portable_ir/run_parity_matrix.sh` — both replay paths vs the Python
  lowerer, byte-identical `.ll`, across kernels × arches.
- `tests/portable_ir/recipe_run.c` — CLI: run a recipe (`--cbor`/`--bundle`) → `.ll`.
- `tests/portable_ir/run_*_demo.sh` — build + comgr → byte-identical HSACO demos.
