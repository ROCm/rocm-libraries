# C++ replay tooling (`cpp/portable_ir/`)

The C++ side of the portable-IR path: it **replays** a serialized artifact
(portable IR, a recipe, or a recipe bundle) through the engine's lowerer, with
**no CPython**. This is the runtime/JIT counterpart to the Python authoring and
record/roll tooling in `python/rocke/portable_ir/` (see that package's
`README.md` for the end-to-end picture, and
`dsl_docs/architecture/portable_ir_schema.md` for the wire schemas).

These are C++20, like the rest of the engine. Because they live under `cpp/`,
the CMake source glob folds them into `librocke_core.a` alongside the engine —
one build, no separate step. The public API stays C-callable via `extern "C"`
headers in `cpp/include/rocke/`, so the Python `ctypes` binding and the
standalone CLI link against the same surface.

## Structure

The replay tooling (this dir) is decoupled from the engine core and from the
tests; headers are the only shared surface.

```text
platform/
├── cpp/include/rocke/                public headers (the shared API surface)
│   ├── json_dom.h   cbor_dom.h      DOM decoders → jd_val_t
│   ├── recipe_vm.h                  recipe VM entry points
│   ├── ir_import.h                  portable-IR importer
│   ├── online.h                     one-call recipe/bundle/IR → .ll wrappers
│   └── ir.h  arena.h  lower_llvm.h  engine API the tooling calls into
│
├── cpp/
│   ├── portable_ir/                 ◀ THIS DIR — the replay tooling
│   │   ├── json_dom.cpp             JSON  → jd_val_t
│   │   ├── cbor_dom.cpp             CBOR  → jd_val_t   (same DOM as JSON)
│   │   ├── recipe_vm.cpp            rocke.recipe/v1  → rocke_kernel_def_t
│   │   ├── ir_import_json.cpp       rocke.ir/v1      → rocke_kernel_def_t
│   │   ├── online.cpp               recipe/bundle/IR → .ll (FFI for online.py)
│   │   └── README.md                (this file)
│   └── core/**/*.cpp                engine (builder, lowerer, arena, isa)
│
└── tests/portable_ir/               ctests, the standalone CLI, pytest wiring
    ├── dom_decoders.cpp             unit tests for the DOM decoders
    ├── recipe_vm_replay.cpp         hermetic recipe VM replay + specialization
    ├── replay_cli.cpp               CLI: artifact → .ll
    └── test_portable_ir.py          CI wiring for the Python-side drivers
```

Dependency direction (one way):

```text
tests/portable_ir/*  ──uses──▶  cpp/portable_ir/*  ──calls──▶  librocke_core.a (cpp/core)
                     (headers in cpp/include/rocke/ are the only shared surface)
```

Internal coupling within this directory:

```text
recipe_vm.cpp ─┐
               ├─▶ json_dom.cpp / cbor_dom.cpp   (parse text/CBOR → jd_val_t)
online.cpp  ───┼─▶ recipe_vm.cpp  +  ir_import.h
               └─▶ lower_llvm.h (rocke_lower_kernel_to_llvm_ex)
ir_import_json.cpp  (self-contained JSON; does not use json_dom.cpp)
```

## Files

| Source | Header | Role |
|---|---|---|
| `json_dom.cpp` | `rocke/json_dom.h` | dependency-free JSON → arena-owned tagged DOM (`jd_val_t`) |
| `cbor_dom.cpp` | `rocke/cbor_dom.h` | CBOR (RFC 8949 subset) → the **same** `jd_val_t` DOM, so consumers run on JSON or CBOR unchanged |
| `recipe_vm.cpp` | `rocke/recipe_vm.h` | the **recipe VM**: interprets `rocke.recipe/v1` (concrete or parametric: `static_for`/`static_if`/intexpr/rolled lists/format names) → `rocke_kernel_def_t` |
| `ir_import_json.cpp` | `rocke/ir_import.h` | the **portable-IR importer**: `rocke.ir/v1` concrete graph → `rocke_kernel_def_t` |
| `online.cpp` | `rocke/online.h` | one-call wrappers (recipe / bundle / IR-JSON → `.ll`, with phase timing); the FFI surface `portable_ir/src/online.py` binds over ctypes |

## Public entry points

```c
/* recipe_vm.h */
rocke_recipe_run_from_json(text, ints, n_ints, strs, n_strs, &builder, &kernel, err, cap);
rocke_recipe_run_from_cbor(data, len, ...);                  /* compact wire form   */
rocke_recipe_run_from_bundle_cbor(data, len, key, arch, ...);/* serve from a bundle */

/* ir_import.h */
rocke_import_kernel_from_json(text, opts, &builder, &kernel, err, cap);

/* online.h -- recipe/bundle/IR -> malloc'd .ll (free with rocke_online_free) */
rocke_online_recipe_cbor_to_llvm(...);  rocke_online_bundle_cbor_to_llvm(...);
rocke_online_ir_json_to_llvm(...);
```

All build a kernel into a caller-provided `rocke_ir_builder_t` (arena-owned); on
success the kernel is then lowered with `rocke_lower_kernel_to_llvm_ex`.

## Two consumers, one rule

- **Importer** (`ir_import_json.cpp`) — concrete portable IR, 1:1 with the built
  graph. Lowers **byte-identical** to the Python lowerer (the exported SSA names
  are applied verbatim).
- **Recipe VM** (`recipe_vm.cpp`) — concrete *or* parametric. For **concrete**
  recipes (empty `spec`) it names each value verbatim from its bind, so the `.ll`
  is byte-identical too; for **rolled** recipes it keeps fresh names (binds repeat
  across unrolled iterations) and parity is checked at the HSACO level.

Opcode resolution in both goes through a small portable-IR alias
(`*_opcode_from_name`): Python emits some dtype-generic spellings carrying an
`elem_type` attr (`tile.buffer_load`), so the alias tries the exact name, then
`name_elem_type`, then `name_f16`. The engine core and opcode registry are never
modified here.

## Building

This tooling is part of the CMake `rocke_core` source glob, so it builds with the
engine into `librocke_core.a` — no separate step. For the shared library the
`ctypes` binding loads:

```bash
cmake -S platform -B build -DCMAKE_BUILD_TYPE=Debug && \
  cmake --build build --target rocke_core -j
c++ -shared -fPIC -Wl,--whole-archive build/librocke_core.a \
  -Wl,--no-whole-archive -lm -o librocke.so
```

`portable_ir/src/online.py::build_lib()` does exactly this on demand.

## Tests

See [`../../tests/portable_ir/README.md`](../../tests/portable_ir/README.md).
Two hermetic ctests cover this directory directly (`dom_decoders`,
`recipe_vm_replay`); the byte-identity gate across every production kernel is
the Python driver `rocke.portable_ir.drivers.parity_matrix`, and device
coverage is `rocke.portable_ir.drivers.gpu_replay`.
