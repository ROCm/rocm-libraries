# Portable CK-DSL IR prototype (Python frontend ↔ C backend)

A working prototype of the path proposed in
`python/ck_dsl/dsl_docs/architecture/portable_ir_schema.md`: a kernel is
**authored in Python**, exported as a structured **portable IR** artifact
(schema `ck.dsl.ir/v1`), then **imported and lowered by the pure-C `ck_dsl_c`
backend** with no embedded CPython. Demonstrated on the unified-attention 2D
scalar MHA/SDPA kernel (the kernel benchmarked in
`dsl_docs/architecture/SDPA_CKDSL_Provider_Comparison 1.md`) on **gfx950**.

## Pieces

| File | Role |
|------|------|
| `python/ck_dsl/core/ir_export.py` | Python: `KernelDef` → portable-IR dict / JSON |
| `include/ckc/ir_import.h`, `src/ir_import_json.c` | C: JSON → `ckc_kernel_def_t` (generic-op path + scf.for/scf.if via real builders) |
| `roundtrip.c` | C: import IR → `ckc_lower_kernel_to_llvm` → print `.ll` |
| `bench_compile.cpp` | C++: import + lower + `libamd_comgr` → gfx950 HSACO, timed |
| `export_parity.py` / `export_mha.py` | export the micro-parity kernels / the MHA kernel |
| `run_portable_ir_parity.sh` | byte-parity gate on scalar/memory/forloop/vector |
| `run_mha_parity.sh` | byte-parity on 6 MHA families + shape-polymorphism analysis |
| `run_compile_bench.sh` | compile-time table across the comparison's MHA variants |
| `run_compare.sh` | pure C-interface vs portable-IR import compile-time comparison |
| `include/ckc/json_dom.h`, `src/json_dom.c` | shared arena-backed JSON DOM parser |
| `include/ckc/recipe_vm.h`, `src/recipe_vm.c` | C "builder recipe" VM (compile-time control flow + spec params) |
| `recipe_toy.py` | authoring: Python reference kernel + the compact recipe artifact |
| `recipe_run.c` / `comgr_compile_ll.c` | run a recipe to `.ll` / compile `.ll` to HSACO |
| `run_recipe_demo.sh` | one recipe -> D64/D128/D256, HSACO byte-identical to the Python reference |

## Run

```bash
cd projects/composablekernel/python/ck_dsl_c
bash tests/portable_ir/run_portable_ir_parity.sh gfx950   # micro-kernel parity
bash tests/portable_ir/run_mha_parity.sh        gfx950    # MHA parity + shape-poly
bash tests/portable_ir/run_compile_bench.sh     gfx950 10 # compile-time (needs libamd_comgr)
bash tests/portable_ir/run_compare.sh           gfx950 10 # native C vs portable IR
bash tests/portable_ir/run_recipe_demo.sh       gfx950    # builder-recipe VM (one artifact, JIT-specialized by D)
```

## Builder-recipe VM (compact per-builder artifact)

The portable-IR import path above ships one concrete IR graph per shape. The
**builder recipe** (`ck.dsl.recipe/v1`) instead ships ONE compact artifact per
*builder family* and specializes it at JIT time in pure C:

- a recipe encodes the builder *algorithm* — including compile-time control flow
  (`static_for` over a spec value, spec-derived constants) and spec inputs;
- the C VM (`recipe_vm.c`) replays it against a runtime spec, doing the
  Python-time unroll in C, so one recipe covers a whole family.

`run_recipe_demo.sh` proves it: a single **3.4 KB** recipe expands to the
D64 / D128 / D256 kernels (462 / 910 / 1806 lines of `.ll`), and each
comgr-compiled HSACO is **byte-identical** to the equivalent Python-authored
reference kernel. This delivers: Python authoring convenience + compact
per-builder artifact + runtime shape flexibility + no CPython in the JIT.

(Scope: the VM currently implements the instruction set the toy family needs —
`param`, `const_i32/f32`, `thread_id_x`, `emit`, `static_for`, `ret`, plus
`spec`/`var`/`add`/`sub`/`mul` int-exprs. Porting a real instance builder to a
recipe is the next step; idiomatic Python builders are not auto-convertible and
must be authored against the recipe model.)

## What it proves

- **Lossless boundary.** Python-lowered `.ll` == C-from-portable-IR `.ll`,
  byte-for-byte, for the 4 micro kernels and all 6 `(dtype, head_size)` MHA
  families (D64/D128/D256 × fp16/bf16). The MHA kernel is ~1000–1900 lines of
  `.ll`; the round trip is exact.
- **gfx950 compile.** The imported IR comgr-compiles to a gfx950 HSACO.
- **Near-native compile time.** With the hash-indexed value table and arena-owned
  JSON DOM, portable-IR import adds roughly **0.4 ms (D64)**, **0.6-0.7 ms
  (D128)**, and **1.2 ms (D256)** over native C-interface build on the Section-3
  MHA variants; the lowered `.ll` remains byte-identical.
- **Shape-polymorphism.** For a fixed `(dtype, head_size, gqa)` family the
  kernel body is invariant across sequence length **except one baked constant**
  (the block-table stride `max_seqlen_k / block_size`) plus the kernel name.
  Promoting that stride to a runtime kernel argument makes a single IR cover
  S2048 / S4096 / S8192. (MHA vs GQA differ in head-index arithmetic, so they
  are distinct families.)

## Known prototype limitations

- JSON is the canonical format here; a CBOR/compressed form (see the schema doc)
  would shrink the artifact (D256 ≈ 840 KB JSON).
- `scf.if`/else and any op carrying a *list* attribute outside `scf.for` are not
  yet handled by the generic path (the kernels here don't need them).
