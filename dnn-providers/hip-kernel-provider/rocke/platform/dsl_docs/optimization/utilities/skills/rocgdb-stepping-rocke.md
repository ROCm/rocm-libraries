---
name: rocgdb-stepping-rocke
description: >
  Step through a HIP-lowered CK DSL kernel line-by-line under rocgdb. Use this
  when ISA inspection and numeric verify are not enough and you need to set a
  breakpoint on a source line, single-step the kernel, and inspect live locals
  with real input data resident. Generates a self-contained, self-checking HIP
  executable (host main() + the lowered .hip) built with debug info.
  Usage: /rocgdb-stepping-rocke
allowed-tools: Read Edit Bash Grep Glob Agent
---

# rocgdb Line Stepping (CK DSL)

Use this when you need to answer "what is this kernel actually doing, register
by register, on this input?" — the question ISA disassembly
(`isa-inspection-rocke.md`) and numeric verify (`elementwise_verify_hip.py`)
cannot answer because neither stops on a source line.

The production path is `LLVM -> COMGR -> .hsaco` (optimized, no host driver), so
it is not steppable. The HIP lowering path emits readable C++, which we compile
**together with a generated host `main()`** into one `-g -O0` executable that
`rocgdb` can break on and single-step.

This is a debug / inspection path only. `-O0` deliberately does **not** match the
production `-O3` `.hsaco` — use it to understand semantics, then confirm
performance separately.

---

## When To Use

- A kernel produces wrong numbers and ISA/verify have not localized it.
- You want to watch a specific lane/thread's index math or accumulator evolve.
- You are bringing up a new op in the HIP lowering path and need to confirm the
  emitted C++ executes as intended.

Do **not** use it for performance work — `-O0` changes the schedule entirely.

---

## Minimal Workflow

Requires `hipcc` + `rocgdb` + a ROCm GPU with the **HIP dev headers**
(`hip/hip_runtime.h`) installed. (A ROCm install with only `amd_comgr` — the
production LLVM/COMGR path — cannot compile the HIP path at all.)

1. Generate the `.hip`, a self-checking `main.cpp`, and build the debug exe:

```bash
export PYTHONPATH=python

python3 -m rocke.examples.common.hip_rocgdb_driver \
    --case elementwise.add --arch gfx942 --out-dir ./rocgdb_dbg --build --run
```

`--run` executes the built binary and prints `... -> PASS`, confirming the
lowered kernel is numerically correct before you debug it. List the available
cases with `--list`.

2. Step through it under rocgdb (the tool prints this recipe after `--build`):

```bash
rocgdb --args ./rocgdb_dbg/elementwise_add_dbg
(rocgdb) break rocke_elementwise_add_f16_b256_v8    # break by kernel name
(rocgdb) break elementwise_add.hip:73               # or by source line
(rocgdb) run
(rocgdb) step                                       # single-step device code
(rocgdb) info locals                                # inspect kernel locals
(rocgdb) print C[0]
```

`info threads` / `thread <n>` switch between wavefronts/lanes; `continue`
resumes.

---

## Adding a Case

The generator is case-based because a `KernelDef` gives parameter *types* and
block size but not buffer sizes, grid, or scalar values. Append a `DebugCase` to
`_CASES` in
`python/rocke/examples/common/hip_rocgdb_driver.py`, supplying the `KernelDef`
builder, `grid`/`block`, the ordered arg list (matching the emitted signature),
and two C++ snippets: `fill_cpp` (populate the `h_<input>` host vectors) and
`ref_cpp` (populate `h_ref` for the checked output). The seeded cases
(`elementwise.add`, `elementwise.relu`) show the binary and unary signature
shapes. GEMM-family kernels launch through the manifest runner rather than a flat
packed signature, so wiring one in means providing a C++ reference GEMM in
`ref_cpp`.

---

## Related

- `isa-inspection-rocke.md` — static ISA disassembly of a `.hsaco` (no GPU run).
- `capture-kernel-trace-rocke.md` — documents a `compile_kernel(..., debug=True)`
  DWARF flag for ATT source mapping on the **LLVM** path (a separate, unimplemented
  convention; not the same as this hipcc `-g` executable path).
