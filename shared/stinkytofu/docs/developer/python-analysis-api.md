# Python module

`stinkytofu` is the Python package for the StinkyTofu library. This page covers the calls that load STIR or assembler source, run named passes, read the instructions back, and emit assembler source.

The package uses the same parsers, pass catalog, printer, and emitter as `stinkytofu-opt`. After a pass runs, the module is still in memory, so you read opcodes and operands from it.

The same package also exposes the logical IR, intrinsics, and `runOptimizationPipeline`. Those bindings are unchanged. The sections below are the analysis API.

## Purpose

Use it to test a pass on a kernel, and to edit a kernel by hand.

1. Load physical STIR, or assembler `.s`, from a string or a file.
2. Run catalog passes, in the order you list them.
3. Read each instruction. Registers stay physical until you run `LiftAsmRegistersToSSAPass`. After that pass, operands are SSA value ids.
4. Print the module back to STIR, or emit assembler source and assemble it.

`stinkytofu run` runs only the passes you name. It does not run the optimization pipeline.

## How to use

Build from `shared/stinkytofu`. This build does not configure TensileLite or the rocm-libraries superbuild. The Python module is on by default here. `python3 tasks.py build --no-python` leaves it out. A TensileLite build does not produce it.

```bash
cd shared/stinkytofu
python3 tasks.py build
```

The package lands in `build/lib/stinkytofu/`. No ROCm install is required. When `amd_comgr` is found, assembler capability probing is linked. Without it, probing reports unsupported, and the architecture tables are still available.

There is no pip package. Put `build/lib` on `PYTHONPATH`, or run the launcher. `build/bin/stinkytofu` finds `build/lib` from its own location. The same script installed under a prefix finds `lib/pythonX.Y/dist-packages`.

```bash
export PYTHONPATH=shared/stinkytofu/build/lib
python3 -c "import stinkytofu; print(stinkytofu.pass_names()[:3])"

shared/stinkytofu/build/bin/stinkytofu passes
python3 -m stinkytofu passes
```

### Command

`passes` prints one catalog name per line.

`run` loads a file, runs the passes you repeat with `--pass`, and writes the result to stdout. A `.s` file loads as assembler source and prints assembler source. Any other file loads as STIR and prints STIR.

```bash
stinkytofu run kernel.stir --arch 12,5,0 \
    --pass LiftAsmRegistersToSSAPass \
    --pass RegisterAllocationPass=apply,classes=vs

stinkytofu run kernel.s --pass StinkyRemoveNopPass -o out.s
```

| Option | Meaning |
| --- | --- |
| `FILE` or `-` | STIR or `.s` file, or stdin |
| `--arch 12,5,0` | `major,minor,stepping`. Default is `12,5,0` |
| `--asm` | Read the input as assembler source. Use it for stdin, or a file without `.s` |
| `--pass NAME[=args]` | Pass to run. Repeat for more than one, in order |
| `--emit-asm` | Print assembler source. The default for assembler input |
| `--ssa` | Print the SSA view. That text cannot be loaded again |
| `-o PATH` | Write the result to a file |

Exit status:

| Code | When |
| --- | --- |
| 0 | The run finished |
| 2 | The file is missing, the text does not load, or the pass name is unknown |
| 1 | A pass refuses the function, or a pass reports an analysis failure |

Remarks from the passes also go to stderr.

Tests for this API are in `python_module/tests/test_stir_api.py`. When pytest is installed, standalone `ctest` runs that file.

## API

```python
import stinkytofu

module = stinkytofu.load_stir(text, arch=(12, 5, 0))
module.run_passes([
    "LiftAsmRegistersToSSAPass",
    "RegisterAllocationPass=apply,classes=vs",
])
physical = module.serialize()       # load_stir accepts this again
view = module.serialize(ssa=True)   # diagnostic only
```

`arch` is `(major, minor, stepping)`. The default is `(12, 5, 0)`.

The text after `=` is split on commas, the same way `stinkytofu-opt --Pass=...` splits it.

`run_passes` uses the same feature defaults as `stinkytofu-opt` in individual-pass mode: GEMM loops are unrolled, and global reads are distributed.

### Load

| Call | Result |
| --- | --- |
| `load_stir(text, arch=(12, 5, 0))` | A module parsed from physical STIR |
| `load_stir_file(path, arch=(12, 5, 0))` | The same, read from a file |
| `load_asm(text, arch=(12, 5, 0), *, ...)` | A module parsed from assembler source, with the parser `stinkytofu-opt` uses for `.s` |
| `load_asm_file(path, arch=(12, 5, 0), *, ...)` | The same, read from a file |
| `module.load_warnings()` | The warnings from that load, one `line:column: warning: ...` string each |
| `pass_names()` | The catalog names `run_passes` accepts |

`load_stir` reads bare `st.func` text, or the `st.module @name { ... }` that `serialize()` prints, and keeps the module name.

`load_asm` keeps the kernel header (`.amdhsa_*` and the code object metadata) for `emit_asm`. A mnemonic the arch does not know stays in the module as text, with a warning. Its keyword options match `stinkytofu-opt` flags:

| Option | Meaning |
| --- | --- |
| `preserve_symbolic_regs` | Keep names such as `v[vgprSerial]` for `emit_asm(symbolic_names=True)` |
| `preserve_comments` | Keep trailing `//` and `;` comments for `emit_asm(comments=True)` |
| `from_label`, `to_label` | Keep only the instructions from one label through the other. Give both. The region has no kernel header |

The module owns its functions. A load raises `ValueError` when the arch is unknown, the text is SSA form, or the parser or converter reports an error. A STIR mnemonic the arch does not know, or a `Successors:` label that names no block, is an error. Each error carries its line and column.

### Run

`run_passes` runs each name on every function, in order.

| What happened | Exception |
| --- | --- |
| Unknown pass name | `ValueError`. The message lists `pass_names()` |
| The pass rejects its arguments | `ValueError` |
| A missed remark, and no success remark | `RuntimeError` with the remark text |
| The pass reports an analysis failure | `RuntimeError` |

### Print and emit

`serialize()` prints physical STIR, as `st.module @name { ... }`. That text round-trips through `load_stir` for a module loaded from STIR.

A module loaded with `load_asm` is one flat block. `serialize()` does not print its labels or the text of its directives, so edit it as assembler source instead: `emit_asm`, change the text, `load_asm`.

`serialize(ssa=True)` prints attached SSA for diagnosis. `load_stir` refuses that text. To see the values the allocator uses, load physical STIR, run `LiftAsmRegistersToSSAPass`, and read `defs` and `uses`.

`emit_asm()` writes every function, entry first, with the emitter `stinkytofu-opt --emit-asm` uses. Its keyword options:

| Option | Default | Meaning |
| --- | --- | --- |
| `comments` | false | Header comments and kept source comments |
| `cycle_info` | false | A cycle note on each instruction |
| `symbolic_names` | false | Symbolic register names that the load kept |
| `signature` | true | The kernel header `load_asm` kept. A STIR module has none |

`assemble()` feeds `emit_asm()` to `amd_comgr`. It returns `None` when the text assembles, and the assembler's messages when it does not. A build without `amd_comgr` raises `RuntimeError`.

```python
module = stinkytofu.load_asm_file("kernel.s")
edited = module.emit_asm().replace("s_nop 0\n", "", 1)
error = stinkytofu.load_asm(edited).assemble()   # None when the kernel assembles
```

### Read instructions

`functions()` copies the module as it is at that call. A later `run_passes` leaves those objects unchanged.

```python
text = """\
st.func @k() {
^entry:
  v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 1 }
  v2 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
}
"""

module = stinkytofu.load_stir(text)
for function in module.functions():          # entry function first
    for block in function.blocks():
        print(block.name)                    # "entry"
        print(block.successors())            # ["label_loop"] when the block branches
        for inst in block.instructions():
            print(inst.opcode)               # "st.v_mov_b32"
            print(inst.defs)                 # [("v", 0, 1)]
            print(inst.uses)                 # [("v", 1, 1)]
            print(inst.modifiers)            # ["mod.ds = { na = 1, ... }"], [] when none

module.run_passes(["LiftAsmRegistersToSSAPass"])
for function in module.functions():
    for block in function.blocks():
        for inst in block.instructions():
            if inst.opcode == "st.v_add_f32":
                value_id, reg_class = inst.defs[0]
                print(value_id, reg_class)   # integer id, "v"
```

| Operand | Python value |
| --- | --- |
| Register | `(class, index, count)`, for example `("v", 0, 1)` |
| SSA value | `(id, class)`, for example `(3, "v")` |
| Hardware register | `("hwreg", id, offset, size)` |
| Literal | A Python number or string |

`inst.modifiers` is the `mod.X = { ... }` text `serialize()` prints for each modifier. This API does not build new instructions.
