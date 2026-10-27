# Python module

`stinkytofu` is the Python package for the StinkyTofu library. This page covers the calls that load STIR or assembler source, run named passes, read the instructions back, and emit assembler source.

The package uses the same parsers, pass catalog, printer, and emitter as `stinkytofu-opt`. After a pass runs, the module is still in memory, so you read opcodes and operands from it.

The same package also exposes the logical IR, intrinsics, and `runOptimizationPipeline`. Those bindings are unchanged. The sections below are the analysis API.

## Purpose

Use it to test a pass on a kernel, and to edit a kernel by hand.

1. Load physical STIR, or assembler `.s`, from a string or a file.
2. Run catalog passes, in the order you list them, and read the remarks and dumps each one left.
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

stinkytofu run kernel.stir --function helper \
    --pass StinkyRemoveNopPass --print-before asm --print-after asm
```

| Option | Meaning |
| --- | --- |
| `FILE` or `-` | STIR or `.s` file, or stdin |
| `--arch 12,5,0` | `major,minor,stepping`. Default is `12,5,0` |
| `--asm` | Read the input as assembler source. Use it for stdin, or a file without `.s` |
| `--pass NAME[=args]` | Pass to run. Repeat for more than one, in order |
| `--function NAME` | Run the passes on that function only. Default is every function |
| `--print-before stir` or `asm` | Write the function to stderr before each pass |
| `--print-after stir` or `asm` | Write the function to stderr after each pass |
| `--emit-asm` | Print assembler source. The default for assembler input |
| `--ssa` | Print the SSA view. That text cannot be loaded again |
| `-o PATH` | Write the result to a file |

Exit status:

| Code | When |
| --- | --- |
| 0 | The run finished |
| 2 | The file is missing, the text does not load, the pass name is unknown, or no function has the `--function` name |
| 1 | A pass refuses the function, or a pass reports an analysis failure |

Remarks from the passes also go to stderr. The `--print-before` and `--print-after` dumps go to stderr too. Stdout is only the module.

Tests for this API are in `python_module/tests/test_stir_api.py`. When pytest is installed, standalone `ctest` runs that file.

## API

```python
import stinkytofu

module = stinkytofu.load_stir(text, arch=(12, 5, 0))
records = module.run_passes([
    "LiftAsmRegistersToSSAPass",
    "RegisterAllocationPass=apply,classes=vs",
])
for record in records:              # one per pass, per function
    for remark in record.remarks:
        print(record.pass_name, remark.kind, remark.remark_name)
physical = module.serialize()       # load_stir accepts this again
view = module.serialize(ssa=True)   # diagnostic only
```

`arch` is `(major, minor, stepping)`. The default is `(12, 5, 0)`.

The text after `=` is split on commas, the same way `stinkytofu-opt --Pass=...` splits it.

`run_passes` uses the same feature defaults as `stinkytofu-opt` in individual-pass mode: GEMM loops are unrolled, and global reads are distributed. The `features` and `tile` options below change them for one call.

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

`run_passes(passes, *, function=None, features=None, tile=None, print_before=None, print_after=None)` runs each name on every function, in order. It returns one `PassRecord` per pass on each function: the entry function's records first, in pass order, then each callee's.

| Option | Meaning |
| --- | --- |
| `function` | Run on the function with this name only. `""` is an unnamed entry function |
| `features` | A dict with `unroll_gemm` and `distribute_global_read`, each a bool. A key left out keeps the default above. Of the catalog passes, only `StinkyDAGSchedulerPass` reads them |
| `tile` | A dict of GEMM tile fields, named as the `stinkytofu-opt` flags are: `TileA0`, `TileB0`, `TileM0`, `NumGRA`, `NumGRB`, `NumGRM`, `NumWaves`. Each is an int from 0 to 4294967295. A field left out stays 0, and `NumWaves` stays 1. The arch is always the module's |
| `print_before`, `print_after` | `"stir"` or `"asm"`. Keep a dump of the function from before or after each pass, in that form |

| `PassRecord` field | Meaning |
| --- | --- |
| `pass_name` | The catalog name you asked for |
| `function` | The function's name |
| `changed` | The function prints different physical STIR after the pass. Only that function is compared, not its callees |
| `remarks` | Every remark the pass emitted on the function, in order. Each has `kind` (`"passed"`, `"missed"`, or `"analysis"`), `remark_name`, and `message` |
| `before`, `after` | The dump `print_before` or `print_after` asked for, or `None` |

`changed` compares physical STIR, so a pass that only adds SSA values reports `False`. `LiftAsmRegistersToSSAPass` is one. Its `LiftedToSSA` remark says what it lifted.

A missed remark stays in the record when the same pass also reported a success. `TieExecMaskedWritesPass` can return `TiedWrites` and `UnmatchedExecSpan` together.

A dump begins with a `*** Before Pass: NAME ***` or `*** After Pass: NAME ***` line. NAME is the name the pass object reports, which is not always the catalog name: register allocation prints `Register Allocation`. A dump of the entry function is followed by each callee, under a `; --- callee Function: NAME ---` line. `"asm"` prints the function as `emit_asm()` does, without the kernel header.

Remarks still go to stderr as the pass runs.

These raise, and the module is left as it was:

| What happened | Exception |
| --- | --- |
| Unknown pass name | `ValueError`. The message lists `pass_names()` |
| The pass rejects its arguments | `ValueError` |
| No function has the `function` name | `ValueError`. The message lists the names |
| An unknown `features` or `tile` key, a tile value out of range, or a dump form other than `"stir"` or `"asm"` | `ValueError` |
| A `features` value that is not a bool, or a `tile` value that is not an int | `TypeError` |

These raise after the pass has run, so that pass and the ones before it may already have changed the module:

| What happened | Exception |
| --- | --- |
| A missed remark, and no success remark | `RuntimeError` with one `missed: <pass>: <message>` line per missed remark. `<pass>` is the name the remark carries, such as `RegisterAllocation` |
| The pass reports an analysis failure | `RuntimeError` |

A call that raises returns no records.

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
