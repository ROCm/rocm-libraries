# Debugging a kernel with rocgdb

`rocgdb` (the ROCm fork of gdb, shipped in `<rocm>/bin`) stops a rocke kernel on a
GPU memory fault and tells you **which authoring line emitted the faulting
instruction**, plus the workgroup and lane that hit it. That is the piece the bare
runtime message doesn't give you:

```text
Memory access fault by GPU node-2 (Agent handle: 0x...) on address 0x7f4067e00000. Reason: Unknown.
```

An address and a node, no location. Everything below closes that gap.

For the failure-mode catalogs — what usually *causes* a fault, and the build-level
things that only look like bugs — see [`testing.md`](./testing.md#debugging-patterns)
and [`troubleshooting.md`](./troubleshooting.md). This file is only about driving the
debugger.

Verified on `gfx950` with ROCm 7.2 (`rocgdb` 16.3) against a kernel compiled
in-process through comgr. Nothing here needs a special build of the engine.

## What you get, and what you don't

Source mapping comes from `ROCKE_DEBUG_LOC=1`, the same DWARF that feeds WaveScope's
Source tab — set it on the process that **builds** the kernel (it is read when
`IRBuilder` constructs the kernel, not at compile time). There is no `-g` and no
`debug=` parameter; a rocke kernel is Python that builds IR, so the locations have to
be captured during the build. See
[`../optimization/utilities/skills/capture-kernel-trace-rocke.md`](../optimization/utilities/skills/capture-kernel-trace-rocke.md).

The compile unit is emitted as `emissionKind: LineTablesOnly`, which decides the
ceiling:

| | |
|---|---|
| Line + file for any PC | yes |
| Inlined call stack above that line | yes |
| Workgroup + lane of the faulting wave | yes |
| Breakpoints on an authoring line | yes |
| `info locals` / `info args` / `print <name>` | **no** — there are no variable DIEs |
| Raw SGPR / VGPR / `exec` inspection | yes (this is the substitute for `print`) |

Two things to keep in mind before you trust a debug run. `ROCKE_DEBUG_LOC=1` changes
the emitted `.ll` bytes, which the IR goldens and the byte-identity gate both pin — so
the object you are debugging is not byte-identical to the one you ship. And the DWARF
describes *Python* (`DW_LANG_Python`), so the "function" in a frame is the authoring
function that emitted the instruction, not a device function.

## The recipe

```bash
export PYTHONPATH=python                     # from rocke/platform/
ROCKE_DEBUG_LOC=1 rocgdb --args python3 your_bench.py
```

```text
(gdb) set pagination off
(gdb) set amdgpu precise-memory on       # NOT optional -- see below
(gdb) run
```

On the fault you get the wave, the lane, the line, and the source text:

```text
Thread 68 "rocke_eleme-2920" received signal SIGSEGV, Segmentation fault.
[Switching to thread 68, lane 0 (AMDGPU Lane 1:1:1:1/0 (508,0,0)[0,0,0])]
0x00007fff524237a0 in build_elementwise () at <...>/rocke/helpers/tensor_view.py:495
495	        b.global_store_vN(self.base, off, value, n)
```

`(508,0,0)[0,0,0]` is the workgroup and the thread within it — feed that straight back
into your host-side index math, since a fault confined to high workgroup ids is the
signature of a tail/partial-tile addressing bug rather than a broken kernel.

## `set amdgpu precise-memory on` is mandatory

Without it the reported location is wrong, and rocgdb says so:

```text
Warning: precise memory violation signal reporting is not enabled, reported
location may not be accurate.  See "show amdgpu precise-memory".
...
0x00007fff52423a40 in rocke_elementwise_add_f16_b256_v8 () at <...>/tensor_view.py:415
415	            b.global_store(self.base, off, value)
warning: Current lane is inactive.
```

That PC is `s_endpgm` — the *end of the kernel*, with `exec` already `0x0` — and the
line it names is a different store than the one that actually faulted. Memory
instructions retire asynchronously, so with imprecise reporting the wave has run on
past the offender by the time the signal lands. Turn it on and the same fault reports
the correct line.

Set it before `run`. Checking it with `show amdgpu precise-memory` before there is a
process prints `is on (currently disabled)`; that is expected, it takes effect once the
agent is attached.

## Finding the exact faulting instruction

Even with precise reporting the PC sits a little past the memory op, so disassemble
backwards from it and look for the load or store just above the `=>`:

```text
(gdb) disassemble $pc-40,$pc+8
   ...+144:	v_lshl_add_u64 v[4:5], s[8:9], 0, v[8:9]
   ...+152:	global_store_dwordx4 v[4:5], v[0:3], off      <-- the faulting store
=> ...+160:	s_andn2_saveexec_b64 s[0:1], s[0:1]
   ...+164:	s_cbranch_execz 166
```

The address it tried to touch is in the operand registers of that instruction —
`v[4:5]` above — so `p/x $v4` and `p/x $v5` give you the computed pointer per lane,
which is how you check an addressing bug without any variable debug info:

```text
(gdb) p/x $v0
$1 = {0x0 <repeats 64 times>}
(gdb) info registers s0 s1 s2 s3
s2             0x1fc               508          # workgroup id, matches the stop banner
```

## Breakpoints on an authoring line

Useful for wrong-output bugs, not just faults. The code object is loaded late (comgr
compiles in-process), so the breakpoint has to be allowed to stay pending:

```text
(gdb) set breakpoint pending on
(gdb) break tensor_view.py:495
Breakpoint 1 (tensor_view.py:495) pending.
(gdb) run
Thread 68 hit Breakpoint 1, with lanes [0-63], store_vec () at <...>/tensor_view.py:495
```

Without `set breakpoint pending on` the `break` is refused with `No symbol table is
loaded`, because at that point the kernel does not exist yet.

## Orienting commands

```text
(gdb) info agents        # the GPU agents, with architecture
(gdb) info dispatches    # grid, workgroup size, kernel name of each live dispatch
(gdb) info threads       # every wave, with the line each one is sitting on
```

`info dispatches` is the quickest confirmation that the launch geometry is what you
intended:

```text
  Id   Target Id                      Grid          Workgroup Fence   Kernel Function
* 1    AMDGPU Dispatch 1:1:1 (PKID 2) [2097152,1,1] [256,1,1] B|Aa|Ra rocke_elementwise_add_f16_b256_v8
```

`info threads` lists one entry per wave and is long; it is worth reading when a fault
is confined to a subset of waves, because the line they share is the suspect.

## Without `ROCKE_DEBUG_LOC`

The debugger still works, it just has nothing to map addresses onto. The frame names
the kernel symbol and reads the code object straight out of the process:

```text
0x00007fff524237a0 in rocke_elementwise_add_f16_b256_v8 () from memory://2842384#offset=0x15d61e0&size=5848
(gdb) info line *$pc
No line number information available for address 0x7fff524237a0 <rocke_elementwise_add_f16_b256_v8+160>
```

`disassemble`, the register file, `info dispatches`, and the workgroup/lane of the
fault all still work — so an ISA-level session is perfectly viable, you just have to
map offsets back to the builder yourself. The `memory://` URI is also the reason no
setup is needed to point rocgdb at a code object: it reads the one comgr loaded.
