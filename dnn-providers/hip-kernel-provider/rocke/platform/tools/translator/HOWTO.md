# Python to C++ translator

The translator (`tr/`) turns a rocKE Python kernel-builder module
(`library/kernels/common/<module>.py`) into its C++ twin
(`platform/cpp/instances/common/<module>.cpp`) and, with the generated ABI,
the module's C header (`platform/cpp/include/rocke/`). It translates every
function it has a rule for, uses a hand-written fill for the rest, and refuses
to write output while any function is still a hole. The byte-identity gate
then proves the C++ emits the same IR as the Python on every parity config.

A generated source says so in its first lines
(`generated from library/kernels/common/<module>.py:<sha> by tr/translate.py`).
Never edit them by hand: the next regenerate overwrites them. Change the
Python, a fill under `holes/`, or a rule in `tr/`, then regenerate.

## Layout

| Path | What it is |
|---|---|
| `tr/` | The translator. `tr/regen.py` is the one command. |
| `platform/cpp/include/rocke/rocke_py.hpp` | The prelude: hand-written C++ helpers (Python semantics) every generated source includes. This is the only copy; edit it there. `tr.regen` does not write it. |
| `platform/cpp/include/rocke/rocke_py_adapters.hpp` | A package's adapters between Python helper shapes and their C twins (listed in `tr/adapters.py`). It ships with that package's generated modules; a generated source includes it only when it uses one. |
| `tools/` | `cscan.py` (C/C++ declaration scanner), `rsrc.py` and `mine_maps.py` (mine the code maps from the tree). |
| `designs/generated_abi.md` | The rules that turn a Python module into its C header. |
| `designs/tagged_values.md` | A value that can be one of several classes becomes one tagged C struct with generated dispatch. |
| `designs/host_clones.md` | Code that runs on host ints and on IR values is generated once per value kind. |
| `data/facts/<module>.json` | Types recorded from the parity configs (`tr.record`). Added with each generated module. |
| `data/abi/<module>.names.json` | Generated names that keep an existing spelling. Added with each generated module. |
| `holes/<module>/` | Hand fills (`<function>.cpp`) and C-only struct fields (`<Class>.fields`). Added with each generated module. |

## Setup

- The Python rocKE uses, with its requirements and the
  `clang-format==18.1.4` wheel (the version the repository's pre-commit hook
  pins; another version can change the formatting). The translator imports the
  real Python modules of the tree it reads.
- The rocKE tree to read: `--rocke <root>` or `ROCKE_ROOT`, else the tree this
  directory sits in. `<root>` is `.../dnn-providers/hip-kernel-provider/rocke`.
- clang-format: `--clang-format <exe>`, else `CLANG_FORMAT`, else the one
  installed next to the running Python, else `clang-format` on PATH.
- Code maps (`sigmap.json`, `twin_map.json`): `TR_MAPS`, else `data/maps`.
  Neither is checked in here, so `tr.regen` mines them from the tree first
  (`tools/mine_maps.py`, a few seconds).
- Naming: `TR_PACKAGE` (or `tr.regen --package`) picks the C naming in
  `tr/package.py`. The default, `attention_bwd`, gives `rocke_attn_bwd_*` and
  `ROCKE_ATTN_BWD_*`. Another package is one more entry in `PACKAGES`.

## The one command

The examples name the attention backward modules, the package the translator
was built on. Those modules and their data arrive with that package. From this
directory:

```
python -m tr.regen _attention_bwd_addr _attention_bwd_valu _attention_bwd_counted attention_bwd_aux
```

For each module this generates the header (`tr.abi`) and the `.cpp`
(`tr.translate`), then writes the sources to
`platform/cpp/instances/common/` and the headers to
`platform/cpp/include/rocke/`. If any hole is left, it lists the holes and
writes nothing (exit 1). On an unchanged tree it reproduces the checked-in
files byte for byte.

To check that the checked-in generated files are current without writing
anything (the suggested CI step):

```
python -m tr.regen --check _attention_bwd_addr _attention_bwd_valu _attention_bwd_counted attention_bwd_aux
```

It regenerates into a temporary tree, compares the 8 generated files with
the checked-in ones, and exits 1 with the list of stale files on any
difference. Temporary directories (mined maps, headers, the check tree) are
removed on exit.

- Name every module whose generated header the others use in the same command
  (addr needs valu, counted and aux, as above).
- `--into <root>` writes into another tree. `--abi-dir <dir>` keeps the
  generated headers. `--hand-abi` translates against the tree's hand-written
  headers and generates none.

## Steps in a port

1. Record types. The translator types unannotated code from a recording of
   the parity configs:
   `python -m tr.record <module>_emit <module>` writes
   `data/facts/<module>.json`.
2. Look before writing:
   `python -m tr.translate <module> --abi <dir> --analyze` lists each function
   as `translated`, `filled` or `hole`, with the reason and the Python line.
   `TR_DEBUG=1` adds tracebacks for internal errors.
3. Fill only the holes it reports (next section).
4. Run `tr.regen`, then the gate.

## Holes and hand fills

A hole is a function with no translation rule. The generated `.cpp` then holds
`#error "hole: <name>: <reason> (line <n>)"` in its place. You write a fill:

- `holes/<module>/<function>.cpp` replaces the C entry of a refused function.
  Methods use `Class_method`. A C-only function with no Python twin (an ABI
  companion such as a `_count` function) uses its C name. The fill is plain
  C++ with the header's signature. Start it with a comment that quotes the
  Python it stands for.
- A dispatch fill (first lines contain `hole fill (dispatch`) is used even
  though the function translates. Its C entry first dispatches on a C-only
  struct field. That field is declared in `holes/<module>/<Class>.fields`,
  which `tr.abi` adds to the generated struct.

Fills and the hand routines in `tr/adapters.py` and the prelude are data
keyed by module and function name.

## The gate

From `platform/`, at each LLVM flavor:

```
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py --only attention
ROCKE_LLVM_FLAVOR=llvm20 python tools/check_byte_identity.py --only attention
```

The generated modules are shared by every attention backward family, so gate
all attention families, not only the module's own parity pair. Build the
engine with gcc and clang as CI does (warnings are errors). A low config count
means the run stopped early, not that coverage is complete.

## Limits

No module in this tree is generated yet. The translator was built and checked
on the attention backward port: its `valu`, `counted`, `aux` and `addr`
modules regenerate byte-identical to the Python on every attention parity
config (31 families, 3,083 configs per LLVM flavor, gcc and clang). Over the
eight larger backward modules (addr, band, sinks, ring, frag, gemm, sched,
lds), about 77% of the Python lines translate mechanically; hand fills count
as not mechanical.

- Naming: `tr/package.py` has one package, `attention_bwd`. Modules that name
  their C entries another way (for example `rocke_fmha_varlen_*`) need another
  entry before the translator pairs their functions with the C twins.
- Types come from a parity driver `<module>_emit` (`tr.record`); a module
  with no driver cannot be recorded.
- Not handled yet: mixed-arity tuples, `next()` over a generator, a
  module-level dataclass instance, an `@cache` keyed by a tuple, `==` on an
  optional int, a staticmethod whose first parameter looks like `self`, `set`
  values, calls into modules with no C twin (`rocke.analysis`), and generators
  as a C ABI (the `_count` / `_round` split is hand-listed in
  `tr/adapters.py` GENERATORS).
