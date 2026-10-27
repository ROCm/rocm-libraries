# rocKE - Python to C++ Translator RFC

- Contributors: John Afaganis
- **Status**: Planned
- **Status notes**: The translator is on a local branch (`platform/tools/translator/`) and is not merged. It was built and checked on the attention backward C++ port (#14365, open). The four backward modules it generates land after that port.
- **Last verified**: 2026-10-27 against `develop` (6ca11545199)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Contract](#5-contract)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Risks](#8-risks)
9. [Execution Plan](#9-execution-plan)
10. [Testing Plan](#10-testing-plan)
11. [Future Considerations](#11-future-considerations)
12. [Glossary](#12-glossary)

## 1. Executive Summary

This RFC defines a translator that writes the C++ twin of a rocKE Python kernel-builder module from the Python itself. rocKE keeps each kernel builder twice: the Python module under `library/kernels/common/` is the authoring surface, and its C++ twin under `platform/cpp/instances/common/` is what the C++ engine and the hipDNN provider link. Today every twin is written by hand.

The translator reads a Python module, the types recorded from its parity configs, and the C headers of the modules it calls. It writes the module's `.cpp` and its C header. Every function either translates by a fixed rule or becomes a hole that a person fills once by hand. The translator refuses to write output while any hole is open. The existing byte-identity gate then proves that the C++ emits the same IR as the Python on every parity config.

The one command is:

```
python -m tr.regen <module>...            # write the files that do not exist yet
python -m tr.regen --pretend <module>...  # list what would be written or skipped
```

This RFC does not change the Python modules, the IR, the byte-identity gate, or any existing C++ twin. It does not make the generated C++ the only copy (see [11. Future Considerations](#11-future-considerations)).

## 2. Problem Statement

The Python builder and its C++ twin must emit byte-identical LLVM IR. The C++ is a line-by-line port with one hard rule: every call that emits IR must run in the same order as in Python. Porting by hand is slow and error-prone:

1. The attention backward port (#14365) cost about 9.5M agent tokens and about 12.5 hours of agent runs for about 17k lines of C++.
2. 61% of porter time was model inference, mostly re-deriving the same order rule and the same C encodings in every file. The most-read files were read 150-190 times.
3. Most review findings were mechanical: stale comments, duplicate symbols, name prefixes, an unused constant that only clang flagged.
4. Every later change to a Python module needs the same change by hand in its twin, and the two drift between gates.

The order rule, the type encodings and the naming are fixed policy. A program can apply fixed policy the same way every time, in seconds. People are needed only for the code the policy does not cover.

## 3. Current System Overview

The pieces the translator builds on are on `develop` today:

1. **Python builders**: `library/kernels/common/<module>.py` build a kernel through `rocke.core.ir.IRBuilder`.
2. **C++ twins**: `platform/cpp/instances/common/<module>.cpp` build the same kernel through the C IR builder (`platform/cpp/include/rocke/ir.h`). Their C headers are `platform/cpp/include/rocke/helper_rocke.instances.common.<module>.h`.
3. **Parity drivers**: `library/tests/parity/<module>_emit.py` and `<module>_emit.c` run one module over its list of configs, in Python and in C.
4. **Byte-identity gate**: `platform/tools/check_byte_identity.py` builds the engine and compares the `.ll` text of both sides, per family and per LLVM flavor (`ROCKE_LLVM_FLAVOR=llvm22` and `llvm20`).

The dual-backend design ([dual_backend_unification_rfc.md](./dual_backend_unification_rfc.md)) keeps both implementations on purpose and uses the gate as the permanent guarantee. This RFC keeps that model. It changes only who writes the C++.

## 4. Proposed Design

### 4.1 Pipeline

```
 Python module ──► Record ──► Lower ──► Emit ──► .cpp + .h ──► Gate
       │             │          │         │
       │      parity configs  rules    prelude
       │       (types)        │         (rocke_py.hpp)
       └──── C headers of callees (sigmap) ──┘
                              │
                        refusal = hole ──► hand fill (holes/<module>/)
```

| Stage | Module | What it does |
|---|---|---|
| Read the tree | `tr/env.py`, `tools/cscan.py`, `tools/mine_maps.py` | Index the C headers and pair each Python function with its C twin (`sigmap.json`, `twin_map.json`, mined from the tree in a few seconds) |
| Record | `tr/record.py` | Run every parity config in-process and record the types each function's arguments and results took (`data/facts/<module>.json`) |
| Type | `tr/ctypes.py`, `tr/kernel.py` | Map each recorded type to a C type from fixed tables. No inference |
| Lower | `tr/lower.py` | Rewrite each function body to C++ in A-normal form: every call that can emit IR is bound to its own local, inner calls first, left to right. Rule dispatch for the dynamic Python features. Anything no rule matches is a refusal |
| Header | `tr/abi.py`, `tr/family.py`, `tr/host.py` | Derive the module's C header from the Python (`designs/generated_abi.md`) |
| Emit | `tr/translate.py` | Write the `.cpp`: internal C++ plus one `extern "C"` entry per export, formatted with clang-format 18.1.4 |
| Regenerate | `tr/regen.py` | Run all of the above for a set of modules and write the tree, or check it |

### 4.2 Type Recording

Python builders are mostly unannotated. The translator does not infer types. `tr.record` runs the module's parity configs with a profile hook at function boundaries and records the types that actually occur. A function whose argument was `int` in some configs and `Value` in others gets the `int | Value` type `rocke_<pkg>_iv_t`. A function no config reaches is a hole, not a guess.

### 4.3 Generated C Header

The translator derives the header from the Python instead of matching a hand header (`designs/generated_abi.md`). The export set is the module's `__all__` plus the methods of exported classes. Each Python shape has one C shape:

| Python | C |
|---|---|
| function `f` | `rocke_<pkg>_f` |
| class `C` | `rocke_<pkg>_<snake C>_t`, dataclass fields in order |
| method `m` / classmethod `m` | `_<m>` with `self` first / without `self` |
| `IRBuilder` | `rocke_ir_builder_t* b` |
| `Value`, `Value \| None` | `rocke_value_t*` (NULL is None) |
| `int`, `bool`, `float`, `str` | `int`, `bool`, `double`, `const char*` |
| a value of one of several classes | one tagged struct, kind tag first, with generated dispatch (`designs/tagged_values.md`) |
| code run on host ints and on IR values | one generated version per value kind (`designs/host_clones.md`) |

Calls into modules that are not regenerated use their hand header through the mined `sigmap.json`.

### 4.4 Holes and Hand Fills

A hole is a function no rule covers. The generated `.cpp` holds `#error "hole: <name>: <reason> (line <n>)"` in its place, so a hole never builds by accident. A person writes the fill once:

- `holes/<module>/<function>.cpp`: the C entry, plain C++ with the header's signature, starting with a comment that quotes the Python it stands for.
- `holes/<module>/<Class>.fields`: C-only struct fields a fill needs.

`python -m tr.translate <module> --analyze` lists every function as `translated`, `filled` or `hole`, with the reason and the Python line, before anything is written.

### 4.5 Prelude

`rocke_py.hpp` holds the C++ helpers with Python semantics that generated code calls: floor division and modulo, negative indexing, `str` indexing and slicing, and `dict` and `list` shapes. Its master copy ships with the translator in `tools/translator/tr/prelude/`. A package may add an adapters header (`rocke_py_adapters.hpp`, package data in `tr/packages/<package>/`) for calls whose C twin has a different shape than the Python; a generated source includes it only when it uses an adapter. Generated sources include both by a quoted same-directory name. `tr.regen` copies the ones a written source includes into `platform/cpp/instances/common/`, never over an existing copy. There they are private: compiled with the sources, not installed and not API.

### 4.6 Packages

`tr/package.py` maps a package to its C naming and its modules. The one package today is `attention_bwd` (`rocke_attn_bwd_*`, `ROCKE_ATTN_BWD_*`). Not yet built: entries for the other packages on `develop`, such as the forward attention modules (`rocke_fmha_varlen_*`) and the convolution modules. Until a package has an entry, the translator cannot pair that package's Python functions with their C twins, and `--analyze` reports every function as a hole.

### 4.7 Generated Files in the Tree

The translator writes a first draft. The `.cpp` and `.h` files are checked in, and their first lines say `first draft from library/kernels/common/<module>.py:<sha> by tr/translate.py; hand-maintained from here`. From then on they are hand code, edited like any other twin. `tr.regen` never writes over an existing source, header or support header unless it is given `--overwrite`.

## 5. Contract

`python -m tr.regen [--overwrite] [--pretend] [--rocke <root>] <module>...` has the following contract:

- `<root>` is `.../dnn-providers/hip-kernel-provider/rocke`. The default is `ROCKE_ROOT`, else the tree the translator sits in.
- Every module named must have a parity driver `<module>_emit` and recorded facts under `data/facts/`.
- Modules whose generated headers the others include must be named in the same command.
- If any function is a hole with no fill, nothing is written and the exit code is 1. The holes are listed.
- On success the `.cpp` files go to `platform/cpp/instances/common/` and the headers to `platform/cpp/include/rocke/`.
- A target that already exists is skipped (`skip <path> (exists)`, exit 0). Running it twice changes nothing. `--overwrite` writes over existing files; `--pretend` writes nothing and lists what would be written or skipped.
- When every module named already has its header in the tree, the sources are translated against those headers and none is generated; `--generated-abi` overrides that.
- The translator does not decide correctness. The byte-identity gate does, at both LLVM flavors, with the engine built by gcc and clang with warnings as errors.

## 6. Key Design Decisions

- **The gate is the only acceptance test.** The translator never claims a translation is right. Byte-identical `.ll` over every parity config, at both LLVM flavors, is the proof. This is the test the hand port already used.
- **Fixed rules or a hole, never a guess.** A rule either emits its fixed C++ template or refuses. A refusal is visible (`--analyze`, `#error`) and is filled once by a person.
- **Types come from running the configs, not from inference.** The recorder sees the types that actually occur. Code no config reaches is a hole.
- **Call order is structural.** A-normal form binds every emitting call to its own local in Python order, so the C argument-evaluation order cannot reorder IR.
- **The header is generated.** Matching hand headers made every invented C shape a hole. Deriving the header from the Python removes those holes, and a new port has no hand header anyway.
- **The translated C++ is a first draft, checked in as hand code.** Readers, the build and CI see normal C++. The translator never writes over it; the byte-identity gate keeps it correct, as for every twin.
- **Policy is data.** Naming, adapters, fills and recorded types are tables keyed by package, module and function name, not code paths.

### 6.1 Measured Results

On the attention backward package:

- `_attention_bwd_valu`, `_attention_bwd_counted`, `attention_bwd_aux` and `_attention_bwd_addr` regenerate byte-identical on all 31 attention families, 3,083 configs per LLVM flavor, with gcc and clang.
- Over the eight larger backward modules (addr, band, sinks, ring, frag, gemm, sched, lds), 77.0% of Python lines (3,945 of 5,122) and 393 of 447 functions translate mechanically. Hand fills count as not mechanical.
- Each module translates in under 2 seconds.

## 7. Alternatives Considered

- **Keep porting by hand.** Known cost: the backward port above. Every later Python change is a second hand change.
- **An LLM ports each file.** This is how the backward port was done. The cost is dominated by re-reading context and re-deriving fixed policy per file, and each port is a new chance to break the order rule.
- **Record and replay the IR (`portable_ir`).** Ship the IR the Python emits instead of a C++ builder. This removes the C++ twin entirely, but it fixes the kernel at record time and does not cover builders whose output depends on runtime arguments. It is a separate decision (see [11. Future Considerations](#11-future-considerations)).
- **Make the C++ the only source.** This drops the Python authoring surface the dual-backend design keeps on purpose.

## 8. Risks

1. **Code unlike the attention backward package**: The rules were built on one package. Another package may hit hole kinds no rule covers. Mitigation: `--analyze` measures the hole count before any work, and a hole is filled by hand, never guessed.
2. **Recorded types miss a case**: A type no parity config produces is not recorded. Mitigation: unreached code is a hole, not a translation; adding a parity config is the fix, and it also strengthens the gate.
3. **Stale twins**: Someone edits the Python and not the C++. Mitigation: the byte-identity gate, as for every twin.
4. **Hand edits to translated files**: These are the normal case. `tr.regen` never writes over an existing file without `--overwrite`.
5. **Generated code is harder to read than hand code**: A-normal form adds locals. Mitigation: clang-format, Python names kept, and the `.cpp` keeps the module's structure function by function.
6. **Translator maintenance**: The translator is about 16k lines of Python. Mitigation: rules are data-driven and keyed by shape; a new hole kind is one rule or one fill, and the gate checks every change.

## 9. Execution Plan

### 9.1 Files

`platform/tools/translator/`

- `tr/`: the translator (sections 4.1-4.7).
- `tools/cscan.py`, `tools/rsrc.py`, `tools/mine_maps.py`: read the tree and mine the code maps.
- `designs/generated_abi.md`, `designs/tagged_values.md`, `designs/host_clones.md`: the header rules.
- `HOWTO.md`: setup, the one command, holes, the gate, limits.

`platform/tools/translator/tr/prelude/rocke_py.hpp`, `platform/tools/translator/tr/packages/attention_bwd/rocke_py_adapters.hpp`

- The prelude and the attention backward adapters (master copies; nothing under `platform/cpp`).

Per generated module, added with that module:

- `platform/cpp/instances/common/<module>.cpp` and its header, regenerated.
- `platform/tools/translator/data/facts/<module>.json`, `data/abi/<module>.names.json`.
- `platform/tools/translator/holes/<module>/`.

### 9.2 Implementation Order

1. Land the translator alone: `platform/tools/translator/`, prelude included. No module is generated and no existing file changes.
2. After #14365 lands: regenerate the four attention backward modules, add their data, fills and the package adapters header, and gate.
3. Add a package entry for one forward module on `develop` and regenerate it. Not yet built.
4. Use the translator for the next port, and measure its cost against the hand port. Not yet built.

## 10. Testing Plan

Translator change (any edit under `platform/tools/translator/` or to `rocke_py.hpp`):

- The self-test (`python -m pytest tools/translator/tests`) passes.
- Byte-identity gate over all families the generated modules serve, at `llvm22` and `llvm20`, with the expected config count.
- Engine build with gcc and clang, `-Wall -Wextra -Werror`.
- `rocke_py.hpp` compiles alone with both compilers.

New generated module:

- `--analyze` shows no open hole.
- Gate as above. A low config count means the run stopped early, not that coverage is complete.
- Running `tr.regen` a second time skips every file and leaves `git status` clean.

Not yet built: unit tests for individual rules, with each rule's counterexamples as cases.

## 11. Future Considerations

- **Hole kit.** Not yet built. An LLM fills a hole from its signature, recorded types and the slice of the op trace it must reproduce, and the gate checks it. Hole kinds that recur become rules.
- **Single source.** Once a package is fully generated, its C++ could be produced at build time instead of checked in. That needs the translator in the build and a decision on reviewing generated code.
- **`portable_ir`.** Record and replay may serve builders whose output does not depend on runtime arguments. It can sit beside the translator.
- **More packages.** Forward attention and convolution have hand twins on `develop` and parity drivers, so they are the next candidates.

## 12. Glossary

- **Twin**: The C++ implementation of a Python builder module that must emit the same IR.
- **Parity config**: One set of builder arguments in a `<module>_emit` driver. The gate runs each one in Python and in C++.
- **Byte-identity gate**: `platform/tools/check_byte_identity.py`, which compares the `.ll` text of both sides.
- **A-normal form (ANF)**: A form where every call is bound to its own local before use, so evaluation order is explicit.
- **Hole**: A function the translator has no rule for. It needs a hand fill.
- **Fill**: The hand-written C++ for a hole, in `holes/<module>/`.
- **Prelude**: `rocke_py.hpp`, the hand-written C++ helpers with Python semantics.
- **Package**: A set of modules that share a C naming scheme (`tr/package.py`).
- **Mechanical share**: The share of Python lines whose C++ the translator writes with no fill.
