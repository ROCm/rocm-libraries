# rocKE - Python to C++ Translator (Porting Assistant) RFC

- Contributors: John Afaganis
- **Status**: Planned
- **Status notes**: Experimental. The translator is a porting assistant on a branch (`platform/tools/translator/`) and is not merged. It writes the first draft of a C++ twin; the checked-in C++ is hand code from its first commit. It adds no file under `platform/cpp`, no CMake target and no CI step. It was built and checked on the attention backward C++ port (#14365, open); the first drafts of four backward modules can land after that port.
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

This RFC defines an experimental porting assistant: a translator that writes the first draft of the C++ twin of a rocKE Python kernel-builder module from the Python itself. rocKE keeps each kernel builder twice: the Python module under `library/kernels/common/` is the authoring surface, and its C++ twin under `platform/cpp/instances/common/` is what the C++ engine and the hipDNN provider link. Today every twin is written by hand.

The translator reads a Python module, the types recorded from its parity configs, and the C headers of the modules it calls. It writes the module's `.cpp` and, for a new port, its C header. Every function either translates by a fixed rule or becomes a hole that a person fills by hand. The translator refuses to write output while any hole is open. The existing byte-identity gate then proves that the C++ emits the same IR as the Python on every parity config.

The output is a first draft. Once it is checked in, the C++ is ordinary hand code: people and agents edit it as they edit every twin today, and the translator never writes over it. The byte-identity gate is the correctness guard, as it is for every hand twin.

The one command is:

```
python -m tr.regen <module>...            # write the files that do not exist yet
python -m tr.regen --pretend <module>...  # list what would be written or skipped
```

This RFC does not change the Python modules, the IR, the byte-identity gate, the build, CI, or any existing C++ twin. The translator lives under `platform/tools/translator/`, owns no file outside it, and nothing in the build or CI runs it: it is a standalone developer tool that can be ignored or removed with no effect on the tree. It does not make the translated C++ the only copy (see [11. Future Considerations](#11-future-considerations)).

## 2. Problem Statement

The Python builder and its C++ twin must emit byte-identical LLVM IR. The C++ is a line-by-line port with one hard rule: every call that emits IR must run in the same order as in Python. Porting by hand is slow and error-prone:

1. The attention backward port (#14365) cost about 9.5M agent tokens and about 12.5 hours of agent runs for about 17k lines of C++.
2. 61% of porter time was model inference, mostly re-deriving the same order rule and the same C encodings in every file. The most-read files were read 150-190 times.
3. Most review findings were mechanical: stale comments, duplicate symbols, name prefixes, an unused constant that only clang flagged.
4. Every later change to a Python module needs the same change by hand in its twin, and the two drift between gates.

The order rule, the type encodings and the naming are fixed policy. A program can apply fixed policy the same way every time, in seconds. People are needed only for the code the policy does not cover. This RFC addresses the first port of a module (points 1-3). Point 4, keeping a twin in step after the first port, stays with the people and agents who edit the twin, guarded by the gate.

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
| Regenerate | `tr/regen.py` | Run all of the above for a set of modules and write the files that do not exist yet |

### 4.2 Type Recording

Python builders are mostly unannotated. The translator does not infer types. `tr.record` runs the module's parity configs with a profile hook at function boundaries and records the types that actually occur. A function whose argument was `int` in some configs and `Value` in others gets the `int | Value` type `rocke_<pkg>_iv_t`. A function no config reaches is a hole, not a guess.

Recorded types are keyed by the function's qualified name (`co_qualname`, with a `#<n>` ordinal for generator expressions, lambdas and comprehensions), not by line number, so moving code does not lose them. Each entry also stamps a hash of the function's source. When a body has changed since the recording, the translator prints `facts stale for <module>.<function>; re-run tr.record` and translates as before; it never drops types silently.

### 4.3 Generated C Header

For a new port, the translator derives the header from the Python instead of matching a hand header (`designs/generated_abi.md`). The export set is the module's `__all__` plus the methods of exported classes. Each Python shape has one C shape:

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

Calls into modules that are not translated use their existing header through the mined `sigmap.json`. When every module named already has its header in the tree, hand-written or an earlier draft, `tr.regen` translates against those headers and generates none, so a new draft fits the existing callers.

### 4.4 Holes and Hand Fills

A hole is a function no rule covers. The translated `.cpp` holds `#error "hole: <name>: <reason> (line <n>)"` in its place, so a hole never builds by accident. A person writes the fill, which the translator reuses for that module:

- `holes/<module>/<function>.cpp`: the C entry, plain C++ with the header's signature, starting with a comment that quotes the Python it stands for.
- `holes/<module>/<Class>.fields`: C-only struct fields a fill needs.

`python -m tr.translate <module> --analyze` lists every function as `translated`, `filled` or `hole`, with the reason and the Python line, before anything is written.

### 4.5 Prelude

`rocke_py.hpp` holds the C++ helpers with Python semantics that translated code calls: floor division and modulo, negative indexing, `str` indexing and slicing, and `dict` and `list` shapes. Its master copy ships with the translator in `tools/translator/tr/prelude/`. A package may add an adapters header (`rocke_py_adapters.hpp`, package data in `tr/packages/<package>/`) for calls whose C twin has a different shape than the Python; a translated source includes it only when it uses an adapter. Translated sources include both by a quoted same-directory name.

`tr.regen` copies the ones a written source includes into `platform/cpp/instances/common/`, never over an existing copy. There they are private: compiled into `rocke_core` with the sources, not installed (the install rule covers only `cpp/include/`) and not API. The in-tree copy is hand code like the sources: a fix to it, for example for a new compiler warning, is made in the tree. The prelude grows with the rules (most changes to it came with a new rule), so it is not a fixed file.

### 4.6 Packages

`tr/package.py` maps a package to its C naming and its modules. The one package today is `attention_bwd` (`rocke_attn_bwd_*`, `ROCKE_ATTN_BWD_*`). Not yet built: entries for the other packages on `develop`, such as the forward attention modules (`rocke_fmha_varlen_*`) and the convolution modules. Until a package has an entry, the translator cannot pair that package's Python functions with their C twins, and `--analyze` reports every function as a hole.

### 4.7 Ownership of Translated Files

The translator writes a first draft. The `.cpp` and `.h` files are checked in, and their first lines say `first draft from library/kernels/common/<module>.py:<sha> by tr/translate.py; hand-maintained from here`. From then on:

- They are hand code, maintained with the module like every other twin, and edited by hand or by agents as today.
- `tr.regen` never writes over an existing source, header or support header unless it is given `--overwrite`.
- The byte-identity gate is the correctness guard, as for every twin.
- There is no check that a file still matches the translator's output and no generator-owned mode. A hand edit is never an error.

### 4.8 What a First Port Adds to the Tree

The translator change itself adds files only under `platform/tools/translator/`. It adds no header and no source under `platform/cpp`, and no CMake, install or CI change.

A first port of a module with the translator adds:

- one `.cpp` in `platform/cpp/instances/common/`;
- the private prelude copy (and the package adapters header, if used) in the same directory, if not already there;
- with a generated header, one `helper_rocke.instances.common.<module>.h` in `platform/cpp/include/rocke/`, which is installed. This is the same per-module header a hand port adds (14 such headers are on `develop`), and callers outside `instances/common` include it by its `rocke/` path. It is no new kind of installed header;
- under `platform/tools/translator/`: `data/facts/<module>.json`, `data/abi/<module>.names.json` and, if needed, `holes/<module>/`.

A hand port of the same module adds the same `.cpp` and header.

## 5. Contract

`python -m tr.regen [--overwrite] [--pretend] [--hand-abi | --generated-abi] [--rocke <root>] <module>...` has the following contract:

- `<root>` is `.../dnn-providers/hip-kernel-provider/rocke`. The default is `ROCKE_ROOT`, else the tree the translator sits in.
- Every module named must have a parity driver `<module>_emit` and recorded facts under `data/facts/`. Facts are keyed by qualified name; a facts entry whose source stamp no longer matches prints a stale warning, and the fix is to re-run `tr.record` after the Python change it names.
- Modules whose generated headers the others include must be named in the same command.
- If any function is a hole with no fill, nothing is written and the exit code is 1. The holes are listed.
- On success the `.cpp` files go to `platform/cpp/instances/common/` and the headers to `platform/cpp/include/rocke/`.
- A target that already exists is skipped (`skip <path> (exists)`, exit 0). Running it twice changes nothing. `--overwrite` writes over existing files; `--pretend` writes nothing and lists what would be written or skipped.
- When every module named already has its header in the tree, the sources are translated against those headers and none is generated; `--generated-abi` overrides that. When only some have one, the command stops and asks for `--hand-abi` or `--generated-abi`.
- The translator does not decide correctness. The byte-identity gate does, at both LLVM flavors, with the engine built by gcc and clang with warnings as errors.

## 6. Key Design Decisions

- **The gate is the only acceptance test.** The translator never claims a translation is right. Byte-identical `.ll` over every parity config, at both LLVM flavors, is the proof. This is the test the hand port already used.
- **Fixed rules or a hole, never a guess.** A rule either emits its fixed C++ template or refuses. A refusal is visible (`--analyze`, `#error`) and is filled once by a person.
- **Types come from running the configs, not from inference.** The recorder sees the types that actually occur. Code no config reaches is a hole.
- **Call order is structural.** A-normal form binds every emitting call to its own local in Python order, so the C argument-evaluation order cannot reorder IR.
- **A new port's header is generated.** Matching hand headers made every invented C shape a hole. Deriving the header from the Python removes those holes, and a new port has no hand header anyway. When a header exists, its ABI is used, so a new draft fits the callers.
- **The translated C++ is a first draft, checked in as hand code.** Readers, the build and CI see normal C++. The translator never writes over it; the byte-identity gate keeps it correct, as for every twin.
- **Hand and agent edits are normal.** In about four months the existing C++ twins took 48 edits after their first port: 42 commits that changed Python and C++ together and 6 C++-only commits. 35 of the 65 commits that touched C++ (54%) carry an agent co-author trailer, and 15 authors edited C++ twins. After the attention backward port, its C++ took 22 C++-only edits, none of them following a Python change. No generator existed, so every one of these edits was manual by construction; the figures show that edits are the normal way twins are kept, not that generation could not do it. A tool that owns its output would turn each of them into a conflict, so the translator owns none.
- **The translator owns no file outside its directory.** Nothing in the build or CI runs it, so a broken or removed translator breaks no build and blocks no change.
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
- **Generator-owned C++ with a CI currency check.** An earlier version marked the files `do not edit`, regenerated them on every change and failed CI when they differed. Rejected: twins are edited by hand and by agents as a matter of course (section 6), every such edit would become a failure or be lost on the next regeneration, and it would put the translator in CI.

## 8. Risks

1. **Code unlike the attention backward package**: The rules were built on one package. Another package may hit hole kinds no rule covers. Mitigation: `--analyze` measures the hole count before any work, and a hole is filled by hand, never guessed.
2. **Recorded types miss a case**: A type no parity config produces is not recorded. Mitigation: unreached code is a hole, not a translation; adding a parity config is the fix, and it also strengthens the gate.
3. **Stale twins**: Someone edits the Python and not the C++. This already happens with hand twins. Mitigation: the byte-identity gate, as for every twin; it catches IR differences for the configs it covers. The translator adds no currency check of its own.
4. **Hand edits to translated files**: These are the normal case. Translated files are hand code from the first commit, and `tr.regen` never writes over an existing file without `--overwrite`. Residual: a hand-shaped file drifts from the translator's shape, so a later draft of the same module is a starting point to compare, not a drop-in replacement.
5. **Generated code is harder to read than hand code**: A-normal form adds locals. Mitigation: clang-format, Python names kept, and the `.cpp` keeps the module's structure function by function.
6. **Translator maintenance**: The translator is about 14k lines of Python in `tr/`, about 2.8k in `tools/`, and a 937-line prelude. Per-module data (facts, names, fills) goes stale with the Python and is re-recorded by whoever ports with it. Mitigation: the translator owns no file outside `platform/tools/translator/`, and nothing in the build or CI depends on it, so a broken translator breaks no build and blocks no change; it can be ignored or removed. Rules are data-driven and keyed by shape; a new hole kind is one rule or one fill. The self-test (section 10) and the gate check every change to it.
7. **Narrow audience**: New library attention work is moving to Python only, so the remaining hand-twin ports are mainly convolution, GEMM and MoE. The translator only pays for itself if those ports use it; until then its cost is the directory itself.

## 9. Execution Plan

### 9.1 Files

`platform/tools/translator/`

- `tr/`: the translator (sections 4.1-4.7).
- `tools/cscan.py`, `tools/rsrc.py`, `tools/mine_maps.py`: read the tree and mine the code maps.
- `designs/generated_abi.md`, `designs/tagged_values.md`, `designs/host_clones.md`: the header rules.
- `tr/packages/<package>/`: per-package support headers.
- `tests/`: the self-test (pytest) and its fixture root (pinned modules, their data, goldens).
- `HOWTO.md`: setup, the one command, holes, the self-test, the gate, limits.

`platform/tools/translator/tr/prelude/rocke_py.hpp`, `platform/tools/translator/tr/packages/attention_bwd/rocke_py_adapters.hpp`

- The prelude and the attention backward adapters (master copies; nothing under `platform/cpp`).

No file under `platform/cpp`, no `CMakeLists.txt` and no CI file changes.

Per translated module, added with that module (section 4.8):

- `platform/cpp/instances/common/<module>.cpp` and its header, a first draft checked in as hand code.
- `platform/tools/translator/data/facts/<module>.json`, `data/abi/<module>.names.json`.
- `platform/tools/translator/holes/<module>/`.

### 9.2 Implementation Order

1. Land the translator alone: `platform/tools/translator/`, prelude included. No module is translated and no existing file changes.
2. After #14365 lands: optionally compare first drafts of the four attention backward modules with the hand port, and add their data and fills to the translator. The checked-in C++ stays hand code.
3. Add a package entry for one forward module on `develop` and regenerate it. Not yet built.
4. Use the translator for the next port, and measure its cost against the hand port. Not yet built.
5. Per-function updates and a drift report (section 11) start only when a second module family is ported with the translator, and only after one maintenance change has been measured with and without them.

## 10. Testing Plan

Translator change (any edit under `platform/tools/translator/`):

- The self-test (`python -m pytest tools/translator/tests`, from `platform/`) passes. It is run by whoever changes the translator; no CMake target and no CI lane runs it. It needs no GPU and no build. It translates pinned copies of two attention backward modules from its own fixture root and compares them with goldens, checks determinism, the facts keys and stale warnings, the never-overwrite rule of `tr.regen`, and that the prelude compiles alone. Goldens change only with `--update-goldens`.
- Byte-identity gate over all families the translated modules serve, at `llvm22` and `llvm20`, with the expected config count.
- Engine build with gcc and clang, `-Wall -Wextra -Werror`.
- `tr/prelude/rocke_py.hpp` compiles alone with both compilers (part of the self-test).

New translated module:

- `--analyze` shows no open hole.
- Gate as above. A low config count means the run stopped early, not that coverage is complete.
- Running `tr.regen` a second time skips every file and leaves `git status` clean.

Not yet built: unit tests for individual rules, with each rule's counterexamples as cases. Today a rule change is checked through the goldens and the gate.

## 11. Future Considerations

- **Hole kit.** Not yet built. An LLM fills a hole from its signature, recorded types and the slice of the op trace it must reproduce, and the gate checks it. Hole kinds that recur become rules.
- **Per-function updates and a drift report.** Not yet built, and started only on the trigger in section 9.2. `--function` would print the translation of one function for a person or agent to merge into the hand file, and a drift report would list translated functions whose Python changed since their draft. Neither would write into the tree.
- **Single source.** Once a package is fully translated, its C++ could be produced at build time instead of checked in. That needs the translator in the build and a decision on reviewing generated code, so it needs its own proposal.
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
- **First draft**: The translator's output for a module. Once checked in it is hand code.
- **Package**: A set of modules that share a C naming scheme (`tr/package.py`).
- **Mechanical share**: The share of Python lines whose C++ the translator writes with no fill.
