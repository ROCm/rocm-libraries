# rocKE — C++ engine style guide

Conventions for the `rocke` C++ engine (`platform/cpp/`). Like the Python guide,
these rules are **derived from the existing code**: the goal is that new code is
indistinguishable from what is already here.

> **Scope.** This guide governs the **rocke C++ engine** under `platform/cpp/` — the
> C99-style, `extern "C"` port that mirrors the Python authoring frontend and emits
> byte-identical LLVM IR. It is self-contained; follow it for all engine C++.

> If a rule here conflicts with `AGENTS.md`, `AGENTS.md` wins — it owns the hard
> invariants (byte-identity, relative paths, cross-platform).

---

## 0. The paradigm (read this first)

The engine is a **C99-style port of the Python engine**, exposed through a **flat
`extern "C"` ABI** (nearly all public headers declare `extern "C"`). C++ is used only at
the *boundary* (exceptions/guards in `ckc::`, `.hpp`). Every translation unit mirrors
a specific Python module 1:1 — file headers literally say
"bucket N of the C99 port of rocke.core.ir" and carry a Python↔C99 correspondence
table. Two consequences drive everything below:

- **Byte-identity is the #1 invariant.** Each C++ TU must emit the same LLVM-IR bytes
  as its Python twin. Mirror every emission change in both engines and re-run the gate
  from `platform/` (build the C++ engine first — see AGENTS.md "Build / test / run"):
  `python tools/check_byte_identity.py`. A "style" cleanup that changes emitted text
  is not a style change.
- **The ABI stays C-callable.** Prefer C-expressible constructs on the public surface;
  keep C++ features to the internal boundary layer.

## 1. Cardinal rules (never break)

**Compliance** (export controls, restricted data, NPI, product/marketing/code names,
performance data, legal/marketing, internal links) is defined in AGENTS.md
"Compliance" — the source of truth, not restated here. Two file-level rules AGENTS.md
does not cover:

- **MIT license header only** — every file starts with the AMD copyright +
  `SPDX-License-Identifier: MIT`. No other license text.
- **ASCII only** in anything that can be serialized for codegen (IR text, hipRTC).

## 2. Files & headers

- **`.h` for headers, `.cpp` for sources.** `.hpp` is reserved for the small
  **C++-only boundary** headers (e.g. `include/rocke/error.hpp`). Do not introduce
  `.hpp` for C-ABI headers, and there is no `_impl.hpp` template split (the engine is
  not template-heavy).
- **`#ifndef` include guards**, named `ROCKE_<PATH>_H` (matching the file). The engine
  does **not** use `#pragma once`.
- **File preface order:** MIT copyright, then a block comment describing the file's
  role **and its Python correspondence**, then the guard. The correspondence block is
  the house style — an ASCII table (use `->`, not a Unicode arrow) mapping each Python
  symbol to its C99 counterpart, e.g.:
  ```text
  Python (target.py)            C99 (this file)
  ArchTarget.from_gfx(gfx)  ->  rocke_arch_target_from_gfx()
  known_arches()            ->  rocke_known_arches()
  ```
  See `include/rocke/arch_target.h` for a full example. A brand-new file with no Python
  twin may omit the table but must still state its role.
- **Public headers wrap declarations in `extern "C"`:**
  ```c
  #ifndef ROCKE_ARCH_TARGET_H
  #define ROCKE_ARCH_TARGET_H
  #include <stdbool.h>
  #include "rocke/ir.h"
  #ifdef __cplusplus
  extern "C" {
  #endif
  /* ... declarations ... */
  #ifdef __cplusplus
  }  /* extern "C" */
  #endif
  #endif  /* ROCKE_ARCH_TARGET_H */
  ```
- **Files terminate with a newline; no trailing whitespace** (clang-format enforces).
- Keep files reasonably sized; the port is already split into per-concern "buckets"
  (`core/ir/ir_arith.cpp`, `ir_mem.cpp`, `ir_tile.cpp`, …) — follow that granularity.

## 3. Includes

- **Order:** the translation unit's **own header first**, then a blank line, then
  C/C++ system headers in `<...>` (e.g. `<stdint.h>`, `<string.h>`), then a blank line,
  then other project headers in `"rocke/..."` (see `core/ir/serialize.cpp`).
  `SortIncludes` only sorts **within** a blank-line-separated group; it does **not**
  move `<...>` ahead of `"..."` (no `IncludeBlocks: Regroup` is configured), so the
  group order is a **manual** convention, not auto-enforced.
- **Project includes are spelled `"rocke/<name>.h"`** — headers live in
  `cpp/include/rocke/` and `cpp/include/` is on the include path, so
  `#include "rocke/ir.h"`, `#include "rocke/ir_internal.h"`. **No relative `../` paths**,
  no absolute paths.
- **Include what you use.** The C99 port is deliberately **libc-only** in many places
  (e.g. arch data is embedded as static tables, no JSON parser). Reserve C++ STL
  (`<string>`, `<exception>`) for the `ckc::` boundary layer, not the extern "C" TUs.

## 4. Naming

The engine is **`snake_case` with layered prefixes**, driven by the flat C-ABI and the
1:1 Python mirror.

| Kind | Convention | Example |
|---|---|---|
| Public ABI / free functions | `rocke_<layer>_<verb>` `snake_case` | `rocke_b_op`, `rocke_ir_parse`, `rocke_ll_lower_kernel` |
| Internal helpers | `rocke_i_*` | `rocke_i_emit`, `rocke_i_set_err` |
| Types | `rocke_<name>_t` | `rocke_ir_builder_t`, `rocke_arch_target_t`, `rocke_mma_op_t` |
| Enums / status / caps | `ROCKE_UPPER_SNAKE` | `ROCKE_OK`, `ROCKE_ERR_VALUE`, `ROCKE_ERR_MSG_CAP` |
| Macros & constants | `ROCKE_UPPER_SNAKE` | `ROCKE_ERR_MSG_CAP` |
| C++ boundary namespace | `ckc` (lowercase) | `namespace ckc` |
| C++ boundary types | `PascalCase` | `ckc::Error`, `ckc::ValueError`, `ckc::NotImplError` |
| C++ boundary functions | `snake_case` | `ckc::raise_status`, `ckc::guard_builder`, `ckc::format_error` |
| Files / directories | `snake_case`, mirroring Python layers | `core/lower_llvm/mma.cpp` |

- **The prefix encodes the layer/module** and is load-bearing. Match the prefix to the
  directory you're editing:
  - `core/ir/` → `rocke_ir_*`, `rocke_op_*`, `rocke_attr_*`, `rocke_value_*`; builder
    entry points `rocke_b_*`, internal helpers `rocke_i_*`
  - `core/lower_llvm/` → `rocke_ll_*` (shared lowering helpers `rocke_lower_*`)
  - `helpers/` → `rocke_h_*`; string-buffer utilities → `rocke_strbuf_*`

  When unsure, grep a sibling file in the same directory and match it.
- Names mirror their Python counterpart where one exists (`ArchTarget.from_gfx` →
  `rocke_arch_target_from_gfx`); keep the mapping obvious.

## 5. Formatting (owned by `.clang-format` — do not hand-fight it)

The authoritative config is `dnn-providers/hip-kernel-provider/.clang-format`
(WebKit-based; **inherited** — clang-format walks up from `cpp/`, there is no local
copy under `platform/` or `cpp/`). It is **already hook-enforced**: the repo-root
`.pre-commit-config.yaml` runs `clang-format` on `.c/.cpp/.h/.hpp` repo-wide, so these
are auto-applied — do not hand-fight them. Key points:

- **4-space indent, no tabs, 100-column** limit.
- **Allman braces** — opening `{` on its **own line** for functions, classes, structs,
  enums, namespaces, and control statements; `else` on its own line.
- **No namespace indentation** (`NamespaceIndentation: None`); align left.
- `PointerAlignment: Left` (`int* p`), `SpaceBeforeParens: Never`
  (`foo(x)`, `if(x)`), `BinPackArguments/Parameters: false` (one per line when
  wrapping).
- Run clang-format before committing; never reformat to something it would undo.

## 6. Error model (the engine's core contract)

Follow the two-channel model exactly (`include/rocke/error.hpp`, `core/ir/core_builder.cpp`):

- **Internal C++ code *throws*** a `ckc::Error` subclass **where the Python reference
  would `raise`** — `ckc::ValueError` ↔ `ValueError`, `ckc::TypeError`,
  `ckc::KeyError`, `ckc::OOMError`, `ckc::NotImplError`. Use `[[noreturn]]` raise
  helpers (`rocke_i_set_err`, `ckc::raise_status`).
- **Public `extern "C"` entry points catch at the boundary** via
  `ckc::guard_builder` / `ckc::guard_status` and translate to the legacy
  `rocke_status_t` + a sticky, `ROCKE_ERR_MSG_CAP`-bounded message on the builder.
  Never let a C++ exception escape the ABI. (The actual symbols are `ckc::guard_*`; a
  stale doc comment in `error_boundary.hpp` calls them `rocke_guard_*` — ignore that.)
- **Error message text must stay byte-identical to the Python reference** — parity and
  validation reject-reasons are compared verbatim. Format with `ckc::format_error`
  (printf-style, bounded); keep existing format strings unchanged.
- Map failure kinds to the matching status: `ROCKE_ERR_VALUE` for bad input,
  `ROCKE_ERR_NOTIMPL` for unsupported-but-valid paths.

## 7. Memory & ownership

- **Arena allocation.** IR nodes live in the builder's arena and are bulk-freed by
  `rocke_ir_builder_free`; do not manually free arena-owned nodes. Unwinding past
  in-progress builder calls leaks nothing because the arena is bulk-freed at the
  entry boundary.
- **Borrow vs own must be documented.** Static SSOT tables (arch data) return
  **borrowed** pointers valid for program lifetime; say so in the header, as
  `arch_target.h` does. State ownership and lifetime for any pointer an ABI function
  returns or stores.

## 8. Types, structs, enums, comments

- Prefer plain **`struct`** for POD-ish descriptor/value types (`rocke_*_t`); most
  engine state is an **opaque handle** behind the ABI (few top-level struct/class defs
  live in headers).
- **Plain `enum`** with `ROCKE_`-prefixed `UPPER_SNAKE` values (the flat C-ABI needs C
  enums).
- **Comments focus on *why*** and on ownership/lifetime/error-model, not what. The
  file-level block comment (role + Python correspondence table) is expected on
  non-trivial TUs. Use C-style `/* */` for block/file comments and `//` inline.
  **Doxygen is not used in the port** (it is a C-ABI mirror); use plain comments.
  Use `TODO:` / `FIXME:` / `NOTE:` tags for future work.
- **C++ standard is C++20** (`CMAKE_CXX_STANDARD 20`); C++20 features (attributes,
  exceptions, `<string>`) are fine in the `ckc::`/`.hpp` boundary. Do not push C++-only
  constructs onto the flat C-callable surface.

## 9. A minimal conforming file

A new ABI entry point, header + source (bodies elided). Verify the exact type/function
names against the current headers before copying — this is a shape guide.

`include/rocke/foo.h`
```c
/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/foo.h -- PUBLIC API for the C99 port of rocke.core.foo.
 *
 * Python (foo.py)     C99 (this file)
 * do_foo(b, x)   ->   rocke_foo_do(b, x)
 */
#ifndef ROCKE_FOO_H
#define ROCKE_FOO_H

#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

rocke_value_t* rocke_foo_do(rocke_ir_builder_t* b, rocke_value_t* x);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* ROCKE_FOO_H */
```

`core/foo.cpp`
```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * foo.cpp -- C99 port of rocke.core.foo.
 */
#include "rocke/foo.h"

#include <stdint.h>

#include "rocke/ir_internal.h"

rocke_value_t* rocke_foo_do(rocke_ir_builder_t* b, rocke_value_t* x)
{
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    /* ... emit ops via rocke_b_* / rocke_i_* ... */
    return x;
}
```

---

## Quick checklist for a new C++ engine change

- [ ] MIT copyright + `SPDX-License-Identifier: MIT`; ASCII only
- [ ] `.h` + `#ifndef ROCKE_*_H` guard; `extern "C"` wrapper on public headers
- [ ] File block comment states role + Python correspondence
- [ ] Includes: own header → `<system>` → `"rocke/..."`; no `../` or absolute paths
- [ ] `snake_case` `rocke_<layer>_*` names; `rocke_*_t` types; `ROCKE_*` enums/macros
- [ ] clang-format clean (4-space, 100-col, Allman braces)
- [ ] Errors: throw `ckc::Error` internally, guard+translate at the ABI, messages
      byte-identical to Python
- [ ] Ownership/lifetime documented for returned/stored pointers; arena nodes not
      hand-freed
- [ ] Emission changed? mirror in Python + re-run `tools/check_byte_identity.py` (from `platform/`)
- [ ] Engine code stays C-callable; C++ features confined to `ckc::`/`.hpp`
