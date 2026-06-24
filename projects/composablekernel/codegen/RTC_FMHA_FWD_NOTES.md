# Enabling hipRTC compilation of the `ck_tile` FMHA forward kernel

This document records everything that was required to get the `ck_tile`
FMHA **forward** kernel (`FmhaFwdWrapper` / `FmhaFwdKernel`) to compile under
**hipRTC** (real-time compilation) **without** using `-isystem` and without a
system C++ standard library on the include path.

The constraint set was:

- No `-isystem` flags. hipRTC runs with no system include directories.
- Embed `rocm-cxx` headers and provide a **minimal set of `std`-named shim
  headers** that bridge `rocm-cxx` functionality into the `std` namespace.
- Keep the original `ck_tile` headers as close to read-only as possible. Where
  edits were unavoidable, restrict them to **host-only code** by wrapping it in
  `#ifndef __HIPCC_RTC__` guards (`__HIPCC_RTC__` is defined by hipRTC).

The work fell into four distinct categories of intervention, described below.

---

## 1. Standard-library shims (`codegen/src/rtc_std_shims.cpp`)

Because there is no system STL, every `std::` name that `ck_tile` (and HIP's own
runtime headers) reference must be supplied by a shim. Shims are registered in a
map (header name -> source text) and embedded into the RTC translation unit.
Three kinds of shims exist:

- **Bridges** onto `rocm-cxx` (e.g. `type_traits`, `utility`, `limits`, `array`,
  `functional`, `iterator`, `bit`) that mostly `#include <rocm/*.hpp>` and pull
  `rocm` names into `namespace std`.
- **Minimal self-contained** shims (e.g. `tuple`, `cmath`, `cstring`,
  `initializer_list`, `cinttypes`).
- **Forward-declaration / empty stubs** for host-only headers that should never
  be needed by device code (e.g. `iostream`, `sstream`, `variant`, `memory`,
  `unordered_map`, ...).

### Concrete shim changes made during this effort

- **Integer type conflicts.** `rocm::size_t` is `unsigned long long`, whereas the
  platform `size_t` is `unsigned long`. To avoid `std::size_t`/`::size_t`
  mismatches, `cstddef`/`cstdint` were aligned to hipRTC's own
  `__hip_internal` fixed-width types for `int8..int64`/`uint8..uint64`, and
  `ptrdiff_t`/`intptr_t`/`uintptr_t` were sourced from `rocm::`/global types
  rather than redefined.
- **Format macros.** `cinttypes` (and `inttypes.h`) was given a full `PRI*`
  macro set (LP64: 64-bit types use the `l` length modifier).
- **`std::make_tuple`** was implemented in the `tuple` shim.
- **`<array>` availability.** Several `ck_tile` headers use `std::array` without
  including `<array>`. Since `<type_traits>` is the near-universal include,
  `#include <array>` was added to the `type_traits` shim.
- **`std::is_invocable`** was added to `type_traits` via the detection idiom.
- **`std::min`/`max`/`clamp`** live in the `type_traits` shim (ck_tile uses them
  without including `<algorithm>`).
- **Arithmetic functors.** `std::multiplies` / `plus` / `minus` / `divides`
  (both the `T` and `void` specializations) were added to the `type_traits`
  shim, because ck_tile uses `std::multiplies<index_t>` in `static_assert`s
  without including `<functional>`.
- **`std::string_view` (key fix).** A real **constexpr** `std::string_view` was
  added to the `<string>` shim, with constexpr `==`/`!=` so it can be used in
  `if constexpr`. This was essential: the forward kernel selects code paths with
  `if constexpr(kPipelineName != "qr_async_trload")`, and a non-functional
  `string_view` broke that dispatch (see Â§4). `<string_view>` now simply
  re-includes `<string>`, and it was removed from the empty-stub list.
- **`<cstring>` host/device attributes (final fix).** The `memcpy`/`memmove`/
  `memset`/`memcmp` shims were marked `__attribute__((host, device))`. ck_tile
  device code (e.g. `amd_buffer_addressing_builtins.hpp`) calls `std::memcpy`
  from `__device__` context; without the device attribute this is
  "reference to `__host__` function in `__device__` function". The `__builtin_*`
  implementations are valid on both host and device.

---

## 2. Trimming the wrapper's includes (highest-leverage change)

`codegen/include/ck/host/device_fmha_fwd/fmha_fwd_wrapper.hpp` originally pulled
the umbrella header `ck_tile/ops/fmha.hpp`, which aggregates the **entire** FMHA
surface: forward, backward, batch-prefill, append-KV, paged-KV, split-KV, v3,
fp8, etc. The forward wrapper only ever uses:

- `FmhaFwdKernel`
- the three forward pipelines `BlockFmhaPipelineQRKSVS{,Async,AsyncTrload}`
- `TileFmhaShape` / `TileFmhaTraits`
- `BlockFmhaPipelineProblem`
- `Default2DEpilogue`

The umbrella include was replaced with a **forward-only include list**. This
single change eliminated essentially all of the host-only-code errors that were
previously being chased file-by-file in the bwd / batch-prefill / split-KV
paths (ostream operators, `std::variant`/`std::pair` kargs builders,
`std::string` name helpers, `std::multiplies` static asserts, etc.).

### Include-ordering follow-up

The umbrella header used to provide some symbols *before* the kernel by virtue of
include order. After trimming, `fmha_fwd_kernel.hpp` referenced
`BlockDropout`/`NullBlockDropout` (from `block_dropout.hpp`) and
`Alibi`/`EmptyPositionEncoding` (from `block_position_encoding.hpp`) before they
were declared. Both headers were added to the wrapper **before** the kernel
include. (These are genuine device-side symbols, so the fix is the include, not
a guard.)

---

## 3. Host-only code guards in `ck_tile` headers (`#ifndef __HIPCC_RTC__`)

Some host-only code lives in headers that are still reachable on the forward
device path. Each was wrapped in `#ifndef __HIPCC_RTC__` so it disappears under
RTC while remaining intact for normal host builds. The forward kernel never
needs any of this code under RTC (the wrapper builds `Kargs` directly and never
calls the host `MakeKargs`/name helpers).

Files and what was guarded:

- **`include/ck_tile/core/utility/gemm_validation.hpp`** — host-only validation;
  blanked/guarded.
- **`include/ck_tile/ops/common/utils.hpp`** — `gemm_prec_str()` and
  `mem_op_string()` (return `std::string`).
- **`include/ck_tile/ops/common/tensor_layout.hpp`** — `operator<<(std::ostream&, ...)`.
- **`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp`** —
  two `operator<<` overloads.
- **`include/ck_tile/host/concat.hpp`** — the entire namespace body of host
  string utilities (`std::string`/`string_view`/`ostringstream`). The
  `#include` of the scheduler header was kept *outside* the guard to preserve
  transitive includes.
- **`include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp`** — both
  `GetName()` overloads (the only consumers of `concat`).
- **`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`** — three
  `GetName()` overloads returning `std::string`.
- **`include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`** —
  `init_dropout` functions and the `MakeKargs` overloads using
  `std::variant`/`std::pair`. (This header is no longer on the trimmed forward
  path, but the guards remain valid and harmless.)
- **`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`** — see Â§4.

---

## 4. Host-only code inside the forward kernel itself

Once the include set was trimmed, the only remaining errors were inside
`fmha_fwd_kernel.hpp`. These required careful separation of host-only vs
device-required code:

- **`init_dropout` (two overloads)** — use `std::floor` in a host context;
  guarded with `#ifndef __HIPCC_RTC__`. The surrounding `Kargs` struct and its
  data members are device-required and were left untouched.
- **`MakeKargsImpl` / `MakeKargs` block** (a large contiguous region of
  `CK_TILE_HOST` launch-argument builders that use
  `std::variant<std::pair<...>>`) — guarded as a single
  `#ifndef __HIPCC_RTC__` region ending right before `GridSize`. The device
  `operator()`/`run_` build and consume `Kargs` directly, so none of this is
  needed under RTC.
- **`kPipelineName` (`std::string_view`) — important lesson.** This was *first*
  guarded out, which turned out to be **wrong**: it is referenced by **device**
  code in `if constexpr(kPipelineName != "qr_async_trload")`. Guarding it broke
  those conditions and caused a cascade of misleading "no member named
  `kKLoadOnce` / `GetSmemSizeK` / `kAlignmentOacc`" errors — those members only
  exist on the async pipelines and live in branches that are supposed to be
  pruned by the `if constexpr`. The correct fix was to make `std::string_view`
  actually work (Â§1) and **revert** the guard. With a functioning constexpr
  `string_view`, the dispatch prunes the async-only branches and all of those
  errors disappear.

---

## Key lessons / heuristics

1. **Trim includes before guarding.** Removing the umbrella FMHA include
   eliminated far more errors in one step than dozens of per-function guards
   would have. Always check whether unused code is even needed before patching
   it.
2. **Distinguish "host-only code" from "missing std feature."** Host-only code
   (`GetName`, `operator<<`, `MakeKargs`, dropout init) should be guarded out;
   genuinely-needed `std` features used by device code (`string_view`,
   `multiplies`, `memcpy`) must be **shimmed**, not guarded.
3. **Beware guarding symbols used by device `if constexpr`.** A name used only to
   select a compile-time branch (like `kPipelineName`) is still device code.
   Guarding it produces confusing downstream "no member" errors from branches
   that should have been pruned.
4. **Type identity matters under hipRTC.** `size_t`/`ptrdiff_t` and the
   fixed-width integer types must match hipRTC's own (`__hip_internal`) types to
   avoid subtle template-mismatch errors.
5. **Host/device attributes on shims.** Any shim function that device code might
   call (notably `<cstring>`) must be `__host__ __device__`.

## Net result

The forward kernel
`ck_tile::FmhaFwdWrapper<fp16_t, ...>` now compiles cleanly under hipRTC with no
`-isystem` and only the embedded `rocm-cxx` + `std`-named shims.
