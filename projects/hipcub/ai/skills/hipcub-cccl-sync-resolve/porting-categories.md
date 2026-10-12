# hipCUB/CCCL porting categories (DRAFT)

> **Status: DRAFT, validated once.** The categories come from the structure
> of hipCUB (a two-backend wrapper, not a copy of CUB), from the 3.0 sync
> ([`72b6de5f86e`](https://github.com/ROCm/rocm-libraries/commit/72b6de5f86e),
> PR #9931), and from lessons the rocThrust family learned the hard way.
> Categories 12-18 and the amendments marked *(PR #13083)* come from
> comparing an AI-driven v3.0.3 → v3.1.4 run of this pipeline against the
> human-made port of the same range
> ([PR #13083](https://github.com/ROCm/rocm-libraries/pull/13083)). Most of
> what the AI run missed was the same mistake: calling a commit "N/A, hipCUB's
> implementation is independent" when the commit's *intent* (an idiom, a
> namespace, a capability) still applied to hipCUB's own code.
>
> Do not treat this as exhaustive. When you hit a case not covered here,
> STOP and ask the human rather than guessing at a rule that isn't written
> down.

## Relationship to `sensitive-files.md`

`sensitive-files.md` (in `hipcub-cccl-sync-investigate`) lists *where*
AMD-specific logic concentrates. This file is about *what to do* once a
commit touches one of those areas. Don't duplicate the pattern list here.

## Categories

### 1. Backend parity: a public API change usually lands in both backends
An upstream change to a public `cub/cub/<dir>/<name>.cuh` header maps to up
to three hipCUB files under `hipcub/include/hipcub/`:

| hipCUB file | Role | What a port usually looks like |
|-------------|------|--------------------------------|
| `<dir>/<name>.hpp` | platform forwarder | rarely changes; only for a new or renamed header |
| `backend/rocprim/<dir>/<name>.hpp` | the API reimplemented on rocPRIM | real work: new overloads, changed signatures, new behavior expressed with rocPRIM |
| `backend/cub/<dir>/<name>.hpp` | thin wrapper over `::cub::` | forwarding: new overloads call the `::cub::` ones |

Most `block/`, `warp/` and several `thread/`/`iterator/` headers have no
`backend/cub/` file: `backend/cub/hipcub.hpp` re-exports the CUB names, so
a new block/warp API reaches the CUB backend with no hipCUB change.

`hipcub-show-upstream-commit.sh`'s backend parity check lists all three
locations for every touched header. Record a disposition for each backend
in the tick-note ("Backend parity: backend/rocprim/... added X;
backend/cub/... forwards X" or "...N/A: internal refactor, no API change").

A change that only reshuffles CUB's own implementation (moves code into
`detail/`, swaps an intrinsic, renames a private member) is N/A for both
backends even though the path is a public header. Decide on what changed in
the API, not on which file changed. But "the API" is wider than function
signatures *(PR #13083)*. Each of these is API-visible and gets mirrored:

- **Class-template shape.** Template parameter renames or removals
  (`LogicalWarpThreads`, dropping `int ARCH = HIPCUB_ARCH`), new partial
  or full specializations (`WarpReduce<T, 1>`), and new `static_assert`s on
  template arguments. The AI run dismissed `WarpReduce<T, 1>` as "an
  internal detail"; the human port added it.
- **Constraints.** Where upstream constrains with `_CCCL_TEMPLATE` /
  `_CCCL_REQUIRES`, use hipCUB's `HIPCUB_TEMPLATE` / `HIPCUB_REQUIRES`
  (and `HIPCUB_TRAIT`); keep upstream's operator helpers spelled the
  hipCUB way (`_HIPCUB_STD::plus<>`, `_HIPCUB_LIBCXX::maximum<>`).
- **Deprecated overloads follow the signature.** When a public signature
  changes (e.g. `int num_items` → `NumItemsT num_items`), update hipCUB's
  deprecated `debug_synchronous` overloads of the same function too. The
  AI run left them on `int`.
- **Namespace moves.** When upstream moves a name into a namespace that
  `backend/cub/hipcub.hpp` doesn't re-export (`cub::detail`, e.g.
  `ThreadScanInclusive`, `reduce::generate_idx_value`, `arg_min`/`arg_max`),
  add a `backend/cub/<dir>/<name>.hpp` (or extend the existing one) with
  `hipcub::detail::` aliases or forwarders, include it from the top-level
  forwarder, and have tests call `hipcub::detail::` on both platforms. Never
  paper over the difference with `#ifdef __HIP_PLATFORM_NVIDIA__` in a test.
- **New public headers.** A new `cub/cub/**/*.cuh` that isn't internal
  (e.g. `thread/thread_simd.cuh` in 3.1) must be added to the
  `#include <cub/...> // IWYU pragma: export` list in
  `backend/cub/hipcub.hpp`, even when the rocPRIM backend gets nothing.
  `hipcub-show-upstream-commit.sh` flags added headers that list lacks.
- **CUB backend include hygiene.** When upstream switches CUB's own headers
  from host STL (`<type_traits>`, `<cstdint>`, `std::`) to libcu++
  (`<cuda/std/...>`, `::cuda::std::`), switch the matching `backend/cub/`
  wrappers to `_HIPCUB_STD_INCLUDE(...)` / `_HIPCUB_STD::` too.

### 2. CUB backend and `CCCL_MINIMUM_VERSION`
`backend/cub/` is compiled with nvcc against whatever CCCL
`cmake/Dependencies.cmake` finds, and that is at least
`CCCL_MINIMUM_VERSION`. Forwarding to an API newer than that minimum breaks
the CUB backend for anyone on the minimum. No skill in this family bumps
`CCCL_MINIMUM_VERSION` or `HIPCUB_CCCL_VERSION_*`; the human does that once,
outside the per-item loop. So when a backend/cub/ forward needs the new CUB:
- write the forward (it's correct for `$TO_TAG`), and
- say in the tick-note that it needs `CCCL_MINIMUM_VERSION >= <tag>`, so
  the finalize audit can collect these into one list for the human.

Don't guard forwards with `#if CCCL_VERSION >= ...` unless hipCUB already
does that for the same header; ask the human first.

The CUB backend can't be compiled without nvcc. If no CUDA toolchain is
available, say so in the tick-note ("CUB backend not compiled").

### 3. Dispatch, agent, detail, tuning, kernels: N/A, or a rocPRIM follow-up
hipCUB has no copy of `cub/cub/device/dispatch/`, `cub/cub/agent/` or
`cub/cub/detail/`; the rocPRIM backend calls `rocprim::` device algorithms
instead. Commits confined to these paths are N/A in hipCUB, with one
question to answer: does the change alter observable algorithm behavior
(results, determinism, supported sizes or types, new modes), as opposed to
CUDA-specific performance, tuning policies, PTX, PDL or launch plumbing?

- Behavior change: record "rocPRIM follow-up: <what rocPRIM would need>,
  see `projects/hipccl/hipccl3/rocprim/.../device_<x>.hpp`". Don't edit
  `projects/hipccl/hipccl3/rocprim/` without asking the human: it's a separate library with
  its own release cycle. PR #9931 did need one rocPRIM change
  (`rocprim/type_traits.hpp`).
- Otherwise: record "rocPRIM follow-up: none (<one-line reason>)".

`hipcub-show-upstream-commit.sh`'s rocPRIM follow-up check names the likely
rocPRIM header.

**What to recommend when you ask** *(PR #13083)*. The human 3.1 port
edited rocPRIM in the same PR for every capability that callers can see:
64-bit `num_items` for run-length encode, 64-bit segment sizes and
`num_segments` for segmented radix sort and segmented reduce, fixed-size
segmented reduce, and the RLE early exit on empty input. The AI run
recorded all of these as follow-ups and shipped hipCUB with `UINT_MAX`
guards instead. So when the change is a new capability or a lifted limit
(sizes, types, a new mode), recommend "edit rocPRIM in this item" and
describe the rocPRIM change *and* the rocPRIM tests it needs
(`projects/hipccl/hipccl3/rocprim/test/rocprim/test_device_<x>.cpp`, large-size cases with
an OOM skip). Recommend "follow-up" only for performance, tuning, or a
change big enough to deserve its own rocPRIM PR. The human still decides.

**Not every "needs something from below" is a rocPRIM item.** A thin
convenience layer over an existing two-phase API belongs in hipCUB
itself. The 3.1 env-based `DeviceReduce::Reduce(..., env)` / `Sum(..., env)`
overloads (query the stream and memory resource from the env, allocate
temp storage, call the two-phase API, free) were implemented in hipCUB
(`backend/rocprim/detail/util_env.hpp`, `detail/temporary_storage.hpp`),
not rocPRIM. The AI run deferred them on the unverified belief that
libhipcxx lacked execution environments and memory resources; see
category 15.

### 4. Tests: Catch2 → Google Test, at parity
hipCUB implements upstream `cub/test/catch2_test_<x>.cu` tests with Google
Test in `test/hipcub/test_hipcub_<x>.cpp` (some are `.cpp.in` templates
plus a `.hpp`). hipCUB keeps `test/hipcub/` at parity with upstream; that's
policy, not a question to ask.

- New or changed test cases: implement the same cases in the counterpart
  file. Don't add `catch2_test_*.cu`, `c2h/` helpers or Catch2 includes.
- A new upstream test file with no counterpart: add a new
  `test_hipcub_<x>.cpp` and register it with `add_hipcub_test(...)` in
  `test/hipcub/CMakeLists.txt` in the same item.
- `c2h/` plumbing, test_*_fail.cu compile-fail tests, NVTX, `link_*`,
  `ptx-json/` and `internal/` tests of CUB-only internals: N/A.
- An upstream test exercising an API the rocPRIM backend doesn't implement
  yet (because it's a rocPRIM follow-up): ask the human whether to add a
  `GTEST_SKIP()`ed test or leave it out.

**Test the capability, not just upstream's test diff** *(PR #13083)*. When
a commit adds or extends a capability (64-bit sizes, more segments, a new
overload, a new env parameter), hipCUB's tests must exercise it even if
upstream's test change for that commit is trivial or lives in a later
commit. The AI run wrote "Test counterpart: none needed" for the
segmented radix sort 64-bit commit because the upstream test diff was a
one-liner; the human port added segments larger than `UINT_MAX` and a
large `num_segments` case. For large-memory tests, use (or add, in
`test/hipcub/common_test_header.hpp`) an OOM-tolerant allocation helper
that skips the test on `hipErrorOutOfMemory` (the 3.1 port added
`HIP_CHECK_MEMORY`) instead of failing.

**Check the test helpers upstream relies on.** An upstream test change
can depend on a trait or helper being true for hipCUB's test types,
e.g. `is_floating_point_v` for `test_utils::half` / `bfloat16`. Grep
`test/hipcub/test_utils_*.hpp` for it and add the specialization in the
same item when it's missing.

Record "Test counterpart: ..." in the tick-note.

### 5. Benchmarks: nvbench → hipCUB's harness
`cub/benchmarks/bench/<algo>/*.cu` are nvbench benchmarks with tuning axes
(`%RANGE%`). hipCUB's counterparts are `benchmark/benchmark_device_<algo>.cpp`
on its own harness, registered with `add_hipcub_benchmark(...)` in
`benchmark/CMakeLists.txt` (no glob).
- A new benchmarked algorithm or input variant: add it to the counterpart,
  or add and register a new file.
- Tuning-axis, `nvbench_helper/`, or CUDA-launch-only changes: N/A.

Record "Benchmark counterpart: ..." in the tick-note.

### 6. Examples
`cub/examples/block/<x>.cu` → `examples/block/<x>.cu`;
`cub/examples/device/<x>.cu` → `examples/device/<x>.cpp`. New examples are
registered with `add_hipcub_example(...)` in that directory's
`CMakeLists.txt`.

Upstream's shared host helper `cub/test/test_util.h` is used by the
examples, and its hipCUB counterpart is `examples/example_utils.hpp`, not
anything under `test/` *(PR #13083)*. The AI run called upstream's
CUDA 13 `cudaDeviceProp` fix (`memoryClockRate` / `memoryBusWidth` →
`cudaDeviceGetAttribute`) N/A; `example_utils.hpp` had the same fields
and the human port switched it to `hipDeviceGetAttribute`.

### 7. Macro rename/removal sweep — content-triggered
When a commit renames or removes a `_CCCL_*` / `CUB_*` macro, the change
may be "N/A on AMD" functionally and still leave stale text in hipCUB.
By hand:
1. grep `projects/hipccl/hipccl3/hipcub/` for the old name (comments, TODOs, `#ifdef`s);
2. grep for hipCUB's own alias of that macro (`HIPCUB_*` in `config.hpp`,
   `util_macro.hpp`, `util_deprecated.hpp`) and update its call sites the
   way upstream updated its own;
3. grep `projects/hipccl/hipccl3/hipcub/.clang-format` for the old and new spellings.

The rocThrust family confirmed all three sub-cases against a real
human-made port; an N/A verdict didn't exempt the commit from the sweep.

**hipCUB must not define CCCL's macros** *(PR #13083)*. hipCUB used to
define its own fallbacks under CCCL's names (`_CCCL_PRAGMA_UNROLL_FULL`,
`_CCCL_PRAGMA_NOUNROLL`, `_CCCL_SORT_MAYBE_UNROLL`, the user knob
`CCCL_AVOID_SORT_UNROLL`) in `config.hpp` and `backend/rocprim/util_macro.hpp`.
Those collide with libhipcxx/CCCL and are invisible to a "did hipCUB change"
check. When a commit touches a macro hipCUB defines a fallback for, rename
hipCUB's to the `HIPCUB_*` spelling (`HIPCUB_PRAGMA_UNROLL_FULL()`,
`HIPCUB_SORT_MAYBE_UNROLL()`, `HIPCUB_AVOID_SORT_UNROLL`) and update every
call site in headers, tests and benchmarks. On the CUB backend, define the
`HIPCUB_*` name as an alias of the CCCL macro (in `backend/cub/util_macro.hpp`)
so shared code compiles on both platforms. To find them:

```bash
git grep -nE '#\s*define\s+_?CCCL_' -- projects/hipccl/hipccl3/hipcub/
```

Anything this prints is a candidate, whether or not the current commit
mentions it; raise the rest with the human once (the investigate report
lists them as baseline parity debt, category 18). Not every hit gets
renamed: the 3.1 human port kept the `_CCCL_HAS_INT128()` fallback in
`backend/rocprim/util_type.hpp` (commented "deprecate once libhipcxx is
available"). Rename the ones whose uses the current commit touches; ask
about the others. Once a hit is dispositioned, write "Identifier sweep:
kept `<macro>` (<reason>)" in the tick-note so the lint stops flagging it.

### 8. Never assume upstream deleted something
Before "fixing" something that looks stale, check the target tag:

```bash
git show "$TO_TAG:cub/cub/<path>" | grep -n '<identifier>'
```

If it's still there at `$TO_TAG`, a later item removes it, not this one.

### 9. Deprecations and removals mirror into both backends
When upstream deprecates a public API, mark the hipCUB one with
`HIPCUB_DEPRECATED_BECAUSE("<upstream's message, adapted>")` in both
backends (the CUB backend may also inherit CUB's own deprecation warning).
When upstream removes one, remove it from both backends and from the tests
and benchmarks that use it, in the same item. Before removing, grep the
rest of `projects/` for users (rocThrust doesn't include hipCUB, but check
anyway).

This includes internals hipCUB happens to expose *(PR #13083)*. When
upstream removes a `cub::internal::` or `cub::detail::` API that hipCUB
mirrors (the pointer-based `internal::ThreadReduce<Length>(T*)` overloads,
`ThreadReduceSequential` / `BinaryTree` / `TernaryTree`, SIMD stubs in
`thread_operators.hpp`), remove hipCUB's mirror too, inlining what the
remaining public API still needs. "It's internal, so N/A" is wrong when
hipCUB's copy is reachable from `hipcub::`.

### 10. Baseline-marked commits (`[BASELINE: =]` / `[BASELINE: ~]`)
CCCL tags releases on `branch/X.Y.x`, so `$CURRENT_TAG..$TO_TAG` includes
main-line commits whose backports are already in `$CURRENT_TAG`. For a
marked item, check whether hipCUB's tree already reflects the change (the
3.0 port may have covered it). If it does, tick it N/A ("already in
`$CURRENT_TAG` via backport <sha>"). `~` means a non-identical backport
cites the same PR: compare the two before concluding.

### 11. Copyright / file-header convention
Unknown for hipCUB. For a ported file that needs a new or updated AMD
copyright header, ask the human the first time and record the answer here.

### 12. Codebase-wide idiom sweeps apply to hipCUB's own code *(PR #13083)*
Many upstream commits are mechanical sweeps across CUB:
`(void)x;` → `[[maybe_unused]]`, `std::` → `::cuda::std::`, host STL
includes → `<cuda/std/...>`, `thrust::` utilities → `::cuda::std::`,
`::cuda::std` spelled out → a namespace macro. They touch CUB files that
have no hipCUB copy, so they look N/A. They aren't: hipCUB has its own
code doing the same thing, and the human port applied each sweep to it.
For a sweep commit:
1. Identify the old idiom from the `-` lines (the pattern, not the files).
2. Grep **all** of `projects/hipccl/hipccl3/hipcub/` for it: both backends,
   `test/`, `benchmark/`, `examples/`.
3. Rewrite the hits the hipCUB way: `std::` → `_HIPCUB_STD::`,
   `<type_traits>` → `_HIPCUB_STD_INCLUDE(type_traits)`, `::cuda::std` →
   `_HIPCUB_STD`, `(void)x` → `[[maybe_unused]]`.
4. Scope the rewrite to what upstream's sweep covered (a sweep over device
   headers doesn't license rewriting all of `test/`), and say in the
   tick-note how many hits were left and why.

Examples from 3.1: `[[maybe_unused]]` in `device_transform.hpp`,
`util_ptx.hpp`, `config.hpp`, `grid_queue.hpp`, benchmarks and tests;
`std::distance` → `_HIPCUB_STD::distance` in `backend/rocprim/device/device_for.hpp`;
`::cuda::std::extents` → `_HIPCUB_STD::extents` in `backend/cub/device/device_for.hpp`.

### 13. Before an N/A verdict, grep hipCUB for the identifiers *(PR #13083)*
"hipCUB's implementation is independent" is a hypothesis, not a verdict.
Before ticking N/A, take the identifiers the commit changes (macro names,
struct fields, function names, removed types) and grep the whole hipCUB
tree, including `examples/`, `benchmark/` and `test/`:

```bash
git grep -nwE '<ident1>|<ident2>' -- projects/hipccl/hipccl3/hipcub/
```

Record the grep in the tick-note ("N/A: no hits for X, Y in hipCUB"). A
hit means hipCUB has its own copy of the thing upstream changed: port it.
`hipcub-show-upstream-commit.sh`'s identifier check does this for macros
and a few known idioms; do the rest by hand.

### 14. Vendor vector-type deprecations *(PR #13083)*
CUDA 13 deprecated `long4`, `ulong4`, `longlong4`, `ulonglong4`, `double4`
in favor of `*4_16a` / `*4_32a`. Don't dismiss such changes with "HIP
doesn't deprecate these": mirror the deprecation suppression around
hipCUB's uses (`HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH/POP` in
`backend/rocprim/util_type.hpp`), and add the new types' traits
(`AlignBytes` etc.) behind a toolkit-version check like
`HIPCUB_CTK_AT_LEAST(13, 0)`. The rocPRIM backend sees CUDA-compatible
type names through HIP, so CUDA's choices tend to reach it.

### 15. Verify libhipcxx capabilities before deferring *(PR #13083)*
If a port looks blocked on "libhipcxx doesn't have X" (execution
environments, `get_stream`, memory resources, `extents`, a trait), check
before saying so:

```bash
# projects/hipccl/hipccl3/libhipcxx is a git submodule; it may be empty.
grep -rlE '<X>' projects/hipccl/hipccl3/libhipcxx/include/ 2>/dev/null | head
# otherwise: an installed copy (/opt/rocm*/include/cuda/) or a build's _deps/
```

`cmake/Dependencies.cmake` says which libhipcxx the build uses. If no copy
is available to grep, say so and ask the human (they can init the
submodule); don't record "libhipcxx lacks X" as a fact without having
looked. Use the hipCUB macros (`_HIPCUB_STD`, `_HIPCUB_STD_EXEC`,
`_HIPCUB_LIBCXX`) to reach it. When only part of a feature is missing
(e.g. determinism requirements), implement the rest and record the gap in
the tick-note ("determinism requirement ignored: not supported by
libhipcxx yet"); a partial gap isn't a reason to defer the item.

### 16. Signature changes reach every overload
See category 1, "Deprecated overloads follow the signature". When
widening a size type, also drop hipCUB-side range guards that only
existed because of the old narrower type (`UINT_MAX` checks before a
rocPRIM call), once rocPRIM accepts the wider type.

### 17. Don't defer an item because "a later commit finishes it"
Category 8 says don't pre-empt a later commit. The converse also holds:
if the current commit's API is complete at this commit, port it now even
if upstream later polishes it. The AI run deferred the env-based
`DeviceReduce` overloads twice; the human port has them.

### 18. Baseline parity debt
Earlier syncs left hipCUB behind `$CURRENT_TAG` in places: in 3.1 the
human port also removed `int ARCH = HIPCUB_ARCH` template parameters
(upstream dropped `LEGACY_PTX_ARCH` in 3.0), the deprecated
`Equality`/`Inequality`/`InequalityWrapper`/`Sum`/`Difference`/`Division`/`Max`/`Min`
operators (gone upstream before 3.0.3), and the pointer `ThreadReduce`
overloads. No commit in `$CURRENT_TAG..$TO_TAG` mentions these, so the
per-commit loop never meets them. `hipcub-cccl-sync-investigate` lists
them in its "Baseline parity debt" section and `hipcub-cccl-sync-todo`
turns them into `[DEBT]` items; port them like any other item, against
`$CURRENT_TAG`'s API.

## What's deliberately not here yet

- An intrinsic/PTX → rocPRIM/HIP mapping catalog. Build one as ported
  commits reveal recurring patterns.
- Per-file rules. None exist until a sync completes through this pipeline.
- AMD-local build decisions. PR #13083 also made libhipcxx a hard
  dependency (dropping `_HIPCUB_HAS_DEVICE_SYSTEM_STD` fallbacks), changed
  how dependencies are fetched, and set `CCCL_MINIMUM_VERSION`. Those are
  human calls outside the per-item loop, not something a commit implies.
