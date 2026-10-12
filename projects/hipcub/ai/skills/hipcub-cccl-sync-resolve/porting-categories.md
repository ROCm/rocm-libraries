# hipCUB/CCCL porting categories (DRAFT)

> **Status: DRAFT, unvalidated.** No hipCUB sync has run through this
> pipeline yet. The categories below come from the structure of hipCUB
> (a two-backend wrapper, not a copy of CUB), from the last hand-made sync
> ([`72b6de5f86e`](https://github.com/ROCm/rocm-libraries/commit/72b6de5f86e),
> PR #9931), and from lessons the rocThrust family learned the hard way.
> Grow it as ported commits produce real lessons.
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
the API, not on which file changed.

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

## What's deliberately not here yet

- An intrinsic/PTX → rocPRIM/HIP mapping catalog. Build one as ported
  commits reveal recurring patterns.
- Per-file rules. None exist until a sync completes through this pipeline.
