# hipCUB/CCCL sensitive files (DRAFT)

> **Status: DRAFT, needs domain-expert review.** A first-pass,
> directory/pattern-granularity list, not a per-file audit. It exists so
> Phase C.4 (cross-referencing upstream commits against AMD-customized
> areas) has something to check against. No completed sync has run through
> this pipeline yet, so there is no incident history behind it. Treat a
> match as "look closer", not "this will conflict".
>
> `hipcub-cccl-sync-resolve/porting-categories.md` holds the resolution
> guidance. Keep patterns here and guidance there, and grow both as ported
> commits produce real lessons.

## Provenance

Seeded from the changed-file set of the last hipCUB CCCL sync,
[`72b6de5f86e`](https://github.com/ROCm/rocm-libraries/commit/72b6de5f86e)
("feat(hipcub): Add CCCL 3.0.x support (copy)",
[PR #9931](https://github.com/ROCm/rocm-libraries/pull/9931), a copy of
[PR #4079](https://github.com/ROCm/rocm-libraries/pull/4079)), and from
hipCUB's `CHANGELOG.md` entry for that release. That commit touched
~100 files under `projects/hipcub/`: 17 under `backend/cub/device/`, 14
under `backend/rocprim/device/`, and the rest spread over the rocPRIM
backend's `thread/`, `block/`, `warp/` and `iterator/` directories, the
`util_*.hpp` headers, `config.hpp`, `libcxx.hpp` and the tests. It also
needed a rocPRIM change (`rocprim/type_traits.hpp`). Its later revert,
[PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464), was a
scheduling decision, not a technical failure.

`backend/rocprim/device/**` is deliberately not listed: nearly every
device-API commit lands there, so a flag would carry no information. The
backend-parity check in `hipcub-show-upstream-commit.sh` covers it instead.

## Patterns (relative to `projects/hipccl/hipccl3/hipcub/`)

- `hipcub/include/hipcub/config.hpp` — namespace macros, platform
  selection (`__HIP_PLATFORM_AMD__` vs nvcc) and the `_CCCL_*` /
  `HIPCUB_*` compatibility macros. Upstream `cub/cub/config.cuh` and
  `util_namespace.cuh` changes land here.
- `hipcub/include/hipcub/libcxx.hpp` — the libcu++/libhipcxx shim
  (`::cuda::std` vs `::hip::std`). Sensitive to upstream changes in how CUB
  pulls in its standard library.
- `hipcub/include/hipcub/util_type.hpp` — shared type traits used by both
  backends.
- `hipcub/include/hipcub/backend/cub/util_type.hpp` and
  `hipcub/include/hipcub/backend/rocprim/util_type.hpp` — per-backend type
  traits (`Traits`, `BaseTraits`, `NumericTraits`); upstream removed parts
  of these in 3.0 and hipCUB followed in both backends.
- `hipcub/include/hipcub/backend/rocprim/util_ptx.hpp`,
  `hipcub/include/hipcub/backend/rocprim/util_macro.hpp` — warp/lane
  intrinsics and macros reimplemented on rocPRIM; upstream intrinsic or
  macro changes have no mechanical translation.
- `hipcub/include/hipcub/backend/rocprim/thread/**` — thread operators and
  reductions; upstream 3.0 deprecated the CUB functors (`Sum`, `Max`, ...)
  in favour of `cuda::std::` ones, and hipCUB maps them to `hip::std::`.
- `hipcub/include/hipcub/backend/rocprim/iterator/**` — iterators upstream
  removed or replaced with `cuda::` / Thrust equivalents.
- `hipcub/include/hipcub/util_deprecated.hpp` — `HIPCUB_DEPRECATED` /
  `HIPCUB_DEPRECATED_BECAUSE`; upstream deprecation-machinery changes
  surface here.
- `cmake/Dependencies.cmake` — `CCCL_MINIMUM_VERSION` and the CCCL
  find/download logic for the CUB backend, plus the rocPRIM dependency.
  Upstream CMake files never translate to this path, so the commit-list
  script never flags it; it's listed for Phase C.4 and for humans.

## Known gaps not yet reflected here

- Per-file patterns for the test suite (`test/hipcub/test_utils*.hpp`)
  and benchmark harness (`benchmark/benchmark_utils.hpp`). PR #9931
  touched every `test_utils*.hpp`; whether that recurs every sync is
  unknown.
- rocPRIM's own sensitive set. Changes that need a rocPRIM follow-up are
  tracked per item in `todo-hipcub.md`, not here.
