# hipBLASLt JIT downstream handoff

Updated September 29, 2026. This is the canonical handoff for the consolidated
hipBLASLt/TensileLite just-in-time (JIT) work. Development continues on the remote-backed branch
below, with no associated pull request. Consolidation, closure of the upstream
review stack, and the branch rename are complete.

## Current checkout and working direction

- Workspace: `/home/jolabega/rocm-libraries-worktrees/call-tensilelite-from-hipblaslt`.
- Branch: `users/jolabega/downstream-hipblaslt-jit-develop`.
- Tracking: `origin/users/jolabega/downstream-hipblaslt-jit-develop`.
- Remote: `git@github.com:ROCm/rocm-libraries.git`;
  [branch on GitHub](https://github.com/ROCm/rocm-libraries/tree/users/jolabega/downstream-hipblaslt-jit-develop).
- Verified local/remote checkpoint before this handoff update:
  `762833d9281a5a3259d39de6f05025acddb00616`. This is a historical checkpoint,
  not the moving branch HEAD; the handoff cleanup is a later documentation commit.
- Complete implementation baseline: `d742375dbbef5ef44e9890e199741de70d573f14`.
  Subsequent consolidation and handoff changes affect Markdown only.

Read the live revision and tracking state from Git rather than treating a saved
SHA as the current tip:

```bash
git branch --show-current
git rev-parse HEAD
git rev-parse --abbrev-ref '@{upstream}'
git status --short
git ls-remote --heads origin refs/heads/users/jolabega/downstream-hipblaslt-jit-develop
```

The user's meaning of downstream is a branch with a remote copy and **no pull request (PR)**,
even though `origin` is the ROCm repository. Continue new requested work here;
do not resume the former upstream review, split, continuous-integration (CI) triage or publication tasks
from older handoffs. Reopening or replacing those PRs is not the default workflow.
Keep the implementation and existing application programming interface (API)/file names; the move downstream did
not request a naming redesign. The old `downstream/hipblaslt-jit` branch was
removed locally and remotely after the renamed branch was verified.

## Implemented behavior and guides

The full work and its history are consolidated: standalone single-solution
generation, explicit-recipe direct API and sample, generic request/backend/
solution API and separate sample, prediction and benchmark integration, and
the validation driver/workflow. Direct and generic APIs share one production
runtime and algorithm registry. Direct `configPath` remains required;
empty-config prediction belongs to the generic provider. Generation returns an
algorithm for existing C/C++ general matrix multiplication (GEMM) execution.
An ordinary matmul call does not initiate this new TensileLite/generic JIT path;
the separate existing rocRoller integration can generate code at runtime.

- [JIT guide](projects/hipblaslt/JIT.md): the plan of record. It separates
  current behavior from the approved target design and carries the roadmap.
- [Standalone builder](projects/hipblaslt/tensilelite/SINGLE_SOLUTION.md).
- [TensileLite backend](projects/hipblaslt/JIT_TENSILELITE.md) and
  [sample 29](projects/hipblaslt/clients/samples/29_hipblaslt_jit_gemm/README.md).
- [Sample 30](projects/hipblaslt/clients/samples/30_hipblaslt_generic_jit_gemm/README.md)
  for the generic API.
- [Benchmark usage](projects/hipblaslt/clients/bench/README.jit.md) and
  [shared validation driver](.github/scripts/test_hipblaslt_jit.py).
- [Design notes](projects/hipblaslt/jit-design/README.md): the September 29
  Confluence-copyable discussion draft of the target design and the
  KernelFromAnywhere (KFA) assessment cover existing type reuse, common producer
  metadata, rocRoller as a future backend, and heuristic-driven generation. These
  are design updates, not source implementation. Timing plans retain their
  historical source anchors.
  The draft revision has not been published; the supplied Confluence URL is a
  format/tone reference, with no destination page identified. Reading that
  reference returned an access-denied response; its contents were not verified.

## Completed upstream transition

All ten PRs below are **closed without merging**. Native stack #12567 is closed
(`open=false`) with membership/order retained. Their original branches remain
locally and on `origin`, and their heads are ancestors of this downstream work.
The shared prerequisite [#12451](https://github.com/ROCm/rocm-libraries/pull/12451)
is merged; it is outside the closed ten-PR stack.

Branch names below have the prefix `users/jolabega/`.

| Closed PR | Retained branch suffix | Preserved head |
| --- | --- | --- |
| [#12459](https://github.com/ROCm/rocm-libraries/pull/12459) | `tensile-single-solution-builder` | `58b4799d28` |
| [#12460](https://github.com/ROCm/rocm-libraries/pull/12460) | `tensile-jit-recipe-selection` | `4984c944b4` |
| [#12552](https://github.com/ROCm/rocm-libraries/pull/12552) | `tensile-jit-process` | `4f89f19693` |
| [#12563](https://github.com/ROCm/rocm-libraries/pull/12563) | `tensile-jit-artifacts` | `0010f19aca` |
| [#12564](https://github.com/ROCm/rocm-libraries/pull/12564) | `tensile-jit-direct-gemm` | `6a1ebb5765` |
| [#12565](https://github.com/ROCm/rocm-libraries/pull/12565) | `tensile-jit-basic-sample` | `b25107056a` |
| [#12461](https://github.com/ROCm/rocm-libraries/pull/12461) | `hipblaslt-generated-algorithms` | `5d9826f416` |
| [#12462](https://github.com/ROCm/rocm-libraries/pull/12462) | `hipblaslt-jit-sample-owner` | `e89931fd16` |
| [#12463](https://github.com/ROCm/rocm-libraries/pull/12463) | `hipblaslt-predicted-jit-bench` | `7a307c190a` |
| [#12430](https://github.com/ROCm/rocm-libraries/pull/12430) | `call-tensilelite-from-hipblaslt` | `d742375dbb` |

Review comments and objections remain historical evidence, not an active review
assignment or an assertion of maintainer acceptance. Unrelated PRs were not
part of this transition.

## Existing environment

The host runs Ubuntu 24.04. The former ROCm 7.1.1 installation is gone. Two
TheRock bundles live under `/opt/rocm-versions`:
`gfx950-10.2.0a20260922` and `gfx1250-10.2.0a20260922` (AMD clang 24.0.0).
Select one with `rocm-use gfx950` or `rocm-use gfx1250`, a bash function
defined in `/etc/profile.d/rocm-helpers.sh`; zsh cannot source that file. Besides
setting `ROCM_PATH`, `PATH` and `LD_LIBRARY_PATH` for the current shell, it
repoints the system-wide `/opt/rocm` symlink with `sudo`, which affects every
other shell and agent on the host. gfx950 work runs natively (8× MI355X). gfx1250
kernels run on the FFM MI450 simulator installed at `/opt/ffm/mi450`; simulator
runs are not native hardware evidence.

Reuse the root `.venv` and `projects/hipblaslt/build/release`. The venv holds
CPython 3.10.12 installed by uv; both configured Python executable keys use root
`.venv/bin/python`. The build tree uses the Unix Makefiles generator (ninja is
not installed) and was configured with the gfx950 bundle's `amdclang`,
`amdclang++` and `amdflang`. The inspected configuration is Release,
`GPU_TARGETS=gfx950`, `HIPBLASLT_ENABLE_JIT=ON`, `HIPBLASLT_ENABLE_CLIENT=ON`,
`HIPBLASLT_BUILD_TESTING=ON`, `HIPBLASLT_ENABLE_MXDATAGENERATOR=ON`,
`HIPBLASLT_ENABLE_DEVICE=OFF` and `HIPBLASLT_ENABLE_ROCROLLER=OFF`. It also sets
`CMAKE_SKIP_INSTALL_RULES=ON`, so an installation check needs a separate build
tree.

The ROCm bundle ships its own `libhipblaslt` and `libtensilelite-host`. Put both
build-tree library directories ahead of the bundle's libraries, or the loader
picks the prebuilt copies. From the repository root, in bash:

```bash
source /etc/profile.d/rocm-helpers.sh
rocm-use gfx950
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
project_python="$project_root/.venv/bin/python"
source "$project_root/.venv/bin/activate"
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
export LD_LIBRARY_PATH="$project_build/library:$project_build/tensilelite:$LD_LIBRARY_PATH"
export PYTHONDONTWRITEBYTECODE=1
export TENSILE_DISABLE_HELPER_CACHE=1
export TENSILE_HELPER_CACHE_DIR="$project_build/helper-cache"
export AMD_COMGR_CACHE=0
export HIP_VISIBLE_DEVICES=0
```

`rocm-use` places the bundle's `lib` and `llvm/lib` directories at the front of
`LD_LIBRARY_PATH`, so the build-tree directories are prepended after it runs.
`HIP_VISIBLE_DEVICES` selects one GPU when other work shares the host.

Use this build's library and rocisa extension with its configured interpreter;
do not mix the host CPython 3.10 module with a different container interpreter.
Keep one owner for changes to the shared build/configuration and GPU runs.
For later source work, read [hipBLASLt guidance](projects/hipblaslt/AGENTS.md),
[build/test reference](projects/hipblaslt/AGENTS_reference.md), and
[TensileLite guidance](projects/hipblaslt/tensilelite/AGENTS.md) as applicable.
Use fresh output directories for generation/tests and preserve existing evidence.

## Validation retained from the implemented source

The recorded tests apply to the implementation baseline above; subsequent
Markdown-only changes preserve executable/build/test/fixture blobs.

| Scope | Recorded result |
| --- | --- |
| Direct Linux gfx950 driver | All 10 routes passed, including C/C++ numerical execution and disabled-JIT behavior. |
| Generic Linux gfx950 driver | All 12 routes passed; the final 13 affected failure checks passed after exception/diagnostic fixes. |
| Separate direct/generic samples | C and C++ checks each covered 32,768 elements with maximum error 0. |
| Prediction/benchmark | 16 targeted checks passed: 3 numerical, 2 expected prediction/recipe failures, 11 negatives; 6 retained API/sample/alternate/OFF routes passed. |
| Disabled/enabled configuration | OFF behavior checked; final build restored to JIT ON. |
| gfx1250 SIA4 | Generation and compilation passed with the retained compatible compiler; no native numerical execution claim. |

Native numerical evidence is Linux gfx950. There is no claim of native Windows
execution, native gfx1250 execution, a full benchmark sweep, or success on every
configured CI architecture. The SIA4 artifact records compiler 24.0.0 at
`/tmp/jolabega/orchestrate-tensilelite-hipblaslt-20260921/gfx1250-toolchain/amdclang++`;
do not assume the default host compiler is the same toolchain.

Historical evidence lives under
`/tmp/jolabega/orchestrate-pr12451-reference-20260923/`:

- `basic-first/independent-completion-report.md` and its JSON companion bind
  validation to source revisions. Its old PR states are superseded by closure.
- `newbasic-first/{b4-native,generic-native,generic-sample-native,predicted-api-native}/summary.json`,
  `newbasic-first/predicted-bench-native/report.json`, and
  `newbasic-first/sia4-gfx1250/bundle/manifest.json` contain the run evidence.
- `downstream-consolidation/post-closure/verification.json`,
  `downstream-consolidation/rename/final-verification.json`, and
  `handoff-refresh/live-state-before.json` record closure, branch rename and
  refreshed remote/PR facts. Full pre/post-closure discussions remain archived.

These scratch paths are local evidence, not prerequisites for understanding the
versioned guides. The incremental backup
`projects/hipblaslt/build/downstream-backup/hipblaslt-jit-20260924-remote.bundle`
retains the pre-rename consolidation checkpoint; it is not the latest branch
backup or a standalone clone. Its prerequisite commits are recorded by
`git bundle verify`. The remote branch holds subsequent committed work.

## Remaining downstream work

The agreed direct/generic/prediction implementation and branch transition are
complete. The subsequent modeled-input change adds the complete applicable
data-parallel Origami contract, with its capability inventory and unsupported
transport cases documented in `projects/hipblaslt/JIT.md`.

The approved target design in `projects/hipblaslt/JIT.md` moves generation behind
the heuristic query. Its [roadmap](projects/hipblaslt/JIT.md#roadmap) has six
steps, and the status column there is authoritative:

1. Demote the public API: make the JIT headers internal and turn samples 29 and
   30 into test binaries (in progress).
2. Add the Jit component, the backend interface, a mock backend, and the
   Predictor and TuningKnowledge interfaces.
3. Build code objects in hipBLASLt through comgr.
4. Add the persistent JIT solution library.
5. Add `HIPBLASLT_JIT` to the heuristic query and remove `hipblaslt-bench --jit-gemm`.
6. Run the validation sweep with heuristic, cache and mode coverage.

The optional timing/progress design uses independent `HIPBLASLT_JIT_DEBUG`
categories `timing` and `progress`, including `timing,progress`. Unset/empty adds
no new collection, observer or files; existing diagnostics and benchmark timing
boundaries remain unchanged until implementation.

The optional KFA plan starts with complete producer metadata, then strict
schema/ABI validation and proof of argument, launch, helper, workspace and
synchronization equivalence before shared dispatch and simplification. Follow
the preserved producer-first plan; its older external-ingestion-first proposal
is superseded. Recheck historical source anchors when beginning that work.
