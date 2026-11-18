# ADR 0003: CI builds against an installed ROCm, and checks enums against the PR's headers

Status:  Accepted
Decision:  Review A, Decisions 6, 11 and 13 (AIHPBLAS-5274)

## Context

The bindings live in rocm-libraries but never go into TheRock's output, so their CI has to get a hipBLASLt from somewhere. The first spike (AIHPBLAS-5272) found that the in-tree headers can't be used on their own: they need three generated headers that exist only in an installed tree (`hipblaslt-version.h`, `hipblaslt-export.h` and `hipblaslt-opt-in-features.h`). Pairing in-tree headers with an older install fails to compile.

The ROCm the Component CI lanes usually install comes from `--latest-release`, which still installs `10.1.0a20260822` (hipBLASLt 1.4.1). It reads a bucket TheRock stopped publishing to, which is TheRock#9530.

## Decision

- **Build against a current install** (Decision 6). Each lane installs TheRock's nightly wheels (`rocm[libraries,devel]`, plus a `device-<gfx>` wheel on GPU lanes) into a fresh venv, and builds the bindings against them the way a user would.
- **Pin the nightly by date in one place** (Decision 11): `HIPBLASLT_PY_ROCM_VERSION` in `.github/workflows/component-ci-hipblaslt-python.yml`. Bump it deliberately. Switch to `--latest-release` once TheRock#9530 is fixed.
- **Check enums against the PR's own headers.** The enum coverage check reads develop's headers through the preprocessor, not the installed ones (it lands with the enum bindings). The lanes run when `projects/hipblaslt/library/include/hipblaslt/` or `projects/hipblas-common/library/include/` change, so a new enum value is caught on the PR that adds it.
- **Accept the remaining gap** (Decision 13). A behavior change in the library itself shows up only once the pinned nightly picks it up. TESTING.md's Known Risks lists it.

## Consequences

- The lanes test the bindings, which is what changes in this stack, against a hipBLASLt users can actually install. They don't test a PR's own hipBLASLt.
- A lane's result depends on the pinned nightly as well as the PR. A bump is its own PR, so a failure that appears with a new nightly is attributed to the nightly, not to an unrelated change.
- Each lane downloads the nightly wheels (several hundred megabytes on GPU lanes). Test time is seconds, so the download dominates a lane's runtime.
