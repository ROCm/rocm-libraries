# Lane: dependency boundaries (`DEPS`)

Find new dependencies and dependencies that cross a boundary they should not.

## Policy

MIOpen links a fixed set of ROCm libraries (the `RUNTIME_DEPS` and `BUILD_DEPS` TheRock declares for it in `ml-libs/CMakeLists.txt`:
HIP, rocBLAS, hipBLAS, hipBLASLt, rocRAND, Composable Kernel, half, SQLite, bzip2, and the header-only Eigen, frugally-deep,
FunctionalPlus and nlohmann-json). Every new dependency needs justification in the PR, and one TheRock does not provide needs a
TheRock change before CI can build it (`ROCK`).

## New dependencies

Search the diff for `find_package`, `FetchContent`, `ExternalProject`, new `target_link_libraries` entries, third-party includes
(your packet lists changed C, C++ and HIP sources as well as manifests), `projects/miopen/requirements.txt`, `dev-requirements.txt`,
`rbuild.ini` and `install_deps.cmake` changes, `projects/miopen/cmake/Dependencies.cmake` and `ThirdParty.cmake`, Python
requirements of the scripts, and CI install steps. For each:

- Is it new to the project, or only to this target?
- Is it justified in the PR description? Tier an unjustified one per [../policy.md](../policy.md) *Tiers*.
- Is it version-pinned the same way in every manifest, and does TheRock provide it? Read TheRock's subproject declaration at its
  pin without cloning it: the pin is the `therock-ref:` value in `.github/actions/ci-env/action.yml` at `$HEAD`
  ([therock.md](therock.md) *Setup*), then `gh api "repos/ROCm/TheRock/contents/ml-libs/CMakeLists.txt?ref=<ref>" --jq .content | base64 -d`
  and the `third-party` listing `gh api "repos/ROCm/TheRock/contents/third-party?ref=<ref>" --jq '.[].name'`; when they cannot be
  read, mark the claim `[INFERENCE]`.
- Is the license compatible and attributed where the project requires it (`projects/miopen/LICENSE.md`)?

## Boundary violations

- A dependency the `HIPNOGPU` backend cannot satisfy pulled into code that backend compiles (device runtime calls, a GPU-only
  library), or a device call reachable from a `HIPNOGPU` path.
- An optional dependency (CK, hipBLASLt, rocBLAS, MLIR) used without the guard that makes it optional (`MIOPEN_USE_COMPOSABLEKERNEL`,
  `MIOPEN_USE_HIPBLASLT`, `MIOPEN_USE_ROCBLAS`, `MIOPEN_USE_MLIR` and their `#if` macros in `projects/miopen/include/miopen/config.h.in`).
- A private dependency leaking through `projects/miopen/include/miopen/miopen.h` or a `PUBLIC` link of the `MIOpen` target; a new
  `PACKAGE_DEPENDS` entry for something consumers do not need.
- Under the public/private split (`MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`), the public `libMIOpen.so` wrapper gaining a link-time
  dependency on hipDNN, which it must load at run time through `hipdnn_frontend_dynamic`, or on anything beyond `MIOpen_private`;
  and any hipDNN relationship at all in a flag-off build (`projects/miopen/rfcs/0001_HipdnnForwardingWrapper.md`, *hipDNN
  dependency posture* and §4.9).
- Test-only dependencies (googletest, the CPU references, rocjitsu) reachable from shipped targets.

Report only what the diff adds: compare `git -C "$CLONE" grep -n -e <term> "$BASE" -- projects/miopen` with the same at `"$HEAD"`,
and the changed targets' link lines. Prove each violation from the include chain or link line at `$HEAD`; cite every hop.
