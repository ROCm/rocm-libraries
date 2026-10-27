# Lane: dependency boundaries (`DEPS`)

Find new dependencies and dependencies that cross a boundary they should not.

## Policy

`projects/hipdnn/CONTRIBUTING.md` (Dependency Management): hipDNN core stays light; avoid new backend library dependencies;
frontend and SDKs stay header-only with no compiled-library requirement; every new dependency needs justification. Providers
are separate projects and may carry their own.

## New dependencies

Search the diff for `find_package`, `FetchContent`, `hipdnn_add_dependency`, `ExternalProject`, new `target_link_libraries`
entries, third-party includes (your packet lists changed C, C++ and HIP sources as well as manifests), `pyproject.toml` and requirements changes, and CI install steps. For each:

- Is it new to the project, or only to this target?
- Is it justified in the PR description? Tier an unjustified one per [../policy.md](../policy.md) *Tiers*.
- Is it version-pinned, in TheRock's third-party set, and fetched only when `ALLOW_FETCH_DEPS` permits
  (`projects/hipdnn/cmake/Dependencies.cmake`)? Read the set at TheRock's pin without cloning it: the pin is the `therock-ref:` value in
  `.github/actions/ci-env/action.yml` at `$HEAD` ([therock.md](therock.md) *Setup*), the listing
  `gh api "repos/ROCm/TheRock/contents/third-party?ref=<ref>" --jq '.[].name'`; when it cannot be read, mark the claim `[INFERENCE]`.
- Is the license compatible and attributed where the project requires it?

## Boundary violations

- Device dependencies in host code. Host use of the HIP runtime API is allowed throughout core (`hip::host`; `data_sdk` and
  `plugin_sdk` headers include `hip/hip_runtime.h`). A device dependency is `enable_language(HIP)`, `.hip` sources, `hip::device`,
  `--offload-arch`, `__global__`/`__device__` code, hiprtc, comgr or device libraries. Core is `project(hipdnn LANGUAGES CXX)`.
  Report only what the diff adds: compare `git -C "$CLONE" grep -n -e <term> "$BASE" -- projects/hipdnn` with the same at
  `"$HEAD"`, and the changed targets' link lines.
- `hipdnn_frontend_dynamic` gaining a link-time dependency on the backend library.
- Header-only SDKs (`data_sdk`, `plugin_sdk`, `flatbuffers_sdk`) gaining a compiled-library requirement.
- A private dependency leaking through a public header or `PUBLIC` link.
- A provider dependency leaking into core, or one provider depending on another.
- Test-only dependencies reachable from shipped targets.

Prove each violation from the include chain or link line at `$HEAD`; cite every hop. Component and linkage rules:
`projects/hipdnn/docs/ai-rules.md` (component boundaries, including `test_sdk`).
