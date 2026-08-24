# Status and roadmap

Status: proposed design with a working noncanonical rocBLAS implementation. This page
separates what is proven today from what is planned. The three tiers below never blur: DONE
means there is code and a passing test; COMMITTED-NEXT means it is the immediate plan;
ASPIRATIONAL means it is a direction, not a promise.

## DONE (code exists, a test proves it)

Each row cites the `ctest` that locks it and the commit that landed it. On a canonical
amdclang++/ld.lld build the applicable rows register and pass; the exact count depends on
which optional linkers and sanitizers are present, so cite names rather than a count.

| Capability | Proven by (ctest) | Commit |
| --- | --- | --- |
| Provider single-symbol export; no libstdc++ leak | `exports` | `a929517` |
| Named ELF version nodes on loader + provider | `exports` | `ba093ad` |
| `exports` node-aware check extended to the narrow loader shadow (`rocblas_narrow_loader_shadow`) | `exports` | `a741064` |
| `exports` provider list auto-derived from a global registry, with an independent recursive enumeration of every provider `MODULE_LIBRARY`, so a new provider cannot silently escape the check | `exports`, `exports_provider_list_complete` | `61f8dc9`, generalized by current real-provider slice |
| Versioned-provider co-residency (each handle resolves its own version node, cross-version lookup nil) and the bare-lookup interposition hazard reproduced when the version node is removed | `abi03_coresidency`, `abi03_interpose_hazard` | `8832c9a` |
| Core versioned-symbol invariants (ordering, `-Bsymbolic` genuineness via a `DT_FLAGS` `DF_SYMBOLIC` assertion against a plain-DSO control, genuine two-default-definition (`@@`) DSO rejected with a single-`@` accept control, ldconfig stub) | `abi04_three_line_order`, `abi04_bsymbolic_inert`, `abi04_multiple_default_def_rejected`, `abi04_ldconfig_stub_preserved` | `897293e`, `-Bsymbolic` discrimination `215ede4`, dup-def strengthening `b7f3f89` |
| Same-node negative control for the ordering proof: three DSOs on the shared `ROCBLAS_ABI_6` node with `ABI_5`/`ABI_7` lookups nil everywhere | `abi04_same_node_negative` (+ `_lld`) | `215ede4` |
| Same invariants under lld | the `abi04_*_lld` mirrors | `01c14eb` |
| Real `sobol32` data-object versioning | `abi06_data_version_node` | `04ade2b` |
| Version node survives an ASan build, asserted per-symbol (`rocblas_sgemm@@ROCBLAS_ABI_6`) with a genuine-ASan nodeless negative control | `abi04_asan_version_node_survives` | `0081ba5`, tightened `c5031f0` |
| C++ mangled + RTTI versioning | `abi05_cpp_mangled_version_node` (+ `_lld`) | `769f4aa`, tightened `f681132` |
| GCC-LTO-plus-lld build refused at configure | `lto_linker_guard_rejects_gnu_lld` + 3 accepts | `3e56e73` |
| Loader/registry concurrency under TSan | `ops04_concurrency` | `df8512b` |
| Table-ABI negotiation: `abi_minor` floor (provider older than the runtime minor rejected) on top of the `dispatch_table_size` prefix floor, with a larger-table/newer-minor (optional tail) accept and one-field-at-a-time discrimination | `table_abi_negotiation` | `fcb7ac7` |
| Default-on API snapshot and rocBLAS-categorization drift gate | `rocm_interfaces.api_snapshot_drift` | `dc2aa30947a` |
| Linked-consumer interposition proof: a consumer whose relocation carries a versioned undefined reference (`rocblas_sgemm@ROCBLAS_ABI_7`, recorded as a `Verneed` on `libprovB.so.7`) binds to that major even though an ABI_6 provider is `NEEDED` first and earlier in scope; the plain control shares the identical link line and differs only by the single `.symver` directive, and is interposed to ABI_6 - so one directive flips the bound major 7/6 | `abi03_linked_consumer_versioned_binds`, `abi03_linked_consumer_plain_interposed` | `3396f66` |
| Boundary of the node defense (stated, not overclaimed): with genuine version nodes present, `dlvsym` reaches `ROCBLAS_ABI_7`, yet a bare unversioned `dlsym(RTLD_DEFAULT, ...)` still takes the first-loaded `ABI_6` - the defense is scoped to versioned relocations and `dlvsym`, not bare global lookups. Discriminates from `abi03_interpose_hazard` (nodeless DSOs) on lookup form alone | `abi03_versioned_bare_lookup_uncovered` | `8235b3c` |
| Default-off root build integration produces the shadow loader and both real provider targets without enabling them in an ordinary root build | `root_opt_in_build` | current real-provider slice |
| Exhaustive real provider binds canonical rocBLAS by DSO handle, preserves direct behavior for version/status/handle policy calls, and fails closed with a host trace for missing or incomplete backends | `rocblas_real_provider_differential`, `real_provider_missing_backend`, `real_provider_incomplete_backend` | current real-provider slice |
| Installed package exports the shadow loader and installs a strict same-directory provider manifest that selects the real provider | `install_consumer`, strict-manifest cases in `unit` | current real-provider slice |
| Real-provider initialization and dispatch are concurrency-safe | `rocblas_real_provider_concurrency` (also run in the TSan configuration) | current real-provider slice |
| First semantic migration: single-batch FP32 AXPY, SCAL, COPY, and SWAP, with both public index widths, execute through the narrow-v2 vector-transform callback into a backend DSO | `rocblas_narrow_v2_real_vector_transform` | current real-provider slice |
| GPU differential harness compares canonical rocBLAS, the exhaustive provider, and narrow-v2 across AXPY/SCAL/COPY/SWAP, both index widths, host/device scalars, two streams, negative increments, quick returns, invalid arguments, asynchronous completion, and strided storage | `rocblas_gpu_differential` and `interfaces-gpu-ci.yml` | current GPU-validation slice |

Foundational, from the POC base (`9bd0d26`): the loader/runtime/protocols architecture, the
recording providers and rocBLAS bridge, and the narrow-v2 facade are each exercised by named
ctests (`rocm_interfaces.unit`, `rocm_interfaces.rocblas_shadow`, `rocm_interfaces.rocblas_narrow_v2`),
and the install-consumer path by `rocm_interfaces.install_consumer`. The 1,219-row
categorization ledger is checked in, and its regeneration check is a dependency of the API
snapshot drift gate. The Clang-LibTooling extraction and snapshot tooling exist as build
targets (`rocm-interfaces-api-snapshots`, `rocm-interfaces-check-api-snapshots`). With
`BUILD_TESTING=ON`, `rocm_interfaces.api_snapshot_drift` is registered by default because
`ROCM_INTERFACES_CHECK_API_DRIFT` defaults to `ON`; `BUILD_TESTING=OFF` registers no CTest.
The check is not part of the default build (`ALL`) or an automatically wired presubmit, and
the build target remains directly runnable. The reconciled snapshots and ledger are a
pre-adoption prototype baseline, not a claim of append-only ABI evolution for a launched
provider table. The `check_api_policy.py` policy check is still unwired.

## COMMITTED-NEXT (the immediate plan)

- Push and merge mechanics (final review, force-with-lease update after the completed rebase,
  and attaching to PR #10272) are tracked in the PR/handoff, not here.

## ASPIRATIONAL (direction, not commitment)

- **More libraries.** The hardening is proven on rocBLAS-shaped and rocRAND-shaped symbols.
  hipSOLVER is a facade with representation coupling that needs its own edge classification
  before it can adopt the boundary (see [audit-findings.md](audit-findings.md)).
- **Canonical cutover.** The root build now has a default-off interfaces switch, but shipping
  canonical library names remains deliberately deferred until all adapters, package-config
  parity, numerical validation, and published-major coexistence criteria are complete.
- **A broader proof suite.** The interposition story is now proven end to end - handle-scoped
  co-residency, the nodeless hazard, the linked-consumer relocation, and the bare-`RTLD_DEFAULT`
  boundary (all in DONE). Further symbol shapes and toolchains are added as they become relevant.
- **API-process evolution.** Promoting the manual doc-quality checklist
  ([STYLE.md](STYLE.md)) to a CI gate if the docs start to drift.

## What is deliberately NOT claimed

- The layer is not adopted and not on the default build path.
- Canonical library names (`librocblas.so`, not `librocblas-loader.so`) are intentionally
  absent until the cutover criteria above are met.
- Legacy in-library Fortran bindings are out of scope; hipfort is the supported direction.
- Platform scope is Linux/ELF only. The ABI hardening and versioning proofs are registered under UNIX AND NOT APPLE; Windows/PE and macOS/Mach-O are unproven and out of scope, and the DLL/PE ABI mechanism is not part of the normative contract. The module-load primitive in runtime/src/module.cpp does carry a LoadLibraryW path, but no ABI-versioning proof covers it.
