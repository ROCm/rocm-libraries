# Status and roadmap

Status: proposed design, prototype-backed. This page separates what is proven today from
what is planned. The three tiers below never blur: DONE means there is code and a passing
test; COMMITTED-NEXT means it is the immediate plan; ASPIRATIONAL means it is a direction,
not a promise.

## DONE (code exists, a test proves it)

Each row cites the `ctest` that locks it and the commit that landed it. On a canonical
amdclang++/ld.lld build the applicable rows register and pass; the exact count depends on
which optional linkers and sanitizers are present, so cite names rather than a count.

| Capability | Proven by (ctest) | Commit |
| --- | --- | --- |
| Provider single-symbol export; no libstdc++ leak | `exports` | `a929517` |
| Named ELF version nodes on loader + provider | `exports` | `ba093ad` |
| Versioned-provider co-residency (each handle resolves its own version node, cross-version lookup nil) and the bare-lookup interposition hazard reproduced when the version node is removed | `abi03_coresidency`, `abi03_interpose_hazard` | `8832c9a` |
| Core versioned-symbol invariants (ordering, `-Bsymbolic` genuineness via a `DT_FLAGS` `DF_SYMBOLIC` assertion against a plain-DSO control, dup-def guard (linker-observed; strengthening tracked), ldconfig stub) | `abi04_three_line_order`, `abi04_bsymbolic_inert`, `abi04_multiple_default_def_rejected`, `abi04_ldconfig_stub_preserved` | `897293e`, `-Bsymbolic` discrimination `215ede4` |
| Same-node negative control for the ordering proof: three DSOs on the shared `ROCBLAS_ABI_6` node with `ABI_5`/`ABI_7` lookups nil everywhere | `abi04_same_node_negative` (+ `_lld`) | `215ede4` |
| Same invariants under lld | the four `abi04_*_lld` | `01c14eb` |
| Real `sobol32` data-object versioning | `abi06_data_version_node` | `04ade2b` |
| Version node survives an ASan build | `abi04_asan_version_node_survives` | `0081ba5` |
| C++ mangled + RTTI versioning | `abi05_cpp_mangled_version_node` (+ `_lld`) | `769f4aa`, tightened `f681132` |
| GCC-LTO-plus-lld build refused at configure | `lto_linker_guard_rejects_gnu_lld` + 3 accepts | `3e56e73` |
| Loader/registry concurrency under TSan | `ops04_concurrency` | `df8512b` |

Foundational, from the POC base (`9bd0d26`): the loader/runtime/protocols architecture, the
recording providers and rocBLAS bridge, and the narrow-v2 facade are each exercised by named
ctests (`rocm_interfaces.unit`, `rocm_interfaces.rocblas_shadow`, `rocm_interfaces.rocblas_narrow_v2`),
and the install-consumer path by `rocm_interfaces.install_consumer`. The 1,213-row
categorization ledger is a checked-in data artifact, not a test. The Clang-LibTooling API
extraction and the snapshot/policy tooling exist as build targets
(`rocm-interfaces-api-snapshots`, `rocm-interfaces-check-api-snapshots`) but are not yet run
under `ctest`; wiring the drift check into the test suite is listed under COMMITTED-NEXT.

## COMMITTED-NEXT (the immediate plan)

- Push and merge mechanics (rebase, merge-tree re-verify, attaching to PR #10272) are tracked
  in the PR/handoff, not here.
- Wire the API snapshot drift check (`rocm-interfaces-check-api-snapshots`) into `ctest` so
  header drift is caught by the test suite rather than only by a manual target.
- **Strengthen the multiple-default-definition check.** `abi04_multiple_default_def_rejected`
  currently passes on any link failure and on a zero- or one-`@@` result
  (`tests/check_multiple_default_def.cmake`), so it observes the linker rather than forcing a
  genuine two-`@@` DSO to be rejected. Construct a real two-default-definition input and assert
  the FATAL fires.
- **Tighten the ASan node test.** `abi04_asan_version_node_survives` currently only
  substring-matches `ROCBLAS_ABI_6` (satisfied by the absolute `ROCBLAS_ABI_6@@ROCBLAS_ABI_6`
  node symbol); change it to assert `rocblas_sgemm@@ROCBLAS_ABI_6` and add a nodeless negative
  control, matching 6a/6c/6e.
- **Extend the `exports` node-aware check to `rocblas_narrow_loader_shadow`.** The static
  11-symbol `loader/rocblas_loader.map` target is currently covered only by the behavioral
  `rocm_interfaces.rocblas_narrow_shadow` test, not by the `exports` version-node assertion.
- **Table-ABI prefix negotiation.** Today only the response `dispatch_table_size` floor is
  checked. Reading the dispatch table's own `abi_header`, honoring `abi_minor`, distinguishing
  a required prefix from an optional appended tail, and a non-vacuous ctest proving a larger
  table's prefix is accepted while an old provider missing a newly required entry is rejected,
  are not yet implemented.
- **Auto-derive the exports-test provider list.** Today `tests/CMakeLists.txt` and
  `tests/check_exports.cmake` carry a duplicated hard-coded list of provider DSOs; a provider
  added via `add_recording_provider` is unchecked until both are edited by hand. Generate the
  list so every registered provider is inspected automatically.

## ASPIRATIONAL (direction, not commitment)

- **More libraries.** The hardening is proven on rocBLAS-shaped and rocRAND-shaped symbols.
  hipSOLVER is a facade with representation coupling that needs its own edge classification
  before it can adopt the boundary (see [audit-findings.md](audit-findings.md)).
- **Root-build wiring.** Today the tree builds standalone (`cmake -S interfaces`). Wiring it
  into the root ROCm build and shipping canonical library names is deliberately deferred
  until every exported declaration is classified, all adapters exist, package-config parity
  is demonstrated, and coexistence tests cover the published majors.
- **A broader proof suite.** More symbol shapes and toolchains as they become relevant.
- **Linked-consumer interposition proof.** The abi03 positive case only exercises handle-scoped dlvsym/dlsym, which resolve each provider's own symbol regardless of version nodes; the causal proof that version nodes alone defeat interposition under a bare RTLD_DEFAULT lookup or a linked-consumer relocation (a consumer needing rocblas_sgemm@ROCBLAS_ABI_x loaded alongside a provider offering a different major) is not yet ported into CTest.
- **API-process evolution.** Promoting the manual doc-quality checklist
  ([STYLE.md](STYLE.md)) to a CI gate if the docs start to drift.

## What is deliberately NOT claimed

- The layer is not adopted and not on the default build path.
- Canonical library names (`librocblas.so`, not `librocblas-loader.so`) are intentionally
  absent until the cutover criteria above are met.
- Legacy in-library Fortran bindings are out of scope; hipfort is the supported direction.
- Platform scope is Linux/ELF only. The ABI hardening and versioning proofs are registered under UNIX AND NOT APPLE; Windows/PE and macOS/Mach-O are unproven and out of scope, and the DLL/PE ABI mechanism is not part of the normative contract. The module-load primitive in runtime/src/module.cpp does carry a LoadLibraryW path, but no ABI-versioning proof covers it.
