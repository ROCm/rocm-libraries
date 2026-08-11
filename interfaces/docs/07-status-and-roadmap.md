# Status and roadmap

Status: proposed design, prototype-backed. This page separates what is proven today from
what is planned. The three tiers below never blur: DONE means there is code and a passing
test; COMMITTED-NEXT means it is the immediate plan; ASPIRATIONAL means it is a direction,
not a promise.

## DONE (code exists, a test proves it)

Each row cites the `ctest` that locks it and the commit that landed it. The stack is 12
commits on `users/davidd-amd/mathlibs-interfaces-impl`, rebased onto `develop`, reproven
green (a canonical amdclang++/ld.lld build registers 26 tests, all passing).

| Capability | Proven by (ctest) | Commit |
| --- | --- | --- |
| Provider single-symbol export; no libstdc++ leak | `exports` | `a929517` |
| Named ELF version nodes on loader + provider | `exports` | `ba093ad` |
| Version nodes defeat interposition; co-residency | `abi03_coresidency`, `abi03_interpose_hazard` | `8832c9a` |
| Core versioned-symbol invariants (ordering, `-Bsymbolic` inert, dup-def rejected, ldconfig stub) | `abi04_three_line_order`, `abi04_bsymbolic_inert`, `abi04_multiple_default_def_rejected`, `abi04_ldconfig_stub_preserved` | `897293e` |
| Same invariants under lld | the four `abi04_*_lld` | `01c14eb` |
| Real `sobol32` data-object versioning | `abi06_data_version_node` | `04ade2b` |
| Version node survives an ASan build | `abi04_asan_version_node_survives` | `0081ba5` |
| C++ mangled + RTTI versioning | `abi05_cpp_mangled_version_node` (+ `_lld`) | `769f4aa`, tightened `f681132` |
| GCC-LTO-plus-lld build refused at configure | `lto_linker_guard_rejects_gnu_lld` + 3 accepts | `3e56e73` |
| Loader/registry concurrency under TSan | `ops04_concurrency` | `df8512b` |

Foundational, from the POC base (`9bd0d26`): the loader/runtime/protocols architecture, the
recording providers and rocBLAS bridge, the 1,213-row categorization ledger, the narrow-v2
facade, the Clang-LibTooling API extraction and snapshot/policy tooling, and the install-
consumer test (`rocm_interfaces.install_consumer`).

## COMMITTED-NEXT (the immediate plan)

- **Push the stack.** It is complete and green but unpushed by standing policy. Awaiting the
  explicit go.
- **Attach to PR #10272** (the POC PR that introduces `interfaces/`) and take it through
  review and merge.
- **Land this documentation set** as one docs change on the branch.
- Before the real rebase-and-push, re-run `git merge-tree` against a freshly fetched
  `origin/develop` (the dry-run was against local `develop`; `interfaces/` is net-new, so the
  merge was conflict-free, but re-verify after a live fetch).

## ASPIRATIONAL (direction, not commitment)

- **More libraries.** The hardening is proven on rocBLAS-shaped and rocRAND-shaped symbols.
  hipSOLVER is a facade with representation coupling that needs its own edge classification
  before it can adopt the boundary (see [audit-findings.md](audit-findings.md)).
- **Root-build wiring.** Today the tree builds standalone (`cmake -S interfaces`). Wiring it
  into the root ROCm build and shipping canonical library names is deliberately deferred
  until every exported declaration is classified, all adapters exist, package-config parity
  is demonstrated, and coexistence tests cover the published majors.
- **A broader proof suite.** More symbol shapes and toolchains as they become relevant.
- **API-process evolution.** Promoting the manual doc-quality checklist
  ([STYLE.md](STYLE.md)) to a CI gate if the docs start to drift.

## What is deliberately NOT claimed

- The layer is not adopted and not on the default build path.
- Canonical library names (`librocblas.so`, not `librocblas-loader.so`) are intentionally
  absent until the cutover criteria above are met.
- Legacy in-library Fortran bindings are out of scope; hipfort is the supported direction.
