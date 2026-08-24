# Extending the interfaces layer

Status: proposed design, prototype-backed. These are the how-to recipes a maintainer runs.
Each one ends where a test begins - if you add a capability, you add the proof that locks
it. The contract these recipes obey is [03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md);
the proofs they mirror are in [04-hardening.md](04-hardening.md).

## Recipe: add a provider

A provider is a `.so` that exports exactly one symbol and hands back a dispatch table. To
add one:

1. Implement the query function `rocm_interfaces_provider_query_v1` (signature in
   `protocols/include/rocm/interfaces/common.h`). Fill in the response: `provider_id`,
   `build_id`, the `dispatch_table` pointer, its `dispatch_table_size`, and the ABI header.
2. Fill the dispatch table for your domain (`blas.h`, `rand.h`, or `solver.h`). Start the
   table with `rocm_interfaces_abi_header` and set `struct_size` to the real size.
3. Register the target in `ROCM_INTERFACES_PROVIDER_TARGETS` and apply
   `providers/provider.map`, which hides everything but the one bootstrap symbol.
4. Route all logging through the host `trace` callback - never `printf`. Use the host
   `allocate`/`deallocate` callbacks for memory that crosses the ABI boundary or that the
   host must own or free; a provider's own private context (created in `create_context`,
   released in `destroy_context`) may use its internal allocator, as the recording
   providers do with `new`/`delete`. Do not retain any pointer from the request record
   after the call returns.

**Lock it.** `rocm_interfaces.exports` derives its provider list from the global
`ROCM_INTERFACES_PROVIDER_TARGETS` build-system property populated by provider targets and
asserts that every listed DSO exports only
`rocm_interfaces_provider_query_v1` under the named version node. The independent
`rocm_interfaces.exports_provider_list_complete` control recursively enumerates all
`MODULE_LIBRARY` targets under `providers/` and requires the two lists to match. A
registration/enumeration mismatch fails the completeness control instead of silently
skipping export inspection.

## Recipe: add a version node (a new ABI major)

You add a node when a public library gains a new incompatible major that must coexist with
the old one.

1. Write a version script that names the exported symbols under the new node with an
   explicit allowlist, one symbol per line, following `loader/rocblas_loader.map` (which
   lists each symbol by name) - list every exported symbol, for example:

   ```
   ROCBLAS_ABI_8 {
     global:
       rocblas_create_handle;
       rocblas_sgemm;
     local:
       *;
   };
   ```

   Do NOT use a wildcard such as
   `global: rocblas_*`: a glob is not a frozen allowlist - once the node ages into the old
   major, any future symbol whose name matches is silently admitted into the supposedly
   frozen ABI. Use an explicit or generated symbol list (the `exports` test already pins
   the loader to the generated `rocblas_bridge.exports` allowlist). For C++ symbols use
   mangled-name globs (`_ZN<len><ns><len><class>*`, plus `_ZTVN..`/`_ZTIN..`/`_ZTSN..` for
   RTTI), not source spellings. Caveat: a class- or namespace-wide mangled glob
   (`_ZN11rocrand_cpp5error*`) has the same aging hazard as `rocblas_*` - once the node is
   the frozen old major, a method added to that class later is silently admitted into the
   frozen ABI. Scope the glob as narrowly as the class allows, and once a class is frozen
   prefer pinning the exact mangled member names (or a generated allowlist) over an open `*`.
2. Apply it with `target_link_options(<tgt> PRIVATE
   "LINKER:--version-script=<file>")`, following the idiom in
   `loader/rocblas_loader.map`'s target.
3. Keep the old major's map, metadata, and generated loader in place. Do not edit the old
   node's symbol set. Adapt the old public call forward at the loader edge.
4. Give the new major its own SONAME. The ELF version-node step above is Linux/ELF-only;
   the Windows/PE DLL-versioning path is unproven and out of scope (see
   [07-status-and-roadmap.md](07-status-and-roadmap.md#what-is-deliberately-not-claimed)).

**Lock it.** Add a co-residency assertion in the shape of
`rocm_interfaces.abi03_coresidency`: load old and new majors together and assert each
resolves to its own node. Add the nodeless negative control (`abi03_anon.map` idiom) so the
test would fail if the nodes were absent.

## Recipe: add an ABI proof

This is the most important recipe, because a vacuous proof is worse than none. Follow the
non-vacuity recipe from
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md): positive, negative
control, genuineness.

1. **Positive.** Build the DSO correctly and assert the exact node is on the exact symbols
   (`nm -D` shows `sym@@NODE`).
2. **Negative control.** Build a second DSO with the node removed or wrong and assert the
   check now fails. If it still passes, stop - your assertion is not discriminating.
3. **Genuineness.** Assert the DSO is what it claims (lld `.comment` stamp, `__asan_` or
   `__tsan_` symbols) so a silent fallback cannot pass for the wrong reason.
4. Write the check as a `cmake -P` driver (for build/link-shape checks) or a runtime test
   (for `dlvsym` resolution), matching whichever sibling is closest.
   `abi05_cpp_mangled_version_node` is one complete template (positive plus three negative
   controls). The other is `abi04_three_line_order` paired with
   `abi04_same_node_negative`; copy both halves so the same-node control remains discrete from
   the positive ordering proof.
5. Register it in `tests/CMakeLists.txt`. Gate it on a configure-time probe if it needs a
   linker or header the base toolchain may lack (see `ROCM_INTERFACES_HAVE_LLD` and
   `ROCM_INTERFACES_HAVE_ROCRAND_CPP`), so it never breaks a bare configure.

Prove non-vacuity by mutation before you trust it: drop the node, remove an assertion's
target, break the ODR-use, and confirm each mutation flips the test to failing - exactly the
four mutations that validated `abi05_cpp_mangled_version_node`.

## Recipe: add or change a public API

This absorbs the former `api-change-process.md`. There are two paths, and the difference is
whether the public call ABI can stay identical.

### Non-breaking addition

1. Add the declaration using existing public types where possible.
2. Give any new enum value an explicit, previously-unused number at the **end** of the enum.
3. Regenerate the AST snapshot (`cmake --build <build> --target
   rocm-interfaces-api-snapshots`) and review the semantic diff.
4. Classify every new declaration and assign each callable to a provider cluster or facade
   target (see [rocblas-provider-clusters.md](rocblas-provider-clusters.md)).
5. Add the loader adapter and a recording-provider test. If the table must grow, append the
   new function pointer to the end (never touch the existing prefix), bump `abi_minor`, and
   raise the loader's required table size. The registry prefix and minor floors are proven by
   `rocm_interfaces.table_abi_negotiation`, but current domain loaders request the full table
   size, so adding a required entry still rejects every older provider. Per-domain
   optional-tail consumption is not implemented; see the implementation-status note in
   [03](03-abi-and-versioning-contract.md#implementation-status-prototype). Do not call an
   appended entry optional until its loader requests only the stable prefix, checks the
   reported tail size, and supplies a fallback.
6. Run the policy, enum-invariant, export, DSO, and package-consumer tests.

Adding a public function never changes an existing provider-table prefix. That is the whole
reason old callers keep working (Mechanism 1 in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md)).

### Breaking source or ABI change

1. Do not edit or remove the old declaration in place.
2. Add the current API spelling; retain the old major's metadata and generated loader.
3. Adapt the old call forward at the loader edge.
4. Cut a new public SONAME major on ELF and give it a new ELF version node per the "add a
   version node" recipe. The Windows/PE DLL-major equivalent is not proven by this contract
   (Linux/ELF only; see
   [07-status-and-roadmap.md](07-status-and-roadmap.md#what-is-deliberately-not-claimed)) -
   do not assume the version-node step transfers to PE.

Existing enum names, values, and underlying types are never changed. Existing record fields
are never reordered. A caller-sized record may consume documented reserved storage only
after size/alignment tests prove every supported old caller stays valid; otherwise the new
major indirects through edge allocation.

### Draft-to-launch drift

During rollout, header drift is caught by the `rocm-interfaces-check-api-snapshots` build
target, which extracts the current source headers and byte-compares them with the draft
snapshots and checks the rocBLAS categorization ledger. With `BUILD_TESTING=ON`, it is wired
into CTest as `rocm_interfaces.api_snapshot_drift` by default because
`ROCM_INTERFACES_CHECK_API_DRIFT` defaults to `ON`; setting that option to `OFF` is an
explicit opt-out. `BUILD_TESTING=OFF` registers no CTest. The target is not part of the
default build (`ALL`) or an automatically wired presubmit, and it can still be run directly
(`cmake --build <build> --target rocm-interfaces-check-api-snapshots`). Wiring
`check_api_policy.py` (which still has no build or test invocation) remains ASPIRATIONAL.

The checked-in snapshots and categorization ledger are code-generation inputs as well as the
draft baseline. Resolve a drift failure by reconciling the snapshot, current-major mapping,
categorization, generators, and any already-created compatibility major together. The current
reconciliation is a pre-adoption prototype rebaseline; it does not claim append-only ABI
evolution for a launched provider table. Immediately before cutover, regenerate against the
exact release branch, audit exports from the built binaries, and archive those snapshots as
the immutable baseline for that major.

No header is "migrated" merely because it lives in a directory named `internal`. Installed,
included, or exported declarations are public until an explicit compatibility decision says
otherwise.
