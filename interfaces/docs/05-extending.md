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
3. Register the target through `add_recording_provider()` (or the equivalent) in
   `providers/recording/CMakeLists.txt` so it inherits the `--version-script` that hides
   everything but the one symbol.
4. Use the host services for allocation, deallocation, and tracing - never `malloc` or
   `printf` directly. Do not retain any pointer from the request record after the call
   returns.

**Lock it.** The `rocm_interfaces.exports` test already iterates every provider DSO and
asserts a single exported symbol, so a new provider is covered the moment it is registered
that way. If it exports more than one symbol, that test fails - which is the point.

## Recipe: add a version node (a new ABI major)

You add a node when a public library gains a new incompatible major that must coexist with
the old one.

1. Write a version script naming the exported symbols under the new node, e.g.
   `ROCBLAS_ABI_8 { global: rocblas_*; local: *; };`. For C++ symbols use mangled-name
   globs (`_ZN<len><ns><len><class>*`, plus `_ZTVN..`/`_ZTIN..`/`_ZTSN..` for RTTI), not
   source spellings.
2. Apply it with `target_link_options(<tgt> PRIVATE
   "LINKER:--version-script=<file>")`, following the idiom in
   `loader/rocblas_loader.map`'s target.
3. Keep the old major's map, metadata, and generated loader in place. Do not edit the old
   node's symbol set. Adapt the old public call forward at the loader edge.
4. Give the new major its own SONAME major.

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
   (for `dlvsym` resolution), matching whichever sibling is closest. `abi04_three_line_order`
   and `abi05_cpp_mangled_version_node` are the two templates.
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
5. Add the loader adapter, append a provider-table tail entry if one is needed (never touch
   the existing prefix; guard the new entry by `struct_size` and bump `abi_minor`), and add
   a recording-provider test.
6. Run the policy, enum-invariant, export, DSO, and package-consumer tests.

Adding a public function never changes an existing provider-table prefix. That is the whole
reason old callers keep working (Mechanism 1 in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md)).

### Breaking source or ABI change

1. Do not edit or remove the old declaration in place.
2. Add the current API spelling; retain the old major's metadata and generated loader.
3. Adapt the old call forward at the loader edge.
4. Cut a new public SONAME/DLL major (the call ABI cannot remain identical), and give it a
   new version node per the "add a version node" recipe.

Existing enum names, values, and underlying types are never changed. Existing record fields
are never reordered. A caller-sized record may consume documented reserved storage only
after size/alignment tests prove every supported old caller stays valid; otherwise the new
major indirects through edge allocation.

### Draft-to-launch drift

During rollout, every presubmit extracts the current source headers and byte-compares them
with the draft snapshots (`rocm-interfaces-check-api-snapshots`). A drift failure is resolved
by updating both the current-major mapping and any already-created compatibility major.
Immediately before cutover, regenerate against the exact release branch, audit exports from
the built binaries, and archive those snapshots as the immutable baseline for that major.

No header is "migrated" merely because it lives in a directory named `internal`. Installed,
included, or exported declarations are public until an explicit compatibility decision says
otherwise.
