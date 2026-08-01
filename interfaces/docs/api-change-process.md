# Adding or changing public APIs

## Nonbreaking addition

1. Add the declaration using existing public types where possible.
2. Give new enum values explicit, previously unused numbers at the end of the enum.
3. Regenerate the AST snapshot and review the semantic diff.
4. Classify every new declaration and assign callable declarations to a provider cluster or
   façade target.
5. Add the loader adapter, provider-table tail entry if required, and recording-provider test.
6. Run policy, enum-invariant, export, DSO, and package-consumer tests.

Adding a public function never changes an existing provider-table prefix. Optional provider
capabilities are appended and guarded by table size.

## Breaking source or ABI change

Do not edit or remove the old declaration in place. Add the current API spelling and retain
the old major's metadata and generated loader. Adapt the old call forward at the loader edge.
A new public SONAME/DLL major is required when the public call ABI cannot remain identical.

Existing enum names, values, and underlying types are never changed. Existing record fields
are never reordered. A caller-sized record can only consume documented reserved storage
after size/alignment tests prove that every supported old caller remains valid; otherwise the
new public major indirects through edge allocation.

## Draft-to-launch drift

During rollout, every presubmit extracts current source headers and compares them with the
draft snapshots. A drift failure must be resolved by updating both the current-major mapping
and any already-created compatibility major. Immediately before cutover, regenerate against
the exact release branch, audit exports from built binaries, and archive those snapshots as
the immutable baseline for that major.

No header is considered migrated merely because it lives in a directory named `internal`.
Installed, included, or exported declarations are public until an explicit compatibility
decision says otherwise.

