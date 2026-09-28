# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
stages into the build tree for the unit and integration binaries and is installed only
under `HIPKERNELPROVIDER_ENABLE_TESTS`. This README is the root's only content, so
production packaging is dormant unless the cache variable below is pointed elsewhere.

## Authoring a bundle

```
descriptors/<producer>/<bundle>/
```

A bundle is authored under the producer that builds its kernels. `rocKE/` is spelled with
that capitalization because the packer preserves an authored subpath verbatim into the
staged and installed trees. A bundle sits one level under its producer, so every
descriptor lands in a child of its shard root and the archive can be written at the root.

Nothing here is registered in CMake: the packer walks this root recursively, so adding a
bundle is dropping files in a folder. Kernel-source *embedding* is a separate mechanism,
required only for `kernel_source.kind == "embedded_source"`.

A bundle here is compiled at pack time, one comgr invocation per variant, on every build
that has this root wired — CI included. Trim an authored variant set to a covering subset
before it lands, and register the symbols its UKDs name in a native pack, or the loader
refuses the engine at provider load and every lowered kernel is wasted build time.

## Root selection and dormancy

`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` is a `CACHE PATH` defaulting to this
directory; a consumer needing a different root overrides that variable rather than adding
CMake. Production wiring is gated on at least one non-hidden `*.kdp.json` here, since a
KDP is what architecture pruning consumes. With none, packaging stays dormant and any
stale product tree is removed — not an error. A KDP that *is* present but prunes on every
architecture is a hard failure for a root the build NAMED, and dormancy for this root
reached as the built-in default.

Do not delete this README: git tracks no empty directory, and the cache variable's
set-but-not-a-directory check is fatal.

## Relationship to the examples tree

`descriptor-packaging/examples/descriptors/` is a **test fixture** tree. Bundles are
authored and proved there against the packaging suite, then relocated onto this root once
a native pack registers the symbols their UKDs name. The Linux superbuild CI lane
overrides `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` to that fixture tree, so the
production packing rule this root would use is exercised there while this root stays
dormant.
