# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
holds the ones it does not: that tree is staged into the build tree for the unit and
integration binaries and is installed only under `HIPKERNELPROVIDER_ENABLE_TESTS`. The two
names are the whole convention — `descriptors/` ships, `test_descriptors/` does not.

This README is the root's only content: no bundle is authored here, so production
packaging is dormant in every build that does not override the cache variable below.

## Producer subfolders

A bundle is authored under the producer that builds its kernels:

```
descriptors/<producer>/<bundle>/
```

`rocKE/` is spelled with that capitalization because the packer preserves an authored
subpath verbatim into the staged and installed trees. A bundle sits one level under its
producer: every descriptor lands in a child of its shard root, so the archive can be
written at the root itself.

Nothing here is registered in CMake. The packer discovers descriptors by walking this root
recursively, so adding a bundle is dropping files in a folder. A kernel-source *embedding*
entry is a different mechanism and is required only for `kernel_source.kind ==
"embedded_source"`; a `rocke` bundle needs none.

A bundle dropped here is compiled at pack time, one comgr invocation per variant, on every
build that has this root wired — CI included. An authored variant set of any size therefore
has to be trimmed to a covering subset before it lands, and a bundle also needs a native
pack registering the symbols its UKDs name, or the loader refuses the engine at provider
load and every lowered kernel is wasted build time.

## How this root is selected, and what an empty one does

`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` is a `CACHE PATH` defaulting to this directory.
A consumer who needs a different root overrides that variable; they do not add CMake.

Production pack wiring is gated on this root holding at least one non-hidden `*.kdp.json`.
A KDP is what architecture pruning consumes, so standalone UKDs, kernel sources and this
README do not by themselves make a pack. With none, production packaging stays dormant and
any stale product tree from an earlier configure is removed — that is not an error. A KDP
that *is* present but is pruned on every architecture remains a hard failure for a root
the build NAMED, since naming it asserts it ships here. This root reached as the built-in
default goes dormant instead, because a build that never mentioned descriptors asked for
nothing and so cannot have failed to get it.

This file also keeps the directory present in a fresh checkout. Git tracks no empty
directory, and the cache variable's set-but-not-a-directory check is a fatal, so deleting
this README turns a clean configure into a `FATAL_ERROR`.

## Relationship to the examples tree

`descriptor-packaging/examples/descriptors/` is a **test fixture** tree, not the default
production source root. Bundles are authored and proved there against the packaging suite, and are
relocated onto this root once a native pack registers the symbols their UKDs name.
The Linux superbuild CI lane overrides `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` to
that fixture tree, so the production packing rule this root would use is exercised
there even while this root stays empty and dormant.
