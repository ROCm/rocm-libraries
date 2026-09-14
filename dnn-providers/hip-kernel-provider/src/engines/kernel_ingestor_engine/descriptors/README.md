# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
holds the ones it does not: that tree is staged into the build tree for the unit and
integration binaries and is installed only under `HIPKERNELPROVIDER_ENABLE_TESTS`. The two
names are the whole convention — `descriptors/` ships, `test_descriptors/` does not.

## Producer subfolders

A bundle is authored under the producer that builds its kernels:

```
descriptors/<producer>/<bundle>/
```

`rocKE/` is spelled with that capitalization because the packer preserves an authored
subpath verbatim into the staged and installed trees, and it is the spelling the in-repo
packaged configs already emit. A bundle is one level under its producer, which keeps the
same one-level-of-nesting rule `test_descriptors/README.md` states for the test sets: every
descriptor lands in a child of its shard root, so the archive can be written at the root
itself.

Nothing here is registered in CMake. The packer discovers descriptors by walking this root
recursively, so adding a bundle is dropping files in a folder. A kernel-source *embedding*
entry is a different mechanism and is required only for `kernel_source.kind ==
"embedded_source"`; the bundle below is kind `rocke` and needs none.

## What this branch ships

```
rocKE/gfx942_attention_dense/     six descriptors, kind `rocke`, arch gfx942
```

Its kernels lower at pack time through the rocKE wheel's
`kernels/gfx942/attention_dense.py`, which ships outside this repository.

## How this root is selected, and what an empty one does

`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` is a `CACHE PATH` defaulting to this directory.
A consumer who needs a different root overrides that variable; they do not add CMake.

Production pack wiring is gated on this root holding at least one non-hidden `*.kdp.json`.
A KDP is what architecture pruning consumes, so standalone UKDs, kernel sources and this
README do not by themselves make a pack. With none, production packaging stays dormant and
any stale product tree from an earlier configure is removed — that is not an error. A KDP
that *is* present but is pruned on every architecture remains a hard failure: the gate
distinguishes "nothing to ship" from "something to ship that did not".

This file also keeps the directory present in a fresh checkout. Git tracks no empty
directory, and the cache variable's set-but-not-a-directory check is a fatal, so deleting
this README turns a clean configure into a `FATAL_ERROR`.

## Relationship to the examples tree

`descriptor-packaging/examples/descriptors/` is a **test fixture** tree, not a production
one. It is read by the packaging suite's real-bundle regressions and is no longer wired as
a production source root.

PR #11509 authors its own copy of `gfx942_attention_dense` under that examples tree and
relocates it onto this root when it rebases.
