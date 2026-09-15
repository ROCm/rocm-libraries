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

### The KDP carries 4 kernels, deliberately

The authored set holds 2,733 variants. It is trimmed to 4 here because every variant is
compiled at pack time, and the full set costs several hundred seconds of comgr work on
every build that has a production root wired — including CI. The variant set is being
reauthored, so this subset is a placeholder and is expected to be replaced wholesale
rather than grown one kernel at a time.

The 4 were chosen by set cover over the tuning knobs, so between them they exhibit every
value the full set gives to `dtype`, `block_m`, `causal`, `persistent`, `use_exp2_fast`,
`head_size` and `waves_per_eu` — 15 distinct knob/value pairs, all covered:

| kernel | dtype | block_m | causal | persistent | exp2 | head_size | waves |
|---|---|---|---|---|---|---|---|
| `…_sq1024_d128_c0_bm256_bn64_w2_p0_ed` | BF16 | 256 | 0 | 0 | 0 | 128 | 2 |
| `…full_00091_…_d64_c1_bm128_e1` | BF16 | 128 | 1 | 1 | 1 | 64 | 4 |
| `…_fp16_…_sq2048_d128_c0_bm64_bn64_w2_p0_e1` | FP16 | 64 | 0 | 0 | 1 | 128 | 2 |
| `…_sq1024_d128_c0_bm128_bn64_w2_p0_e1` | BF16 | 128 | 0 | 0 | 1 | 128 | 2 |

Four rather than one: the packer only enters its parallel path above a single variant
(`pipeline.py`, `len(jobs) < 2` returns early), so a one-kernel set would take the
worker path out of every build. Keeping the metadata tuples distinct also keeps the
loader's duplicate-name and duplicate-tuple refusals doing real work instead of passing
vacuously.

There is no config in `IngestorGenerator/configs/` for this set, so it cannot be
regenerated from the tree — the trim was applied to the KDP directly, and the table above
is the only record of which variants survived.

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
