# Support Claims Bring-Up Guide

End-to-end walkthrough for adding a new asic (or a brand-new engine) to
the support-claims system. See [schema](support-claims-schema.md) for
field-by-field reference and [failures](support-claims-failures.md) for
the failure-mode runbook.

The example uses miopen-provider; substitute your provider name for
other plugins as they adopt the system.

## TL;DR

```bash
./hipdnn_integration_tests \
    --te MIOPEN_ENGINE \
    --ta path/to/libmiopen_plugin.so \
    --tc dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --write-support-claims

# Review the diff
git diff dnn-providers/miopen-provider/config/MIOPEN_ENGINE.supported.toml

# Commit both files together
git add dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
        dnn-providers/miopen-provider/config/MIOPEN_ENGINE.supported.toml
git commit -m "miopen: claim support on gfx<N>"
```

That's the whole loop. The rest of this page explains *why* and *what
to look for* when reviewing the diff.

## Prerequisites

Before running `--write-support-claims`, the target arch must be a
device the binary can actually use. The tool reads the live
`gcnArchName` from the device and writes a block scoped to that arch
token. Running on `gfx942` produces the gfx942 block. To populate
multiple arches, run on each in turn — the tool only touches the block
for the current arch and platform, leaving the others alone.

The build must be **release** (`-DCMAKE_BUILD_TYPE=Release` or the
generated `NDEBUG`-defining preset). Debug builds can produce different
gtest parameter strings, which would make the recorded `op_chain` text
diverge from what `--enforce-support-claims` sees later.

## What `--write-support-claims` actually does

1. Runs the full integration suite (no `--gtest_filter`, no shard env)
   with the support-matrix collector enabled.
2. For every observed `(op_chain, io_dtype, layout)` tuple, classifies
   into `S` (engine supports) or `U` (engine returned empty support).
3. For each op_chain in `S`, computes the largest `io_dtype × layout`
   sub-rectangle whose cross-product is fully in `S` and disjoint from
   `U`. This is the op_chain's **safe rectangle** — claiming anything
   beyond it risks Rule A on the next run.
4. Groups op_chains that share the same safe rectangle into a single
   matcher. Pure set operations — no tries, no token-splitting.
5. Atomically replaces the `[[supported]]` block(s) for the current
   `(arch, platform)` in `<EngineName>.supported.toml`. Other arches
   and the entire main TOML are untouched.

If any tuples were in `U`, the tool prints up to 10 of them on stderr so
you can sanity-check the carve-outs ("yes, the engine genuinely has no
CK kernel for NHWC bf16 — that's expected").

## Reviewing the diff

Three things to check before committing.

**1. The matcher count is small.** A typical asic block lands in the
single digits — three to six matchers covering plain Conv,
Conv+Activation, Conv+Bias+Activation, and so on. If the tool emitted
twenty matchers, the engine's support is fragmented across
`(io_dtype, layout)` combinations and is worth a closer look (and
likely a bug). The condensation pass groups by *identical* safe
rectangle; fragmentation means the rectangles don't line up.

**2. The op_chains list matches your mental model.** If you see an
unexpected op_chain (a node type you don't think the engine supports),
the engine's `get_ranked_engine_ids` probably says "yes" when it should
say "no". Fix the engine, regenerate.

**3. The stderr "U tuples" list matches your mental model.** These are
observed combinations the engine refused. If something you *expected*
to be supported shows up here, the engine has a gap — file an issue
before committing the diff that bakes in the gap.

## CI integration

Once a sidecar has at least one `[[supported]]` block for an asic, flip
the integration-test invocation on that asic to use
`--enforce-support-claims`:

```bash
./hipdnn_integration_tests \
    --te MIOPEN_ENGINE \
    --ta path/to/libmiopen_plugin.so \
    --tc dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --enforce-support-claims
```

The verifier evaluates after `RUN_ALL_TESTS` and exits non-zero on
Rule A/B/C failures. Output is grouped by rule and also written to
`support_claim_failures.txt` for the CI artifact upload.

A sidecar with no block for the current arch is treated as "not
enforced" — that's the safe state during staged rollout. You can ship
`--enforce-support-claims` to every CI job before populating every asic,
and only the populated asics will actually be checked.

## Adding a brand-new engine

The same workflow plus two prerequisites:

1. Set `[meta].engine` in the main `<EngineName>.toml`. The loader
   refuses to load a sidecar whose `[meta].engine` doesn't match the
   binary's `--test-engine` flag.
2. Create an empty sidecar with just `[meta]` (no `[[supported]]`
   blocks). The tool needs a file to atomic-rename onto. Or, the tool
   creates one from scratch if the file is absent — either works.

## When to regenerate vs. hand-edit

- **Regenerate** when the engine gained or lost real capability —
  shipping a new solver, dropping a deprecated dtype, rolling back a
  feature. The tool produces a clean diff that matches reality.
- **Hand-edit** to add a `[[test_skips]]` (in the main file, not the
  sidecar) for a broken-but-supported combination. Skips and matchers
  compose: a skipped test is excluded from claim evaluation, so the
  matcher can keep claiming the tuple cleanly.
- **Don't hand-edit the sidecar** to add a missing matcher — the tool
  will regenerate over your edit on the next run, and you might miss
  the chance to inspect what got dropped. Regen instead.
