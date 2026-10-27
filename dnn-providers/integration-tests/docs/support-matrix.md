# Support Matrix and Feature Coverage

A bundle can carry a **support claim**: a sidecar asserting that a named engine
supports that exact graph on a given `(arch, platform)`. Claims live beside the
graph — `{Name}.support.json` for a single-graph bundle, one `support.json`
keyed per `cases[].id` for a sweep — and the integration tests enforce them, so
a claim that stops holding fails a test. The spec is
[RFC 0015](../../../projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md).

Two scripts read them, and neither needs a GPU, a plugin, or a DVC pull — the
`.json` and `.support.json` files are all in git:

```bash
# Validate every sidecar: schema, enforcement level, orphans, unknown case ids.
# Also runs as the verify-support-claims pre-commit hook.
python3 dnn-providers/integration-tests/scripts/verify_support_claims.py

# Render the human-readable matrix of who supports what, where. With no
# arguments this writes dnn-providers/integration-tests/SUPPORT_MATRIX.md.
python3 dnn-providers/integration-tests/scripts/render_support_matrix.py
```

## Rendering the support matrix

`SUPPORT_MATRIX.md` is **generated, not committed** — it is large and grows
with every bundle and every gfx target, which would mean a large diff on every
sidecar edit and a file near the size at which GitHub stops rendering markdown.
Run the command above to produce your own copy; it is git-ignored, so it will
not show up in `git status`. The matrix is a pure view of the sidecars, so two
people rendering the same checkout get byte-identical files. Every
`hipdnn-superbuild-ci` Linux run also renders it, with its `--overview-only`
variant and the committed overview, and uploads all three as the
`hipdnn-support-matrix` artifact, so you can download it from a PR's checks
instead of rendering it.

The renderer is three files in `scripts/`: `support_matrix_units.py` collects
one unit per graph from the bundle tree, `support_matrix_markdown.py` renders
those units, and `render_support_matrix.py` is the command line over both.
They read bundle JSON through the same `bundle_discovery.py` loader as
`verify_support_claims.py`, so the two tools agree on which files are readable.

`SUPPORT_MATRIX_OVERVIEW.md` **is committed** — it is the by-option view
from `support_features/feature_matrix.py` (see
[Feature coverage per op](#feature-coverage-per-op)), small enough to render on
GitHub. CI fails if it is stale; regenerate it in the same commit as the
sidecar change:

```bash
python3 dnn-providers/integration-tests/scripts/support_features/feature_matrix.py
```

`harvest_support_claims.py --write` regenerates it for you. The graph-count
`--overview-only` render of `render_support_matrix.py` is git-ignored, like the
full matrix.

Bundle data is checked in CI, not at commit time. The top-level exclude in
`.pre-commit-config.yaml` hides `integration-test-bundles/` from every hook,
so the real tree is checked by `test_render_support_matrix.py` instead. It
runs as the `hipdnn_support_matrix_renderer_python_tests` ctest in
`hipdnn-superbuild-ci` on every change under `dnn-providers/`, and the full
matrix must render. `test_support_features.py`, the
`hipdnn_support_features_python_tests` ctest, does the same for the committed
overview, which must match a fresh render. To check locally before pushing,
run those ctests, or run `feature_matrix.py --check`.

Three pre-commit hooks give quick feedback when a commit changes the scripts
themselves (or, for the last one, the overview file):

| Hook | Fails when |
|------|------------|
| `verify-support-claims` | a sidecar is malformed, orphaned, or names an unknown case id |
| `render-support-matrix` | the full matrix cannot be rendered — a malformed or orphaned sidecar, a claim on an unknown case, two tier copies of one bundle holding different graphs, or a bundle not under `<tier>/<Op>/`. Output is discarded |
| `check-support-matrix-overview` | the committed `SUPPORT_MATRIX_OVERVIEW.md` differs from a fresh `feature_matrix.py` render |

Moving a bundle between tiers does not change the overview and needs no
regenerate: a case is identified by
its op and case id (or its path below the tier), not by its tier, and a case
present in several tiers is counted once. The tier shows up only in the full
matrix's provenance comments.

Useful flags:

| Flag | Effect |
|------|--------|
| `--output PATH` | write somewhere other than the default path; `-` means stdout |
| `--bundles-dir PATH` | render a tree other than `integration-test-bundles/` |
| `--max-case-ids N` | inline up to N sweep case ids per bundle in the provenance comments (0 = none, the default) |
| `--overview-only` | emit only the per-target overview tables — no collapsible details, no per-variant drilldowns; writes to the git-ignored `SUPPORT_MATRIX_GRAPH_OVERVIEW.md` by default |
| `--check` | compare against an existing file and exit 1 if they differ; for keeping a local copy in sync |

Nothing is filtered: every `(arch, platform)` target gets its own section, and
within it every engine is a column and every op family is a row, whether or not
anything claims them. A family that vanished when unclaimed would be
indistinguishable from one nobody has looked at.

Reading it: each overview row is one op family. In the full
`SUPPORT_MATRIX.md`, disclosure triangles expand a row into per-variant and then
per-(variant, dtype) rows; the `--overview-only` render has the family rows only. A
`—` means the graph is **unclaimed**, which is not the same as *known
unsupported* — the engine may reject it, or nobody has written the claim yet.
The Reading guide at the end of the full document spells out the rest,
including the `[multi_batch]` / `[grouped]` / `[stride]` / `[dilation]` /
`[padding]` variant tags.

## Feature coverage per op

The matrix counts graphs; it cannot show that an op has a feature no bundle
touches. `scripts/support_features/` counts features instead: for each op, the
share of its API features that at least one bundle graph exercises (*tested*),
and the share that at least one claimed graph exercises on each arch. Features
count singly, not in combination.

```bash
python3 dnn-providers/integration-tests/scripts/support_features/coverage.py
```

`feature_matrix.py` lays the same shares out like the support matrix: op
groups by GPU and OS, then one table per engine, where a cell is the share of
the op's features claimed there rather than the share of its graphs. The
rendered document calls features *options*. Its *Options per op* section lists
every option counted, split into those a graph exercises and those none does.
Its output is the committed `SUPPORT_MATRIX_OVERVIEW.md`; `--output -` prints
it instead, and `--check` compares without writing.

The denominator is the op's API, not the bundles: each op module
(`support_features/sdpa.py`, ...) catalogs the features of one node type in
the flatbuffers schema, `support_features/registry.py` lists every node type
in the schema, and `test_support_features.py` fails if a schema field is
neither used by a feature nor listed with a reason it is not one. A graph that
uses a value its catalog does not name — a new dtype, an uncatalogued head dim
— is an error, not a silent omission, and both scripts render nothing until it
is fixed. The tests run as the `hipdnn_support_features_python_tests` ctest.

## See Also

- [`README.md`](README.md) — index of the integration-test documentation.
- [Support Claim Enforcement](support-claim-enforcement.md) — how claims are checked at run time.
- [File Formats](file-formats.md#support-claim-sidecars--namesupportjson-and-supportjson) — the sidecar schemas.
