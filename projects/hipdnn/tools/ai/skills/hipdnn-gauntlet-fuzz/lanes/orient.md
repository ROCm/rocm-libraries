# Lane: orient (`ORIENT`)

**Trigger:** every round. **Agent:** one `task` (it writes the brief). **Output:** `<run-dir>/engines/<engine-slug>/BRIEF.md` (not a lane report).

Write, or update, the engine brief that every later lane reads instead of rediscovering the engine.

- **write** mode: no brief exists. Write all nine sections and the code cards.
- **update** mode: a brief exists with `Describes: <old sha>`. Read `git diff <old>..<new> --stat` over the brief's cited paths, and
  rewrite only the sections and cards whose sources changed. Leave a `Changed in <new sha12>:` line under each rewritten heading.

## Format

```markdown
# <engine id>: <op> on <arches>
Describes: <full sha of the PR build>
```

Then nine sections, `path:line` for every claim, about 1,500 words in all:

1. **Semantics.** What the op computes, every attribute, defaults. Sources: the semantics register
   (`projects/hipdnn/docs/SdpaSemanticsRegister.md` or the op's equivalent), `api-cudnn-parity.md`, `KNOWN_DIVERGENCES.md`.
2. **Frontend validation.** Every check the frontend makes on this node (`projects/hipdnn/frontend/include/.../node/<Op>*`) and
   every attribute it does not check.
3. **Backend path.** Frontend to serialization (`flatbuffers_sdk`) to backend (`projects/hipdnn/backend/src`) to the plugin: optional
   vs default fields, int widths, pass-by-value scalars, dtype mapping, engine selection.
4. **References.** The reference executors (CPU `test_sdk`, GPU `integration-tests/src/harness/gpu-graph-executor`) for the op, the
   features each serves, and the features with no oracle. Whether an independent truth exists (SDPA backward: the AOTriton grader).
5. **Conditioning.** Inputs that make a correct answer unstable to the rounding of its own inputs (scores into `exp()` against a given
   LSE, saved statistics, `1/(1-p)`, huge scales) and hardware number-format limits (subnormal flush on some arches). A bound a
   space can encode.
6. **Existing coverage.** The engine's spaces (`gauntlet/spaces/<Op>.<Engine>*.space.json`), axes and constraints, the generator
   (`gauntlet/generators/`), and the probe families with their `expect` and rule ids.
7. **Engine inventory.** Matcher or applicability files, the coverage doc and its rule ids, kernel sources and generators,
   descriptor lists or catalogs, tuning tables, knobs, and how the engine logs its kernel (or that it does not).
8. **Already known.** From the engine's docs, earlier rounds under `<run-dir>`, reviews and the engine ledger: findings, hazards,
   documented divergences, harness limits. Cite ledger IDs.
9. **Harness reach.** What the harness can build, fill and grade for this op (for SDPA backward
   `dnn-providers/integration-tests/src/harness/input-init/SdpaBackwardInputs.cpp`: padding, ragged, explicit masks, ALiBi, bias,
   Philox), and the fill limits (`HIPDNN_TEST_HOST_INPUT_FILL=1` fills inputs on the host).

## Code cards

After the nine sections, a section `## Code cards` that lets every later lane skip reading whole files. One card per file or function
the lanes keep needing: the engine's applicability checks, the plan's geometry and stride derivation, the kernel's indexing and launch,
the frontend validation, the reference's computation. Each card is `### <path>:<first>-<last>` followed by the code (or a faithful
excerpt with elisions marked `...`), at most about 80 lines per card and 600 lines in all. Quote real lines only; a lane that quotes a
card cites its `path:line`. Put also the knowledge-base rows that concern this op here, as short facts with their IDs, so lanes do not
open knowledge-base articles. The cards are the part of the brief that goes stale first: update them whenever the cited lines move.
