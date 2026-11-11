# Harness core: method shared by the StinkyTofu and hipBLASLt harnesses

Read this when you run an A/B, add or retire a harness component, adapt the harness to a new model
generation, or write run results. The project files (`HARNESS.md`) hold the scope, tiers, rules, and
the components themselves. This file holds the method they share, so it exists once.

## Components

Each component is one row in the project's `HARNESS.md` table. `assumes` is the limitation of the model
it covers. A row with no `evidence` is a candidate for removal.

**Adding a component:** add a row with `assumes` and `evidence`, set it to `on-trial`, run the same
real task with and without it, and keep it only if the outcome is better.

**New model generation:** run a real task on the full harness, then disable one component at a time
and rerun the same task. Keep a component only if removing it makes the outcome worse. Record the model
and the result in the last column of the row. Also try adding capabilities the new model may now
handle. Do not remove several components at once.

## What "identical" means (tier 1)

"Asm identical" is a content comparison, never a hash alone.

```text
- compare file by file with diff or cmp. Every file must match. A combined hash does not count.
- report how many files were compared and how many differ. Zero files compared is a failure.
- generate each side in its own working directory (`Tensile <yaml> <working directory>`, with
  KeepBuildTmp on so the asm is kept). The folder names inside look the same on both sides, so the
  working directory is the only way to tell them apart. Print both paths and check they differ.
- build the baseline and the candidate separately. Rebuild the candidate after the last edit (see
  the "fresh binding" note in the StinkyTofu HARNESS.md).
- if anything is normalized before comparing (paths, timestamps), list exactly what, and show a
  diff proving nothing else was removed.
- on any difference, show the diff. Do not call it "equivalent" unless the developer agrees.
```

## Tier 1 config set

The inputs are the perf YAMLs under `projects/hipblaslt/tensilelite/tensilelite/Utilities/tensile_generator/perf_yamls/<target>/`.

```text
- the target is a directory name there (a machine variant, not just an arch). Ask the developer
  which target the change is for. Do not guess. If it has no directory, ask.
- run every YAML under that target's maf/, mab/, and medium/ directories, including the finetune,
  k128k, and other variants. Ignore tuning/ subdirectories: perf YAMLs are for perf tests only.
  No sampling: the full set is already the minimum.
- one tier 1 result per YAML, not one for the whole set
```

## State files

| File | Format | Written by |
| --- | --- | --- |
| `HARNESS.md`, `PROFILING.md` | Markdown | humans |
| `results.json` (gitignored) | JSON, one entry per config; agents change only result fields | agents |
| `progress.md` (gitignored) | Markdown, append-only handoff log | agents |
| run logs (gitignored) | text, one file per run | scripts |

`results.json` and `progress.md` can contain confidential or private information. Never commit them
to this repo, and never paste them into a public PR, issue, or comment.

Print little to the terminal and write full detail to the log file. Format an error as one line,
`ERROR: <reason>`, so it can be grepped.
