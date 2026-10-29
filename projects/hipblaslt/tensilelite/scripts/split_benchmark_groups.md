# Splitting benchmark YAMLs

`split_benchmark_groups.py` plans and renders partitions of TensileLite benchmark configurations. It accepts directories anywhere inside a Git repository. Run it with Python 3.10 or newer, PyYAML, and Git available; it does not import TensileLite or require a GPU. Commands below run from the repository root. Use `--repo` before the subcommand to select another repository.

The indivisible unit is a problem type plus one complete benchmark group under `BenchmarkProblems`. Each unit retains its parameter sweep and all problem sizes. The tool copies the other top-level settings into every output, keeps comments with their groups, and defines or expands YAML anchors when their definitions would otherwise be missing. It checks parsed group multiplicities and settings before writing.

## Reproduce the checked-in split

The checked-in plan records a reviewed selection of gfx1250 fixtures. Its source commit contains the unsplit inputs. From the PR checkout, verify the generated files with:

```sh
python projects/hipblaslt/tensilelite/scripts/split_benchmark_groups.py render \
  projects/hipblaslt/tensilelite/scripts/benchmark_group_splits.json \
  --verify --strict-bytes
```

Generate the same files from an unsplit checkout containing the tool and plan with:

```sh
python projects/hipblaslt/tensilelite/scripts/split_benchmark_groups.py render \
  projects/hipblaslt/tensilelite/scripts/benchmark_group_splits.json \
  --write --verify --strict-bytes
```

An unchanged plan can be replayed in an already split checkout. Changed files, including locally deleted files, must be clean in Git before the tool will replace them. It never deletes files.

Verification checks every planned output. It also rejects YAML files in the recorded directories that belong to neither the unsplit source commit nor the output plan. This detects obsolete partitions after a plan is reduced or renamed. It intentionally also rejects unrelated new YAMLs absent from the recorded source commit; use a source commit containing those originals or a clean unsplit checkout. It does not validate the contents of unsplit files outside the plan.

## Propose splits in another directory

First commit the unsplit inputs. For example, from an unsplit checkout:

```sh
python projects/hipblaslt/tensilelite/scripts/split_benchmark_groups.py scan \
  projects/hipblaslt/tensilelite/tensilelite/Tests/common/streamk/gfx1250 \
  --source-ref HEAD --timings timings.json --plan-out proposed-splits.json
```

`timings.json` maps repository-relative YAML paths to positive elapsed seconds measured with a consistent execution setup. Multiple directories may follow `scan`. Named Git refs are resolved to commit IDs in the saved plan. Omitting `--source-ref` permits inspection of local inputs, but the resulting plan cannot be written in place until it names saved unsplit inputs.

The planner repeatedly bisects the estimated longest range of complete groups, choosing a boundary that balances their weights. Each weight combines the number of fork-parameter combinations, the operation count of supported Exact GEMM sizes, and configuration text size. These are relative estimates, especially for sparse operations, unsupported size forms and rejected kernel candidates.

For timed files, splitting stops at the target duration, the maximum number of parts, or the added-startup budget. Defaults are 60 seconds per part, at most 16 parts, and 3.5 seconds of startup per extra file within 5% of the original elapsed time. Those defaults are assumptions to calibrate on the intended runner, not performance guarantees. Untimed files stay whole unless `--untimed-max-parts` is explicitly increased; their proposals still need timing.

Review the JSON report and plan before rendering. Whole-file expected failures, disabled configurations, sections following `BenchmarkProblems`, and StreamK schedule assertions are excluded from automatic proposals. The checked-in plan is a reviewed selection, not the output of running this example command with default settings.

## Worked example: three input YAMLs

Suppose a committed `cases/` directory contains these three configurations, with no whole-file assertions or expected-failure marks. Each weight represents the estimated work of one complete group, in source order. These small weights and whole-file times are hypothetical, chosen to make the arithmetic visible; they are not benchmark results.

| Source | Group weights | Whole-file time in `timings.json` |
| --- | --- | --- |
| `cases/a.yaml` | `[4, 1, 1, 2]` | 240 seconds |
| `cases/b.yaml` | `[1, 1]` | 65 seconds |
| `cases/c.yaml` | `[1, 1, 1, 1]` | No timing entry |

Run the scan with the default budgets:

```sh
python projects/hipblaslt/tensilelite/scripts/split_benchmark_groups.py scan cases \
  --source-ref HEAD --timings timings.json --plan-out proposed-splits.json
```

The algorithm considers each source independently. For `a.yaml`, the total weight is 8, so one weight unit represents `240 / 8 = 30` estimated seconds. Its extra-startup budget is `240 * 0.05 = 12` seconds.

1. Start with all four groups in one part: `[4, 1, 1, 2]`, estimated at 240 seconds.
2. Split after the first group, giving `[4] | [1, 1, 2]`. Each side has weight 4 (120 seconds). This balances work, not group counts. One extra file costs an estimated 3.5 seconds, within the 12-second budget.
3. The first part is indivisible. Split the second part into `[1, 1] | [2]`, each with weight 2 (60 seconds). There are now three parts: `[4] | [1, 1] | [2]`, with 7 seconds of added startup.
4. Stop: the only part that still contains multiple groups is already at the 60-second target. The first part remains at 120 seconds because the algorithm never divides a complete group. The target is therefore not a promised upper bound on file duration.

For `b.yaml`, splitting could bring both parts below 60 seconds, but its startup budget is only `65 * 0.05 = 3.25` seconds. One extra file would cost 3.5 seconds, so it stays whole. `c.yaml` also stays whole: without a timing entry, the default `untimed_max_parts=1` prohibits a split regardless of its weights.

The saved plan contains only `a.yaml`, with `sizes: [1, 2, 1]`: these are group counts, not the weights `[4, 2, 2]`. Scan records the source commit and settings digest but does not modify any YAML. After reviewing the plan, run:

```sh
python projects/hipblaslt/tensilelite/scripts/split_benchmark_groups.py render \
  proposed-splits.json --write --verify --strict-bytes
```

Assuming no filename collisions, the renderer puts group 1 in `a.yaml`, groups 2–3 in `a_part2.yaml`, and group 4 in `a_part3.yaml`. Each output keeps the applicable problem types, global settings, test settings and complete group contents. Together with unchanged `b.yaml` and `c.yaml`, three inputs become five files. The estimated serial cost for `a.yaml` grows from 240 to 247 seconds; whether the smaller scheduling items reduce parallel wall time must be measured.

## Plan fields

| Field | Meaning |
| --- | --- |
| `version` | Plan format, currently `1`. |
| `source_ref` | Commit containing the unsplit inputs. Required for writes. |
| `directories` | Original scan scope, relative to the repository. Retain it when removing split records so obsolete outputs remain detectable. |
| `splits[].path` | Repository-relative source YAML inside a recorded directory. |
| `splits[].sizes` | Positive counts of complete groups, in source order, whose sum equals the source's group count. |
| `splits[].settings_sha256` | Digest of parsed top-level settings other than `BenchmarkProblems`. A mismatch requires review. |

The first part normally keeps the source filename and later parts receive `_partN.yaml` names, avoiding names already present in the source commit. An optional `names` list provides explicit YAML basenames and must include the original name. Alternatively, `slots` permutes the automatically assigned output names. Names must preserve path-derived architecture and expected-failure policy.

`expand_all_anchors` expands all aliases instead of only those missing a definition. `allow_streamk_schedules` permits rendering a schedule-bearing configuration after each output's runtime assertions have been reviewed. Neither option relaxes the check that every parsed group and non-group setting is preserved.

## Change the plan or regenerate after a rebase

Reconstruct the generated commit from a clean checkout of the new unsplit base. Carry over the tool commit and the plan, but do not apply the old generated YAML commit. This avoids obsolete outputs and YAML merge conflicts.

Update the plan's `source_ref` to the new base commit. Review upstream changes to the selected sources: update group counts if groups were added or removed, and update settings digests only after inspecting changed settings. A fresh `scan` can supply proposed counts and digests, but its selection does not replace a reviewed plan automatically. Keep the recorded directory scope when dropping sources. Then run the write and verification commands above and inspect the generated diff.

The gfx1250 split also has two integration edits outside the renderer: `test_zgemm_gfx1250_codegen.py` must exercise both ZGEMM outputs, and the comment in `streamk/gfx1250/core/sk_halfplr_f8gemm_tdm.yaml` names the multicast fixtures. Carry these over and review any new filename references when reconstructing that commit. Confirm reference coverage, the strict input-YAML corpus and the splitter tests before publishing.
