# Dynamic StreamK partials validation

The `sk_dynamic_partials*.yaml` configs exercise both `DynamicWorkQueue` and
the dynamic sub-path of `Hybrid`, with work stealing off and on. They cover
batched tiles, M/N edges, uneven K chunks, zero alpha, and zero workspace.
The largest split case needs exactly 2 MiB of partials workspace. All outputs
are checked, bounds checking is enabled, and the synchronizer is checked for
leftover queue counters and ready flags. `ValidateAfterEnqueues` checks the
final output and synchronizer state after each group of four consecutive
launches, without inserting readbacks between those launches. Two benchmark
passes exercise reuse across problems and solutions.

Run from `projects/hipblaslt/tensilelite` in the build venv. Use a full
eight-XCD gfx942 or gfx950 device: the dynamic queue requires the runtime XCD
count to match the compiled queue count.

```bash
python -m pytest tensilelite/Tests/common/test_config.py \
  --gpu-targets gfx942 -k sk_dynamic_partials \
  --prebuilt-client "$PWD/build_tmp/tensilelite/client/tensilelite-client"
```

Repeat the normal-alpha split config with `--global-parameters
MaxWorkspaceSize=2097151`. Its 16-tile case must fall back to whole tiles when
the workspace is one byte short, while the smaller cases can still split.
Use `TENSILE_DB=0x200000` for a diagnostic run and check `skTiles`, `skSplit`,
the workspace size, and the fallback fields in the launch summary so a
passing GEMM cannot silently validate the wrong path.

On a machine without a GPU, build portable artifacts instead:

```bash
python -m pytest tensilelite/Tests/common/test_config_build.py \
  --build-only --gpu-targets gfx942 --artifact-dir /tmp/dynamic-partials-gfx942 \
  -k sk_dynamic_partials \
  --prebuilt-client "$PWD/build_tmp/tensilelite/client/tensilelite-client"
```

Run those artifacts on the matching GPU with `test_config_run.py --use-cache`
and the same artifact directory, target, selection, and prebuilt-client
arguments. Repeat for gfx950. Building an artifact does not validate device
execution or performance.

For performance, compare separate builds of the partial-tile feature commit
and its parent (which already includes the client validation option) on
the same GPU, using the same configs and inputs. Copy the configs into the
parent checkout. Include the normal-alpha split config and the larger
whole-tile cases in `sk_dynamic.yaml`. Override
`NumBenchmarks=10 SyncsPerBenchmark=5 EnqueuesPerSync=20 KernelTime=True
CheckSynchronizer=False ValidateAfterEnqueues=False BoundsCheck=0` for timing runs after correctness
passes, and leave launch-summary logging off. Compare per-problem,
per-kernel median times; retain both improvements and regressions, including
the workspace-starved cases. Record the GPU partition, clocks, ROCm version,
and both commit IDs with the results. An environment override that disables
splitting on this branch is useful diagnostically but does not replace the
parent comparison, because this branch also changes grid selection.
