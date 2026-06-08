# Convolution Heuristic - Routing Investigation and Next Steps

## Current Status

`conv_heuristic.cpp` classifies convolution forward graphs and logs the regime, tensor dims,
stride, padding, data type, and candidate engine IDs passed through the heuristic ABI.

The heuristic is detection-only:

```cpp
*out_applied = 0;
```

This is intentional until production MIOpen convolution engine IDs are visible in the
candidate list.

## Engine ID Probe Results

Two standalone verifier probes have run on gfx942:

| Probe | Shape | Regime | Candidate count | Candidate IDs |
|-------|-------|--------|-----------------|---------------|
| `conv_engine_id_probe.log` | N=4 C=4 H=4 W=4, K=4 R=1 S=1, stride=1 | `GEMM_CONV` | 1 | `-3` |
| `conv_realistic_probe.log` | N=32 C=64 H=56 W=56, K=64 R=3 S=3, stride=1 | `WINOGRAD` | 1 | `-3` |

`engine_id=-3` is not a production MIOpen engine. It maps to the test engine plugin:

```text
projects/hipdnn/tests/test_plugins/TestPluginEngineIdMap.hpp:
HIPDNN_MAP_TO_ID(GoodDefaultPlugin, -3);
```

The current verifier path uses `libtest_good_default_plugin.so`, so the only candidate ID
it can expose is the test plugin's fixed ID. This means the probes validate that
`conv_heuristic` sees and classifies convolution graphs, but they do not reveal production
Winograd, GEMM, or direct MIOpen engine IDs.

## Conclusion

Routing is not actionable from the current verifier results.

The heuristic can classify regimes correctly, but the runtime candidate list in the current
test path contains a single test engine candidate. There is no choice to make, and the ID is
not a production convolution algorithm ID.

Before implementing routing, use a verifier path that loads the production MIOpen engine
plugin and produces real convolution candidates. The key question is whether production
conv graphs ever pass multiple candidate IDs through `PolicySetEngineIds`.

## Required Next Probe

Run the same conv heuristic against the production MIOpen engine plugin instead of
`libtest_good_default_plugin.so`.

Expected evidence needed before routing:

```text
[CONV_HEURISTIC] PolicySetEngineIds count=N
[CONV_HEURISTIC]   candidate engine_id[0] = <production-id>
[CONV_HEURISTIC]   candidate engine_id[1] = <production-id>
[CONV_HEURISTIC] regime=WINOGRAD ...
```

If `count > 1`, map each ID back to its engine name/algorithm. If `count == 1` even with
the production plugin, convolution routing is not viable yet and the performance issue is
not a heuristic selection problem.

## Intended Routing Plan Once IDs Are Known

Use the same pattern as `Q3_ROUTING_SPEC.md`:

- `WINOGRAD_ENGINE_ID`: activate for `regime=WINOGRAD` (3x3 stride-1, not depthwise)
- `GEMM_ENGINE_ID`: activate for `regime=GEMM_CONV` (1x1)
- Leave `DIRECT`, `DEPTHWISE`, and `GENERAL_CONV` as fallthrough until benchmarks prove a
  reorder improves them

In `PolicyFinalize`:

```cpp
if(regime == WINOGRAD && candidate list contains WINOGRAD_ENGINE_ID)
{
    routed_engine_id = WINOGRAD_ENGINE_ID;
    *out_applied = 1;
}
else if(regime == GEMM_CONV && candidate list contains GEMM_ENGINE_ID)
{
    routed_engine_id = GEMM_ENGINE_ID;
    *out_applied = 1;
}
else
{
    *out_applied = 0;
}
```

In `PolicyGetSortedEngineIds`, rotate the chosen engine ID to the front of the candidate
list and return the reordered list.

## Benchmark Template

After production IDs are known and routing is active:

```bash
cat > ~/slurm_jobs/bench_conv_routed_gfx942.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=bench_conv_routed
#SBATCH --partition=defq
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --constraint=MARKHAM
#SBATCH --time=01:00:00
#SBATCH --output=/home/AMD/ysoliman/slurm_jobs/%x_%j.out
#SBATCH --error=/home/AMD/ysoliman/slurm_jobs/%x_%j.err

module add ubuntu-24
module load rocm/7.2.70200

# TODO: run the production conv benchmark for:
# - 1x1 GEMM_CONV shapes
# - 3x3 stride-1 WINOGRAD shapes
EOF
sbatch ~/slurm_jobs/bench_conv_routed_gfx942.sh
```

## Commit Message Template

```text
feat: conv heuristic routing for GEMM and Winograd regimes

- PolicyFinalize returns out_applied=1 for GEMM_CONV and WINOGRAD when their engine IDs are present
- PolicyGetSortedEngineIds rotates the selected production engine ID to the front
- DIRECT, DEPTHWISE, and GENERAL_CONV remain fallthrough
- Benchmarks: 1x1 <before> -> <after>, 3x3 stride-1 <before> -> <after>

Experimental branch only. No PR.
```
