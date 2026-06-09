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

Standalone verifier probes have run on gfx942:

| Probe | Shape | Regime | Candidate count | Candidate IDs |
|-------|-------|--------|-----------------|---------------|
| `conv_engine_id_probe.log` | N=4 C=4 H=4 W=4, K=4 R=1 S=1, stride=1 | `GEMM_CONV` | 1 | `-3` |
| `conv_realistic_probe.log` | N=32 C=64 H=56 W=56, K=64 R=3 S=3, stride=1 | `WINOGRAD` | 1 | `-3` |
| `conv_miopen712_probe.log` | N=32 C=64 H=56 W=56, K=64 R=3 S=3, stride=1 | `WINOGRAD` | 2 | `-6748551569128940061`, `1563989756945604898` |

`engine_id=-3` is not a production MIOpen engine. It maps to the test engine plugin:

```text
projects/hipdnn/tests/test_plugins/TestPluginEngineIdMap.hpp:
HIPDNN_MAP_TO_ID(GoodDefaultPlugin, -3);
```

The current verifier path uses `libtest_good_default_plugin.so`, so the only candidate ID
it can expose is the test plugin's fixed ID. This means the probes validate that
`conv_heuristic` sees and classifies convolution graphs, but they do not reveal production
Winograd, GEMM, or direct MIOpen engine IDs.

The production MIOpen provider path was validated by `verify_conv_miopen`, which loads the
prebuilt ROCm 7.12 provider plugin:

```text
/cluster/apps/ubuntu-24/rocm/rocm-7.12.0.60610/lib/hipdnn_plugins/engines/libmiopen_plugin.so
```

That path produced real hipDNN engine IDs:

```text
[CONV_HEURISTIC] PolicySetEngineIds count=2
[CONV_HEURISTIC]   candidate engine_id[0] = -6748551569128940061
[CONV_HEURISTIC]   candidate engine_id[1] = 1563989756945604898
[CONV_HEURISTIC] regime=WINOGRAD N=32 C=64 H=56 W=56 K=64 filterC=64 R=3 S=3 stride=1x1 pad=1x1 dtype=FLOAT
```

These IDs map to the registered engine names in `EngineNames.hpp`:

```text
MIOPEN_ENGINE_DETERMINISTIC = -6748551569128940061
MIOPEN_ENGINE               =  1563989756945604898
```

## Conclusion

Routing is now actionable at the coarse MIOpen-engine level.

The test-plugin verifier path is non-production evidence only, but the ROCm 7.12 MIOpen
provider path proves hipDNN can present multiple production engine candidates to the
heuristic. For the realistic 3x3 stride-1 WINOGRAD-class shape, the candidate order was:

```text
MIOPEN_ENGINE_DETERMINISTIC
MIOPEN_ENGINE
```

This still does not expose separate internal MIOpen solver IDs for Winograd, GEMM, or
direct kernels. The heuristic can route between the two hipDNN-level MIOpen engines, but it
cannot directly select an internal MIOpen solver unless the provider exposes more granular
engine IDs in the future.

## Required Next Probe

Run additional production MIOpen provider probes for 1x1 and large-filter shapes using
`verify_conv_miopen` variants. Confirm whether candidate order changes by regime and
whether `MIOPEN_ENGINE_DETERMINISTIC` should be preferred only for WINOGRAD-like shapes or
for all convolution shapes.

The production provider target is `miopen_plugin` in:

```text
dnn-providers/miopen-provider/CMakeLists.txt
```

Its expected build output is:

```text
<provider-build>/lib/hipdnn_plugins/engines/libmiopen_plugin.so
```

As of the Session 12 investigation, no production MIOpen engine plugin `.so` exists in
the current hipDNN build tree. The configured hipDNN build only contains the manually
installed test engine plugin:

```text
projects/hipdnn/build/lib/hipdnn_plugins/engines/libhipdnn_test_plugin1.so
```

Standalone provider build attempts were made against ROCm 7.1.1 and ROCm 7.2.70200. Both
configured after pointing CMake at the hipDNN SDK package configs, but both failed to build
`miopen_plugin` because the installed MIOpen headers do not expose batchnorm APIs expected
by this provider source:

```text
miopenBatchNormForwardInferenceActivationInvVariance
miopenBatchNormalizationForwardInferenceInvVariance
```

There is no provider CMake option to build only convolution plans or exclude the batchnorm
plan files from `miopen_plugin_impl`, so the production plugin cannot be produced from this
checkout without either:

- a matching newer MIOpen SDK/header package that provides those APIs, or
- a provider source/build adjustment that excludes or gates the newer batchnorm plan code.

Evidence format:

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

- `MIOPEN_ENGINE_DETERMINISTIC`: candidate observed first for `regime=WINOGRAD`
- `MIOPEN_ENGINE`: general MIOpen engine candidate observed second
- Leave actual routing off until before/after benchmarks prove that changing the
  hipDNN-level engine order improves a specific regime

In `PolicyFinalize`:

```cpp
if(regime == WINOGRAD && candidate list contains MIOPEN_ENGINE_DETERMINISTIC)
{
    routed_engine_id = MIOPEN_ENGINE_DETERMINISTIC;
    *out_applied = 1;
}
else if(regime == GEMM_CONV && candidate list contains MIOPEN_ENGINE)
{
    routed_engine_id = MIOPEN_ENGINE;
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
