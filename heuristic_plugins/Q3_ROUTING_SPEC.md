# SDPA Heuristic - Q3 Engine Routing Integration Spec

## Background

The SDPA regime classifier (`sdpa_heuristic.cpp`) is deployed and confirmed on gfx942.
The gfx950 validation path uses the same CPU-side heuristic plugin and is tracked by the
Slurm verifier job recorded in `heuristic_build_log.md`.
It currently returns `out_applied=0` for all shapes - detection and logging only.

Three regimes have critical performance gaps that routing can address once the CK SDPA
engine ships:

| Regime       | Condition       | Current perf (gfx942) | Root cause     |
|--------------|-----------------|------------------------|----------------|
| GQA_DECODE   | Sq == 1         | 0.09-0.17 TFLOPS       | Kernel gap     |
| GQA_PREFILL  | Hq > Hkv        | 3-6 TFLOPS             | Kernel gap     |
| D256_PREFILL | D == 256        | ~29 TFLOPS             | Tile mismatch  |
| MHA_PREFILL  | else            | 55-70 TFLOPS           | OK - no action |

## Step 1: Discover the CK SDPA Engine ID

When the CK SDPA engine lands on the branch, find its registered engine ID:

```bash
# Search engine registration in the hipDNN backend
rg -n "ASM_SDPA|CK_SDPA|ck_sdpa|asm_sdpa" \
  ~/rocm-libraries-heuristic-prototype/projects/hipdnn/backend/src/

# Or look at engine ID declarations directly
rg -n "EngineId|engine_id" \
  ~/rocm-libraries-heuristic-prototype/projects/hipdnn
```

Record the integer or enum value for the CK SDPA engine. Call it `CK_SDPA_ENGINE_ID` below.

## Step 2: Changes to `sdpa_heuristic.cpp`

The routing change is isolated to `PolicyFinalize` and `PolicyGetSortedEngineIds`.
Everything else (`SetEngineIds`, `SetSerializedGraph`, regime parsing) stays identical.

### In `PolicyFinalize`

Replace the current fallthrough logic:

```cpp
// CURRENT (detection only):
out_applied = 0;
```

with active routing:

```cpp
if(Sq == 1 || Hq > Hkv)
{
    // GQA_DECODE or GQA_PREFILL: route to CK SDPA engine.
    g_routed_engine_id = CK_SDPA_ENGINE_ID;
    out_applied = 1;
    fprintf(stderr,
            "[SDPA_HEURISTIC] routing to CK_SDPA engine_id=%d\n",
            CK_SDPA_ENGINE_ID);
}
else
{
    out_applied = 0;
}
```

Keep `out_applied=0` for `MHA_PREFILL` (already good) and `D256_PREFILL` (tile mismatch is
a kernel property, not a selection gap - routing will not help until the kernel is fixed).

Add a file-scope variable to pass the chosen ID to `PolicyGetSortedEngineIds`:

```cpp
static int32_t g_routed_engine_id = -1;
```

## Step 3: Update `PolicyGetSortedEngineIds`

The current implementation returns input engine IDs unchanged. Replace that with:

```cpp
if(g_routed_engine_id >= 0)
{
    // Move the chosen engine to the front.
    auto it = std::find(out_engine_ids.begin(), out_engine_ids.end(), g_routed_engine_id);
    if(it != out_engine_ids.end())
    {
        std::rotate(out_engine_ids.begin(), it, it + 1);
    }
    else
    {
        fprintf(stderr,
                "[SDPA_HEURISTIC] WARNING: CK_SDPA engine_id=%d not in candidate list\n",
                g_routed_engine_id);
        // Fall through: return unchanged.
    }
    g_routed_engine_id = -1;
}
```

## Step 4: Build and Validate

```bash
# Build on a compute node via srun.
srun --partition=defq --time=00:20:00 --pty bash
cd ~/heuristic_plugins/sdpa_heuristic/build
ninja -v
cp libsdpa_heuristic.so \
  /home/AMD/ysoliman/rocm-libraries-heuristic-prototype/projects/hipdnn/build/lib/hipdnn_plugins/heuristics/

# Verify routing fires:
# Run the verifier with a GQA_DECODE shape (Sq=1) and confirm:
# [SDPA_HEURISTIC] routing to CK_SDPA engine_id=<N>
```

## Step 5: Slurm Benchmark to Measure Impact

After routing is active, benchmark GQA_DECODE and GQA_PREFILL shapes before and after:

```bash
cat > ~/slurm_jobs/bench_sdpa_routed_gfx942.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=bench_sdpa_routed
#SBATCH --partition=defq
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --constraint=MARKHAM
#SBATCH --time=01:00:00
#SBATCH --output=/home/AMD/ysoliman/slurm_jobs/%x_%j.out
#SBATCH --error=/home/AMD/ysoliman/slurm_jobs/%x_%j.err

module add ubuntu-24
module load rocm/7.2.70200
source ~/venvs/pytorch-rocm62/bin/activate

python3 ~/bench_sdpa_inline.py --arch gfx942 --filter "gqa_decode,gqa_prefill"
EOF
sbatch ~/slurm_jobs/bench_sdpa_routed_gfx942.sh
```

Compare output against `gfx942_sdpa.csv` (baseline, pre-routing).
Success criterion: `GQA_DECODE` recovers from 0.09 TFLOPS to a target above 10 TFLOPS.

## Step 6: Commit Message Template

```text
feat: SDPA heuristic routing active for GQA regimes

- PolicyFinalize returns out_applied=1 for GQA_DECODE (Sq=1) and GQA_PREFILL (Hq>Hkv)
- PolicyGetSortedEngineIds moves CK_SDPA_ENGINE_ID to front of candidate list
- MHA_PREFILL and D256_PREFILL still fall through (out_applied=0)
- Benchmark: GQA_DECODE <before> -> <after> TFLOPS on gfx942
- Benchmark: GQA_PREFILL <before> -> <after> TFLOPS on gfx942

Depends on: CK SDPA engine (commit <sha>)
Experimental branch only. No PR.
```

## Notes

- `D256_PREFILL` tile mismatch is not fixed by routing - it needs a kernel change. Flag to CK team.
- `g_routed_engine_id` uses static storage: this is fine for single-threaded test usage but should be replaced with `thread_local` or a proper context struct before any production merge.
- Do not set `out_applied=1` before the CK SDPA engine is in the candidate list. If the engine ID is absent from `in_engine_ids`, `GetSortedEngineIds` can fall through safely, but `Finalize` returning `out_applied=1` with no valid reorder is undefined behavior territory.
