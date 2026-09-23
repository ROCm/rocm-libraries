# WGM Debug Feature (hipBLASLt)

Visualize how GEMM **workgroups are mapped to output tiles** and **distributed across XCDs (XCC)** — the effect of `WorkGroupMapping` (WGM) and `WorkGroupMappingXCC`. Useful for understanding L2 locality and chiplet load balancing during tuning.

> ⚠️ **Kernels built with `EnableWGMDebug: 1` produce INCORRECT GEMM results.** The instrumentation overwrites each output tile's top-left element with diagnostics. Use only for visualization, never in production.

## What it does

When a solution has `EnableWGMDebug: 1`, the kernel writes a dedicated **16-byte record** to the top-left element of every workgroup's output tile in `D` (a raw `buffer_store_dwordx4`, lane‑0 only), independent of the output data type:

| dword | bytes | contents |
|------:|------:|----------|
| 0 | 0–3   | original **pre-WGM 1D workgroup id** |
| 1 | 4–7   | packed **post-WGM** `(WorkGroup0 << 16) \| WorkGroup1` |
| 2 | 8–11  | **XCC id** (`HW_REG_XCC_ID`) — which XCD ran the workgroup |
| 3 | 12–15 | reserved (0); the WGM value is in the kernel name |

Works for **all `DestDataType`s** (fp32, bf16, fp16, fp8, int8, …): the 16 bytes are written raw at the tile origin, so they survive regardless of element size.

## How to use

### 1. Enable per kernel (YAML)
Add to the solution parameters of any Tensile tuning/logic YAML:
```yaml
- EnableWGMDebug: [1]        # ForkParameters
# or in a hand-written logic solution:
  EnableWGMDebug: 1
```

### 2. Build
No special flags — it's a per-kernel parameter, works in any build mode:
```bash
cd projects/hipblaslt
./install.sh -c --skip_rocroller -a gfx950     # Release; add -k for RelWithDebInfo
```

### 3. Run and dump `D`
Run any GEMM that selects the instrumented kernel through `hipblaslt-bench` **with a validation flag** (so `D` is copied to host) and the dump env var set:
```bash
HIPBLASLT_DEBUG_WGM_DUMP=d.bin \
  ./hipblaslt-bench --yaml my_gemm.yaml --norm_check 1
```
This writes `d.bin` = a small header + the raw `D` bytes. Header layout:
```
int32 magic 'WGMD' (0x57474D44), int32 M, int32 N, int32 ldd, int32 bytesPerElement
```
Only the first batch of the first GEMM is dumped. If you have a custom tuned
library, point at it with `HIPBLASLT_TENSILE_LIBPATH=<libdir>/gfx950`.

### 4. Visualize
```bash
python3 scripts/plot_wgm.py d.bin --mt0 <MacroTile0> --mt1 <MacroTile1> -o out.jpg
```
- `--mt0/--mt1` = the kernel's macro tile (M×N), e.g. `256 256` from `..._MT256x256x...`.
- Produces a plot: **XCC id per workgroup** (chiplet distribution) and the **post‑WGM WorkGroup0 + launch‑order walk** (tile mapping). Decodes any `bytesPerElement`.

## Notes & caveats
- Results are intentionally wrong — a validation flag (`--norm_check`/`--unit_check`/`--allclose_check`) will report failures; that's expected and only used to trigger the host copy + dump.
- `WorkGroupMappingXCC <= 1` means the kernel does **no** XCC remap, so the XCC map you see is the hardware's default WG→XCD assignment (read live from `HW_REG_XCC_ID`).
- Very large workgroup grids (thousands of WGs) may stress some profilers/tools; the plot itself scales (per-cell text only for small grids).

## Files
- `tensilelite/Tensile/Common/ValidParameters.py` — `EnableWGMDebug` parameter
- `tensilelite/Tensile/KernelWriter.py` — persistent `WGMDebugOrigWG0` sgpr
- `tensilelite/Tensile/KernelWriterAssembly.py` — pre-WGM snapshot + `wgmDebugRawStore()`
- `tensilelite/Tensile/Components/GlobalWriteBatch.py` — per-kernel store hook
- `clients/common/include/testing_matmul.hpp` — `debug_wgm_dump_d` (env `HIPBLASLT_DEBUG_WGM_DUMP`)
- `scripts/plot_wgm.py` — decoder/visualizer
