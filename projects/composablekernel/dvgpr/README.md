# Dynamic-VGPR fusion experiment (AICK-1303)

Tooling to test whether gfx1250 dynamic VGPR (`s_alloc_vgpr` + `ENABLE_DYNAMIC_VGPR`)
lets a single fused-VectorSize conv kernel recover the per-path occupancy that plain
fusion loses. Full analysis: `~/vault/CK/projects/ck/issues/gfx12-buffer-inst-prefetch/dynamic-vgpr-feasibility.md`.

Sources in `../example/ck_tile/20_grouped_convolution/`:
- `fused_vectorsize_probe.hpp` - kernel-type machinery (the real conv kernel for VS 1/2/4/8).
- `fused_vectorsize_probe.cpp` - the `__global__` kernels (four solos + `fused_conv`); device
  code source for the `.hsaco` and the static analysis.
- `fused_vectorsize_kernels.inc` - the kernel definitions (shared by probe + harness).
- `fused_vectorsize_harness.cpp` - host program: sets up a real fp16 2D NHWGC conv, builds
  kargs, and launches a kernel via `<<<>>>` (compiler-handled ABI), times it, prints a checksum.
  Vanilla (as-compiled) only; the dynamic-VGPR patched path is not runnable here (see below).

## Comparison matrix

Two device objects, five kernels each, giving "each solo plus fused, with and without patch":

- `out/vanilla.o` - kernels as compiled (static VGPR per kernel).
- `out/patched.o` - kernels transformed to dynamic VGPR (small creation count + per-path
  `s_alloc_vgpr(peak)` + `MSG_DEALLOC_VGPRS` removed + RSRC3 bit 17 set).

## Scripts

| Script | Purpose | Runs now? |
|---|---|---|
| `build_variants.sh` | Compile the probe; emit `out/vanilla.o` + `out/patched.o` (static analysis). | yes |
| `build_runnable.sh` | Build `out/vanilla.hsaco`, `out/patched.hsaco`, and `out/harness`. | yes (no GPU) |
| `run_profile.sh` | Run the harness per variant on MI450; time + checksum (+ rocprofv3 if `PROFILE=1`). | needs MI450 |
| `dvgpr_transform.py` | Per-kernel `.s` transform: insert `s_alloc_vgpr`, lower `next_free_vgpr`, drop dealloc. | yes |
| `patch_dvgpr.py` | Set `ENABLE_DYNAMIC_VGPR` (RSRC3 bit 17) on named `.kd` (handles .o and .hsaco). | yes |
| `verify_dvgpr.py` | Read-only: report `dvgpr_en` per `.kd` in an object. | yes |
| `extract_metrics.py` | Static comparison: per-kernel/per-path creation vs steady VGPR and est. waves/SIMD. | yes |
| `cfg_attribute.py`, `highreg_locate.py`, `analyze_conv.py` | Per-path register attribution of the fused kernel. | yes |

## Usage

Static comparison (no GPU):
```bash
bash dvgpr/build_variants.sh
python3 dvgpr/extract_metrics.py dvgpr/out/vanilla.s dvgpr/out/patched.s
```

Runtime comparison (on the MI450 node):
```bash
bash dvgpr/build_runnable.sh        # builds out/harness (kernels compiled in, <<<>>>)
bash dvgpr/run_profile.sh           # fused path i vs solo i; PROFILE=1 adds rocprofv3
```

Scripts derive compile flags from the build's `compile_commands.json` (set `BUILD=` if not
`./build`). `run_profile.sh` sweeps several shapes (small/large/spatial/deep) and reports each
fused path's overhead vs its solo - i.e. the runtime cost of plain fusion - with a checksum-match
column. The harness shape is set by env vars (`CONV_N/K/C/HI/WI/FY/FX/STRIDE/PAD`, C/K multiples
of 8); run one shape directly, e.g. `CONV_N=256 CONV_C=256 CONV_K=256 dvgpr/out/harness solo8 0 1000`.

## Static result

Plain fusion pins every path to the VS1 budget (259 VGPR, ~5 waves/SIMD). Dynamic VGPR
lets each path run at its own allocation: VS8 174/~8 waves, VS2 200/~7, VS4 216/~7, VS1
259/~5. So the lighter paths recover occupancy a single fused instance would otherwise lose.
Estimate uses 1536 VGPR/SIMD (gfx1250 wave32); achieved values need hardware.

## Dynamic VGPR on hardware: unsupported on this ROCm (MI450, 2026-06-16)

The dynamic-VGPR patched path does not run on this node's ROCm. Evidence:
- `libhsa-runtime64.so` has no dynamic-vgpr / dvgpr strings - the runtime never puts a dispatch
  into dynamic-VGPR mode, so `ENABLE_DYNAMIC_VGPR` in the descriptor is not honored.
- Running the patched code object hangs the GPU (GPU Hang, core dumped): with the mode off,
  `s_alloc_vgpr` is illegal per the ISA and the wave stalls.

So the descriptor-patch approach cannot work until AMD wires the launch path (queue/MES/firmware)
for compute dynamic VGPR. The static analysis (build_variants.sh + extract_metrics.py) and the
patched `.o`/`.hsaco` build remain valid as the design + a HW-ready artifact for when support lands.

The runnable harness was therefore switched to vanilla `<<<>>>` launches, measuring the
as-compiled kernels: the plain fused kernel (one instance, runtime sel - the 4->1 enumeration win)
and each fused path's overhead vs its solo (the occupancy cost plain fusion pays, which dynamic
VGPR would remove).
