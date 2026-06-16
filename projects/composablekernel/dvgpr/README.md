# Dynamic-VGPR fusion experiment (AICK-1303)

Tooling to test whether gfx1250 dynamic VGPR (`s_alloc_vgpr` + `ENABLE_DYNAMIC_VGPR`)
lets a single fused-VectorSize conv kernel recover the per-path occupancy that plain
fusion loses. Full analysis: `~/vault/CK/projects/ck/issues/gfx12-buffer-inst-prefetch/dynamic-vgpr-feasibility.md`.

Sources in `../example/ck_tile/20_grouped_convolution/`:
- `fused_vectorsize_probe.hpp` - kernel-type machinery (the real conv kernel for VS 1/2/4/8).
- `fused_vectorsize_probe.cpp` - the `__global__` kernels (four solos + `fused_conv`); device
  code source for the `.hsaco` and the static analysis.
- `fused_vectorsize_harness.cpp` - host program: sets up a real fp16 2D NHWGC conv, builds
  kargs, loads a `.hsaco`, launches a kernel via `hipModuleLaunchKernel`, times it, prints a
  checksum. The same harness runs vanilla or patched - only the code object differs.

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
bash dvgpr/build_runnable.sh        # builds out/{vanilla.hsaco,patched.hsaco,harness}
bash dvgpr/run_profile.sh           # time + checksum per variant; PROFILE=1 adds rocprofv3
```

Scripts derive compile flags from the build's `compile_commands.json` (set `BUILD=` if not
`./build`). The harness uses a fixed shape (G=1 N=64 K=128 C=128, 3x3, 28x28, stride1 pad1);
edit `fused_vectorsize_harness.cpp` to change it.

## Static result

Plain fusion pins every path to the VS1 budget (259 VGPR, ~5 waves/SIMD). Dynamic VGPR
lets each path run at its own allocation: VS8 174/~8 waves, VS2 200/~7, VS4 216/~7, VS1
259/~5. So the lighter paths recover occupancy a single fused instance would otherwise lose.
Estimate uses 1536 VGPR/SIMD (gfx1250 wave32); achieved values need hardware.

## Running on hardware

`build_runnable.sh` + `run_profile.sh` are the runtime path. `run_profile.sh` prints, per
kernel/sel, vanilla vs patched time and whether the output checksum matches (the patched code
must produce the same result). The patched run only differs from vanilla if the MI450 CP/MES
honors `ENABLE_DYNAMIC_VGPR`. Open correctness items if results mismatch or it faults:
`s_alloc_vgpr` can fail (SCC=0) and may need a retry loop (the inserts don't loop yet), and the
dynamic-VGPR segment granularity must match the creation block size (32). The harness launches
via `hipModuleLaunchKernel` with a kargs buffer mirroring the kernel signature; if the fused
result is wrong but solos are right, suspect the kernarg layout for the 4-struct `fused_conv`
signature.
