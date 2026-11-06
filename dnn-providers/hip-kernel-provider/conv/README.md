# conv: the HIP convolution engine group

HIP convolution kernels served by the kernel ingestor, beside the `flydsl/` and `rocke/`
groups. Kernels here are plain HIP source, embedded in the plugin and compiled at runtime
through HIPRTC for whatever device a plan is built on, so one source serves every
architecture. This is where convolution kernels brought over from MIOpen land until faster
engines (rocKE and others) replace them.

| Folder | Engines | What it is |
|---|---|---|
| `naive/` | `hipkernel:CONV_NAIVE_FWD`, `hipkernel:CONV_NAIVE_BWD`, `hipkernel:CONV_NAIVE_WRW` | Direct convolution, one thread per output element, fp32 accumulation. Correct for every layout, group count, padding, stride, dilation, 1-D/2-D/3-D shape and both convolution modes, in fp32, fp16 and bf16. Accepted only for small problems: see `NAIVE_MACS_PER_LANE_BUDGET` in `packs/ConvNaiveProblem.hpp`. |

## Where the pieces live

| Piece | Path |
|---|---|
| Kernel sources | `conv/<family>/` (this folder) |
| Descriptors | `src/engines/kernel_ingestor_engine/descriptors/conv/<family>/` |
| Native pack (matching, dispatch) | `src/engines/kernel_ingestor_engine/packs/Conv<Family>*.cpp` |
| Tests | `src/tests/engines/kernel_ingestor_engine/packs/TestConv<Family>*.cpp` |

## Building

Off by default. Enable with `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_CONV=ON`.
With the option OFF, the packs are not compiled, the kernels are not embedded and the
`conv/` descriptor folder is not packed (`HKP_DESCRIPTOR_FAMILIES` in
`descriptor-packaging/cmake/HkpPackaging.cmake`).

## Adding a kernel family

1. Put the HIP source under `conv/<family>/`, with every shape a runtime argument.
2. Register it for embedding in `HKP_CONV_KERNEL_SOURCES` / `HKP_CONV_KERNEL_KEYS`
   (`src/engines/kernel_ingestor_engine/CMakeLists.txt`). Embedded headers are keyed by bare
   file name, so give each a unique one.
3. Author its descriptors under `descriptors/conv/<family>/` with `kernel_source.kind`
   `embedded_source` and the key as `source_file`.
4. Add the native pack and its row in `IngestorPacks.cpp`, inside `#ifdef HIPDNN_ENGINE_CONV`.
