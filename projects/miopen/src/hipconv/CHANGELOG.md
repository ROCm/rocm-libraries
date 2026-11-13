# Changelog for hipconv

## v0.5.0 - 2026-11-04

### Added

* `direct_flex` (CDNA 4) runs dense conv2d and conv3d fprop and dgrad in fp16 and bf16, with C (K in
  dgrad) a multiple of 8, filters up to 16 per axis and any stride, dilation and padding, except
  strided dilated dgrad and dgrad past 8 stride phases. Dilated layers had no kernel.
* `direct_flex` (CDNA 4) splits the reduction of 9-tap filters, such as 3×3, across a workgroup's
  waves in fprop and unit-stride dgrad, on some layers with at most one tile per CU.
* `direct_l1` (CDNA 4) runs conv3d fprop and stride-1 dgrad for 3×3×3 filters in fp16 and bf16,
  with split-K for layers too small to fill the GPU. Such a layer had no kernel.
* `direct_l1` (CDNA 4) runs tf32 conv3d fprop and stride-1 dgrad for 3×3×3 filters, including odd
  channel counts. Such a layer had no kernel.
* `direct_l1` (CDNA 4) has a K(64) split-K tile for 3×3 conv2d and 3×3×3 conv3d in fp16 and bf16,
  for layers that underfill the GPU.
* `direct_l1` (CDNA 4) runs fprop and dgrad for 1×3 and 1×5 filters at stride 1 and pad_h 0, in
  fp16 and bf16. Such a layer had no kernel.
* `direct_l1` (CDNA 4) runs tf32 fprop and dgrad for 3×1 filters, including single images past
  2 GiB. Such a layer had no kernel.
* `direct_l1` (CDNA 4) has a chunked unfold tile for 3×3 conv2d fprop and dgrad at output widths
  above 8 in fp16 and bf16, and a K192 tile for output channels divisible by 192 but not 256.
* `direct_wgrad` (CDNA 4) runs 5×1, 1×3 and 1×5 filters in fp16 and bf16, and 1×3 also in tf32,
  at stride 1 and dilation 1 for tensors within 2 GiB, grouped layers included. Such a layer had
  no kernel.
* `direct_wgrad` and `grouped_wgrad` (CDNA 4) and the explicit-GEMM `pointwise` write dW in the
  operand type (`weight_grad_type` fp16 or bf16) for 16-bit operands, so a caller can drop its cast
  from fp32. `grouped_wgrad` needs a workspace of 4 bytes per dW element for it.
* `grouped` (CDNA 4) runs conv3d fprop, dgrad and wgrad at 4, 8, 16 and 32 channels per group in
  fp16, bf16 and tf32 at any depth stride, padding and dilation, with kd ≤ 7 in fprop and dgrad.
  Such a layer had no kernel unless it unfolded to a conv2d or was a 3×3×3 stride-1 fprop or dgrad.
* `grouped` (CDNA 4) runs stride-2 dgrad at 8 channels per group and `grouped_wgrad` (CDNA 4)
  stride-2 wgrad at 4, 8, 16 and 32, both in fp16, bf16 and tf32. Such a layer had no kernel.
* `grouped` (CDNA 4) runs fprop and dgrad on layers whose tensors pass 2 GiB, single images past
  2 GiB included, at 4, 8, 16 and 32 channels per group in conv2d and conv3d.
* `depthwise_1d_toeplitz` and `depthwise_wgrad_hankel` (CDNA 4) run depthwise conv3d in fp16, bf16
  and tf32 at any depth stride, padding and dilation; wgrad at any square filter, fprop and dgrad
  at kd ≤ 7 and plane stride 1 or 2, with smaller plane filters at larger kd and in tf32.
* `depthwise_wgrad_hankel` (CDNA 4) runs tf32 wgrad with fp32 dW, computed as three bf16 MFMAs
  on a (big, small) bf16 split.
* `depthwise_1d_toeplitz` and `depthwise_wgrad_hankel` (CDNA 4) run 1×kw (kw 2–127) and kh×1
  (kh 2–31) filters in fprop, dgrad and wgrad at stride 1 or 2; in tf32, kh×1 stops at 25 for
  fprop and dgrad and at 29 for stride-1 wgrad. Such layers had no kernel.
* `depthwise_1d_toeplitz` and `depthwise_wgrad_hankel` (CDNA 4 and CDNA 5) run layers whose image
  exceeds 2 GiB, and on CDNA 5 wgrad layers whose whole tensor does. Such a layer had no kernel.
* `patch_embed` (CDNA 4) runs fprop, dgrad and wgrad for 1×1 filters at stride 1, which only the
  explicit-GEMM `pointwise` served, for strides larger than the filter, which had no kernel, and
  for filter rows shorter than 8 elements (`kw·c < 8`).
* `patch_embed` (CDNA 4) and the explicit-GEMM `patch_embed` run tf32 fprop, dgrad and wgrad.
  A tf32 patch embedding had no kernel.
* The explicit-GEMM `pointwise` runs grouped 1×1 convolution, depthwise included, in fp16, bf16 and
  tf32, except 16-bit wgrad with fp32 dW on narrow groups, which hipBLASLt has no algorithm for. On
  CDNA 5 it ranks above `direct` for grouped 1×1 fprop and dgrad.
* `direct` (CDNA 5) runs 1×3, 3×1, 1×5, 5×1 and 7×1 filters, and `direct_wgrad` (CDNA 5) runs
  1×5 and 5×1.
* The CMake option `HIPCONV_DISABLE_ASAN` (default `OFF`) compiles hipconv without ASAN
  instrumentation, for a consumer built with `-fsanitize=address` whose link otherwise fails with
  an out-of-range relocation.
* `describe_kernel(kernel)` returns `family[full description]`, or the bare family when the full
  description is empty, and `find_kernel(arch, description)` returns the kernel it names.
* `is_deterministic(kernel, device, par)` reports whether two launches on the same inputs write
  bitwise-identical outputs; it is false for a kernel that adds into dW atomically.
* `has_device_code(hip_device)` reports whether the loaded binary holds hipconv's device code for
  a HIP device.

### Changed

* libhipconv is a shared library when `BUILD_SHARED_LIBS` is `ON`. A consumer that links it into a
  shared library of its own sets `HIPCONV_FORCE_STATIC` to `ON` before `add_subdirectory(hipconv)`;
  a shared libhipconv does not configure on Windows.
* `hipconv/export.hpp` is generated in the build directory; a target that includes hipconv headers
  without linking `hipconv` links the new `hipconv_headers` interface target.
* Selection and launch take a `hipconv::Device { ArchHandle arch; int cu_count; }`: in place of the
  `ArchHandle` in `get_valid_configs` and `find_config`, and after the kernel handle elsewhere.
  Selection makes no HIP calls, and a null `arch` or a `cu_count` below 1 is rejected.
* `resolve_arch` no longer checks the current device for device code; it is the same registry
  lookup as `find_arch`. Without device code for the GPU, `launch()` fails; `has_device_code`
  reports beforehand whether a HIP device has it.
* `launch()` requires the stream's device to be current, as hipBLASLt does.
* `ConvLaunch::make` no longer asks hipBLASLt for an algorithm. If hipBLASLt has none for the
  layer, the first `launch()` returns `hipErrorNotSupported`.
* The explicit-GEMM kernels give hipBLASLt a workspace for wgrad, so it can choose Stream-K
  algorithms. Fprop and dgrad get none, since hipBLASLt before rocm-libraries #11691 can return
  wrong results from Stream-K's fix-up on gfx950.
* `direct_l1` (CDNA 4) adds a narrow K64 tile for 3×3 conv2d fprop and ranks every valid config
  for a layer, best first, where it claimed one per shape; MIOpen's Find and tuning reach them all.
* `direct_l1` (CDNA 4) ranks its K64 tiles for 3×3 conv2d by estimated launch time, so a layer whose
  output channel count is not a multiple of 256 no longer defaults to a tile padded in K.
* `direct_l1` (CDNA 4) divides the input channels of a 3×3 conv2d layer (output channels in dgrad)
  among workgroups when its tiles alone would leave the GPU idle, as it does for 3×3×3.
* `direct_l1` (CDNA 4) runs its K(64) tiles in C(64) steps in place of C(32), for 2×2 to 5×5 conv2d,
  3×3×3 conv3d and reductions of at most 64 channels, in fp16 and bf16.
* `direct_wgrad` (CDNA 4) no longer zeroes dW and adds into it atomically: an unsplit grid stores dW
  directly, with `no_splits` in the full description, and a split one sums fp32 partials from a
  workspace in a second kernel, so `get_workspace_size` is nonzero for it.
* `direct_wgrad` (CDNA 4) divides an image of height H into up to about √H row segments where its
  column blocks alone would leave workgroups idle or unevenly loaded, on 3×3, 3×1 and 5×5 filters in
  16-bit and tf32, and processes 16 or 32 images together on 1×3 and 1×5 wgrad at input height 1.
* The full description of a CDNA 4 `grouped_wgrad` kernel carries `elem_bytes`, so its fp16/bf16
  and tf32 kernels no longer share a description.
* `grouped` (CDNA 4) at 32 channels per group keeps its input row loads in flight across rows.
* `grouped` (CDNA 4) fprop and dgrad split each image's rows across the GPU's XCDs when a layer
  has fewer than 8 images (output planes in conv3d), which had left XCDs idle.
* `grouped` (CDNA 5) splits a group's image height across waves in stride-1 fprop and dgrad,
  choosing the split and waves per SIMD from the CU count; full descriptions carry `h_split` and
  `waves_per_simd`.

### Fixed

* A build that names no GPU hipconv has kernels for still compiles one empty kernel for each of its
  GPUs, which TheRock's artifact split requires.
* A shared libhipconv exports `sizeof_data_type`, the `to_string` overloads, `parse_algorithm`,
  `parse_data_type`, `parse_direction_mask` and `ConvParams::unfolded`, which lacked `HIPCONV_API`.
* `is_applicable` and `ConvLaunch::make` accept only the kernels `get_valid_configs` lists, where
  they also accepted others, such as a depthwise kernel for a dense layer.
* `get_tolerance`, `ConvLaunch::get_tolerance` and `get_recursive_summation_tolerance` returned
  `atol` 0 for an fp16 result, so a correctly rounded subnormal output could fail the bound; it is
  now 2^-25. `depthwise_1d_toeplitz` drops its 1e-6 floor on `atol` and returns the same.
* Explicit-GEMM launches failed on every device but the first, which owned the one process-wide
  hipBLASLt handle. Handles are now per device, keyed by the stream's device.
* Selecting a CDNA 4 `depthwise_wgrad_hankel` kernel no longer initializes the HIP runtime to read
  the CU count.
* CDNA 4 `direct_wgrad` ignores a `HIPCONV_WGRAD_UNFOLD_N` other than 1, 2 or 4, which forced one
  image per column block.
* `direct_wgrad` (CDNA 4) could write wrong dW when compiled with clang 23, on layers whose C or K
  does not fill the last tile. v0.4.0 has the same defect.
* `direct_wgrad` (CDNA 4) closes a race on the LDS ring of input rows it stages.
* `direct_l1`, `direct_wgrad` and `grouped` (CDNA 4) close three synchronization gaps that held only
  under the compiler's current instruction order or the current configs.
* `direct` (CDNA 4) closes two LDS races: it adds a missing LDS drain, and each wave waits for
  exactly the input loads it issued.
* `direct` (CDNA 5) closes a race on the input ring it stages in LDS.
* `direct` (CDNA 5) no longer crashes on layers with large padding.
* `grouped_wgrad` (CDNA 5) 3 × 3 wgrad in fp16, bf16 and tf32 no longer races a Tensor Data Mover
  load into LDS with the same wave's earlier LDS stores and loads, which could corrupt dW.
* A Debug build compiles and links, with or without UBSan in `CMAKE_CXX_FLAGS_DEBUG`, and the
  build-time generators no longer need `libamdhip64` to run. Device code compiles at `-O3` in
  every build type, with no debug info beyond the line tables that `HIPCONV_LINE_TABLES` enables.
* `direct` and `direct_l1` (CDNA 4) reject an fprop or dgrad layer whose weights per group take
  4 GiB or more. Before, both accepted such a layer and the launch faulted.

## v0.4.0 - 2026-10-06

### Added

| Kernel         | Architecture     | Layer type            | Direction | Data type           | Filter, stride                          |
|----------------|------------------|-----------------------|-----------|---------------------|-----------------------------------------|
| `direct_wgrad` | CDNA 5 (gfx1250) | dense, grouped conv2d | wgrad     | fp16, bf16; dW fp32 | 2×2, 3×3, 4×4, 5×5, 3×1, 1×3; stride 1  |

* `describe_config(kernel, full)`: `full` includes default values, so the description selects
  exactly that config through `matches_descriptor()`.
* `version()`, this build's version as `git describe --tags` names it.
* `config_version()`, the major and minor version (e.g. `v0.4`), for keying a stored config
  description. Within a minor version, a full description always selects the same config.

### Fixed

* Add the missing config descriptors to `grouped`, `grouped_wgrad`, and `depthwise_2d_toeplitz`.
  Change the `depthwise_1d_toeplitz` descriptors to use the standard form. Add the missing
  `row_pad` field to the `patch_embed` descriptor.
* Every config descriptor registers every field of its kernel's `Config`, so a `--config` spec
  can name any of them. Fields added to `direct` (CDNA 4 and CDNA 5), `direct_l1`,
  `direct_wgrad`, and `depthwise_wgrad_hankel` (CDNA 4 and CDNA 5). The short description of
  every `direct` config now includes `kh` and `kw`, and on CDNA 4 also `tile_size_c`.
* `ConvParams::is_valid()` rejects a layer whose dilated filter is wider than its padded
  input on some axis (`output_is_empty()`). Such a layer reached kernel matching and scored
  a NaN weighted throughput index.
* `ConvLaunch::make` returns `nullopt`, and `launch()` returns `hipErrorNotSupported`, for
  an explicit-GEMM pointwise or patch-embedding layer whose GEMM hipBLASLt has no algorithm
  for on the current device. The launch failed with `HIPBLAS_STATUS_INTERNAL_ERROR` instead,
  and ended the driver's run. Selection is unchanged and still needs no GPU; the driver
  binds the next ranked kernel or reports `no_match`.
* `launch()` and `ConvLaunch::launch` return `hipErrorUnknown` when any other hipBLASLt call
  in an explicit-GEMM launch fails. The hipBLASLt exception escaped both instead.

## v0.3.2 - 2026-10-02

### Added

| Kernel                   | Architecture     | Layer type             | Direction           | Data type           | Filter, stride                        |
|--------------------------|------------------|------------------------|---------------------|---------------------|---------------------------------------|
| `patch_embed`            | CDNA 4 (gfx950)  | patch-embedding conv2d | fprop, dgrad, wgrad | fp16, bf16; dW fp32 | stride = filter, no padding           |
| `depthwise_wgrad_hankel` | CDNA 5 (gfx1250) | depthwise conv2d       | wgrad               | fp16, bf16; dW fp32 | 3×3 … 11×11 odd square, stride 1 or 2 |

* tf32 on three CDNA 4 kernels, stored as fp32 and computed as three bf16 MFMAs on a
  (big, small) bf16 split: `depthwise_1d_toeplitz` fprop and dgrad, `direct_l1`
  fprop and dgrad at 2×2 … 5×5, and `direct_wgrad` at 2×2, 3×1, 3×3, 4×4 and 5×5.
* In builds with `HIPCONV_ENABLE_EXPLICIT_GEMM=ON` (default OFF; MIOpen leaves it off),
  tf32 on pointwise layers, stored as fp32 and computed by hipBLASLt's
  `HIPBLAS_COMPUTE_32F_FAST_TF32`. That compute type asks for "at least tf32 precision"
  and leaves the method to the backend, so the accuracy it delivers is
  architecture-dependent: an architecture without xf32 MFMA decomposes into bf16 and
  lands inside the error bound hipconv derives for tf32, while one with xf32 computes a
  single lower-precision pass that the bound does not cover.
* `find_arch()`, which resolves a GFX name to an `ArchHandle` without the device-code
  check `resolve_arch()` makes, and `arch_names()`, which lists the architectures the
  registry holds. Neither needs a device or a HIP context, so a host with no gfx1250 can
  enumerate which layers the gfx1250 kernels claim. Call `resolve_arch()` before a launch.

### Changed

* Renamed the `pointwise` algorithm to `explicit_gemm`, which now serves non-overlapping
  patch-embedding convolutions as well as 1×1 through hipBLASLt. Its CMake option
  `HIPCONV_ENABLE_POINTWISE` is now `HIPCONV_ENABLE_EXPLICIT_GEMM`, still default OFF and
  left off by MIOpen. CMake ignores the old name without a warning, so a consumer that sets
  it must rename it.
* CDNA 4 `depthwise_wgrad_hankel` stores each partition's dW with plain stores into a
  channel-major workspace, and a fold kernel sums the partitions, replacing per-tap
  atomics: geometric mean +88% over fifteen shapes. With `weight_grad_type` equal
  to the input type, it writes dW as fp16 or bf16 in that final store, so a caller needs
  no separate cast kernel.
* CDNA 5 `direct` launches a one-dimensional grid and unfolds the block index into batch,
  group, spatial and K tiles. It rejects depthwise layers (`k == c`, one channel per
  group) and configs whose thread count exceeds 2^32.

### Fixed

* CDNA 5 `direct` put groups × batch tiles in the grid's z dimension, so a layer with more
  than 65535 of them crashed. Reported as ALMIOPEN-2624.
* CDNA 5 `direct`: a TDM load could overwrite an LDS buffer another wave was still
  reading, because the compiler placed that wave's `s_wait_dscnt` in the following stage.
  The kernel now drains `dscnt` after each tile load, and the odd-K dgrad restriction
  from v0.3.1 is lifted.
* CDNA 5 `direct` sized its output staging LDS for one image of a batch-folded tile
  (`tile_size_n > 1`).
* CDNA 4 `direct_wgrad` ordered its staging reuse only between partitions, so on
  single-partition configs clang 24 let a lane's next-position stores reach LDS words
  other lanes were still draining, and dW came back holding the next position's gradient.
  Already in MIOpen develop through rocm-libraries #12406; this release is its first
  hipconv version.
* Fixed builds with compilers that reject implicit conversions between vector types
  (`-flax-vector-conversions=none`): builtin operands are now converted explicitly.

## v0.3.1 - 2026-09-17

### Added

* `gfx1250-strict` among the CDNA 5 targets, so a build serving MI455 A0 gets its own
  offload bundle and registry row, and the `GPU_TARGETS` filter keeps the `-strict` suffix
  it had dropped as a non-offload target. rocm-libraries #12085 made this change to the
  vendored copy; carrying it here is what keeps it across snapshots.
* `CHANGELOG.md` ships with the snapshot.

### Changed

* `-fno-offload-lto` on hipconv's own device targets, since device LTO costs up to 8× on
  these kernels. `-DHIPCONV_OFFLOAD_LTO=ON` opts back in. Reported as
  ROCm/llvm-project#4434.

### Fixed

* CDNA 5 `direct` dgrad rejects shapes whose input or weight extent ends mid-dword, where
  a TDM load corrupts a concurrent `ds_load_tr16_b128`.
* Moved `direct_wgrad`'s ladder identity out of a `static_assert` and into a test. The
  compile-time search cost four seconds of each shard's front end and sat close enough to
  clang's constexpr step limit that GCC 14's libstdc++ pushed it over. Reported as
  rocm-libraries #11990, and patched there in the vendored copy by rocm-libraries #11989.

## v0.3.0 - 2026-09-11

### Added

| Kernel                   | Architecture     | Layer type       | Direction    | Data type           | Filter, stride                        |
|--------------------------|------------------|------------------|--------------|---------------------|---------------------------------------|
| `depthwise_wgrad_hankel` | CDNA 4 (gfx950)  | depthwise conv2d | wgrad        | fp16, bf16; dW fp32 | 3×3 … 11×11 odd square, stride 1 or 2 |
| `depthwise_1d_toeplitz`  | CDNA 5 (gfx1250) | depthwise conv2d | fprop, dgrad | fp16, bf16, tf32    | 3×3 … 11×11 odd square, stride 1 or 2 |

* tf32 on every CDNA 5 kernel, stored as fp32 and computed by a three-way bf16
  decomposition.
* conv3d descriptors, and the unfolding that reduces a degenerate 3D convolution to an
  equivalent conv2d so those layers reach the existing 2D kernels.
* An option to emit DWARF line tables for device code.

### Changed

* Take an unserved architecture's offload target from a declared GPU.
* Removed 1×1 from the CDNA 4 `direct` filter set.

### Fixed

* Worked around the gfx1250 WMMA C-operand hazard.

## v0.2.8 - 2026-09-08

### Changed

* Take an unserved architecture's offload target from a declared GPU, rather than the
  build machine's.

## v0.2.7 - 2026-08-24

* clang-format over the CDNA 5 grouped kernels.

## v0.2.6 - 2026-08-24

### Fixed

* CDNA 5 grouped convolution: `s_wait_tensorcnt` does not order a `ds_store` against a TDM
  into the same slot, which corrupted the staged tile.
* CDNA 5 grouped wgrad failure on gfx1250.

## v0.2.5 - 2026-08-20

### Changed

* Build the kernels for `GPU_TARGETS` rather than the build machine's devices.
* Derive `direct_wgrad`'s error tolerance from its blocked accumulation.

## v0.2.4 - 2026-08-16

### Changed

* Compile the device code against Microsoft's standard library.

## v0.2.3 - 2026-08-16

### Changed

* Build `libautoshard` as a static library.

## v0.2.2 - 2026-08-14

### Fixed

* Guard `direct_wgrad`'s kernel entry on the gfx950 builtins it needs, so it no longer
  fails to compile for another target.

## v0.2.1 - 2026-08-14

* Formatting.

## v0.2.0 - 2026-08-14

### Added

| Kernel                  | Architecture    | Layer type       | Direction    | Data type           | Filter, stride                        |
|-------------------------|-----------------|------------------|--------------|---------------------|---------------------------------------|
| `direct_wgrad`          | CDNA 4 (gfx950) | dense conv2d     | wgrad        | fp16, bf16; dW fp32 | 2×2, 3×3, 4×4, 5×5, 3×1; stride 1     |
| `depthwise_1d_toeplitz` | CDNA 4 (gfx950) | depthwise conv2d | fprop, dgrad | fp16, bf16          | 3×3 … 11×11 odd square, stride 1 or 2 |
| `depthwise_2d_toeplitz` | CDNA 4 (gfx950) | depthwise conv2d | fprop        | fp16, bf16          | 3×3, 5×5, 7×7; stride 1 or 2          |

* The `Depthwise` algorithm and its backend gate, `groups == C == K`.
  `depthwise_2d_toeplitz` is opt-in behind `HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ` and never
  wins the auto-pick even when built.
* CDNA 5 `direct` filters to 5×5, up from 4×4, on a row ring buffer.

### Changed

* Replaced the preprocessor architecture guard with feature guards at the call sites, and
  built every architecture in every configuration rather than only the one the build
  targets.
* Reworked the CDNA 5 `direct` inner loop: NHW folding, a Gray-code MFMA walk order (also
  enabled on CDNA 4), modulo LDS indexing in place of the prologue and the TDM zero
  writes, and bank-conflict removal on fprop and dgrad.
* Let spec layers declare their own directions and data type.

## v0.1.2 - 2026-07-24

### Changed

* Ship `.clang-format` with the snapshot so the consumer's format check applies hipconv's
  style, and adopt the pre-commit framework.

## v0.1.1 - 2026-07-22

### Changed

* Dropped references to an internal document from shipped comments.

## v0.1.0 - 2026-07-22

### Added

| Kernel          | Architecture     | Layer type       | Direction           | Data type                 | Filter, stride                                   |
|-----------------|------------------|------------------|---------------------|---------------------------|--------------------------------------------------|
| `pointwise`     | CDNA 3, 4, 5     | pointwise conv2d | fprop, dgrad, wgrad | fp16, bf16; dW fp32       | 1×1, stride 1                                    |
| `direct_l1`     | CDNA 4 (gfx950)  | dense conv2d     | fprop, dgrad        | fp16, bf16                | 2×2 … 5×5, stride 1                              |
| `direct`        | CDNA 4 (gfx950)  | dense conv2d     | fprop, dgrad        | fp16, bf16                | 1×1, 3×3 … 7×7; stride 1                         |
| `grouped`       | CDNA 4 (gfx950)  | grouped conv2d   | fprop, dgrad        | fp16, bf16, tf32          | 3×3, stride 1 or 2; 4/8/16/32 channels per group |
| `grouped_wgrad` | CDNA 4 (gfx950)  | grouped conv2d   | wgrad               | fp16, bf16, tf32; dW fp32 | 3×3, stride 1; 4/8/16/32 channels per group      |
| `direct`        | CDNA 5 (gfx1250) | dense conv2d     | fprop, dgrad        | fp16, bf16                | 1×1 … 4×4, stride 1                              |
| `grouped`       | CDNA 5 (gfx1250) | grouped conv2d   | fprop, dgrad        | fp16, bf16, tf32          | 3×3, stride 1 or 2; 4/8/16/32 channels per group |
| `grouped_wgrad` | CDNA 5 (gfx1250) | grouped conv2d   | wgrad               | fp16, bf16, tf32; dW fp32 | 3×3, stride 1; 4/8/16/32 channels per group      |

* `pointwise` is a hipBLASLt GEMM behind the `ConvKernel` interface, off by default.
  A library vendoring hipconv serves 1×1 through its own GEMM.
* The multi-architecture registry mapping a GFX name to an `ArchHandle` and a
  per-architecture algorithm list, the `ConvKernel` interface with its weighted throughput
  index, per-kernel configuration descriptors, the autoshard build that splits each kernel
  family across translation units, and CDNA 3 stub backends.
