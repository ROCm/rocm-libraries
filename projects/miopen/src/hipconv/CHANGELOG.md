# Changelog for hipconv

## v0.4.0 - 2026-10-06

### Added

| Kernel         | Architecture     | Layer type            | Direction | Data type           | Filter, stride                          | PR         |
|----------------|------------------|-----------------------|-----------|---------------------|-----------------------------------------|------------|
| `direct_wgrad` | CDNA 5 (gfx1250) | dense, grouped conv2d | wgrad     | fp16, bf16; dW fp32 | 2×2, 3×3, 4×4, 5×5, 3×1, 1×3; stride 1  | #212, #297 |

* `describe_config(kernel, full)`: `full` includes default values, so the description selects
  exactly that config through `matches_descriptor()` (#291).
* `version()`, this build's version as `git describe --tags` names it (#291).
* `config_version()`, the major and minor version (e.g. `v0.4`), for keying a stored config
  description. Within a minor version, a full description always selects the same config
  (#291).

### Fixed

* Add the missing config descriptors to `grouped`, `grouped_wgrad`, and `depthwise_2d_toeplitz`.
  Change the `depthwise_1d_toeplitz` descriptors to use the standard form. Add the missing
  `row_pad` field to the `patch_embed` descriptor (#291).
* Every config descriptor registers every field of its kernel's `Config`, so a `--config` spec
  can name any of them. Fields added to `direct` (CDNA 4 and CDNA 5), `direct_l1`,
  `direct_wgrad`, and `depthwise_wgrad_hankel` (CDNA 4 and CDNA 5). The short description of
  every `direct` config now includes `kh` and `kw`, and on CDNA 4 also `tile_size_c` (#291).
* `ConvParams::is_valid()` rejects a layer whose dilated filter is wider than its padded
  input on some axis (`output_is_empty()`). Such a layer reached kernel matching and scored
  a NaN weighted throughput index (#167, #288).
* `ConvLaunch::make` returns `nullopt`, and `launch()` returns `hipErrorNotSupported`, for
  an explicit-GEMM pointwise or patch-embedding layer whose GEMM hipBLASLt has no algorithm
  for on the current device. The launch failed with `HIPBLAS_STATUS_INTERNAL_ERROR` instead,
  and ended the driver's run. Selection is unchanged and still needs no GPU; the driver
  binds the next ranked kernel or reports `no_match` (#244, #281).
* `launch()` and `ConvLaunch::launch` return `hipErrorUnknown` when any other hipBLASLt call
  in an explicit-GEMM launch fails. The hipBLASLt exception escaped both instead (#281).

## v0.3.2 - 2026-10-02

### Added

| Kernel                   | Architecture     | Layer type             | Direction           | Data type           | Filter, stride                        |
|--------------------------|------------------|------------------------|---------------------|---------------------|---------------------------------------|
| `patch_embed`            | CDNA 4 (gfx950)  | patch-embedding conv2d | fprop, dgrad, wgrad | fp16, bf16; dW fp32 | stride = filter, no padding           |
| `depthwise_wgrad_hankel` | CDNA 5 (gfx1250) | depthwise conv2d       | wgrad               | fp16, bf16; dW fp32 | 3×3 … 11×11 odd square, stride 1 or 2 |

* tf32 on three CDNA 4 kernels, stored as fp32 and computed as three bf16 MFMAs on a
  (big, small) bf16 split: `depthwise_1d_toeplitz` fprop and dgrad (#223), `direct_l1`
  fprop and dgrad at 2×2 … 5×5 (#226), and `direct_wgrad` at 2×2, 3×1, 3×3, 4×4 and 5×5
  (#234).
* In builds with `HIPCONV_ENABLE_EXPLICIT_GEMM=ON` (default OFF; MIOpen leaves it off),
  tf32 on pointwise layers, stored as fp32 and computed by hipBLASLt's
  `HIPBLAS_COMPUTE_32F_FAST_TF32`. That compute type asks for "at least tf32 precision"
  and leaves the method to the backend, so the accuracy it delivers is
  architecture-dependent: an architecture without xf32 MFMA decomposes into bf16 and
  lands inside the error bound hipconv derives for tf32, while one with xf32 computes a
  single lower-precision pass that the bound does not cover (#229).
* `find_arch()`, which resolves a GFX name to an `ArchHandle` without the device-code
  check `resolve_arch()` makes, and `arch_names()`, which lists the architectures the
  registry holds. Neither needs a device or a HIP context, so a host with no gfx1250 can
  enumerate which layers the gfx1250 kernels claim. Call `resolve_arch()` before a launch
  (#245).

### Changed

* Renamed the `pointwise` algorithm to `explicit_gemm`, which now serves non-overlapping
  patch-embedding convolutions as well as 1×1 through hipBLASLt. Its CMake option
  `HIPCONV_ENABLE_POINTWISE` is now `HIPCONV_ENABLE_EXPLICIT_GEMM`, still default OFF and
  left off by MIOpen. CMake ignores the old name without a warning, so a consumer that sets
  it must rename it (#237).
* CDNA 4 `depthwise_wgrad_hankel` stores each partition's dW with plain stores into a
  channel-major workspace, and a fold kernel sums the partitions, replacing per-tap
  atomics: geometric mean +88% over fifteen shapes (#227). With `weight_grad_type` equal
  to the input type, it writes dW as fp16 or bf16 in that final store, so a caller needs
  no separate cast kernel (#235).
* CDNA 5 `direct` launches a one-dimensional grid and unfolds the block index into batch,
  group, spatial and K tiles. It rejects depthwise layers (`k == c`, one channel per
  group) and configs whose thread count exceeds 2^32 (#274).

### Fixed

* CDNA 5 `direct` put groups × batch tiles in the grid's z dimension, so a layer with more
  than 65535 of them crashed. Reported as ALMIOPEN-2624 (#265, #274).
* CDNA 5 `direct`: a TDM load could overwrite an LDS buffer another wave was still
  reading, because the compiler placed that wave's `s_wait_dscnt` in the following stage.
  The kernel now drains `dscnt` after each tile load, and the odd-K dgrad restriction
  from #224 is lifted (#248).
* CDNA 5 `direct` sized its output staging LDS for one image of a batch-folded tile
  (`tile_size_n > 1`).
* CDNA 4 `direct_wgrad` ordered its staging reuse only between partitions, so on
  single-partition configs clang 24 let a lane's next-position stores reach LDS words
  other lanes were still draining, and dW came back holding the next position's gradient.
  Already in MIOpen develop through rocm-libraries #12406; this release is its first
  hipconv version (#231).
* Fixed builds with compilers that reject implicit conversions between vector types
  (`-flax-vector-conversions=none`): builtin operands are now converted explicitly (#264).

## v0.3.1 - 2026-09-17

### Added

* `gfx1250-strict` among the CDNA 5 targets, so a build serving MI455 A0 gets its own
  offload bundle and registry row, and the `GPU_TARGETS` filter keeps the `-strict` suffix
  it had dropped as a non-offload target. rocm-libraries #12085 made this change to the
  vendored copy; carrying it here is what keeps it across snapshots.
* `CHANGELOG.md` ships with the snapshot (#225).

### Changed

* `-fno-offload-lto` on hipconv's own device targets, since device LTO costs up to 8x on
  these kernels. `-DHIPCONV_OFFLOAD_LTO=ON` opts back in. Reported as
  ROCm/llvm-project#4434 (#221).

### Fixed

* CDNA 5 `direct` dgrad rejects shapes whose input or weight extent ends mid-dword, where
  a TDM load corrupts a concurrent `ds_load_tr16_b128` (#224).
* Moved `direct_wgrad`'s ladder identity out of a `static_assert` and into a test. The
  compile-time search cost four seconds of each shard's front end and sat close enough to
  clang's constexpr step limit that GCC 14's libstdc++ pushed it over. Reported as
  rocm-libraries #11990, and patched there in the vendored copy by rocm-libraries #11989
  (#218).

## v0.3.0 - 2026-09-11

### Added

| Kernel                   | Architecture     | Layer type       | Direction    | Data type           | Filter, stride                        |
|--------------------------|------------------|------------------|--------------|---------------------|---------------------------------------|
| `depthwise_wgrad_hankel` | CDNA 4 (gfx950)  | depthwise conv2d | wgrad        | fp16, bf16; dW fp32 | 3×3 … 11×11 odd square, stride 1 or 2 |
| `depthwise_1d_toeplitz`  | CDNA 5 (gfx1250) | depthwise conv2d | fprop, dgrad | fp16, bf16, tf32    | 3×3 … 11×11 odd square, stride 1 or 2 |

* tf32 on every CDNA 5 kernel, stored as fp32 and computed by a three-way bf16
  decomposition (#201).
* conv3d descriptors, and the unfolding that reduces a degenerate 3D convolution to an
  equivalent conv2d so those layers reach the existing 2D kernels (#189).
* An option to emit DWARF line tables for device code (#196).

### Changed

* Take an unserved architecture's offload target from a declared GPU (#211).
* Removed 1×1 from the CDNA 4 `direct` filter set (#213, #214).

### Fixed

* Worked around the gfx1250 WMMA C-operand hazard (#215).

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

* Build the kernels for `GPU_TARGETS` rather than the build machine's devices (#190).
* Derive `direct_wgrad`'s error tolerance from its blocked accumulation (#169).

## v0.2.4 - 2026-08-16

### Changed

* Compile the device code against Microsoft's standard library (#188).

## v0.2.3 - 2026-08-16

### Changed

* Build `libautoshard` as a static library (#187).

## v0.2.2 - 2026-08-14

### Fixed

* Guard `direct_wgrad`'s kernel entry on the gfx950 builtins it needs, so it no longer
  fails to compile for another target (#186).

## v0.2.1 - 2026-08-14

* Formatting (#185).

## v0.2.0 - 2026-08-14

### Added

| Kernel                  | Architecture    | Layer type       | Direction    | Data type           | Filter, stride                        |
|-------------------------|-----------------|------------------|--------------|---------------------|---------------------------------------|
| `direct_wgrad`          | CDNA 4 (gfx950) | dense conv2d     | wgrad        | fp16, bf16; dW fp32 | 2×2, 3×3, 4×4, 5×5, 3×1; stride 1     |
| `depthwise_1d_toeplitz` | CDNA 4 (gfx950) | depthwise conv2d | fprop, dgrad | fp16, bf16          | 3×3 … 11×11 odd square, stride 1 or 2 |
| `depthwise_2d_toeplitz` | CDNA 4 (gfx950) | depthwise conv2d | fprop        | fp16, bf16          | 3×3, 5×5, 7×7; stride 1 or 2          |

* The `Depthwise` algorithm and its backend gate, `groups == C == K` (#128, #157).
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
* Let spec layers declare their own directions and data type (#154).

## v0.1.2 - 2026-07-24

### Changed

* Ship `.clang-format` with the snapshot so the consumer's format check applies hipconv's
  style, and adopt the pre-commit framework (#147).

## v0.1.1 - 2026-07-22

### Changed

* Dropped references to an internal document from shipped comments (#144).

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

* `pointwise` (#115) is a hipBLASLt GEMM behind the `ConvKernel` interface, off by default.
  A library vendoring hipconv serves 1×1 through its own GEMM.
* The multi-architecture registry mapping a GFX name to an `ArchHandle` and a
  per-architecture algorithm list, the `ConvKernel` interface with its weighted throughput
  index, per-kernel configuration descriptors, the autoshard build that splits each kernel
  family across translation units, and CDNA 3 stub backends.
