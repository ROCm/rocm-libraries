# Changelog for hipconv

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
