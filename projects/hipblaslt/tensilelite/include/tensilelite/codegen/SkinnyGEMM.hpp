// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_SKINNYGEMM_HPP
#define ROCM_TENSILELITE_CODEGEN_SKINNYGEMM_HPP

#include <tensilelite/codegen/Target.hpp>

#include <string>

namespace rocm::tensilelite::codegen {

/// Compile-time knobs of the skinny MXFP8 GEMM, named after the constants of
/// the MLIR reference (gemm_mxfp8_skinny.mlir).
struct SkinnyGEMMConfig {
  /// BM: rows of the skinny operand handled per workgroup.
  int bm = 1;
  /// TN: output columns of the wide operand per lane group.
  int tn = 1;
  /// LANES_K: lanes that split the k dimension; one of 4, 8, 16 and 64.
  int lanesK = 64;
  /// WAVES: wavefronts per workgroup, so a block has 64 * waves lanes.
  int waves = 4;
  /// PF: prefetch ring depth; 0 means no ring.
  int prefetch = 2;
  /// UP_ITERS: 0 rings the scales, otherwise the up-front scale vector size.
  int upIters = 0;
  /// PARTIAL_STEPS: 0, 1 or 2, derived by the host from k and spk.
  int partialSteps = 0;
  /// GROUP: workgroup grouping factor of the grid swizzle.
  int group = 1;
  /// NUM_XCDS: number of XCDs the workgroups are distributed over.
  int numXcds = 1;
  /// GRID_3D: false selects a persistent 1-D launch.
  bool grid3d = true;
};

/// Returns the symbol of the GEMM kernel, in the reference's format:
/// gemm_BM<v>_TN<v>_LANES_K<v>_WAVES<v>_UP_ITERS<v>_PARTIAL_STEPS<v>_PF<v>_
/// GROUP<v>_NUM_XCDS<v>_GRID_3D<v>. Throws std::invalid_argument for invalid
/// knobs.
std::string skinnyGEMMKernelName(const SkinnyGEMMConfig &config);

/// Emits the GEMM kernel as the assembly text of one kernel, for
/// assembleToHSACO, replicating the behavior and the schedule of the MLIR
/// reference for the 3-D launch with ring scales and no partial K steps: grid3d
/// with group = 1 and numXcds = 1, upIters = 0 and partialSteps = 0, so k and
/// spk must be multiples of 64 * lanesK. The kernargs are the pointers D, A, B,
/// scalesA, scalesB and ws, then ldD, ldA, ldB, m, n, k and spk (i32), then the
/// hidden group sizes x, y, z and block counts x, y, z, in the layout of the
/// reference. The block is (64 * waves, 1, 1) and the grid is (ceil(max(m, n) /
/// BN), ceil(min(m, n) / bm), ceil(k / spk)) with BN = waves * (64 / lanesK) *
/// tn. With a single split the kernel stores bf16 into D, else each split
/// stores its f32 slice into the workspace, which splitk_reduce sums. Throws
/// std::invalid_argument for invalid knobs, for the features that are not
/// emitted yet (upIters > 0, partialSteps != 0 and grid3d = false, each
/// naming the knob) and when the configuration exceeds the register budget.
std::string emitSkinnyGEMM(const Target &target,
                           const SkinnyGEMMConfig &config);

/// Emits the split-K reduction kernel splitk_reduce:
/// D[i, j] = the bf16 truncation (top 16 bits) of the f32 sum over z = 0..
/// splits-1 of ws[z * m * n + j * m + i], summed in split order from 0.0. The
/// kernargs are the pointers D (bf16, column stride ldD) and ws (f32), then
/// ldD, m, n and splits (i32), then the hidden group sizes x, y, z and block
/// counts x, y, z (4 bytes each, offsets 32 to 52, segment size 56). The block
/// is (256, 1, 1) and the grid is (ceil(m / 256), n, 1); splits must be at
/// least 1.
std::string emitSplitKReduce(const Target &target);

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_SKINNYGEMM_HPP
