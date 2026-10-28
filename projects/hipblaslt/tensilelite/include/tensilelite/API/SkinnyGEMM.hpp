// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_API_SKINNYGEMM_HPP
#define ROCM_TENSILELITE_API_SKINNYGEMM_HPP

#include <tensilelite/Codegen/KernelAbi.hpp>
#include <tensilelite/Codegen/Target.hpp>

#include <cstdint>
#include <string>

namespace rocm::tensilelite {

/// Compile-time knobs of the skinny MXFP8 GEMM, named after the constants of
/// the MLIR reference (gemm_mxfp8_skinny.mlir).
struct SkinnyGEMMConfig {
  /// BM: rows of the skinny operand handled per workgroup.
  int32_t bm = 1;
  /// TN: output columns of the wide operand per lane group.
  int32_t tn = 1;
  /// LANES_K: lanes that split the k dimension; one of 4, 8, 16 and 64.
  int32_t lanesK = 64;
  /// WAVES: wavefronts per workgroup, so a block has 64 * waves lanes.
  int32_t waves = 4;
  /// PF: prefetch ring depth; 0 means no ring.
  int32_t prefetch = 2;
  /// UP_ITERS: 0 rings the scales, otherwise the up-front scale vector size.
  int32_t upIters = 0;
  /// PARTIAL_STEPS: 0 or 1, derived by the host from k and spk: 1 when some
  /// split's K extent min(spk, k - kz * spk) is not a multiple of 64 * lanesK,
  /// so a partial tail step runs (guarded at run time by a non-empty
  /// remainder); 0 when every split's extent is a multiple, with no tail code.
  int32_t partialSteps = 0;
  /// GROUP: workgroup grouping factor of the grid swizzle.
  int32_t group = 1;
  /// NUM_XCDS: number of XCDs the workgroups are distributed over.
  int32_t numXcds = 1;
  /// GRID_3D: false selects a persistent 1-D launch.
  bool grid3d = true;
  /// USE_BETA: whether the kernel computes beta * C, see emitSkinnyGEMM.
  bool useBeta = false;
};

/// Returns the symbol of the GEMM kernel, in the reference's format:
/// gemm_BM<v>_TN<v>_LANES_K<v>_WAVES<v>_UP_ITERS<v>_PARTIAL_STEPS<v>_PF<v>_
/// GROUP<v>_NUM_XCDS<v>_GRID_3D<v>_USE_BETA<v>. Throws std::invalid_argument
/// for invalid knobs.
std::string skinnyGEMMKernelName(const SkinnyGEMMConfig &config);

/// Emits the GEMM kernel as the assembly text of one kernel, for
/// assembleToHSACO, replicating the behavior and the schedule of the MLIR
/// reference, for any upIters and partialSteps. The strides ldD, ldA, ldB and
/// ldC are BYTE strides: ldD and ldC (bf16) must be even, and ldA and ldB are
/// the fp8 element strides, which equal their byte strides. ldA and ldB must
/// also be multiples of 4 (the operand extent is padded to 4 bytes). ldD and
/// ldC need no alignment beyond being even: the kernel stores and loads vectors
/// of up to 16 B at 2-byte-aligned addresses and relies on unaligned buffer
/// access, which ROCm enables on gfx9+. The kernargs, with
/// their byte offsets, are the pointers D (0), A (8), B (16), scalesA (24),
/// scalesB (32), ws (40) and C (48), then the i32 ldD (56), ldA (60), ldB (64),
/// m (68), n (72), k (76), spk (80) and ldC (84), then the f32 alpha (88) and
/// beta (92), then the i32 numWorkGroupsX (96), numWorkGroupsY (100) and
/// numWorkGroupsZ (104), the grid dimensions that the host (or the launcher)
/// fills with the launched workgroup counts, for a segment size of 112. There
/// are no hidden arguments at any code object version. abi selects the code
/// object version (v4, v5 or v6) of the directives and the metadata, and
/// maxPreloadDwords, which preloads the leading pointers into SGPRs (the first
/// 14 dwords are D, A, B, scalesA, scalesB, ws and C): each is copied into its
/// resource and its register released right after, so the registers are
/// reusable, and with 0 the emitted kernel is unchanged. The block is
/// (64 * waves, 1, 1).
/// BN = waves * (64 / lanesK) * tn. Both values of useBeta share this ABI.
///
/// Split kz covers the K range [kz * spk, min((kz + 1) * spk, k)), in
/// iterations of 64 * lanesK counted from its own start, followed by a partial
/// tail step when partialSteps is 1. With several splits (k > spk), spk must be
/// a multiple of 256, the K group of the MX scales, as every iteration must
/// start on a scale group boundary; a single split accepts any spk >= k and any
/// k.
///
/// With several splits (k > spk) the kernel stores each split's raw f32 slice
/// into the workspace, without alpha and without C, which splitk_reduce sums
/// and scales (with beta and C), whatever useBeta is. With a single split
/// (k <= spk) the kernel stores D, the bf16 truncation (top 16 bits) of:
///
/// useBeta = false: alpha * acc. The kernel never reads C, ldC or beta, which
/// are only in the ABI so that both variants share it.
///
/// useBeta = true: fma(beta, C, alpha * acc) when beta is not zero (the bit
/// test (beta & 0x7fffffff) != 0, so -0.0 counts as zero), with C bf16 read at
/// the byte offset j * ldC + 2 * i in the orientation of D. With beta zero the
/// kernel never loads C and issues no FMA, only a uniform scalar test per tile,
/// and stores alpha * acc, so C may be uninitialized; the same holds with
/// several splits. C may alias D. The loads of C are prefetched at the start of
/// each tile into dedicated VGPRs, after the tile decode and before the
/// prologue loads of the ring, so that their latency overlaps the K loop; the
/// variant therefore holds those registers for the whole kernel, which is why
/// useBeta = false exists.
///
/// grid3d = true: the grid is (ceil(max(m, n) / BN), ceil(min(m, n) / bm),
/// ceil(k / spk)) and the block (x, y, z) is the tile (tx, ty, kz). group and
/// numXcds are ignored, as in the reference, so the kernel is the one of group
/// = 1 and numXcds = 1 under another name. Block x runs the wide tile of an
/// XCD grouping: the 32 / BN tiles that share scale lines (when BN is a power
/// of two below 32) run on one XCD, and the tail of the grid that is not a
/// multiple of 8 * (32 / BN) keeps the identity.
///
/// grid3d = false: the persistent launch with the grid (B, 1, 1) for any B >=
/// 1. Block b runs the tiles b, b + B, ... of the flattened tile space, of
/// ceil(max(m, n) / BN) * ceil(min(m, n) / bm) * ceil(k / spk) tiles, in the
/// order of the XCD remap (numXcds) and the grouped decode (group); blocks past
/// the tile count do nothing. The kernel computes the split count in the
/// kernel and reads only numWorkGroupsX.
///
/// Throws std::invalid_argument for invalid knobs and when the configuration
/// exceeds the LDS or the register budget.
std::string emitSkinnyGEMM(const codegen::Target &target,
                           const SkinnyGEMMConfig &config,
                           const codegen::KernelAbi &abi = {});

/// Emits the split-K reduction kernel splitk_reduce: with sum[i, j] the f32
/// sum over z = 0..splits-1 of ws[z * m * n + j * m + i], summed in split order
/// from 0.0, D[i, j] = the bf16 truncation (top 16 bits) of fma(beta, C[i, j],
/// alpha * sum[i, j]). ldD and ldC are BYTE strides and must be even, with no
/// other alignment (unaligned buffer access is relied on). C is
/// read, as bf16 at the byte offset j * ldC + 2 * i, only when beta is not
/// zero (-0.0 counts as zero), so it may be uninitialized otherwise; it may
/// alias D, and ldC is its own column stride. The kernargs, with their byte
/// offsets, are the pointers D (0, bf16, byte column stride ldD), ws (8, f32)
/// and C (16), then the i32 ldD (24), m (28), n (32), splits (36) and ldC
/// (40), then the f32 alpha (44) and beta (48), for a segment size of 56 (the
/// 52 bytes of arguments rounded up to 8). It has no hidden arguments and needs
/// no workgroup counts. The block is (256, 1, 1) and the grid is (ceil(m /
/// 256), n, 1); splits must be at least 1. The abi is as for emitSkinnyGEMM,
/// and so is the preloading; its 13 dwords of arguments all fit in 14, and
/// only the pointers are released after their copy.
std::string emitSplitKReduce(const codegen::Target &target,
                             const codegen::KernelAbi &abi = {});

} // namespace rocm::tensilelite

#endif // ROCM_TENSILELITE_API_SKINNYGEMM_HPP
