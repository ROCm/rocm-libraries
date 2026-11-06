// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_API_SKINNYGEMM_HPP
#define ROCM_TENSILELITE_API_SKINNYGEMM_HPP

#include <tensilelite/Codegen/KernelAbi.hpp>
#include <tensilelite/Codegen/Target.hpp>

#include <cstdint>
#include <string>

namespace rocm::tensilelite {

/// Compile-time knobs of the skinny MXFP8 GEMM. With mfma the knobs change
/// meaning, as each field documents and emitSkinnyGEMM details.
struct SkinnyGEMMConfig {
  /// BM: rows of the skinny operand handled per workgroup. With mfma, the
  /// skinny columns per block, 16 or 32, padded (see mfma).
  int32_t bm = 1;
  /// TN: output columns of the wide operand per lane group. With mfma, the
  /// 16-column wide MFMA tiles per wave, so a block covers 16 * TN wide
  /// columns.
  int32_t tn = 1;
  /// LANES_K: lanes that split the k dimension; one of 4, 8, 16 and 64. With
  /// mfma it is unused and must be 64, so that kernel names stay unambiguous.
  int32_t lanesK = 64;
  /// WAVES: wavefronts per workgroup, so a block has 64 * waves lanes. With
  /// mfma, the waves also split K inside the block.
  int32_t waves = 4;
  /// PF: prefetch ring depth; 0 means no ring. With mfma, the ring depth in
  /// iterations of 256 K, at least 1.
  int32_t prefetch = 2;
  /// UP_ITERS: 0 rings the scales, otherwise the up-front scale vector size.
  /// With mfma it is unused and must be 0.
  int32_t upIters = 0;
  /// GROUP: workgroup grouping factor of the grid swizzle.
  int32_t group = 1;
  /// NUM_XCDS: number of XCDs the workgroups are distributed over.
  int32_t numXcds = 1;
  /// GRID_3D: false selects a persistent 1-D launch.
  bool grid3d = true;
  /// USE_BETA: whether the kernel computes beta * C, see emitSkinnyGEMM.
  bool useBeta = false;
  /// Selects the scaled-MFMA compute path (v_mfma_scale_f32_16x16x128_f8f6f4,
  /// gfx950), which consumes the fp8 data and the E8M0 scales directly instead
  /// of converting to bf16 and using dot2; see emitSkinnyGEMM for its contract.
  bool mfma = false;
  /// With mfma, stages the operand tiles through LDS: the loads are coalesced
  /// (8 columns of 128 contiguous bytes per instruction) and go straight to a
  /// private LDS ring of every wave, from which the MFMA operands are read;
  /// see emitSkinnyGEMM. Requires mfma.
  bool lds = false;
  /// USE_SPLIT_K: whether the kernel splits K at run time by spk and stores the
  /// raw slices of several splits into the workspace (see emitSkinnyGEMM);
  /// without it the kernel covers the whole K and never reads spk or ws.
  bool useSplitK = false;
};

/// Returns the symbol of the GEMM kernel, in the format:
/// gemm_BM<v>_TN<v>_LANES_K<v>_WAVES<v>_UP_ITERS<v>_PF<v>_GROUP<v>_NUM_XCDS<v>_
/// GRID_3D<v>_USE_BETA<v>, with the suffix _MFMA1 when
/// mfma is set, then _LDS1 when lds is set, then _USE_SPLIT_K1 when useSplitK
/// is set (the names without them are unchanged). Throws std::invalid_argument
/// for invalid knobs.
std::string skinnyGEMMKernelName(const SkinnyGEMMConfig &config);

/// Emits the GEMM kernel as the assembly text of one kernel, for
/// assembleToHSACO, for any upIters. The strides ldD, ldA, ldB
/// and ldC are BYTE strides: ldD and ldC (bf16) must be even, and ldA and ldB
/// are the fp8 element strides, which equal their byte strides. ldA and ldB
/// must also be multiples of 4 (the operand extent is padded to 4 bytes). ldD
/// and ldC need no alignment beyond being even: the kernel stores and loads
/// vectors of up to 16 B at 2-byte-aligned addresses and relies on unaligned
/// buffer access, which ROCm enables on gfx9+. The kernargs, with their byte
/// offsets, are the pointers D (0), A (8), B (16), scalesA (24), scalesB (32),
/// ws (40) and C (48), then the i32 ldD (56), ldA (60), ldB (64), m (68), n
/// (72), k (76), spk (80) and ldC (84), then the f32 alpha (88) and beta (92),
/// then the i32 numWorkGroupsX (96), the launched workgroup count that the host
/// (or the launcher) fills and only the persistent launch reads, then the i32
/// wholeTiles (100) and pieces (104) of the remainder narrowing (below), zero
/// to disable it, for a segment size of 112 (the 108 bytes of arguments rounded
/// up to 8). The ABI is unchanged and shared by every variant: spk and ws stay
/// in it, and without useSplitK the kernel ignores them (ws may be null and spk
/// any value), like C, ldC and beta without useBeta. There are no hidden
/// arguments at any code object version. abi selects the code object version
/// (v4, v5 or v6) of the directives and the metadata, and maxPreloadDwords,
/// which preloads the leading pointers into SGPRs (the first 14 dwords are D,
/// A, B, scalesA, scalesB, ws and C): each is copied into its resource and its
/// register released right after, so the registers are reusable, and with 0 the
/// emitted kernel is unchanged. The block is (64 * waves, 1, 1). BN = waves *
/// (64 / lanesK) * tn, or 16 * tn with mfma. Both values of useBeta share this
/// ABI.
///
/// Without useSplitK the kernel covers the K range [0, k), in iterations of 64
/// * lanesK (256 with mfma) followed by the partial tail step, always emitted
/// and skipped at run time when k is a multiple of the iteration K. It always
/// stores D, the bf16 (round to nearest even) of the values below, as a single
/// split does with useSplitK. With up-front scales (upIters), UP_ITERS then
/// bounds the iterations of the whole K, ceil(k / (64 * lanesK)), instead of
/// those of a split.
///
/// With useSplitK, split kz covers the K range [kz * spk, min((kz + 1) * spk,
/// k)), in iterations of 64 * lanesK counted from its own start, followed by a
/// partial tail step, always emitted and skipped at run time when the extent is
/// a multiple of the iteration K. With several splits (k > spk), spk must be
/// a multiple of 256, the K group of the MX scales, as every iteration must
/// start on a scale group boundary; a single split accepts any spk >= k and any
/// k. With several splits the kernel stores each split's raw f32 slice into the
/// workspace, without alpha and without C, which splitk_reduce sums and scales
/// (with beta and C), whatever useBeta is. With a single split (k <= spk) the
/// kernel stores D, the bf16 (round to nearest even) of:
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
/// grid3d = true: the grid is the 1-D (tilesX * tilesY, 1, 1), with tilesX =
/// ceil(max(m, n) / BN) and tilesY = ceil(min(m, n) / bm); with useSplitK it is
/// (tilesX * tilesY * S, 1, 1), with S = ceil(k / spk). The kernel recomputes
/// these from m, n (and k and spk with useSplitK) and decodes workgroup w into
/// the tile (tx, ty) with tx = w mod tilesX and ty = (w / tilesX) mod tilesY,
/// and with useSplitK also the split kz = w / (tilesX * tilesY), the order in
/// which a 3-D grid would be linearized; numWorkGroupsX is not read. group and
/// numXcds are ignored, so the kernel is the one of group = 1 and numXcds = 1
/// under another name. The wide tile tx runs with an XCD grouping: the 32 / BN
/// tiles that share scale lines (when BN is a power of two below 32) run on one
/// XCD, and the tail of the grid that is not a multiple of 8 * (32 / BN) keeps
/// the identity.
///
/// grid3d = false: the persistent launch with the grid (B, 1, 1) for any B >=
/// 1. Block b runs the tiles b, b + B, ... of the flattened tile space, of
/// ceil(max(m, n) / BN) * ceil(min(m, n) / bm) tiles, times ceil(k / spk) with
/// useSplitK, in the order of the XCD remap (numXcds) and the grouped decode
/// (group); blocks past the tile count do nothing. The kernel computes the
/// split count in the kernel and reads numWorkGroupsX as B.
///
/// mfma = true selects the scaled-MFMA compute path, with the ABI, the
/// orientations (m <= n and m > n), alpha, useBeta, useSplitK and its
/// workspace, the grid3d and persistent grids and the skinny extent of the
/// paragraphs above. The knobs then mean: bm, the skinny columns per block, 16
/// or 32 (columns past the extent are zero-filled on load and masked on store);
/// tn, the 16-column wide MFMA tiles per wave, so a block covers 16 * tn wide
/// columns; waves, the waves per block, which split K inside the block (wave w
/// takes the 256-byte K iterations w, w + waves, ...) and reduce through LDS at
/// the end; prefetch, the ring depth in iterations, at least 1; group, numXcds
/// and useBeta keep their meaning. lanesK must be 64 and upIters 0; lanesK and
/// upIters are unused, and constrained to their defaults so that kernel names
/// stay unambiguous. An iteration is 256 K, two MFMA K steps of 128. With
/// several splits (useSplitK) spk must be a multiple of 256, so only the last
/// split (or the single one) can have a partial last iteration. The split's
/// whole iterations run in the K loop. A guarded tail iteration always follows
/// it. It is run by the wave (whole iterations) mod waves, the one that the
/// round-robin gives the next iteration, and skipped by the others and by the
/// splits with no remainder. The loads of the tail are steered out of range or
/// byte-masked in registers on both operands, so the bytes at or past the end
/// of the split contribute zeros, and the scales of the K blocks that start
/// past the end are forced to 2^0, so that padding scales cannot poison the
/// products. No buffer padding is required beyond the contract of the other
/// paths.
///
/// The kernel structure with mfma: each wave holds tn wide 16-column tiles x
/// (bm / 16) skinny 16-column tiles of 16x16 f32 accumulators, fed by
/// v_mfma_scale_f32_16x16x128_f8f6f4 with the wide operand as MFMA src0 and
/// the skinny operand as src1. The column of a lane is lane % 16 and its lane
/// group lg = lane / 16 holds, as the hardware lays out the operands, the K
/// bytes [16 lg, 16 lg + 16) in VGPRs 0-3 and [64 + 16 lg, 64 + 16 lg + 16) in
/// VGPRs 4-7 of the 128 bytes of an MFMA step, so a column is read as 128
/// contiguous bytes per step with two dwordx4 loads per lane. The scales are
/// the pre-swizzled gfx950 dwords, one dword per lane per 256 K per 32 columns.
/// The lane group lg supplies the scale operand for the K block 4 * step + lg
/// of its row (the matrix core takes the scale of the K block b from the lane
/// group b), independent of which data bytes the lane holds, and the byte (row
/// half, K step) of the dword is selected by op_sel.
/// The intra-block K split reduces through LDS, (waves - 1) * tn * (bm / 16) *
/// 1024 bytes. The epilogue stores per element: bf16 D with alpha and beta, or
/// the raw f32 workspace slice with several splits (useSplitK).
///
/// lds = true (with mfma) changes how the operands reach the matrix core, not
/// what it computes, and every knob above keeps its meaning. A lane of the
/// MFMA operand layout reads 16 bytes of a column and only 4 lanes share a
/// column, so a direct load touches 16 columns x 64 bytes per instruction, a
/// pattern that the vector memory address path serves slowly. With lds every
/// wave instead loads 8 columns x 128 contiguous bytes per instruction, with
/// the buffer-to-LDS load (buffer_load_dwordx4 ... lds), into a private LDS
/// ring of its own: one slot per iteration holds, per 16-column tile (the wide
/// tiles, then the skinny ones), 4 KiB, a blob of 1 KiB per (K step, column
/// half) in lane order. The 16-byte chunk x of the 128-byte K row of a column
/// c sits at the slot x xor (c / 2) of its row, so that the operand reads of a
/// wave (ds_read_b128 of the chunks lg and 4 + lg of the column lane % 16 for
/// the lane group lg) hit distinct banks. prefetch is then the ring depth in
/// LDS: a wave's ring is prefetch slots of 4 KiB * (tn + bm / 16), and the LDS
/// of the block is the rings of its waves followed by the area of the K
/// reduction. The scale dwords still load into VGPRs. A wave only reads the
/// LDS bytes it loaded, so the K loop has no barrier: the data is visible once
/// the wave waits for the load.
/// The tail iteration is loaded after the K loop, inside the tail guard, into
/// ring slot 0 (the scales into the registers of slot 0 and the data into the
/// LDS of slot 0), as on the other path; it is then read from LDS and masked in
/// registers exactly as there.
///
/// Remainder narrowing (mfma with grid3d and a power of two tn only; the VALU
/// and the persistent kernels ignore wholeTiles and pieces). When the tiles of
/// the problem do not fill the compute units evenly, the last tiles can run on
/// idle ones: with T = tilesX * tilesY tiles, a tile being one workgroup tile
/// (tx, ty) of the grid, and C compute units, the host launches the 1-D grid
/// (W + R * S, 1, 1), for W = floor(T / C) * C, R = T mod C and S = pieces,
/// and passes wholeTiles = W and pieces = S. Workgroup w < W runs the tile t =
/// w whole, as without the narrowing; the others run, for r = w - W, the piece p
/// = r mod S of the tile t = W + r / S, with t = tx + ty * tilesX, tilesX =
/// ceilDiv(max(m, n), 16 * tn). The piece p owns the pw = 16 * tn / S
/// contiguous wide columns [tx * 16 * tn + p * pw, + pw) of the tile, over the
/// whole K, with the epilogue, alpha and beta * C of a whole tile, and stores
/// only those columns of D. Every output element is computed by the same MFMA
/// sequence over K as without the narrowing, so D is bit-identical to the one
/// of the unsplit kernel; there is no reduction, workspace or atomic. The
/// remap of the XCDs applies to tx after this decode. A piece without columns
/// (of the last tile, past the output) ends its workgroup right away. The
/// contract is that pieces is a power of two in [2, 16 * tn] (pieces < 2
/// disables the narrowing, and wholeTiles is then ignored), that tn is a power
/// of two (else the kernel ignores pieces), and a single K split: no useSplitK,
/// or spk >= k. With lds a piece of pw columns only loads the 8-column units of
/// the wide operand that it owns, pw / 8 of them or one when pw < 8, through
/// variants of the K loop selected at run time (log2(2 * tn) + 1 copies of it),
/// as an issued load costs the same when it is dead; without lds the dead
/// columns are only masked, so the narrowing gives no gain there.
///
/// With mfma every buffer must be smaller than 2 GiB (each operand, the scales,
/// D and C, and with useSplitK the workspace of splits * m * n * 4 bytes), as
/// the accesses that must be out of range use the voffset 0x80000000. The sizes
/// are runtime values, so the host must enforce this.
///
/// Throws std::invalid_argument for invalid knobs and when the configuration
/// exceeds the LDS or the register budget.
std::string emitSkinnyGEMM(const codegen::Target &target,
                           const SkinnyGEMMConfig &config,
                           const codegen::KernelAbi &abi = {});

/// Emits the split-K reduction kernel splitk_reduce: with sum[i, j] the f32
/// sum over z = 0..splits-1 of ws[z * m * n + j * m + i], summed in split order
/// from 0.0, D[i, j] = the bf16 (round to nearest even) of fma(beta, C[i, j],
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
