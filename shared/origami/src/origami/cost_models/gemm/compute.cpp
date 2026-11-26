// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi) {
  // Compute the number of Matrix Instructions required in each dim.
  size_t num_m_instrs = math::safe_ceil_div(mt.m, mi.m);
  size_t num_n_instrs = math::safe_ceil_div(mt.n, mi.n);
  size_t num_k_instrs = math::safe_ceil_div(mt.k, mi.k);

  // Total number of matrix instructions.
  size_t num_matrix_instrs = num_m_instrs * num_n_instrs * num_k_instrs;

  return num_matrix_instrs;
}

// Compute arithmic intensity
double arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

// Computes Emulated arithmetic intensity for TF32 (assumes 3xBF16).
double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 3.0 * 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 3.0 * 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

// Compute cvt overhead in x1 tf32 emulation
// TODO: We can generalize the same routine to cover more GEMMs that perform conversion
double compute_cvt_overhead_x1(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config) {
  // In X1 TF32 GEMMs, we do:
  // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
  // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
  // ds_write_b64
  // That is, the extra instructions that we need to account for are the two cvt_pk ops
  // per wavefront tile

  // However, these extra ops should not be added up to the overal tile latency becuase
  // they can be run in parallel to Matix and Memory operations (given they are not dependent).
  // So, We should ideally take L_tile = max{Mem, Comp, Vec (cvt latencies)}.
  // Since, Vec latency is not modeled yet, we somehow model that into the current logic
  // by scaling according to MFMA latencies and putting some heuristics to model the fact
  // that these vector operations can be hidden (read interleaved) with the other memory
  // or MFMA instructions.

  // --- Shorthands -----------------------------------------------------------
  const double MI_M = static_cast<double>(config.mi.m);
  const double MI_N = static_cast<double>(config.mi.n);
  const double MI_K = static_cast<double>(config.mi.k);

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);

  const dim3_t wave        = compute_wave_share(config);
  const double wave_tile_m = static_cast<double>(wave.m);
  const double wave_tile_n = static_cast<double>(wave.n);
  const double wave_tile_k = static_cast<double>(wave.k) / MI_K;

  // MFMA count
  const double N_MI     = (wave_tile_m / MI_M) * (wave_tile_n / MI_N) * wave_tile_k;
  const double num_mfma = 1.0 * N_MI;
  // Cycle scale per MI
  const double L_MI = hardware.get_mi_latency(MI_M, MI_N, MI_K, problem.mi_dtype);

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  const double bytesA = wave_tile_m * wave.k * static_cast<double>(a_bytes);
  const double bytesB = wave_tile_n * wave.k * static_cast<double>(b_bytes);

  // 3) Modeled transfer quanta (LDS->VGPR transfer width)
  //      dsA = bytesA / (LDS_XFER * MI_M)
  //      dsB = bytesB / (LDS_XFER * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  constexpr double lds_xfer = static_cast<double>(heuristic_defaults_t::LDS_XFER_BYTES);
  const double dsA          = (bytesA / lds_xfer) / MI_M;  // LDS->VGPR for A
  const double dsB          = (bytesB / lds_xfer) / MI_N;  // LDS->VGPR for B
  const double GR           = dsA;                         // Global->LDS reads
  const double LR           = dsA + dsB;                   // total DS->VGPR

  // 5) Exposed vs hidden CVT
  // spare MFMA
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  // 2 cvt per each ds_write (this for SS_BSS -- should be revised for other datatypes)
  // Each cvt occupies the SIMD for one vector-ALU instruction (4 clocks for wave64
  // on SIMD16). It is scaled by the MI Latency
  // Note: change 16.0 based on mi_data_type if we want to generalize this for all
  // casting GEMMs.
  const double cvt =
      (2.0 * static_cast<double>(kernel_valu_cycles(hardware, config)) / 16.0 * L_MI) * LR;
  // cvt ops are interleaved in main loop and don't stall matrix or memory units.
  // Heuristically, we set
  const double H        = (8.0 / 16.0 * L_MI) * spare_mfma + (4.0 / 16.0) * L_MI * (LR + GR);
  const double overhead = std::max(cvt - H, 0.0);

  return overhead;
}

// The type A has in LDS. A kernel writes A to LDS as it reads it, unless it converts it in
// between: a W4A16 kernel dequantizes its int4 A between the global read and the LDS write, so
// the tile sits in LDS (and is read from it) as the MI type. Any other quantized-A scheme
// (W4A8, ...) belongs here.
data_type_t lds_dtype_a(const problem_t& problem, const hardware_t& hardware) {
  return is_w4a16(problem, hardware) ? problem.mi_dtype : problem.a_dtype;
}

// W4A16 dequantize cost per MT_MxMT_NxMT_K iteration, in the same serial-MI cycle
// units as compute_mt_compute_latency. Every workgroup expands its MT_M x MT_K int4 A
// tile into the MI type between the global read and the LDS write, with VALU
// instructions that issue on the SIMDs that run the MIs (RDNA has no separate matrix
// pipe), so the cost adds to the MI time rather than hiding under it. Relative to the
// MI time it is 4 * I / MT_N, I being VALU instructions per dequantized element: the A
// tile is expanded once per output tile, so a wider N tile spreads it over more MI
// work. Only for W4A16 problems: the caller checks is_w4a16.
double compute_w4a16_dequant_latency(const problem_t& problem,
                                     [[maybe_unused]] const hardware_t& hardware,
                                     const config_t& config) {
  assert(is_w4a16(problem, hardware));
  const double per_elem  = problem.mi_dtype == data_type_t::BFloat16
                               ? w4a16_params_t::BF16_DEQUANT_VALU_PER_ELEM
                               : w4a16_params_t::F16_DEQUANT_VALU_PER_ELEM;
  constexpr double lanes = 32.0;  // wave32: one VALU instruction covers 32 elements
  return static_cast<double>(config.mt.m) * static_cast<double>(config.mt.k) * per_elem / lanes;
}

// Compute cvt overhead in tf32 emulation
double compute_cvt_overhead(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) {
  // Wavefront tile sizes
  const dim3_t wave        = compute_wave_share(config);
  const double wave_tile_m = static_cast<double>(wave.m);
  const double wave_tile_n = static_cast<double>(wave.n);
  const double wave_tile_k = static_cast<double>(wave.k) / static_cast<double>(config.mi.k);

  // MFMA count and cycles
  const double N_MI = (wave_tile_m / config.mi.m) * (wave_tile_n / config.mi.n) * wave_tile_k;

  // TF32 emu: 3× BF16 MI issue slots
  const double num_mfma = 3.0 * static_cast<double>(N_MI);

  // Cycle scale per MI (use BF16 MI latency as the basic timing quantum)
  const double L_MI_bf16 =
      hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, data_type_t::BFloat16);

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  auto a_bytes = data_type_to_bytes(problem.a_dtype);
  auto b_bytes = data_type_to_bytes(problem.b_dtype);

  const double bytesA = static_cast<double>(wave_tile_m) * wave.k * a_bytes;
  const double bytesB = static_cast<double>(wave_tile_n) * wave.k * b_bytes;

  // 3) Modeled transfer quanta (LDS->VGPR transfer width)
  //      dsA = bytesA / (LDS_XFER * MI_M)
  //      dsB = bytesB / (LDS_XFER * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  constexpr double lds_xfer = static_cast<double>(heuristic_defaults_t::LDS_XFER_BYTES);
  const double dsA = (bytesA / lds_xfer) / static_cast<double>(config.mi.m);  // LDS->VGPR for A
  const double dsB = (bytesB / lds_xfer) / static_cast<double>(config.mi.n);  // LDS->VGPR for B
  const double GR  = dsA;                                                     // Global->LDS reads
  const double LR  = dsA + dsB;                                               // total DS->VGPR

  // 4) Heuristic weights, scaled to MI latency (calibrated at L_MI_bf16 == 16 to
  //    A = 26 vector-ALU instructions, B = 8 and C = 4 cycles).
  // CVT per LR-sized chunk (DS->VGPR).
  const double A =
      (26.0 / 16.0) * L_MI_bf16 * static_cast<double>(kernel_valu_cycles(hardware, config));
  const double B = (8.0 / 16.0) * L_MI_bf16;  // hidden per spare MFMA slot
  const double C = (4.0 / 16.0) * L_MI_bf16;  // hidden per (LR+GR) slot

  // 5) Exposed vs hidden CVT
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  const double cvt        = A * dsA;                         // only DS->VGPR contributes CVT
  const double H          = B * spare_mfma + C * (LR + GR);  // hidden cycles
  const double overhead   = std::max(cvt - H, 0.0);

  return overhead;
}

// Determine the compute latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
size_t compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  // Compute the number of matrix instructions
  size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  // Latency of a single MT_MxMT_NxMT_k tile is the latency of one MI multiplied by
  // number of MI per MT_MxMT_NxMT_k.
  size_t L_MI = hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);

  size_t L_MT = L_MI * N_MI;

  return L_MT;
}

double compute_local_read_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  bool full_width) {
  const auto& tp    = tparams(config);
  const double wg_m = std::max(1.0, static_cast<double>(tp.wave_group_m));
  const double wg_n = std::max(1.0, static_cast<double>(tp.wave_group_n));
  const double mt_k = static_cast<double>(config.mt.k);
  // A lane's LDS access occupies a whole bank, so a read narrower than one bank
  // (e.g. a VW=1 16-bit read) still costs a full bank slot.
  constexpr double lds_bank_bytes = 4.0;

  auto operand_bytes = [&](double mt_x,
                           double wave_rereads,
                           data_type_t dtype,
                           bool direct_to_vgpr,
                           int unroll_major,
                           bool default_unroll_major,
                           bool lds_transpose_read,
                           int vector_width) {
    if (direct_to_vgpr) return 0.0;
    const double bpe   = data_type_to_bytes(dtype);
    const double bytes = wave_rereads * mt_x * mt_k * bpe;
    const bool tile_major =
        !lds_transpose_read && (unroll_major < 0 ? !default_unroll_major : unroll_major == 0);
    if (full_width || !tile_major) return bytes;
    const double read_bytes = static_cast<double>(std::max(vector_width, 1)) * bpe;
    return bytes * std::max(1.0, lds_bank_bytes / read_bytes);
  };

  // TensileLite's default TransposeLDS keeps each operand in its global layout:
  // A is K-contiguous when transposed, B when not.
  const double lds_a = operand_bytes(static_cast<double>(config.mt.m),
                                     wg_n,
                                     lds_dtype_a(problem, hardware),
                                     tp.direct_to_vgpr_a,
                                     tp.unroll_major_lds_a,
                                     problem.a_transpose == transpose_t::T,
                                     tp.lds_transpose_read_a,
                                     config.vector_width_a);
  const double lds_b = operand_bytes(static_cast<double>(config.mt.n),
                                     wg_m,
                                     problem.b_dtype,
                                     tp.direct_to_vgpr_b,
                                     tp.unroll_major_lds_b,
                                     problem.b_transpose == transpose_t::N,
                                     tp.lds_transpose_read_b,
                                     config.vector_width_b);
  return (lds_a + lds_b) /
         static_cast<double>(hardware_t::get_lds_read_bytes_per_clk(hardware.arch));
}

// LocalSplitU reduction cost: the lsu waves' partial MT_MxMT_N tiles are reduced
// through LDS (write -> barrier -> read + accumulate). Raw cycles to match
// L_compute; charged once per output tile.
double base_model::lsu_reduction(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config,
                                 const context_t& context) {
  const dim3_t& waves = context.wave_group;

  const double num_threads = static_cast<double>(waves.m * waves.n * waves.k) *
                             static_cast<double>(context.wavefront_size);
  const double tile_elems = static_cast<double>(config.mt.m) * static_cast<double>(config.mt.n);
  const double elems_per_thread = tile_elems / num_threads;

  const double svw = std::max(1.0, static_cast<double>(config.gwvw_d));
  const double bpe = std::max(1.0, static_cast<double>(data_type_to_bytes(problem.mi_dtype)));

  // Local-write cost per element (cycles), keyed on store-width x bytes-per-elem,
  // then doubled for the write pass.
  double lw_cycle;
  switch (static_cast<int>(svw * bpe)) {
    case 32: lw_cycle = 20.0 * 2.0; break;  // widest store splits into two
    case 16: lw_cycle = 20.0; break;
    case 8: lw_cycle = 12.0; break;
    default: lw_cycle = 8.0; break;
  }
  // The LDS costs were fitted for wave64 and scale with the lanes one instruction
  // moves; the adds are vector-ALU instructions.
  const double lds_lane_scale =
      static_cast<double>(context.wavefront_size) /
      static_cast<double>(hardware_t::get_wavefront_size(hardware_t::architecture_t::gfx950));
  const double local_write = elems_per_thread * lw_cycle * 2.0 * lds_lane_scale;
  const double local_read  = elems_per_thread / svw * 4.0 * 2.0 * lds_lane_scale;
  const double reduction   = elems_per_thread * static_cast<double>(waves.k - 1) *
                           static_cast<double>(context.valu_cycles);

  return local_write + local_read + reduction;
}

}  // namespace gemm
}  // namespace origami
