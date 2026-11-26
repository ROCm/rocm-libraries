// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <cstdint>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace {

namespace cms_speedup {
constexpr uint8_t X100 = 100;
constexpr uint8_t X105 = 105;
constexpr uint8_t X110 = 110;
constexpr uint8_t X115 = 115;
constexpr uint8_t X120 = 120;
constexpr uint8_t X123 = 123;
constexpr uint8_t X126 = 126;
}  // namespace cms_speedup

constexpr double cms_efficiency(uint8_t speedup_x100) {
  return 100.0 / static_cast<double>(speedup_x100);
}

struct cms_kernel_entry_t {
  origami::data_type_t mi_dtype;
  origami::transpose_t trans_a;
  origami::transpose_t trans_b;
  uint16_t m;
  uint16_t n;
  uint16_t k;
  uint8_t speedup_x100;
};

// Hand-optimized (CMS) kernels of the gfx950 library and their measured speedups.
constexpr std::array<cms_kernel_entry_t, 37> gfx950_cms_kernels = {{
    // BF16 NT
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::T,
     160,
     256,
     64,
     cms_speedup::X120},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::T,
     192,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::T,
     208,
     256,
     64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::T,
     256,
     160,
     64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::T,
     256,
     192,
     64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::T,
     256,
     256,
     64,
     cms_speedup::X115},
    // BF16 NN
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::N,
     160,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::N,
     208,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::N,
     origami::transpose_t::N,
     256,
     256,
     64,
     cms_speedup::X120},
    // BF16 TN
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::N,
     160,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::N,
     192,
     256,
     64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     96,
     64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     192,
     64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     224,
     64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     256,
     64,
     cms_speedup::X115},
    // BF16 TT
    {origami::data_type_t::BFloat16,
     origami::transpose_t::T,
     origami::transpose_t::T,
     256,
     256,
     64,
     cms_speedup::X110},
    // FP16 NT
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::T,
     192,
     320,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::T,
     208,
     256,
     64,
     cms_speedup::X120},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::T,
     224,
     128,
     64,
     cms_speedup::X120},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::T,
     256,
     192,
     64,
     cms_speedup::X120},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::T,
     256,
     256,
     64,
     cms_speedup::X115},
    // FP16 NN
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::N,
     128,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::N,
     160,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::N,
     256,
     160,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::N,
     192,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::N,
     256,
     192,
     64,
     cms_speedup::X100},
    {origami::data_type_t::Half,
     origami::transpose_t::N,
     origami::transpose_t::N,
     256,
     256,
     64,
     cms_speedup::X105},
    // FP16 TN
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::N,
     160,
     256,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::N,
     192,
     256,
     64,
     cms_speedup::X105},
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     96,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     192,
     64,
     cms_speedup::X110},
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     224,
     64,
     cms_speedup::X105},
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::N,
     256,
     256,
     64,
     cms_speedup::X105},
    // FP16 TT
    {origami::data_type_t::Half,
     origami::transpose_t::T,
     origami::transpose_t::T,
     256,
     256,
     64,
     cms_speedup::X110},
    // TF32 NN
    {origami::data_type_t::XFloat32,
     origami::transpose_t::N,
     origami::transpose_t::N,
     192,
     256,
     32,
     cms_speedup::X123},
    // TF32 TN
    {origami::data_type_t::XFloat32,
     origami::transpose_t::T,
     origami::transpose_t::N,
     128,
     256,
     32,
     cms_speedup::X126},
    {origami::data_type_t::XFloat32,
     origami::transpose_t::T,
     origami::transpose_t::N,
     192,
     256,
     32,
     cms_speedup::X123},
}};

}  // namespace

namespace origami {
namespace gemm {

size_t cdna4_model::hand_optimized_kernel_count() { return gfx950_cms_kernels.size(); }

void cdna4_model::register_heuristics(heuristics_database_t& db) {
  for (const auto& entry : gfx950_cms_kernels) {
    db.add_hand_optimized_efficiency(hand_optimized_kernel_key_t{hardware_t::architecture_t::gfx950,
                                                                 entry.mi_dtype,
                                                                 entry.trans_a,
                                                                 entry.trans_b,
                                                                 entry.m,
                                                                 entry.n,
                                                                 entry.k},
                                     cms_efficiency(entry.speedup_x100));
  }
}

// TF32 emulation: the hand-optimized 256x256x32 kernels beat the model by a factor that depends
// on the problem's arithmetic intensity, so it cannot be precomputed in the database.
void cdna4_model::adjust_heuristics(heuristic_params_t& params,
                                    const problem_t& problem,
                                    const hardware_t& hardware,
                                    const config_t& config) {
  if (problem.mi_dtype != data_type_t::XFloat32) return;

  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);

  // Compute arithmetic intensity for this specific problem
  double arith     = emulated_tf32_arithmetic_intensity(M, N, K, static_cast<double>(a_bytes));
  double threshold = constants(hardware).TF32_ARITH_INTENSITY_THRESHOLD;

  // Custom kernel optimizations based on transpose mode and tile config
  // NT: N-transpose configuration
  if ((!a_trans && b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
    if (arith < threshold) {
      params.weight_tile_total *= 0.6;
    } else {
      params.weight_tile_total *= 0.4;
    }
  }

  // NN: No-transpose configuration
  if ((!a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
    if (arith < threshold) {
      params.weight_tile_total *= 0.8;
    } else {
      params.weight_tile_total *= 0.4;
    }
  }

  // TN: Transpose-A configuration
  if ((a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
    if (arith < threshold) {
      params.weight_tile_total *= 0.8;
    } else {
      params.weight_tile_total *= 0.4;
    }
  }
}

}  // namespace gemm
}  // namespace origami
