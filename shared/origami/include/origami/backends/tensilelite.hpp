// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "origami/core/math.hpp"

namespace origami {

/**
 * @brief Tensile/TensileLite-specific configuration parameters.
 *
 * Contains parameters specific to TensileLite-generated GEMM kernels,
 * used by the Formocast simulation model. These parameters are ignored
 * by the estimation-based prediction model.
 */
struct tensile_params_t {
  /// Depth unroll factor (0 = use mt.k)
  std::size_t depth_u = 0;

  /// Global split-K factor
  std::int16_t global_split_u = 1;

  /// GSU accumulation method (0=none, 2=MultiBuffer, 3=MultiBufferSingleKernel)
  int global_accumulation = 0;

  /// Local split-K factor
  int local_split_u = 1;

  /// DirectToVGPR flags - bypass LDS for register file
  bool direct_to_vgpr_a = false;
  bool direct_to_vgpr_b = false;

  /// DirectToLDS flags - direct global memory to LDS
  bool direct_to_lds_a = false;
  bool direct_to_lds_b = false;

  /// Number of loads that can be coalesced
  int num_loads_coalesced_a = 1;
  int num_loads_coalesced_b = 1;

  /// Number of waves per workgroup
  std::size_t wave_num = 4;

  /// Threads per wavefront the kernel was built for (0 = the architecture's default)
  int wavefront_size = 0;

  /// Wave group dimensions [wave_group_m, wave_group_n]
  int wave_group_m = 2;
  int wave_group_n = 2;

  /// Prefetch global read depth
  int prefetch_global_read = 2;

  /// Buffer load/store flags - use buffer instructions for global memory access
  bool buffer_load  = true;
  bool buffer_store = true;

  /// Math clocks per unrolled loop iteration (0 = auto-calculate)
  int math_clocks_unrolled_loop = 0;

  /// Swizzled memory layout flags
  bool swizzle_a = false;
  bool swizzle_b = false;

  /// Workgroup mapping XCC parameters
  int workgroup_mapping_xcc           = 1;
  int workgroup_mapping_xcc_group     = 0;
  bool global_split_u_coalesced       = false;
  bool global_split_u_wgm_round_robin = false;
  bool one_lds_buffer                 = false;
  bool source_swap                    = false;

  /// LDS layout per operand (TensileLite UnrollMajorLDS): 1 = K contiguous,
  /// 0 = M/N contiguous, -1 = unknown (derived from the problem's transpose using
  /// TensileLite's default TransposeLDS).
  int unroll_major_lds_a = -1;
  int unroll_major_lds_b = -1;

  /// Operand is read from LDS with transpose loads.
  bool lds_transpose_read_a = false;
  bool lds_transpose_read_b = false;

  /// TensileLite TDMInst: A/B move global-to-LDS by tensor DMA (0 = per-lane loads, 3 = both).
  int tdm_inst = 0;

  /// TensileLite PrefetchGL2: tiles the GL2 prefetch runs ahead of the global-read front.
  int prefetch_gl2 = 0;

  /// TensileLite PrefetchLocalRead. -1 = unknown (the local-read exposure term is skipped).
  int prefetch_local_read = -1;

  /// TensileLite HalfPLR bitmask: 1 = A, 2 = B, 3 = both read only half ahead.
  int half_plr = 0;

  /// Cluster peers share A/B tiles by multicast (TensileLite Multicast).
  bool multicast = false;

  /// TensileLite AdaptiveGemm: edge-tile stores pick the widest vector width the remainder
  /// allows instead of the scalar edge path.
  int adaptive_gemm = 0;

  /// TensileLite TemporalHintA / B / D as built (gfx1250 TH field: 0 RT, 1 NT, 2 HT, 3 LU on loads
  /// and WB on stores, 4..7 near/far splits).  -1 = unknown: the cache_hints_* code decides (>= 4
  /// is NT).
  int temporal_hint_a = -1;
  int temporal_hint_b = -1;
  int temporal_hint_d = -1;

  /// TensileLite TDMFuse.  0: A and B ride one tensor_load_to_lds per wave, which carries a hint
  /// only when TemporalHintA == TemporalHintB.  1-3: A and B have their own instructions and hints.
  int tdm_fuse = 0;

  /// Every parameter, in declaration order.  operator== and hash() are both defined over this
  /// list, so a new parameter is added here once.
  // clang-format off
  constexpr auto identity() const noexcept {
    return std::tie(depth_u, global_split_u, global_accumulation, local_split_u,
                    direct_to_vgpr_a, direct_to_vgpr_b, direct_to_lds_a, direct_to_lds_b,
                    num_loads_coalesced_a, num_loads_coalesced_b, wave_num, wavefront_size,
                    wave_group_m, wave_group_n, prefetch_global_read, buffer_load, buffer_store,
                    math_clocks_unrolled_loop, swizzle_a, swizzle_b, workgroup_mapping_xcc,
                    workgroup_mapping_xcc_group, global_split_u_coalesced,
                    global_split_u_wgm_round_robin, one_lds_buffer, source_swap,
                    unroll_major_lds_a, unroll_major_lds_b, lds_transpose_read_a,
                    lds_transpose_read_b, tdm_inst, prefetch_gl2, prefetch_local_read, half_plr,
                    multicast, adaptive_gemm, temporal_hint_a, temporal_hint_b, temporal_hint_d,
                    tdm_fuse);
  }
  // clang-format on

  /// Field-wise equality over every parameter.
  constexpr bool operator==(const tensile_params_t& o) const noexcept {
    return identity() == o.identity();
  }

  /// @return Hash over every parameter, in declaration order.
  std::size_t hash() const {
    return std::apply([](const auto&... field) { return math::hash_combine(field...); },
                      identity());
  }
};

}  // namespace origami
