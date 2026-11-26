// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// The other split groups pass their f32 partial tiles through LDS at its read bandwidth (write,
// barrier, read), and one group adds them with one vector-ALU instruction per element.  The sum
// is scaled by the fit to the ranking of measured candidates.
double cdna5_model::lsu_reduction(const problem_t&,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  const context_t& context) {
  const cdna5_fit_t& fit = cdna5_model::fit(hardware);
  const dim3_t& waves    = context.wave_group;

  const double partial_bytes = static_cast<double>(waves.k - 1) * static_cast<double>(config.mt.m) *
                               static_cast<double>(config.mt.n) *
                               heuristic_defaults_t::EPILOGUE_WORKSPACE_BYTES_PER_ELEM;
  const double lanes =
      static_cast<double>(waves.m * waves.n) * static_cast<double>(context.wavefront_size);
  return (2.0 * partial_bytes /
              static_cast<double>(hardware_t::get_lds_read_bytes_per_clk(hardware.arch)) +
          heuristic_defaults_t::L_BARRIER +
          static_cast<double>(config.mt.m * config.mt.n) * static_cast<double>(waves.k - 1) /
              lanes * static_cast<double>(context.valu_cycles)) *
         fit.LSU_REDUCTION_SCALE;
}

}  // namespace gemm
}  // namespace origami
