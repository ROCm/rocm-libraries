// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>

#include "origami/gemm.hpp"
#include "origami/simulator/tensilelite/formocast_simulator.hpp"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

double compute_formocast_latency(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config) {
  // Create Formocast simulator instance
  Formocast formocast;

  // Convert problem_t to Formocast::ProblemInfo
  Formocast::ProblemInfo prob_info;
  prob_info.M              = static_cast<double>(problem.size.m);
  prob_info.N              = static_cast<double>(problem.size.n);
  prob_info.K              = static_cast<double>(problem.size.k);
  prob_info.NumBatches     = static_cast<double>(problem.batch);
  prob_info.bpeA           = static_cast<uint32_t>(datatype_to_bits(problem.a_dtype) / 8);
  prob_info.bpeB           = static_cast<uint32_t>(datatype_to_bits(problem.b_dtype) / 8);
  prob_info.bpeD           = static_cast<uint32_t>(datatype_to_bits(problem.d_dtype) / 8);
  prob_info.bpeCompute     = static_cast<uint32_t>(datatype_to_bits(problem.mi_dtype) / 8);
  prob_info.transA         = (problem.a_transpose == transpose_t::T);
  prob_info.transB         = (problem.b_transpose == transpose_t::T);
  prob_info.swizzleTensorA = config.tensile().swizzle_a;
  prob_info.swizzleTensorB = config.tensile().swizzle_b;
  prob_info.dataType       = problem.mi_dtype;

  // Convert config_t to Formocast::SizeMapping
  Formocast::SizeMapping size_mapping;
  size_mapping.macroTile[0]         = static_cast<int>(config.mt.m);
  size_mapping.macroTile[1]         = static_cast<int>(config.mt.n);
  size_mapping.macroTile[2]         = static_cast<int>(config.mt.k);
  size_mapping.matrixInstruction[0] = static_cast<int>(config.mi.m);
  size_mapping.matrixInstruction[1] = static_cast<int>(config.mi.n);
  size_mapping.matrixInstruction[2] = static_cast<int>(config.mi.k);
  size_mapping.matrixInstruction[3] = 1;  // Default

  // Use depth_u if set, otherwise use mt.k
  size_mapping.depthU = (config.tensile().depth_u > 0) ? config.tensile().depth_u : config.mt.k;

  size_mapping.globalSplitU       = config.tensile().global_split_u;
  size_mapping.globalAccumulation = config.tensile().global_accumulation;
  size_mapping.LocalSplitU        = config.tensile().local_split_u;

  size_mapping.grvwA = config.tensile().grvw_a;
  size_mapping.grvwB = config.tensile().grvw_b;
  size_mapping.gwvwD = config.tensile().gwvw_d;
  size_mapping.gwvwC = config.tensile().gwvw_d;  // Typically same as D

  size_mapping.DirectToVgprA = config.tensile().direct_to_vgpr_a;
  size_mapping.DirectToVgprB = config.tensile().direct_to_vgpr_b;
  size_mapping.DirectToLdsA  = config.tensile().direct_to_lds_a;
  size_mapping.DirectToLdsB  = config.tensile().direct_to_lds_b;

  size_mapping.NumLoadsCoalescedA = config.tensile().num_loads_coalesced_a;
  size_mapping.NumLoadsCoalescedB = config.tensile().num_loads_coalesced_b;
  size_mapping.VectorWidthA       = config.tensile().vector_width_a;
  size_mapping.VectorWidthB       = config.tensile().vector_width_b;

  size_mapping.waveNum      = config.tensile().wave_num;
  size_mapping.waveGroup[0] = config.tensile().wave_group_m;
  size_mapping.waveGroup[1] = config.tensile().wave_group_n;

  size_mapping.workGroupMapping         = config.workgroup_mapping;
  size_mapping.workGroupMappingXCC      = config.tensile().workgroup_mapping_xcc;
  size_mapping.workGroupMappingXCCGroup = config.tensile().workgroup_mapping_xcc_group;
  size_mapping.globalSplitUCoalesced    = config.tensile().global_split_u_coalesced;
  size_mapping.globalSplitUWorkGroupMappingRoundRobin =
      config.tensile().global_split_u_wgm_round_robin;

  size_mapping.CUOccupancy            = config.occupancy;
  size_mapping.PrefetchGlobalRead     = config.tensile().prefetch_global_read;
  size_mapping.MathClocksUnrolledLoop = config.tensile().math_clocks_unrolled_loop;

  // Set problem, solution, and hardware in Formocast
  formocast.setProblem(prob_info);
  formocast.setSolution(size_mapping);
  formocast.setHardware(hardware.arch);

  // Get predicted performance
  Formocast::PredictedPerformance perf = formocast.predictedPerformance();

  // Return latency in microseconds
  return perf.microSeconds;
}

}  // namespace gemm
}  // namespace origami
