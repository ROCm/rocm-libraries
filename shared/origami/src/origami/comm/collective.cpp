/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "origami/comm/collective.hpp"

namespace origami::comm {

double ring_step_overhead_cycles(primitive_t primitive,
                                 const collective_algorithm_t& algorithm,
                                 const heuristics_t& heur,
                                 const comm_hardware_t& fabric) {
  if (!algorithm.is_ring_class()) return 0.0;
  // The heuristic is host wall time (ns); convert to cycles at the actual GPU
  // clock. clock_ghz is cycles-per-ns, so cycles = ns × clock_ghz exactly.
  const double per_step_ns = heur.ring_step_overhead_ns_for(primitive);
  const double total_ns    = per_step_ns * static_cast<double>(algorithm.num_timesteps());
  return total_ns * fabric.clock_ghz;
}

double compute_ring_latency(const collective_algorithm_t& algorithm,
                            const comm_problem_t& problem,
                            const comm_config_t& config,
                            const system_t& system,
                            int my_rank,
                            const heuristics_t& heur) {
  const hardware_t& hw           = system.gpu;
  const comm_hardware_t& comm_hw = system.fabric;
  const int num_timesteps        = algorithm.num_timesteps();
  const std::size_t CL           = hw.cacheline_bytes;

  // A ring moves one chunk per step; over num_timesteps steps each GPU pushes
  // num_timesteps such chunks across its outgoing link. That product is the
  // total bytes this rank puts on the wire — the numerator of the throughput
  // model.
  const std::size_t gpu_timestep_tile_bytes =
      problem.gpu_tile_cachelines(hw.cacheline_bytes) * CL /
      static_cast<std::size_t>(algorithm.chunks_per_timestep());
  const std::size_t total_wire_bytes =
      gpu_timestep_tile_bytes * static_cast<std::size_t>(num_timesteps);

  const int eff_wgs = config.effective_num_wgs(gpu_timestep_tile_bytes);

  // Per-WG remote throughput is the same latency cap as in latency.hpp:
  // outstanding misses (mshr_depth × waves × CL) drained every RTT.
  const double mshr_bw_per_wg =
      (static_cast<double>(hw.mshr_depth_per_wave) * hw.waves_per_wg * static_cast<double>(CL)) /
      hw.xgmi_latency_cycles;

  // The ring's sustainable rate is whichever ceiling binds first: the physical
  // link width, or the combined latency-limited throughput of the WGs feeding
  // it (eff_wgs × per-WG cap). Few WGs -> CU-limited; many WGs -> link-limited.
  const double aggregate_bw =
      std::min(comm_hw.link_bw, static_cast<double>(eff_wgs) * mshr_bw_per_wg);

  const double T_transfer = static_cast<double>(total_wire_bytes) / aggregate_bw;

  // Sync: count signal_t+wait_t ops in the work graph times atomic_latency_cycles.
  const auto entry = algorithm.link_of(/*pid=*/0, /*timestep=*/0, my_rank);
  int sync_ops     = 0;
  for (const auto& op : entry.work_graph) {
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, signal_t> || std::is_same_v<T, wait_t>) { ++sync_ops; }
        },
        op);
  }
  // Each ring step needs its own handshake, and the steps are serialized by
  // the dependency chain, so sync cost accrues per step.
  const double T_sync_per_step = static_cast<double>(sync_ops) * comm_hw.atomic_latency_cycles;
  const double T_sync_total    = static_cast<double>(num_timesteps) * T_sync_per_step;

  // The same bytes that cross the fabric must also be read from / written to
  // local HBM, which has its own (CU-count-scaled) aggregate ceiling. The ring
  // can be bound by either resource, so take the slower of fabric and HBM.
  const double hbm_bw_agg = hw.hbm_read_bw * hw.bw_fraction(eff_wgs);
  const double T_hbm      = static_cast<double>(total_wire_bytes) / hbm_bw_agg;

  const double T_transfer_total = std::max(T_transfer, T_hbm);

  // Per-step proxy/handshake overhead the bandwidth model cannot see (CPU-
  // mediated; empirical, from heuristics). Keyed by the collective being run.
  const double T_step_overhead =
      ring_step_overhead_cycles(problem.collective, algorithm, heur, comm_hw);

  // Fixed launch floor + the throughput-bound transfer + serial sync + per-step
  // overhead. Launch dominates tiny messages; transfer dominates large ones.
  return comm_hw.launch_overhead_cycles + T_transfer_total + T_sync_total + T_step_overhead;
}

double compute_sequential_latency(const collective_algorithm_t& algorithm,
                                  const comm_problem_t& problem,
                                  const comm_config_t& config,
                                  const system_t& system,
                                  int my_rank,
                                  const heuristics_t& heur) {
  const hardware_t& hw           = system.gpu;
  const comm_hardware_t& comm_hw = system.fabric;
  const int chunks_per_timestep  = algorithm.chunks_per_timestep();
  // The collective being implemented; threaded down to the per-tile heuristics
  // (k_xgmi_write / ring_step_overhead) which take only the primitive, not the
  // full problem.
  const primitive_t primitive = problem.collective;

  const tile_shape_t gpu_tile = problem.gpu_tile_shape();
  const tile_shape_t gpu_timestep_tile =
      gpu_tile.divide_byte_equal(static_cast<std::size_t>(chunks_per_timestep));

  const int eff_wgs = config.effective_num_wgs(gpu_timestep_tile.bytes());

  const tile_shape_t wg_tile =
      gpu_timestep_tile.divide_byte_equal(static_cast<std::size_t>(eff_wgs));
  const wg_tile_geometry_t wg_geometry =
      wg_tile_geometry_t::from_shape(wg_tile, hw.cacheline_bytes);
  const latency_context_t lat_ctx{config, system, heur, primitive};

  // This loop is the iterative driver of the schedule: the algorithm methods are
  // closed-form (one timestep per call), and stepping through the timeline happens
  // here, not inside them. Timesteps are data-dependent, so their latencies add up.
  // Within a timestep, however, the links run in parallel — so a timestep costs the
  // *slowest* link, not their sum (T_link_max below).
  double T_timesteps = 0.0;
  for (int timestep = 0; timestep < algorithm.num_timesteps(); ++timestep) {
    const auto entry = algorithm.link_of(/*pid=*/0, timestep, my_rank);

    if (entry.is_self) {
      // A "self" step is a local copy (no peer): bound by local HBM, so the
      // per-WG budget is the HBM per-CU share rather than a link share.
      const double bw_per_wg = hw.hbm_read_bw_per_cu(eff_wgs);
      const auto breakdown   = compute_wg_tile_latency(entry.work_graph,
                                                       wg_geometry,
                                                       bw_per_wg,
                                                       /*active_cus=*/eff_wgs,
                                                       lat_ctx);
      T_timesteps += breakdown.T_total_cycles;
    } else {
      // A remote step may light up several links at once; wgs_per_active_link
      // reports how the eff_wgs workgroups are distributed over them (one count
      // per active link). Each link's WGs share that link's width evenly, and the
      // timestep waits for the most congested link to finish — hence the max over
      // links.
      const auto link_wg_counts = algorithm.wgs_per_active_link(timestep, eff_wgs);

      double T_link_max = 0.0;
      for (const int wgs_on_link : link_wg_counts) {
        const double bw_per_wg = comm_hw.link_bw / static_cast<double>(std::max(wgs_on_link, 1));

        const auto breakdown = compute_wg_tile_latency(entry.work_graph,
                                                       wg_geometry,
                                                       bw_per_wg,
                                                       /*active_cus=*/eff_wgs,
                                                       lat_ctx);
        T_link_max           = std::max(T_link_max, breakdown.T_total_cycles);
      }
      T_timesteps += T_link_max;
    }
  }

  const double T_step_overhead = ring_step_overhead_cycles(primitive, algorithm, heur, comm_hw);
  return comm_hw.launch_overhead_cycles + T_timesteps + T_step_overhead;
}

double compute_collective_latency_for_rank(const comm_problem_t& problem,
                                           const comm_config_t& config,
                                           const system_t& system,
                                           int my_rank,
                                           const heuristics_t& heur) {
  std::unique_ptr<collective_algorithm_t> owned;
  const collective_algorithm_t* A = config.algorithm_override;
  if (!A) {
    owned = resolve_algorithm(problem, config);
    A     = owned.get();
  }

  if (A->is_ring_pipeline()) {
    return compute_ring_latency(*A, problem, config, system, my_rank, heur);
  }
  return compute_sequential_latency(*A, problem, config, system, my_rank, heur);
}

double compute_collective_latency(const comm_problem_t& problem,
                                  const comm_config_t& config,
                                  const system_t& system,
                                  const heuristics_t& heur) {
  if (heur.assume_rank_symmetry) {
    return compute_collective_latency_for_rank(problem, config, system, /*my_rank=*/0, heur);
  }

  double T_max = 0.0;
  for (int rank = 0; rank < problem.num_gpus; ++rank) {
    T_max =
        std::max(T_max, compute_collective_latency_for_rank(problem, config, system, rank, heur));
  }
  return T_max;
}

double predict_row(std::string_view primitive,
                   std::size_t msg_bytes,
                   int world_size,
                   int nchannels,
                   const system_t& system,
                   std::size_t M,
                   std::size_t N,
                   int split_dim,
                   const heuristics_t& heur) {
  constexpr data_type_t dtype = data_type_t::BFloat16;

  if (M == 0 || N == 0) {
    const std::size_t total_elements = msg_bytes / static_cast<std::size_t>(dtype_bytes(dtype));
    std::size_t per_rank_elements;
    if (primitive == "reduce_scatter") {
      per_rank_elements =
          std::max<std::size_t>(total_elements / static_cast<std::size_t>(world_size), 1);
    } else {
      per_rank_elements = total_elements;
    }
    M = 1;
    N = per_rank_elements;
  }

  comm_problem_t problem{M, N, world_size, dtype, split_dim};
  problem.collective = primitive_from_name(primitive);  // string → enum at the edge
  comm_config_t config{};
  config.num_wgs          = nchannels;
  config.load_width       = load_width_t::DWORDX16;
  config.vgprs_for_data   = 128;
  config.min_bytes_per_wg = heur.min_bytes_per_wg;

  const double T_cycles = compute_collective_latency(problem, config, system, heur);

  return system.gpu.cycles_to_us(T_cycles);
}

}  // namespace origami::comm
