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

// origami::comm — analytical communication cost model
//
// hardware_t constants — the bottom of the cost model: the physical ceilings
// every higher layer divides work by.
//
// Why everything is in cycles, not seconds:
//   The model reasons about contention between functional units (VMEM issue,
//   TCP, L2, MALL, HBM, xGMI, VALU) that all advance on the same GPU clock.
//   Expressing every rate as bytes-per-cycle and every latency as cycles lets
//   the bottleneck comparison in latency.hpp be a plain max() of like units,
//   and makes the model clock-portable: retargeting to a different frequency
//   (overclock studies, a future part) only changes clock_ghz, not the
//   per-cycle physics. The single cycles→seconds conversion happens once, at
//   the public API boundary (predict_row in collective.hpp).
//
// Unit identity used throughout: a rate quoted in GB/s equals
//   (GB/s) / clock_ghz  bytes-per-cycle,
// because bytes/cycle = (bytes/ns) / (cycles/ns) = (GB/s) / clock_ghz. That is
// why make_system() converts each calibrated peak aggregate rate (held in
// arch_ceilings_t in native GB/s) to bytes/cycle as "<GB/s> / clock", and each
// native-ns latency to cycles as "<ns> * clock".
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include "origami/architecture.hpp"

namespace origami::comm {

// ─── hardware_t (per-CU and per-XCD compute / memory) ──────────────
/**
 * @brief Per-CU and per-XCD compute and memory ceilings for one GPU.
 *
 * The physical floor of the cost model: every higher layer divides its work by
 * these throughput ceilings and latencies. All bandwidths are expressed in
 * bytes-per-cycle and all latencies in cycles (see the unit identity at the top
 * of this file) so the bottleneck comparison in latency.hpp is a max() of like
 * units.
 */
struct hardware_t {
  /// GPU architecture identity. Reuses the canonical origami::architecture_t
  /// enum (from the HIP-free origami/architecture.hpp) rather than a private
  /// string label, so "which GPU" has a single source of truth across the GEMM
  /// and comm models.
  architecture_t arch;

  // Die structure
  int num_cu;        ///< Total compute units across the whole device.
  int num_xcd;       ///< Number of accelerator complex dies (XCDs).
  int cu_per_xcd;    ///< Compute units per XCD (num_cu / num_xcd).
  double clock_ghz;  ///< cycles per ns

  // Per-CU throughput ceilings
  double vmem_issue_rate;  ///< VMEM instructions per CU per cycle
  double valu_rate;        ///< VALU lane-elements per CU per cycle

  // TCP / vL1D (per CU)
  std::size_t tcp_capacity_bytes;  ///< Per-CU vL1D/TCP cache capacity in bytes.
  double tcp_bw;                   ///< bytes per CU per cycle

  // Outstanding request limits
  int mshr_depth_per_wave;     ///< Outstanding misses a wave issues before it stalls.
  int waves_per_wg;            ///< Waves co-resident per workgroup feeding misses.
  double xgmi_latency_cycles;  ///< round-trip latency for remote load

  // L2 / TCC (per XCD)
  std::size_t l2_capacity_bytes;  ///< Per-XCD L2/TCC capacity in bytes.
  double l2_bw_per_cu;            ///< bytes per CU per cycle

  // MALL / Infinity Cache (device-wide)
  std::size_t mall_capacity_bytes;  ///< Device-wide MALL/Infinity Cache capacity in bytes.
  double mall_bw;                   ///< bytes per cycle aggregate

  // HBM
  double hbm_read_bw;              ///< bytes per cycle aggregate
  double hbm_write_bw;             ///< bytes per cycle aggregate
  std::size_t hbm_capacity_bytes;  ///< Total HBM capacity in bytes.

  /// HBM bandwidth-utilization polynomial: fraction = a*N^2 + b*N + c of peak,
  /// clamped to [0,1], where N is the number of CUs concurrently streaming.
  ///
  /// First principle: a single CU cannot saturate HBM. Peak HBM bandwidth is
  /// only reached once enough independent CUs issue in parallel to keep every
  /// channel/arbiter busy. The fabric arbitrates per-request with no per-CU
  /// isolation, so sustained bandwidth ramps roughly linearly with active CUs
  /// until the channels saturate. The default {0,0.015,0} encodes
  /// fraction ≈ 0.015·N, i.e. each CU contributes ~1.5% of peak and the array
  /// reaches full peak near N≈67 CUs. (This linear fit is calibrated for the
  /// many-CU collective regime; it knowingly understates single-CU sustained
  /// bandwidth — see day2-atom-validation — which is irrelevant once dozens of
  /// workgroups stream a collective.)
  std::array<double, 3> mem_bw_coeffs = {0.0, 0.015, 0.0};

  /// Cache-line size in bytes: the granularity at which TCP/L2/MALL/HBM and the
  /// xGMI fabric tag, fetch, and evict data. The model rounds all traffic up to
  /// whole lines because that is what the silicon actually moves. This is a
  /// per-architecture property (64 B on CDNA), not a global constant, so a
  /// future GPU with a different line size only edits its hardware table entry.
  std::size_t cacheline_bytes = 64;

  // ── BW polynomial ──────────────────────────────────────────
  /**
   * @brief Fraction of peak HBM bandwidth sustainable with `active_cus` streaming.
   *
   * Evaluates the utilization polynomial a*N^2 + b*N + c (N = active CUs) and
   * clamps the result to [0, 1]. See @ref mem_bw_coeffs for the physical
   * rationale behind the coefficients.
   *
   * @param active_cus Number of CUs concurrently streaming HBM.
   * @return double Sustainable fraction of peak bandwidth in [0, 1].
   */
  constexpr double bw_fraction(int active_cus) const noexcept {
    const double N = static_cast<double>(active_cus);
    const double f = mem_bw_coeffs[0] * N * N + mem_bw_coeffs[1] * N + mem_bw_coeffs[2];
    return std::clamp(f, 0.0, 1.0);
  }

  /**
   * @brief Per-CU share of HBM read bandwidth at a given active-CU count.
   *
   * Per-CU share of HBM read bandwidth: take the aggregate ceiling, discount
   * it by how much of peak N active CUs can actually sustain (bw_fraction),
   * then split that sustained aggregate evenly across the N contenders. With
   * the linear polynomial the two N's partly cancel, so per-CU bandwidth is
   * nearly flat across the saturated range — the realistic behaviour of a
   * shared memory system once it is busy.
   *
   * @param active_cus Number of CUs concurrently streaming (-1 uses num_cu).
   * @return double Per-CU read bandwidth in bytes per cycle.
   */
  constexpr double hbm_read_bw_per_cu(int active_cus = -1) const noexcept {
    const int n    = (active_cus < 0) ? num_cu : active_cus;
    const double f = bw_fraction(n);
    return hbm_read_bw * f / static_cast<double>(n);
  }

  /**
   * @brief Per-CU share of HBM write bandwidth at a given active-CU count.
   *
   * The write-path analogue of @ref hbm_read_bw_per_cu: the aggregate write
   * ceiling is discounted by bw_fraction and split evenly across the N active
   * contenders.
   *
   * @param active_cus Number of CUs concurrently streaming (-1 uses num_cu).
   * @return double Per-CU write bandwidth in bytes per cycle.
   */
  constexpr double hbm_write_bw_per_cu(int active_cus = -1) const noexcept {
    const int n    = (active_cus < 0) ? num_cu : active_cus;
    const double f = bw_fraction(n);
    return hbm_write_bw * f / static_cast<double>(n);
  }

  /**
   * @brief Per-CU L2 bandwidth, scaled for partial XCD occupancy.
   *
   * Per-CU L2 bandwidth, scaled for partial XCD occupancy. l2_bw_per_cu is
   * calibrated for a fully-occupied XCD (all cu_per_xcd CUs sharing the TCC
   * crossbar). When only n < cu_per_xcd CUs are active, the same crossbar
   * bandwidth is shared among fewer consumers, so each active CU gets a larger
   * slice — hence the (cu_per_xcd / n) upscaling. This is the inverse of the
   * HBM polynomial: HBM under-delivers when under-subscribed, whereas the L2
   * crossbar is per-CU-bounded so an idle CU's share is reclaimed.
   *
   * @param active_cus_on_xcd CUs active on the XCD (-1 uses cu_per_xcd).
   * @return double Per-CU L2 bandwidth in bytes per cycle.
   */
  constexpr double l2_bw_per_cu_scaled(int active_cus_on_xcd = -1) const noexcept {
    int n = (active_cus_on_xcd < 0) ? cu_per_xcd : active_cus_on_xcd;
    n     = std::min(n, cu_per_xcd);
    return l2_bw_per_cu * (static_cast<double>(cu_per_xcd) / std::max(n, 1));
  }

  // ── Frequency / cycle ↔ time conversion ────────────────────
  /**
   * @brief Clock frequency in hertz.
   *
   * @return double Clock frequency in Hz (clock_ghz * 1e9).
   */
  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  /**
   * @brief Convert a duration in cycles to seconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in seconds.
   */
  constexpr double cycles_to_seconds(double cycles) const noexcept { return cycles / clock_hz(); }
  /**
   * @brief Convert a duration in cycles to nanoseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in nanoseconds.
   */
  constexpr double cycles_to_ns(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e9;
  }
  /**
   * @brief Convert a duration in cycles to microseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in microseconds.
   */
  constexpr double cycles_to_us(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e6;
  }
  /**
   * @brief Convert a duration in seconds to cycles.
   *
   * @param s Duration in seconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double seconds_to_cycles(double s) const noexcept { return s * clock_hz(); }
  /**
   * @brief Convert a duration in nanoseconds to cycles.
   *
   * @param ns Duration in nanoseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double ns_to_cycles(double ns) const noexcept { return seconds_to_cycles(ns * 1e-9); }
  /**
   * @brief Convert a duration in microseconds to cycles.
   *
   * @param us Duration in microseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double us_to_cycles(double us) const noexcept { return seconds_to_cycles(us * 1e-6); }

  /**
   * @brief Convert a per-cycle rate to a per-second rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per second.
   */
  constexpr double rate_per_second(double per_cycle) const noexcept {
    return per_cycle * clock_hz();
  }
  /**
   * @brief Convert a per-cycle rate to a per-nanosecond rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per nanosecond.
   */
  constexpr double rate_per_ns(double per_cycle) const noexcept {
    return rate_per_second(per_cycle) * 1e-9;
  }

  /**
   * @brief Convert a per-second rate to a per-cycle rate.
   *
   * @param per_s Rate expressed per second.
   * @return double Equivalent rate per clock cycle.
   */
  constexpr double rate_per_cycle_from_per_second(double per_s) const noexcept {
    return per_s / clock_hz();
  }
  /**
   * @brief Convert a per-nanosecond rate to a per-cycle rate.
   *
   * @param per_ns Rate expressed per nanosecond.
   * @return double Equivalent rate per clock cycle.
   */
  constexpr double rate_per_cycle_from_per_ns(double per_ns) const noexcept {
    return rate_per_cycle_from_per_second(per_ns * 1e9);
  }
};

// ─── comm_hardware_t (inter-GPU communication) ──────────────────────
/**
 * @brief Inter-GPU fabric ceilings: xGMI links, SDMA engines, and protocol overheads.
 *
 * Captures everything the cost model needs about the communication fabric that
 * joins one GPU to its peers. As with @ref hardware_t, bandwidths are in
 * bytes-per-cycle and latencies in cycles.
 */
struct comm_hardware_t {
  // xGMI link
  double link_bw;      ///< bytes per cycle per link, unidirectional
  int num_peer_links;  ///< links to other GPUs

  // SDMA
  int num_sdma_engines;  ///< Number of SDMA (DMA-copy) engines.
  double sdma_read_bw;   ///< bytes per cycle per link
  double sdma_write_bw;  ///< bytes per cycle per link

  // Protocol overhead
  double atomic_latency_cycles;   ///< Latency of one fabric atomic (signal/wait), in cycles.
  double launch_overhead_cycles;  ///< Fixed per-collective kernel launch/setup floor, in cycles.

  double clock_ghz = 2.0;  ///< companion clock for ns display helpers

  /**
   * @brief Clock frequency in hertz.
   *
   * @return double Clock frequency in Hz (clock_ghz * 1e9).
   */
  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  /**
   * @brief Convert a duration in cycles to seconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in seconds.
   */
  constexpr double cycles_to_seconds(double cycles) const noexcept { return cycles / clock_hz(); }
  /**
   * @brief Convert a duration in cycles to nanoseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in nanoseconds.
   */
  constexpr double cycles_to_ns(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e9;
  }
  /**
   * @brief Convert a duration in cycles to microseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in microseconds.
   */
  constexpr double cycles_to_us(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e6;
  }
  /**
   * @brief Convert a duration in seconds to cycles.
   *
   * @param s Duration in seconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double seconds_to_cycles(double s) const noexcept { return s * clock_hz(); }
  /**
   * @brief Convert a duration in nanoseconds to cycles.
   *
   * @param ns Duration in nanoseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double ns_to_cycles(double ns) const noexcept { return seconds_to_cycles(ns * 1e-9); }
  /**
   * @brief Convert a duration in microseconds to cycles.
   *
   * @param us Duration in microseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double us_to_cycles(double us) const noexcept { return seconds_to_cycles(us * 1e-6); }

  /**
   * @brief Convert a per-cycle rate to a per-second rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per second.
   */
  constexpr double rate_per_second(double per_cycle) const noexcept {
    return per_cycle * clock_hz();
  }
  /**
   * @brief Convert a per-cycle rate to a per-nanosecond rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per nanosecond.
   */
  constexpr double rate_per_ns(double per_cycle) const noexcept {
    return rate_per_second(per_cycle) * 1e-9;
  }
  /**
   * @brief Convert a per-nanosecond rate to a per-cycle rate.
   *
   * @param per_ns Rate expressed per nanosecond.
   * @return double Equivalent rate per clock cycle.
   */
  constexpr double rate_per_cycle_from_per_ns(double per_ns) const noexcept {
    return per_ns * 1e9 / clock_hz();
  }
};

// ─── system_t (the physical machine) ────────────────────────────────
/**
 * @brief The physical machine: a GPU plus the fabric that joins it to its peers.
 *
 * A GPU plus the fabric that joins it to its peers. The model is currently
 * homogeneous — every rank is assumed to be `gpu` — so a single hardware_t
 * suffices rather than one per GPU; the GPU *count* is the communicator size
 * and lives in comm_problem_t::num_gpus, not here. If heterogeneous nodes or a
 * non-uniform link topology ever need modelling, `gpu` grows into a per-rank
 * container and `fabric` into a bandwidth matrix; until then this stays the
 * minimal honest representation of what the cost model actually consumes.
 */
struct system_t {
  hardware_t gpu;          ///< one GPU's compute/memory ceilings
  comm_hardware_t fabric;  ///< the xGMI mesh between GPUs
};

// ─── gpu_topology_t (live, per-device shape) ────────────────────────
/**
 * @brief Per-device GPU topology: the part of the machine description that comes
 *        from the actual chip in the box, not from calibration.
 *
 * These are the quantities a runtime query (hipDeviceProp_t, via
 * origami::hardware_t) reports and that differ between otherwise-identical
 * silicon — most importantly the CU/XCD counts under CPX-style partitioning,
 * where the same gfx942 part exposes fewer CUs and XCDs. Pairing this with the
 * calibrated @ref arch_ceilings_t lets @ref make_system build a system_t for the
 * device actually about to run, instead of a hardcoded nominal one.
 *
 * @see origami::comm::system_from_device / system_from_hardware
 *      (origami/comm/hardware_device.hpp).
 */
struct gpu_topology_t {
  architecture_t arch;            ///< Architecture identity (keys the ceilings table).
  std::size_t num_cu;             ///< Total compute units exposed by this device/partition.
  std::size_t num_xcd;            ///< Number of XCDs exposed by this device/partition.
  std::size_t cu_per_xcd;         ///< Compute units per XCD (num_cu / num_xcd).
  std::size_t l2_capacity_bytes;  ///< Per-XCD L2/TCC capacity in bytes.
};

// ─── arch_ceilings_t (calibrated, per-architecture) ─────────────────
/**
 * @brief Per-architecture calibrated ceilings, in native units (GB/s, ns,
 *        bytes), independent of clock and of device partitioning.
 *
 * The communication analogue of origami::architecture_constants: the empirical
 * data origami owns for an architecture, separated from the live topology. It is
 * stored in *native* units — bandwidths in GB/s, latencies in nanoseconds — so
 * the table reads in the units the microbenchmarks report and carries no clock
 * assumption. @ref make_system converts these to the model's bytes-per-cycle and
 * cycle units at build time using the target clock (see the unit identity at the
 * top of this file): a GB/s rate becomes (GB/s)/clock bytes/cycle, and an ns
 * latency becomes ns*clock cycles. Per-cycle rates, counts, capacities and the
 * BW polynomial are already clock-free and pass through unchanged.
 */
struct arch_ceilings_t {
  // Per-CU compute ceilings (clock-free: per-cycle or dimensionless).
  double vmem_issue_rate;  ///< VMEM instructions per CU per cycle.
  double valu_rate;        ///< VALU lane-elements per CU per cycle.

  // Cache / memory structure (clock-free).
  std::size_t tcp_capacity_bytes;       ///< Per-CU vL1D/TCP capacity.
  double tcp_bw;                        ///< bytes per CU per cycle.
  int mshr_depth_per_wave;              ///< Outstanding misses per wave before stall.
  int waves_per_wg;                     ///< Waves co-resident per workgroup.
  std::size_t mall_capacity_bytes;      ///< Device-wide MALL/Infinity Cache capacity.
  std::size_t hbm_capacity_bytes;       ///< Total HBM capacity.
  std::array<double, 3> mem_bw_coeffs;  ///< HBM utilization-vs-active-CU polynomial.
  std::size_t cacheline_bytes;          ///< Cache-line / fabric transfer granularity.

  // Measured aggregate GPU bandwidths, native GB/s (→ bytes/cycle at build).
  double l2_bw_per_cu_GBps;  ///< Per-CU L2 bandwidth at full XCD occupancy.
  double mall_bw_GBps;       ///< Aggregate MALL bandwidth.
  double hbm_read_GBps;      ///< Aggregate HBM read bandwidth.
  double hbm_write_GBps;     ///< Aggregate HBM write bandwidth.

  // Measured GPU latency, native ns (→ cycles at build).
  double xgmi_latency_ns;  ///< Remote-load round-trip latency.

  // Fabric link / engine counts (clock-free).
  int num_peer_links;    ///< xGMI links to peer GPUs.
  int num_sdma_engines;  ///< SDMA (DMA-copy) engines.

  // Measured fabric bandwidths, native GB/s (→ bytes/cycle at build).
  // link_GBps is the payload rate (already discounted for wire/framing overhead).
  double link_GBps;        ///< Per-link payload bandwidth.
  double sdma_read_GBps;   ///< Per-engine SDMA read bandwidth.
  double sdma_write_GBps;  ///< Per-engine SDMA write bandwidth.

  // Measured fabric protocol latencies, native ns (→ cycles at build).
  double atomic_latency_ns;   ///< One signal/wait fabric atomic.
  double launch_overhead_ns;  ///< Fixed per-collective kernel launch/setup floor.
};

/**
 * @brief HBM bandwidth-utilization polynomial for an architecture, in comm's
 *        std::array form, sourced from the shared GEMM calibration.
 *
 * The BW-vs-active-CU polynomial is the one piece of comm's calibration GEMM
 * already measures — origami::architecture_constants::mem_bw_per_wg_coefficients.
 * Reading it from there instead of re-typing the numbers keeps a single source
 * of truth (and picks up per-architecture curves GEMM has but comm had not, e.g.
 * gfx950). Returned as std::array to match @ref hardware_t::mem_bw_coeffs;
 * std::array copy-assignment is not reliably constexpr in C++17, so callers
 * assigning into an existing array must copy element-wise.
 *
 * @param arch Architecture enum value.
 * @return std::array<double, 3> {a, b, c} of fraction = a*N^2 + b*N + c.
 */
constexpr std::array<double, 3> mem_bw_coeffs_from_constants(architecture_t arch) {
  const auto t = get_arch_constants(arch).mem_bw_per_wg_coefficients;
  return {std::get<0>(t), std::get<1>(t), std::get<2>(t)};
}

/**
 * @brief Calibrated communication ceilings for an architecture, in native units.
 *
 * The communication analogue of origami::get_arch_constants. Only the
 * architectures origami has microbenchmarked for collectives appear; today that
 * is gfx942 (MI300X, CDNA3). The values are the per-link and aggregate rates and
 * latencies measured on that part, in GB/s and ns — @ref make_system applies the
 * clock conversion. The HBM BW-vs-active-CU polynomial is not re-typed here; it
 * is pulled from the shared GEMM calibration via @ref mem_bw_coeffs_from_constants.
 *
 * @param arch Architecture enum value.
 * @return arch_ceilings_t Native-unit ceilings for @p arch.
 * @throws std::invalid_argument If no comm ceilings are calibrated for @p arch.
 */
constexpr arch_ceilings_t get_arch_ceilings(architecture_t arch) {
  switch (arch) {
    case architecture_t::gfx942:
      // MI300X (CDNA3). vmem_issue_rate=1.0: one VMEM instr/CU/cycle.
      // valu_rate=2.10*64: 64 lanes/SIMD × ~2.10 elem/lane/cycle. tcp_bw=64:
      // one global_load_dwordx16/cycle. mshr_depth=12 (the measured N=13 cliff).
      // xGMI 660 ns remote-load RTT. l2/mall/hbm GB/s are measured aggregate
      // peaks (write peak > read on this part). link = 49.1 GiB/s wire rate →
      // decimal bytes/s, discounted 1.23× wire-to-payload. 7 single-hop peers,
      // 14 SDMA engines. atomic ~100 ns/handshake, ~45 µs launch floor.
      return arch_ceilings_t{
          /* vmem_issue_rate     */ 1.0,
          /* valu_rate           */ 2.10 * 64.0,
          /* tcp_capacity_bytes  */ 32ULL * 1024ULL,
          /* tcp_bw              */ 64.0,
          /* mshr_depth_per_wave */ 12,
          /* waves_per_wg        */ 10,
          /* mall_capacity_bytes */ 256ULL * 1024ULL * 1024ULL,
          /* hbm_capacity_bytes  */ 192ULL * 1024ULL * 1024ULL * 1024ULL,
          /* mem_bw_coeffs       */ mem_bw_coeffs_from_constants(architecture_t::gfx942),
          /* cacheline_bytes     */ 64,
          /* l2_bw_per_cu_GBps   */ 83.6,
          /* mall_bw_GBps        */ 4730.0,
          /* hbm_read_GBps       */ 4730.0,
          /* hbm_write_GBps      */ 5140.0,
          /* xgmi_latency_ns     */ 660.0,
          /* num_peer_links      */ 7,
          /* num_sdma_engines    */ 14,
          /* link_GBps           */ 49.1 * (1024.0 * 1024.0 * 1024.0) / 1e9 / 1.23,
          /* sdma_read_GBps      */ 49.5,
          /* sdma_write_GBps     */ 23.6,
          /* atomic_latency_ns   */ 100.0,
          /* launch_overhead_ns  */ 45000.0,
      };
    case architecture_t::gfx950: {
      // TODO(uncalibrated): MI350 series (CDNA4) placeholder — NOT microbenchmarked.
      // Only the publicly known HBM capacity and peak bandwidth are updated; every
      // other ceiling is carried over from gfx942 and MUST be re-measured before
      // this is trusted for MI350 predictions. In particular the xGMI link rate,
      // SDMA rates, MALL/L2 bandwidth, and all latencies differ on CDNA4. This
      // mirrors how the GEMM model handles its own not-yet-calibrated
      // architectures in get_arch_constants().
      arch_ceilings_t c = get_arch_ceilings(architecture_t::gfx942);
      // MI355X ships 288 GB HBM3E (public spec).
      c.hbm_capacity_bytes = 288ULL * 1024ULL * 1024ULL * 1024ULL;
      // ~8 TB/s peak HBM3E vs MI300X's ~5.3 TB/s: scale gfx942's calibrated
      // sustained read/write aggregates by that peak ratio as a rough stand-in
      // until MI350 sustained rates are measured.
      c.hbm_read_GBps  = 4730.0 * (8.0 / 5.3);
      c.hbm_write_GBps = 5140.0 * (8.0 / 5.3);
      // Unlike the HBM/fabric ceilings above, the BW-vs-active-CU polynomial IS
      // calibrated for gfx950 in the shared GEMM constants, so use the real
      // gfx950 curve rather than the gfx942 one carried over above. Element-wise
      // because std::array copy-assignment is not reliably constexpr in C++17.
      const auto k       = mem_bw_coeffs_from_constants(architecture_t::gfx950);
      c.mem_bw_coeffs[0] = k[0];
      c.mem_bw_coeffs[1] = k[1];
      c.mem_bw_coeffs[2] = k[2];
      return c;
    }
    default:
      throw std::invalid_argument(
          "origami::comm has no calibrated arch_ceilings_t for this architecture");
  }
}

/**
 * @brief Fuse calibrated ceilings, live topology, and a clock into a system_t.
 *
 * The single place native-unit ceilings (GB/s, ns) and a device's topology meet
 * the model's cycle units. Bandwidths are divided by the clock to get
 * bytes/cycle and latencies multiplied by it to get cycles (the unit identity at
 * the top of this file); per-cycle rates, counts, capacities and the BW
 * polynomial pass through unchanged. Topology fields (CU/XCD counts, L2
 * capacity) come from @p topo, so a CPX partition models its own reduced shape
 * rather than the full part.
 *
 * @param ceilings Calibrated per-architecture ceilings in native units.
 * @param topo Live GPU topology (from a device query or an explicit fixture).
 * @param clock_ghz Target compute clock in GHz used for the unit conversion.
 * @return system_t The assembled GPU + fabric machine description.
 */
constexpr system_t make_system(const arch_ceilings_t& ceilings,
                               const gpu_topology_t& topo,
                               double clock_ghz) {
  hardware_t gpu{};
  gpu.arch                = topo.arch;
  gpu.num_cu              = static_cast<int>(topo.num_cu);
  gpu.num_xcd             = static_cast<int>(topo.num_xcd);
  gpu.cu_per_xcd          = static_cast<int>(topo.cu_per_xcd);
  gpu.clock_ghz           = clock_ghz;
  gpu.vmem_issue_rate     = ceilings.vmem_issue_rate;
  gpu.valu_rate           = ceilings.valu_rate;
  gpu.tcp_capacity_bytes  = ceilings.tcp_capacity_bytes;
  gpu.tcp_bw              = ceilings.tcp_bw;
  gpu.mshr_depth_per_wave = ceilings.mshr_depth_per_wave;
  gpu.waves_per_wg        = ceilings.waves_per_wg;
  gpu.xgmi_latency_cycles = ceilings.xgmi_latency_ns * clock_ghz;
  gpu.l2_capacity_bytes   = topo.l2_capacity_bytes;
  gpu.l2_bw_per_cu        = ceilings.l2_bw_per_cu_GBps / clock_ghz;
  gpu.mall_capacity_bytes = ceilings.mall_capacity_bytes;
  gpu.mall_bw             = ceilings.mall_bw_GBps / clock_ghz;
  gpu.hbm_read_bw         = ceilings.hbm_read_GBps / clock_ghz;
  gpu.hbm_write_bw        = ceilings.hbm_write_GBps / clock_ghz;
  gpu.hbm_capacity_bytes  = ceilings.hbm_capacity_bytes;
  // Element-wise (std::array copy-assignment is not reliably constexpr in C++17).
  gpu.mem_bw_coeffs[0] = ceilings.mem_bw_coeffs[0];
  gpu.mem_bw_coeffs[1] = ceilings.mem_bw_coeffs[1];
  gpu.mem_bw_coeffs[2] = ceilings.mem_bw_coeffs[2];
  gpu.cacheline_bytes  = ceilings.cacheline_bytes;

  comm_hardware_t fabric{};
  fabric.link_bw                = ceilings.link_GBps / clock_ghz;
  fabric.num_peer_links         = ceilings.num_peer_links;
  fabric.num_sdma_engines       = ceilings.num_sdma_engines;
  fabric.sdma_read_bw           = ceilings.sdma_read_GBps / clock_ghz;
  fabric.sdma_write_bw          = ceilings.sdma_write_GBps / clock_ghz;
  fabric.atomic_latency_cycles  = ceilings.atomic_latency_ns * clock_ghz;
  fabric.launch_overhead_cycles = ceilings.launch_overhead_ns * clock_ghz;
  fabric.clock_ghz              = clock_ghz;

  return system_t{gpu, fabric};
}

}  // namespace origami::comm
