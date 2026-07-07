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
// Empirical heuristic weights.
//
// Why this file exists separately from hardware.hpp: the roofline model
// captures the *bandwidth-bound* limits exactly, but real collectives also pay
// costs the first-principles model cannot derive from silicon datasheets —
// CPU-mediated launch and per-step handshakes, sub-saturation link efficiency,
// framework call overhead. Those are captured here as a small set of empirical
// factors, each fitted to a measured RCCL sweep and each with a stated
// physical interpretation. Isolating them means: (a) the physics in
// hardware.hpp stays clean and portable, and (b) these values are understood
// to drift as measurement coverage grows and should be re-fit, not trusted as
// constants of nature.
//
//   hardware.hpp   — measured datasheet constants (facts about silicon)
//   heuristics.hpp — this file: empirical fits expected to drift
#pragma once

#include "origami/comm/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace origami::comm {

// primitive_t / PRIMITIVE_NAMES / primitive_name live in types.hpp (the
// operation is a problem property); this table keys its weights off them.

// ─── framework_t enum + name table ─────────────────────────────────
/**
 * @brief Calling framework whose host-side launch overhead is modeled.
 *
 * Indexes the @ref heuristics_t::framework_overhead_ns table; the framework's
 * software stack adds a fixed host-side cost before the GPU kernel runs.
 */
enum class framework_t : std::uint8_t {
  raw,    ///< No framework — launch straight into the collective.
  rccl,   ///< ROCm Communication Collectives Library.
  nccl,   ///< NVIDIA Collective Communications Library.
  torch,  ///< PyTorch (the only framework with a measurable host floor).
  jax,    ///< JAX.
  mpi,    ///< Message Passing Interface.
};

/// Lowercase string names for each @ref framework_t, indexed by enum value.
inline constexpr std::array<std::string_view, 6> FRAMEWORK_NAMES = {
    "raw",
    "rccl",
    "nccl",
    "torch",
    "jax",
    "mpi",
};

/**
 * @brief Look up the string name of a framework.
 *
 * @param f Framework enumerator.
 * @return std::string_view The framework's lowercase name.
 */
constexpr std::string_view framework_name(framework_t f) noexcept {
  return FRAMEWORK_NAMES[static_cast<std::size_t>(f)];
}

/**
 * @brief Empirical fudge factors layered on top of the roofline model.
 *
 * The single home for empirical fudge factors. Override per-study by
 * constructing a custom instance and passing it through
 * `predict_tensor_collective`.
 */
struct heuristics_t {
  // ── WG-cap heuristic ────────────────────────────────────────
  /// Minimum useful work per workgroup (see comm_config_t::effective_num_wgs).
  /// Below this, the per-WG launch/sync constants outweigh the bandwidth a WG
  /// adds, so over-launched channels are counted as idle. 16 KiB is both the
  /// NCCL/RCCL LL128 minimum chunk size and the empirical best in a 0/64/…/64 KiB
  /// cap sweep over the 4 KB–1 MiB regime; it does not move the >1 MiB regime.
  int min_bytes_per_wg = 16'384;

  // ── Rank-symmetry shortcut ──────────────────────────────────
  /// A collective finishes only when its *slowest* rank finishes, so the honest
  /// cost is the max latency over all ranks. compute_collective_latency takes
  /// that max by default. Every algorithm shipped today is rank-symmetric in
  /// cost (each rank's is_self/peer schedule has the same shape, just rotated),
  /// so the max equals rank 0 — but that is a property of the current
  /// algorithms, not a guarantee. Setting this true asserts the symmetry and
  /// evaluates rank 0 only, an N× speedup that is exact for symmetric algorithms
  /// and an approximation for any future asymmetric one. Left false so the
  /// engine is correct by construction for algorithms we have not yet written.
  bool assume_rank_symmetry = false;

  // ── framework_t overhead floor (nanoseconds, HOST wall time) ──
  /// A fixed host-side cost the *caller's* software stack adds before the GPU
  /// kernel even runs (Python dispatch, stream setup, etc.). It is host wall
  /// time, not GPU cycles, so it is added after the cycles→µs conversion and
  /// never scaled by clock. Only `torch` has a measurable floor (~400 µs);
  /// raw/RCCL/MPI launch straight into the collective.
  std::array<double, 6> framework_overhead_ns = {
      0.0,        // raw
      0.0,        // rccl
      0.0,        // nccl
      400'000.0,  // torch (~400 µs MI300X floor — see calibration caveat)
      0.0,        // jax
      0.0,        // mpi
  };

  // ── Per-ring-step proxy/sync overhead (host nanoseconds) ────
  /// Every ring step incurs a CPU-mediated proxy handshake the bandwidth model
  /// cannot see; left at zero it produces a flat ~130 µs underestimate on small
  /// all-gather. The fit charges it per step: AG ~10 µs, RS ~4 µs, others
  /// negligible (AG is the most write/handshake-bound). Adding this term drops
  /// AG MdAPE from ~45% to single digits. Stored in nanoseconds (host wall time,
  /// the unit it is measured in); the engine converts to cycles at the target
  /// GPU clock, so the value is clock-invariant and carries no MI300X assumption.
  std::array<double, 5> ring_step_overhead_ns = {
      /* all_gather     */ 10'000.0,
      /* reduce_scatter */ 4'000.0,
      /* broadcast      */ 0.0,
      /* all_reduce     */ 0.0,
      /* all_to_all     */ 0.0,
  };

  // ── xGMI write concentration efficiency ─────────────────────
  /// The saturation rate k in util(wgs) = 1 − exp(−wgs/k) used by latency.hpp's
  /// xGMI-write block. Measured: one WG saturates only ~21% of a link, two
  /// ~38%, five ~65%, nine+ ~83% — the link needs many concurrent writers to
  /// hide framing/turnaround. Smaller k -> saturates with fewer WGs. Values are
  /// per primitive because their write-concentration patterns differ (e.g.
  /// all-reduce spreads writes thinner, so it needs a larger k).
  double xgmi_write_concentration_k_default = 4.0;
  /// Per-primitive saturation rate k (see @ref xgmi_write_concentration_k_default).
  std::array<double, 5> xgmi_write_concentration_k_by_primitive = {
      /* all_gather     */ 4.0,
      /* reduce_scatter */ 3.0,
      /* broadcast      */ 3.5,
      /* all_reduce     */ 6.0,
      /* all_to_all     */ 4.0,
  };

  /**
   * @brief xGMI write-concentration saturation rate k for a primitive.
   *
   * @param p Communication primitive.
   * @return double Saturation rate k for use in util(wgs) = 1 − exp(−wgs/k).
   */
  constexpr double k_xgmi_write(primitive_t p) const noexcept {
    return xgmi_write_concentration_k_by_primitive[static_cast<std::size_t>(p)];
  }

  /**
   * @brief xGMI write-concentration saturation rate k, looked up by name.
   *
   * String-keyed overload — used at the public API edge where the
   * caller passes a name (falls back to default on unknown).
   *
   * @param name Primitive name (see PRIMITIVE_NAMES).
   * @return double Saturation rate k, or xgmi_write_concentration_k_default if
   *         the name is unknown.
   */
  constexpr double k_xgmi_write(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < PRIMITIVE_NAMES.size(); ++i) {
      if (PRIMITIVE_NAMES[i] == name) { return xgmi_write_concentration_k_by_primitive[i]; }
    }
    return xgmi_write_concentration_k_default;
  }

  /**
   * @brief Per-ring-step proxy/sync overhead for a primitive, in nanoseconds.
   *
   * @param p Communication primitive.
   * @return double Per-step overhead in nanoseconds (host wall time).
   */
  constexpr double ring_step_overhead_ns_for(primitive_t p) const noexcept {
    return ring_step_overhead_ns[static_cast<std::size_t>(p)];
  }
  /**
   * @brief Per-ring-step proxy/sync overhead, looked up by name, in nanoseconds.
   *
   * @param name Primitive name (see PRIMITIVE_NAMES).
   * @return double Per-step overhead in nanoseconds, or 0 if the name is unknown.
   */
  constexpr double ring_step_overhead_ns_for(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < PRIMITIVE_NAMES.size(); ++i) {
      if (PRIMITIVE_NAMES[i] == name) { return ring_step_overhead_ns[i]; }
    }
    return 0.0;
  }

  /**
   * @brief Host-side framework launch overhead floor, in microseconds.
   *
   * @param f Calling framework.
   * @return double Host wall-time overhead in microseconds.
   */
  constexpr double framework_overhead_us(framework_t f) const noexcept {
    return framework_overhead_ns[static_cast<std::size_t>(f)] / 1000.0;
  }
  /**
   * @brief Host-side framework launch overhead floor, looked up by name.
   *
   * @param name Framework name (see FRAMEWORK_NAMES).
   * @return double Host wall-time overhead in microseconds, or 0 if the name is
   *         unknown.
   */
  constexpr double framework_overhead_us(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < FRAMEWORK_NAMES.size(); ++i) {
      if (FRAMEWORK_NAMES[i] == name) { return framework_overhead_ns[i] / 1000.0; }
    }
    return 0.0;
  }
};

/// Default-constructed @ref heuristics_t with the shipped empirical fits.
inline constexpr heuristics_t DEFAULT_HEURISTICS{};

}  // namespace origami::comm
