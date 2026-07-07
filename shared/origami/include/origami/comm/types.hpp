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

/**
 * @file
 * @brief origami::comm — analytical communication cost model.
 *
 * Core type definitions for the communication cost model.
 *
 * Two ideas thread through every type here:
 *   1. The cache line is the indivisible unit of memory traffic. The hardware
 *      never moves a partial line, so the model counts work in *cache lines*,
 *      not bytes, and tracks how efficiently a payload packs into them
 *      (cacheline_efficiency). Wasted line bytes are real wasted bandwidth.
 *   2. Work is decomposed top-down by even division: a [M,N] tensor is split
 *      across GPUs, each GPU's tile is split across timesteps, each timestep
 *      tile across workgroups, each WG tile streamed in fixed-size iterations.
 *      Every divide_* below preserves total bytes while choosing an axis that
 *      keeps rows contiguous (and therefore cache-line-aligned) when it can.
 *
 * Header-only; constexpr where possible so the whole division tree can fold
 * at compile time.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include "origami/math.hpp"
#include "origami/types.hpp"

namespace origami::comm {

/**
 * @brief Custom collective-algorithm interface (full definition in algorithms.hpp).
 *
 * Forward declaration so comm_config_t can carry an algorithm-override pointer
 * without including algorithms.hpp (which depends on this header).
 */
class collective_algorithm_t;

/**
 * @brief Alias for the canonical Origami data-type enum.
 *
 * Reuse the canonical Origami data-type enum rather than defining a second,
 * incompatible one. `origami::data_type_t` resolves unqualified here via
 * enclosing-namespace lookup (origami::comm is nested in origami).
 */
using origami::data_type_t;

/**
 * @brief Element size in bytes for a comm-supported data type.
 *
 * Kept header-only on purpose: origami::data_type_to_bytes() is backed by
 * datatype_to_bits() in the (HIP-linked) origami library, so calling it would
 * force comm consumers to link roc::origami and pull in HIP. The comm model
 * only needs byte sizes for the dtypes it supports, so we keep a local switch
 * and preserve the header-only / HIP-free contract.
 *
 * @param dt Data type to size.
 * @return int Bytes per element, or 0 for unsupported types.
 */
constexpr int dtype_bytes(data_type_t dt) noexcept {
  switch (dt) {
    case data_type_t::Float8:
    case data_type_t::Int8: return 1;
    case data_type_t::Half:
    case data_type_t::BFloat16: return 2;
    case data_type_t::Float: return 4;
    case data_type_t::Double: return 8;
    default: return 0;
  }
}

/**
 * @brief Bytes moved by a single VMEM instruction.
 *
 * How many bytes a single VMEM instruction moves. A wider load (dwordx16 =
 * 64 B) fills a whole cache line in one instruction, so it minimizes the VMEM
 * *issue* count — and VMEM issue (one instr/CU/cycle) is a distinct ceiling
 * from memory bandwidth. Narrow loads can make issue, not bandwidth, the
 * bottleneck. The enum value *is* the byte width, so casting recovers it.
 */
enum class load_width_t : std::uint8_t {
  DWORD    = 4,
  DWORDX4  = 16,
  DWORDX16 = 64,
};

/**
 * @brief Recover the byte width encoded in a load_width_t value.
 *
 * @param w Load width.
 * @return int Bytes moved per VMEM instruction.
 */
constexpr int load_width_bytes(load_width_t w) noexcept { return static_cast<int>(w); }

/**
 * @brief VMEM instructions needed to touch one cache line.
 *
 * Instructions needed to touch one cache line = cacheline_bytes / bytes-per-instr.
 * This converts a cache-line count (the bandwidth view) into a VMEM-issue count
 * (the issue-rate view) so latency.hpp can compare the two ceilings.
 *
 * @param w Load width.
 * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
 * @return int Instructions per cache line.
 */
constexpr int instrs_per_cacheline(load_width_t w, std::size_t cacheline_bytes) noexcept {
  return static_cast<int>(cacheline_bytes) / load_width_bytes(w);
}

// ─── direction_t / reduce_op_t (carried through op_t resolution) ───────
/** @brief Data-movement direction for a transfer primitive (push vs pull). */
enum class direction_t : std::uint8_t { PUSH, PULL };
/** @brief Reduction operator applied when combining peer contributions. */
enum class reduce_op_t : std::uint8_t { SUM, MAX, MIN, PROD };

// ─── primitive_t: the collective OPERATION ─────────────────────────────
/**
 * @brief The collective operation being performed (a property of the problem).
 *
 * Which collective is being performed. This is a property of the *problem*,
 * not the config: it determines the correct result — an all-gather and a
 * reduce-scatter of the same buffer produce different answers — whereas the
 * algorithm that *implements* it (ring vs two-shot, …) is a performance choice
 * and therefore lives in comm_config_t. Lives in types.hpp because both the
 * problem type and the heuristics table are keyed by it.
 */
enum class primitive_t : std::uint8_t {
  all_gather,
  reduce_scatter,
  broadcast,
  all_reduce,
  all_to_all,
};

/** @brief Canonical string names for each primitive_t, indexed by enum value. */
inline constexpr std::array<std::string_view, 5> PRIMITIVE_NAMES = {
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_reduce",
    "all_to_all",
};

/**
 * @brief Look up the canonical name for a collective primitive.
 *
 * @param p Collective primitive.
 * @return std::string_view Canonical name.
 */
constexpr std::string_view primitive_name(primitive_t p) noexcept {
  return PRIMITIVE_NAMES[static_cast<std::size_t>(p)];
}

/**
 * @brief Parse a canonical collective name into the enum.
 *
 * Used only at the public string edge (predict_row / predict_tensor_collective);
 * throws on an unknown name, which preserves the original string-keyed algorithm
 * factory's behaviour.
 *
 * @param name Canonical collective name.
 * @return primitive_t Matching collective primitive.
 * @throws std::invalid_argument If the name is not a known collective.
 */
inline primitive_t primitive_from_name(std::string_view name) {
  for (std::size_t i = 0; i < PRIMITIVE_NAMES.size(); ++i) {
    if (PRIMITIVE_NAMES[i] == name) return static_cast<primitive_t>(i);
  }
  throw std::invalid_argument(std::string{"unknown collective: "} + std::string{name});
}

// ─── algorithm_t: the collective IMPLEMENTATION ────────────────────────
/**
 * @brief How a collective is carried out — the dataflow pattern over the ranks.
 *
 * A performance choice (every valid algorithm yields the same result, just at a
 * different cost), so it lives in comm_config_t, never in the problem.
 *
 * Crucially, an algorithm is only meaningful *for a particular collective*:
 * the (collective, algorithm) pair must be one resolve_algorithm() defines, or
 * it is rejected. `automatic` always resolves to the canonical algorithm for
 * the problem's collective, so the common path needs no explicit choice. The
 * only collective with a real menu today is all_reduce {one_shot, two_shot,
 * ring}; the rest have a single algorithm and accept only automatic (or its
 * explicit name).
 */
enum class algorithm_t : std::uint8_t {
  automatic,  ///< resolve the canonical algorithm for the problem's collective
  ring,       ///< neighbour-ring pipeline (all_gather/reduce_scatter/broadcast/all_reduce)
  one_shot,   ///< all_reduce: direct gather-and-reduce from every peer
  two_shot,   ///< all_reduce: reduce-scatter shot then all-gather shot
  direct,     ///< all_to_all: pid-staggered pairwise exchange
};

/** @brief Canonical string names for each algorithm_t, indexed by enum value. */
inline constexpr std::array<std::string_view, 5> ALGORITHM_NAMES = {
    "automatic",
    "ring",
    "one_shot",
    "two_shot",
    "direct",
};

/**
 * @brief Look up the canonical name for an algorithm.
 *
 * @param a Algorithm.
 * @return std::string_view Canonical name.
 */
constexpr std::string_view algorithm_name(algorithm_t a) noexcept {
  return ALGORITHM_NAMES[static_cast<std::size_t>(a)];
}

/**
 * @brief Parse a canonical algorithm name into the enum.
 *
 * @param name Canonical algorithm name.
 * @return algorithm_t Matching algorithm.
 * @throws std::invalid_argument If the name is not a known algorithm.
 */
inline algorithm_t algorithm_from_name(std::string_view name) {
  for (std::size_t i = 0; i < ALGORITHM_NAMES.size(); ++i) {
    if (ALGORITHM_NAMES[i] == name) return static_cast<algorithm_t>(i);
  }
  throw std::invalid_argument(std::string{"unknown algorithm: "} + std::string{name});
}

// ─── ceil_div: ceil(a/b) for positive integers ─────
/**
 * @brief Ceiling division ceil(a/b) for non-negative integers.
 *
 * Thin wrapper over origami::math::safe_ceil_div (the shared, overflow-safe
 * implementation used throughout the base GEMM model) that preserves comm's
 * common-type return. Delegating keeps a single source of truth for the
 * ceil-div logic and inherits its overflow / zero-denominator guards (b == 0
 * yields 0 rather than undefined behaviour).
 *
 * @tparam A Numerator type.
 * @tparam B Denominator type.
 * @param a Numerator.
 * @param b Denominator.
 * @return std::common_type_t<A, B> Smallest integer >= a / b (0 if b == 0).
 */
template <typename A, typename B>
constexpr auto ceil_div(A a, B b) noexcept -> std::common_type_t<A, B> {
  using U = std::common_type_t<A, B>;
  return math::safe_ceil_div(static_cast<U>(a), static_cast<U>(b));
}

/**
 * @brief A 2D row-major tile (m × n × dtype) with a contiguity bit.
 *
 * The contiguous/non-contiguous regimes and their costs are described
 * on the member accessors below.
 */
struct tile_shape_t {
  std::size_t m     = 1;                      ///< Tile row count.
  std::size_t n     = 1;                      ///< Tile column count.
  data_type_t dtype = data_type_t::BFloat16;  ///< Element data type.
  int split_dim     = 0;                      ///< Axis the tile was split along (0=row, 1=col).
  bool contiguous   = true;                   ///< True if the tile is one dense, flat byte run.

  /** @brief Construct a unit (1 × 1) tile with default dtype. */
  constexpr tile_shape_t() noexcept = default;

  /**
   * @brief Construct a tile, clamping the row/column counts to at least 1.
   *
   * @param m_ Row count (clamped to >= 1).
   * @param n_ Column count (clamped to >= 1).
   * @param dt Element data type.
   * @param split Axis the tile was split along (0=row, 1=col).
   * @param contig Whether the tile is a contiguous, dense byte run.
   */
  constexpr tile_shape_t(std::size_t m_,
                         std::size_t n_,
                         data_type_t dt,
                         int split   = 0,
                         bool contig = true) noexcept
      : m{std::max<std::size_t>(m_, 1)}
      , n{std::max<std::size_t>(n_, 1)}
      , dtype{dt}
      , split_dim{split}
      , contiguous{contig} {}

  /** @brief Bytes per element of this tile's data type. */
  constexpr int element_bytes() const noexcept { return dtype_bytes(dtype); }
  /** @brief Total element count (m × n). */
  constexpr std::size_t elements() const noexcept { return m * n; }
  /** @brief Total payload size in bytes (elements × element_bytes). */
  constexpr std::size_t bytes() const noexcept { return elements() * element_bytes(); }

  /**
   * @brief Cache lines spanned by a single row, rounding up.
   *
   * A row whose byte length is not a whole multiple of the cache-line size still
   * consumes a whole final line. This rounding is the entire source of
   * cache-line inefficiency for strided tiles.
   *
   * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
   * @return std::size_t Cache lines per row.
   */
  constexpr std::size_t cl_per_row(std::size_t cacheline_bytes) const noexcept {
    return ceil_div(n * static_cast<std::size_t>(element_bytes()), cacheline_bytes);
  }

  /**
   * @brief Total cache lines transferred for this tile.
   *
   * The contiguity bit captures a real memory-layout fact: a contiguous tile
   * is one flat byte run, so only the *single* final line is partially wasted
   * (ceil over total bytes). A non-contiguous (column-striped) tile has m
   * independent rows, each of which pays its own partial-line rounding —
   * m × cl_per_row — which can be far more traffic than the same bytes laid
   * out contiguously.
   *
   * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
   * @return std::size_t Total cache lines moved.
   */
  constexpr std::size_t cachelines(std::size_t cacheline_bytes) const noexcept {
    if (contiguous) { return std::max<std::size_t>(ceil_div(bytes(), cacheline_bytes), 1); }
    return m * cl_per_row(cacheline_bytes);
  }

  /**
   * @brief Logical 2D cacheline footprint.
   *
   * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
   * @return std::pair<std::size_t, std::size_t> (rows, cache lines per row);
   *         a contiguous tile collapses to a single row.
   */
  constexpr std::pair<std::size_t, std::size_t> cacheline_shape(
      std::size_t cacheline_bytes) const noexcept {
    if (contiguous) return {1, cachelines(cacheline_bytes)};
    return {m, cl_per_row(cacheline_bytes)};
  }

  /**
   * @brief Packing efficiency: useful bytes ÷ bytes actually moved.
   *
   * 1.0 means perfect packing; < 1.0 means bandwidth is spent on padding.
   * Higher layers pay for transferred-line bandwidth, so this ratio is the
   * lever that makes a poorly-aligned strided collective slower than its byte
   * count suggests.
   *
   * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
   * @return double Packing efficiency in (0, 1].
   */
  constexpr double cacheline_efficiency(std::size_t cacheline_bytes) const noexcept {
    const std::size_t transferred = cachelines(cacheline_bytes) * cacheline_bytes;
    if (transferred == 0) return 1.0;
    return static_cast<double>(bytes()) / static_cast<double>(transferred);
  }

  /**
   * @brief Split a tile into `factor` equal chunks along one axis.
   *
   * axis=0 row-stripe : fewer rows, same row length → stays contiguous and
   *                     cache-line-aligned (the cheap split).
   * axis=1 column-stripe : shorter rows → each row re-pays partial-line
   *                     rounding, so the result is marked non-contiguous.
   *
   * @param factor Number of chunks (clamped to >= 1).
   * @param axis Split axis (0=rows, 1=columns).
   * @return tile_shape_t One chunk of the split.
   */
  constexpr tile_shape_t divide(std::size_t factor, int axis = 0) const noexcept {
    factor = std::max<std::size_t>(factor, 1);
    if (axis == 0) {
      return tile_shape_t{
          std::max<std::size_t>(ceil_div(m, factor), 1), n, dtype, split_dim, contiguous};
    }
    return tile_shape_t{
        m,
        std::max<std::size_t>(ceil_div(n, factor), 1),
        dtype,
        /*split_dim=*/1,
        /*contiguous=*/false,
    };
  }

  /**
   * @brief Split into `factor` chunks, preferring the cheaper contiguous axis.
   *
   * Pick the cheaper split when possible: if there are at least `factor` rows,
   * row-stripe (axis 0) to keep contiguity; otherwise we are forced to cut
   * within rows (axis 1) and accept the per-row line padding. This is why a
   * tall tensor parallelizes across WGs more cheaply than a wide one.
   *
   * @param factor Number of chunks (clamped to >= 1).
   * @return tile_shape_t One chunk of the split.
   */
  constexpr tile_shape_t divide_byte_equal(std::size_t factor) const noexcept {
    factor = std::max<std::size_t>(factor, 1);
    if (factor == 1) return *this;
    if (m >= factor) return divide(factor, /*axis=*/0);
    return divide(factor, /*axis=*/1);
  }
};

/**
 * @brief 2D tensor [M, N] distributed across num_gpus, split along split_dim.
 *
 * split_dim selects which axis the data is partitioned on: split_dim=0 gives
 * each rank a horizontal stripe of rows (ceil(M/num_gpus) × N, still dense and
 * contiguous), split_dim=1 gives a vertical stripe of columns. The choice
 * follows the collective's data-distribution convention and feeds the same
 * contiguity reasoning as tile_shape_t.
 */
struct comm_problem_t {
  std::size_t M;                              ///< Global tensor row count.
  std::size_t N;                              ///< Global tensor column count.
  int num_gpus;                               ///< Number of ranks the tensor is split across.
  data_type_t dtype = data_type_t::BFloat16;  ///< Element data type.
  int split_dim     = 0;                      ///< Partition axis (0=rows, 1=columns).
  /// The operation to perform. Correctness-determining, hence part of the
  /// problem (kept last so existing 5-field aggregate initializers still bind).
  primitive_t collective = primitive_t::all_reduce;

  /** @brief Bytes per element of this problem's data type. */
  constexpr int element_bytes() const noexcept { return dtype_bytes(dtype); }
  /** @brief Total message size in bytes (M × N × element_bytes). */
  constexpr std::size_t message_bytes() const noexcept { return M * N * element_bytes(); }
  /** @brief Total element count (M × N). */
  constexpr std::size_t total_elements() const noexcept { return M * N; }

  /** @brief Per-rank tile row count after splitting along split_dim. */
  constexpr std::size_t gpu_tile_m() const noexcept {
    return (split_dim == 0) ? ceil_div(M, static_cast<std::size_t>(num_gpus)) : M;
  }
  /** @brief Per-rank tile column count after splitting along split_dim. */
  constexpr std::size_t gpu_tile_n() const noexcept {
    return (split_dim == 1) ? ceil_div(N, static_cast<std::size_t>(num_gpus)) : N;
  }
  /** @brief Per-rank tile element count (gpu_tile_m × gpu_tile_n). */
  constexpr std::size_t gpu_tile_elements() const noexcept { return gpu_tile_m() * gpu_tile_n(); }
  /** @brief Per-rank tile size in bytes. */
  constexpr std::size_t gpu_tile_bytes() const noexcept {
    return gpu_tile_elements() * element_bytes();
  }

  /**
   * @brief Shape of one rank's tile.
   *
   * Per-rank buffer is always dense → contiguous=true.
   *
   * @return tile_shape_t The per-rank tile shape.
   */
  constexpr tile_shape_t gpu_tile_shape() const noexcept {
    return tile_shape_t{gpu_tile_m(), gpu_tile_n(), dtype, split_dim, /*contiguous=*/true};
  }

  /** @brief Cache lines moved for one rank's tile. */
  constexpr std::size_t gpu_tile_cachelines(std::size_t cacheline_bytes) const noexcept {
    return gpu_tile_shape().cachelines(cacheline_bytes);
  }

  /** @brief Packing efficiency of one rank's tile (useful ÷ transferred bytes). */
  constexpr double cacheline_efficiency(std::size_t cacheline_bytes) const noexcept {
    const std::size_t transferred = gpu_tile_cachelines(cacheline_bytes) * cacheline_bytes;
    if (transferred == 0) return 1.0;
    return static_cast<double>(gpu_tile_bytes()) / static_cast<double>(transferred);
  }
};

/**
 * @brief Workgroup-level execution config.
 *
 * `min_bytes_per_wg` is sourced from heuristics::DEFAULT_HEURISTICS by the
 * factory below — the constructor takes an explicit value to keep this header
 * standalone.
 */
struct comm_config_t {
  int num_wgs;                                       ///< Requested workgroup (channel) count.
  load_width_t load_width = load_width_t::DWORDX16;  ///< VMEM load width per instruction.
  int vgprs_for_data      = 128;                     ///< VGPRs each WG dedicates to in-flight data.
  int min_bytes_per_wg    = 16'384;  ///< Default mirrors heuristics_t.min_bytes_per_wg.

  /// Which algorithm implements the problem's collective. `automatic` resolves
  /// to the canonical algorithm for that collective; a named value selects a
  /// specific implementation and is rejected by resolve_algorithm() unless it
  /// is defined for the collective. A performance choice (same result, different
  /// cost), so it belongs in the config, not the problem.
  algorithm_t algorithm = algorithm_t::automatic;

  /// Escape hatch for a caller-supplied custom algorithm object (experiments,
  /// bespoke schedules). When non-null it wins over `algorithm`. It must
  /// correctly implement problem.collective — this pointer bypasses the
  /// (collective, algorithm) validity check that resolve_algorithm() enforces.
  const collective_algorithm_t* algorithm_override = nullptr;

  /**
   * @brief Bytes a workgroup moves per software-pipelined iteration.
   *
   * Equals the register budget the WG dedicates to in-flight data
   * (vgprs_for_data × 4 B/VGPR). This is the depth of the load→compute→store
   * pipeline the WG can keep outstanding; it sets how the WG tile is chopped
   * into iterations in latency.hpp.
   *
   * @return int Bytes moved per iteration.
   */
  constexpr int bytes_per_iter() const noexcept { return vgprs_for_data * 4; }
  /**
   * @brief Cache lines moved per pipelined iteration.
   * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
   */
  constexpr int cl_per_iter(std::size_t cacheline_bytes) const noexcept {
    return bytes_per_iter() / static_cast<int>(cacheline_bytes);
  }
  /**
   * @brief VMEM instructions needed per cache line at the configured load width.
   * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
   */
  constexpr int instrs_per_cl(std::size_t cacheline_bytes) const noexcept {
    return instrs_per_cacheline(load_width, cacheline_bytes);
  }

  /**
   * @brief Effective (not requested) workgroup count for bandwidth scaling.
   *
   * A caller may launch many channels, but if the per-timestep tile is small,
   * splitting it across all of them leaves each WG with a sliver of work. Below
   * ~min_bytes_per_wg the launch/sync cost per WG dominates and extra WGs do not
   * add bandwidth, so the model caps the count at floor(tile_bytes /
   * min_bytes_per_wg) for all bandwidth-scaling purposes. The default 16 KiB ≈
   * the NCCL LL128 minimum chunk, the point below which more channels stop
   * helping in practice. (Launch + sync overheads still charge the full
   * requested num_wgs.)
   *
   * @param tile_bytes Bytes in the tile being divided across workgroups.
   * @return int Effective workgroup count for bandwidth scaling.
   */
  constexpr int effective_num_wgs(std::size_t tile_bytes) const noexcept {
    if (min_bytes_per_wg <= 0) return num_wgs;
    const auto fit =
        std::max<std::size_t>(tile_bytes / static_cast<std::size_t>(min_bytes_per_wg), 1);
    return std::min(static_cast<std::size_t>(num_wgs), fit) == 0
               ? num_wgs
               : static_cast<int>(std::min(static_cast<std::size_t>(num_wgs), fit));
  }
};

/**
 * @brief The work a single iteration imposes on each hardware functional unit.
 *
 * Counted in each unit's native unit: cache lines for the memory/fabric stages
 * (TCP, L2, MALL, HBM, xGMI), issued instructions for VMEM, lane-ops for
 * VALU, and discrete fabric atomics for sync. Keeping the units separate is
 * the whole point: latency.hpp divides each count by that unit's independent
 * throughput and takes the max, modelling the FUs as parallel contending
 * resources rather than one fused "bandwidth". operator+ lets a work graph
 * accumulate the contributions of its primitives. This is the atom that
 * resolve_work_graph builds up.
 */
struct functional_unit_work_t {
  std::int64_t vmem_read_instrs  = 0;  ///< VMEM read instructions issued.
  std::int64_t vmem_write_instrs = 0;  ///< VMEM write instructions issued.
  std::int64_t tcp_read_cl       = 0;  ///< TCP (L1) read cache lines.
  std::int64_t tcp_write_cl      = 0;  ///< TCP (L1) write cache lines.
  std::int64_t l2_read_cl        = 0;  ///< L2 read cache lines.
  std::int64_t l2_write_cl       = 0;  ///< L2 write cache lines.
  std::int64_t mall_read_cl      = 0;  ///< MALL read cache lines.
  std::int64_t mall_write_cl     = 0;  ///< MALL write cache lines.
  std::int64_t hbm_read_cl       = 0;  ///< HBM read cache lines.
  std::int64_t hbm_write_cl      = 0;  ///< HBM write cache lines.
  std::int64_t xgmi_read_cl      = 0;  ///< xGMI fabric read cache lines.
  std::int64_t xgmi_write_cl     = 0;  ///< xGMI fabric write cache lines.
  std::int64_t valu_ops          = 0;  ///< VALU lane-operations.
  std::int64_t atomic_count      = 0;  ///< Discrete fabric atomics (sync).

  /** @brief A zero-initialized work tally. */
  static constexpr functional_unit_work_t zero() noexcept { return {}; }

  /**
   * @brief Accumulate another tally into this one, field by field.
   *
   * @param o Work to add.
   * @return functional_unit_work_t& Reference to this updated tally.
   */
  constexpr functional_unit_work_t& operator+=(const functional_unit_work_t& o) noexcept {
    vmem_read_instrs += o.vmem_read_instrs;
    vmem_write_instrs += o.vmem_write_instrs;
    tcp_read_cl += o.tcp_read_cl;
    tcp_write_cl += o.tcp_write_cl;
    l2_read_cl += o.l2_read_cl;
    l2_write_cl += o.l2_write_cl;
    mall_read_cl += o.mall_read_cl;
    mall_write_cl += o.mall_write_cl;
    hbm_read_cl += o.hbm_read_cl;
    hbm_write_cl += o.hbm_write_cl;
    xgmi_read_cl += o.xgmi_read_cl;
    xgmi_write_cl += o.xgmi_write_cl;
    valu_ops += o.valu_ops;
    atomic_count += o.atomic_count;
    return *this;
  }

  /**
   * @brief Field-wise sum of two work tallies.
   *
   * @param a First tally (taken by value, used as the accumulator).
   * @param b Second tally to add.
   * @return functional_unit_work_t Combined work.
   */
  friend constexpr functional_unit_work_t operator+(functional_unit_work_t a,
                                                    const functional_unit_work_t& b) noexcept {
    a += b;
    return a;
  }
};

/**
 * @brief Full wg_tile transfer latency in GPU cycles, plus per-FU breakdowns.
 *
 * ns helpers go through `clock_ghz` (defaulted to 2.0 = MI300X).
 */
struct wg_tile_latency_breakdown_t {
  double T_total_cycles    = 0.0;  ///< Total wg_tile transfer latency (cycles).
  double T_wlt_cycles      = 0.0;  ///< Steady-state work-loop transfer latency (cycles).
  double T_prologue_cycles = 0.0;  ///< Pipeline fill / prologue latency (cycles).
  double T_epilogue_cycles = 0.0;  ///< Pipeline drain / epilogue latency (cycles).
  double T_sync_cycles     = 0.0;  ///< Synchronization latency (cycles).
  std::size_t num_iters    = 0;    ///< Number of pipelined iterations.

  double T_vmem_cycles       = 0.0;  ///< VMEM-issue-bound latency (cycles).
  double T_tcp_cycles        = 0.0;  ///< TCP (L1)-bound latency (cycles).
  double T_l2_cycles         = 0.0;  ///< L2-bound latency (cycles).
  double T_mall_cycles       = 0.0;  ///< MALL-bound latency (cycles).
  double T_hbm_read_cycles   = 0.0;  ///< HBM-read-bound latency (cycles).
  double T_hbm_write_cycles  = 0.0;  ///< HBM-write-bound latency (cycles).
  double T_xgmi_read_cycles  = 0.0;  ///< xGMI-read-bound latency (cycles).
  double T_xgmi_write_cycles = 0.0;  ///< xGMI-write-bound latency (cycles).
  double T_valu_cycles       = 0.0;  ///< VALU-bound latency (cycles).
  std::string bottleneck;            ///< Name of the dominant (bottleneck) functional unit.

  double clock_ghz = 2.0;  ///< GPU clock in GHz (default 2.0 = MI300X).

  /** @brief GPU clock in Hz. */
  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  /**
   * @brief Convert a cycle count to nanoseconds at the configured clock.
   *
   * @param cycles Cycle count.
   * @return double Equivalent time in nanoseconds.
   */
  constexpr double cycles_to_ns(double cycles) const noexcept { return cycles / clock_hz() * 1e9; }

  // ns accessors for display; the *_cycles fields are the source of truth.
  /** @brief Total latency in nanoseconds. */
  constexpr double T_total() const noexcept { return cycles_to_ns(T_total_cycles); }
  /** @brief Steady-state work-loop transfer latency in nanoseconds. */
  constexpr double T_wlt() const noexcept { return cycles_to_ns(T_wlt_cycles); }
  /** @brief Prologue (pipeline fill) latency in nanoseconds. */
  constexpr double T_prologue() const noexcept { return cycles_to_ns(T_prologue_cycles); }
  /** @brief Epilogue (pipeline drain) latency in nanoseconds. */
  constexpr double T_epilogue() const noexcept { return cycles_to_ns(T_epilogue_cycles); }
  /** @brief Synchronization latency in nanoseconds. */
  constexpr double T_sync() const noexcept { return cycles_to_ns(T_sync_cycles); }
  /** @brief VMEM-bound latency in nanoseconds. */
  constexpr double T_vmem() const noexcept { return cycles_to_ns(T_vmem_cycles); }
  /** @brief TCP-bound latency in nanoseconds. */
  constexpr double T_tcp() const noexcept { return cycles_to_ns(T_tcp_cycles); }
  /** @brief L2-bound latency in nanoseconds. */
  constexpr double T_l2() const noexcept { return cycles_to_ns(T_l2_cycles); }
  /** @brief MALL-bound latency in nanoseconds. */
  constexpr double T_mall() const noexcept { return cycles_to_ns(T_mall_cycles); }
  /** @brief HBM-read-bound latency in nanoseconds. */
  constexpr double T_hbm_read() const noexcept { return cycles_to_ns(T_hbm_read_cycles); }
  /** @brief HBM-write-bound latency in nanoseconds. */
  constexpr double T_hbm_write() const noexcept { return cycles_to_ns(T_hbm_write_cycles); }
  /** @brief xGMI-read-bound latency in nanoseconds. */
  constexpr double T_xgmi_read() const noexcept { return cycles_to_ns(T_xgmi_read_cycles); }
  /** @brief xGMI-write-bound latency in nanoseconds. */
  constexpr double T_xgmi_write() const noexcept { return cycles_to_ns(T_xgmi_write_cycles); }
  /** @brief VALU-bound latency in nanoseconds. */
  constexpr double T_valu() const noexcept { return cycles_to_ns(T_valu_cycles); }
};

}  // namespace origami::comm
