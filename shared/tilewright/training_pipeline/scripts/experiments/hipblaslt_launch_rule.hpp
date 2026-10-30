// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipBLASLt's launch of its kernels as a tilewright::LaunchModel: the rule of
// training_pipeline/lib/hipblaslt_launch.py (launch_plan), which MLREC_v3
// models record as "hipblaslt:<arch>:<RULE_VERSION>", with its launch_key.
// Both must give the same plan and key for every input; bench_shipped_rank
// --dump-plans prints these for the comparison.

#pragma once

#include "tilewright/types.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipblaslt_launch {

inline constexpr int kRuleVersion = 1;

inline constexpr std::int64_t kSmTargetGranule           = 1;
inline constexpr std::int64_t kMinTilesForDynamic        = 480;
inline constexpr std::int64_t kMaxOccupancyAlwaysDynamic = 3;
inline constexpr double kTilesPerCuDynamic               = 8.41;
inline constexpr std::int64_t kMaxDynamicOccupancy       = 3;
inline constexpr std::int64_t kMaxPartialWorkspace       = 128LL * 1024 * 1024;
inline constexpr std::int64_t kMinItersPerCu             = 8;
inline constexpr std::int64_t kMaxTreeIters              = 65536;
inline constexpr std::int64_t kStreamKFlagElements       = 2048;
inline constexpr std::int64_t kCacheLineBytes            = 128;
inline constexpr std::int64_t kFlagBytes                 = 4;
inline constexpr std::int64_t kU32Items                  = 1LL << 32;

// QUEUE_COUNTS of hipblaslt_launch.py; 0 for an arch missing from it.
inline std::int64_t queue_count(const std::string& arch) {
  static constexpr const char* kOneXcd[] = {"gfx90a",
                                            "gfx1100",
                                            "gfx1101",
                                            "gfx1150",
                                            "gfx1151",
                                            "gfx1152",
                                            "gfx1153",
                                            "gfx1200",
                                            "gfx1201"};
  const std::string base                 = arch.substr(0, arch.find(':'));
  if (base == "gfx942" || base == "gfx950") return 8;
  return std::find(std::begin(kOneXcd), std::end(kOneXcd), base) != std::end(kOneXcd) ? 1 : 0;
}

// Flags of the Stream-K fix-up a launch can index; throws for a dynamic-queue
// launch on an arch without a queue count.
inline std::int64_t flag_entries(bool dynamic, std::int64_t queues) {
  if (!dynamic) return kStreamKFlagElements;
  if (queues == 0)
    throw std::invalid_argument(
        "no XCD count for this arch, so the flags its dynamic-queue launches can index are "
        "unknown");
  const std::int64_t prefix = queues * kCacheLineBytes / kFlagBytes;
  return prefix >= kStreamKFlagElements ? kStreamKFlagElements : kStreamKFlagElements - prefix;
}

inline std::int64_t ceil_div(std::int64_t n, std::int64_t d) {
  return d == 0 ? 0 : (n + d - 1) / d;
}

inline std::int64_t attribute(const tilewright::Config& c, const char* name) {
  for (const auto& a : c.attributes)
    if (a.name == name) return a.value;
  return 0;
}

inline std::int64_t partial_bytes(tilewright::DataType mi) {
  using D = tilewright::DataType;
  switch (mi) {
    case D::Double:
    case D::ComplexFloat:
    case D::Int64: return 8;
    case D::ComplexDouble: return 16;
    default: return 4;
  }
}

inline bool hybrid_dynamic(tilewright::Schedule schedule,
                           bool runtime_choice,
                           std::int64_t tiles,
                           std::int64_t sm_target,
                           std::int64_t n_cu,
                           std::int64_t occupancy) {
  if (schedule == tilewright::Schedule::Dynamic) return true;
  if (schedule == tilewright::Schedule::Default && sm_target == 0) return false;
  if (!runtime_choice || tiles <= kMinTilesForDynamic) return false;
  const std::int64_t available = sm_target > 0 ? std::min(sm_target, n_cu) : n_cu;
  if (available == 0 || available >= n_cu) return false;
  if (occupancy <= kMaxOccupancyAlwaysDynamic) return true;
  return static_cast<double>(tiles) / static_cast<double>(available) > kTilesPerCuDynamic;
}

inline std::int64_t static_grid(std::int64_t tiles,
                                std::int64_t cu_count,
                                std::int64_t n_cu,
                                std::int64_t occupancy,
                                std::int64_t iters,
                                bool parallel,
                                std::int64_t tile_bytes,
                                std::int64_t batch) {
  std::int64_t grid = tiles;
  if (cu_count < n_cu) grid = std::min(grid, cu_count);
  if (tiles > cu_count) {
    const std::int64_t virt     = occupancy > 1 ? cu_count * occupancy : cu_count;
    const std::int64_t min_even = std::max<std::int64_t>(1, tiles / virt);
    for (double frac : {0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0}) {
      const auto g = static_cast<std::int64_t>(
          static_cast<double>(tiles) / (static_cast<double>(min_even) + frac) + 0.5);
      if (g != 0 && tiles % g != 0 && tile_bytes * g > kMaxPartialWorkspace) continue;
      if (g <= virt) {
        grid = g;
        break;
      }
    }
  } else if (tiles < cu_count) {
    if (parallel) {
      grid = tiles * std::min(cu_count / tiles, iters / kMinItersPerCu);
    } else {
      for (std::int64_t frac : {16, 12, 8, 6, 4, 3, 2, 1}) {
        if (tiles * frac <= cu_count && iters / frac >= kMinItersPerCu) {
          grid = tiles * frac;
          break;
        }
      }
    }
  }
  if (tiles <= cu_count || grid == 0 || grid >= tiles || batch > 1) return grid;
  if (iters <= 1) return (cu_count * 2 <= tiles && tiles < cu_count * 32) ? tiles : grid;
  const std::int64_t dp_steps = ceil_div(tiles, cu_count) * cu_count;
  if (static_cast<double>(tiles) < 0.8 * static_cast<double>(dp_steps)) return grid;
  return ((tiles * iters / grid) % iters != 0) ? tiles : grid;
}

// The attributes plan() reads, looked up once per kernel.
struct KernelInputs {
  std::int64_t strategy = 0;
  std::int64_t assign   = 0;
  bool atomic           = false;
  std::int64_t threads  = 1;
  bool hand_written     = false;

  bool operator==(const KernelInputs& o) const {
    return strategy == o.strategy && assign == o.assign && atomic == o.atomic &&
           threads == o.threads && hand_written == o.hand_written;
  }
};

inline KernelInputs kernel_inputs(const tilewright::Config& config) {
  KernelInputs in;
  in.strategy = attribute(config, "tile_processing_strategy");
  in.assign   = attribute(config, "work_assignment");
  in.atomic   = attribute(config, "stream_k_atomic") != 0;
  for (const char* name : {"work_group_x", "work_group_y", "work_group_z"})
    in.threads *= std::max<std::int64_t>(attribute(config, name), 0);
  in.hand_written = attribute(config, "hand_written") != 0;
  return in;
}

// A hand-written Stream-K kernel that can run on a static grid (any
// assignment but DynamicWorkQueue): TensileLite reduces its partial tiles in
// a tree only.
inline bool tree_only(const KernelInputs& in) {
  return in.strategy == 2 && in.assign != 1 && in.hand_written;
}

// launch_inputs: what plan() reads besides the tile sizes, with the values it
// does not distinguish made equal, and a sixth value, 1, for a tree-only
// kernel.
struct LaunchInputs {
  std::array<std::int64_t, 6> values{};
  std::size_t size = 5;

  const std::int64_t* begin() const { return values.data(); }
  const std::int64_t* end() const { return values.data() + size; }
};

inline LaunchInputs launch_inputs(const KernelInputs& in, const tilewright::Config& config) {
  LaunchInputs out;
  if (in.strategy != 1 && in.strategy != 2) return out;
  out.values = {in.strategy,
                in.assign,
                in.atomic && in.strategy == 2 ? 1 : 0,
                in.threads,
                std::max<std::int64_t>(config.occupancy, 1),
                1};
  out.size   = tree_only(in) ? 6 : 5;
  return out;
}

inline LaunchInputs launch_inputs(const tilewright::Config& config) {
  return launch_inputs(kernel_inputs(config), config);
}

// FNV-1a 64 of the launch inputs as 8-byte little-endian two's complement
// (features.view_key_hash).
inline std::uint64_t launch_key(const KernelInputs& in, const tilewright::Config& config) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const std::int64_t v : launch_inputs(in, config)) {
    const auto u = static_cast<std::uint64_t>(v);
    for (int byte = 0; byte < 8; ++byte) {
      h ^= (u >> (8 * byte)) & 0xFFu;
      h *= 0x100000001b3ULL;
    }
  }
  return h;
}

inline std::uint64_t launch_key(const tilewright::Config& config) {
  return launch_key(kernel_inputs(config), config);
}

inline tilewright::LaunchPlan plan(const KernelInputs& in,
                                   const tilewright::Config& config,
                                   const tilewright::Problem& problem,
                                   const tilewright::ExecutionContext& context,
                                   std::int64_t n_cu,
                                   bool runtime_choice,
                                   std::int64_t queues) {
  using PR                     = tilewright::PartialReduction;
  const std::int64_t strategy  = in.strategy;
  const std::int64_t assign    = in.assign;
  const bool atomic            = in.atomic;
  const std::int64_t threads   = in.threads;
  const std::int64_t mt_k      = std::max<std::int64_t>(static_cast<std::int64_t>(config.mt.k), 1);
  const std::int64_t mt_m      = std::max<std::int64_t>(static_cast<std::int64_t>(config.mt.m), 1);
  const std::int64_t mt_n      = std::max<std::int64_t>(static_cast<std::int64_t>(config.mt.n), 1);
  const std::int64_t occupancy = std::max<std::int64_t>(config.occupancy, 1);
  const auto k                 = static_cast<std::int64_t>(problem.size.k);
  const auto batch             = static_cast<std::int64_t>(problem.batch);
  const std::int64_t tiles     = ceil_div(static_cast<std::int64_t>(problem.size.m), mt_m) *
                             ceil_div(static_cast<std::int64_t>(problem.size.n), mt_n) * batch;

  if (strategy == 1 && assign != 0)
    throw std::invalid_argument(
        "a DataParallel kernel takes WorkAssignment StaticGrid only; TensileLite does not load "
        "one with a work queue");
  tilewright::LaunchPlan out;
  if ((strategy != 1 && strategy != 2) || n_cu == 0) {
    out.grid_wgs = static_cast<std::size_t>(tiles);
    return out;
  }
  const std::int64_t sm_target =
      static_cast<std::int64_t>(context.cu_budget) / kSmTargetGranule * kSmTargetGranule;
  const std::int64_t cu_count = (sm_target > 0 && sm_target < n_cu) ? sm_target : n_cu;
  const std::int64_t iters    = ceil_div(k, mt_k);
  const bool dynamic =
      assign == 1 ||
      (assign == 2 &&
       hybrid_dynamic(context.schedule, runtime_choice, tiles, sm_target, n_cu, occupancy));
  bool parallel = !tree_only(in) && tiles < cu_count && std::max<std::int64_t>(iters, 1) >= 64 &&
                  tiles <= cu_count / 4;

  std::int64_t grid = tiles;
  if (dynamic) {
    grid = std::min(tiles, n_cu * std::min(occupancy, kMaxDynamicOccupancy));
  } else if (tiles > 0) {
    grid = static_grid(tiles,
                       cu_count,
                       n_cu,
                       occupancy,
                       iters,
                       strategy == 1 ? false : parallel,
                       mt_m * mt_n * partial_bytes(problem.mi_dtype),
                       batch);
  }
  out.dynamic_queue = dynamic;
  if (strategy == 1) {
    const std::int64_t max_grid = (kU32Items - 1) / std::max<std::int64_t>(threads, 1);
    if (grid > max_grid) grid = std::min(n_cu * occupancy, max_grid);
    out.grid_wgs = static_cast<std::size_t>(grid);
    return out;
  }
  if (!parallel && grid > 0) {
    const std::int64_t max_tiles = kU32Items / std::max<std::int64_t>(threads, 1);
    if (iters >= kMaxTreeIters || tiles * iters / grid >= kMaxTreeIters ||
        tiles * iters >= max_tiles)
      grid = tiles >= max_tiles ? n_cu * occupancy : tiles;
  }
  const bool partial = grid > 0 && tiles % grid != 0;
  if (!atomic && !parallel && partial) grid = std::min(grid, flag_entries(dynamic, queues));
  out.grid_wgs = static_cast<std::size_t>(grid);
  if (!dynamic && grid > 0 && tiles % grid != 0)
    out.partial_reduction = atomic ? PR::Atomic : (parallel ? PR::Parallel : PR::Tree);
  return out;
}

inline tilewright::LaunchPlan plan(const tilewright::Config& config,
                                   const tilewright::Problem& problem,
                                   const tilewright::ExecutionContext& context,
                                   std::int64_t n_cu,
                                   bool runtime_choice,
                                   std::int64_t queues) {
  return plan(kernel_inputs(config), config, problem, context, n_cu, runtime_choice, queues);
}

// Given the configs it will see, a Launch looks their attributes up once, by
// Config::index: a config is then taken to be the one of that index it was
// given. Indices it was not given, or given with different attributes, are
// looked up per call. plan() is called for every fully scored config, so an
// integration should likewise not search attributes by name there: hold
// KernelInputs per kernel, or read struct fields as TensileLite's adapter does.
class Launch final : public tilewright::LaunchModel {
 public:
  Launch(std::string arch, std::size_t n_cu)
      : arch_(std::move(arch))
      , n_cu_(static_cast<std::int64_t>(n_cu))
      , runtime_choice_(arch_ == "gfx950")
      , queues_(queue_count(arch_)) {}

  Launch(std::string arch, std::size_t n_cu, const std::vector<tilewright::Config>& configs)
      : Launch(std::move(arch), n_cu) {
    for (const tilewright::Config& c : configs) {
      if (c.index >= inputs_.size()) {
        inputs_.resize(c.index + 1);
        state_.resize(c.index + 1, kNone);
      }
      const KernelInputs in = kernel_inputs(c);
      if (state_[c.index] == kNone) {
        inputs_[c.index] = in;
        state_[c.index]  = kHeld;
      } else if (!(inputs_[c.index] == in)) {
        state_[c.index] = kMixed;
      }
    }
  }

  tilewright::LaunchPlan plan(const tilewright::Config& config,
                              const tilewright::Problem& problem,
                              const tilewright::ExecutionContext& context) const override {
    if (held(config))
      return hipblaslt_launch::plan(
          inputs_[config.index], config, problem, context, n_cu_, runtime_choice_, queues_);
    return hipblaslt_launch::plan(config, problem, context, n_cu_, runtime_choice_, queues_);
  }

  std::uint64_t launch_key(const tilewright::Config& config) const override {
    return held(config) ? hipblaslt_launch::launch_key(inputs_[config.index], config)
                        : hipblaslt_launch::launch_key(config);
  }

  std::string id() const override {
    return "hipblaslt:" + arch_ + ":" + std::to_string(kRuleVersion);
  }

 private:
  enum : unsigned char { kNone, kHeld, kMixed };

  bool held(const tilewright::Config& config) const {
    return config.index < state_.size() && state_[config.index] == kHeld;
  }

  std::string arch_;
  std::int64_t n_cu_;
  bool runtime_choice_;
  std::int64_t queues_;
  std::vector<KernelInputs> inputs_;
  std::vector<unsigned char> state_;
};

}  // namespace hipblaslt_launch
