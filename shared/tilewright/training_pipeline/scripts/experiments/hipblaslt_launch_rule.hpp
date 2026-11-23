// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipBLASLt's launch of its kernels as a tilewright::LaunchModel: the rule of
// training_pipeline/lib/hipblaslt_launch.py (launch_plan), which MLREC_v3
// models record as "hipblaslt:<arch>:<rule version>", with its launch_key.
// Both must give the same plan and key for every input and rule version;
// bench_shipped_rank --dump-plans prints these for the comparison.

#pragma once

#include "tilewright/types.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipblaslt_launch {

// The last of RULE_VERSIONS of hipblaslt_launch.py; plan() implements versions
// 1 to it.
inline constexpr int kRuleVersion = 4;
// RULE_VERSION of hipblaslt_launch.py: the version new models train on, but on
// the architectures rule_version() names.
inline constexpr int kTrainedRuleVersion = 3;
// A workspace that holds every partial tile (ExecutionContext::workspace_bytes'
// default), which versions 1 and 2 plan with.
inline constexpr std::uint64_t kUnboundedWorkspace = std::numeric_limits<std::uint64_t>::max();

inline constexpr std::int64_t kSmTargetGranule           = 1;
inline constexpr std::int64_t kMinTilesForDynamic        = 480;
inline constexpr std::int64_t kMaxOccupancyAlwaysDynamic = 3;
inline constexpr double kTilesPerCuDynamic               = 8.41;
inline constexpr std::int64_t kMaxDynamicOccupancy       = 3;
inline constexpr std::int64_t kMaxPartialWorkspace       = 128LL * 1024 * 1024;
inline constexpr std::int64_t kMinItersPerCu             = 8;
inline constexpr std::int64_t kMaxTreeIters              = 65536;
inline constexpr std::int64_t kMaxTreeDividend           = 1LL << 24;
inline constexpr std::int64_t kSkFullTiles               = 1;
inline constexpr std::int64_t kSpatialClusterLayout      = 2;
inline constexpr std::int64_t kStreamKFlagElements       = 2048;
inline constexpr std::int64_t kCacheLineBytes            = 128;
inline constexpr std::int64_t kFlagBytes                 = 4;
inline constexpr std::int64_t kU32Items                  = 1LL << 32;

// canonical_arch of lib/arch.py: the compiler target of an arch name.
inline std::string canonical_arch(const std::string& name) {
  std::string arch = name.substr(0, name.find(':'));
  if (arch.rfind("gfx", 0) == 0) arch = arch.substr(0, arch.find('-'));
  return arch;
}

// rule_version of hipblaslt_launch.py: the rule version models of `arch`
// train on (1 for RULE_1_ARCHS, which holds none, 4 for RULE_4_ARCHS, else
// kTrainedRuleVersion). Those tables and TensileLite's
// TilewrightLaunch::ruleVersion() hold the same: change them together.
inline int rule_version(const std::string& arch) {
  return canonical_arch(arch) == "gfx1250" ? 4 : kTrainedRuleVersion;
}

// served_rule_version of hipblaslt_launch.py for a launch id
// "hipblaslt:<arch>:<version>": true, with its arch and version, when that
// version ranks a model recording the id (any version plan() implements, else
// the arch's rule_version), false for any other id.
inline bool served_launch(const std::string& id, std::string* arch, int* version) {
  const std::string head  = "hipblaslt:";
  const std::size_t colon = id.rfind(':');
  if (id.compare(0, head.size(), head) != 0 || colon == std::string::npos || colon <= head.size())
    return false;
  const std::string name = id.substr(head.size(), colon - head.size());
  const std::string tail = id.substr(colon + 1);
  int served             = rule_version(name);
  for (int v = 1; v <= kRuleVersion; ++v)
    if (tail == std::to_string(v)) served = v;
  if (name.find(':') != std::string::npos || tail != std::to_string(served)) return false;
  *arch    = name;
  *version = served;
  return true;
}

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

inline bool has_attribute(const tilewright::Config& c, const char* name) {
  for (const auto& a : c.attributes)
    if (a.name == name) return true;
  return false;
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
                                std::int64_t batch,
                                int version,
                                std::uint64_t workspace = kUnboundedWorkspace) {
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
  // A grid that splits tiles needs their partial tiles in the workspace.
  if (grid > 0 && tiles % grid != 0 &&
      static_cast<std::uint64_t>(tile_bytes) * static_cast<std::uint64_t>(grid) > workspace)
    grid = tiles;
  if (tiles <= cu_count || grid == 0 || grid >= tiles || batch > 1) return grid;
  if (iters <= 1) {
    if (version >= 2) return grid > cu_count * 2 ? tiles : grid;
    return (cu_count * 2 <= tiles && tiles < cu_count * 32) ? tiles : grid;
  }
  const std::int64_t dp_steps = ceil_div(tiles, cu_count) * cu_count;
  if (static_cast<double>(tiles) < 0.8 * static_cast<double>(dp_steps)) return grid;
  return ((tiles * iters / grid) % iters != 0) ? tiles : grid;
}

// The attributes plan() reads, looked up once per kernel: those of
// LAUNCH_ATTRIBUTES, and those LAUNCH_ATTRIBUTES_V3 and _V4 add.
struct KernelInputs {
  std::int64_t strategy = 0;
  std::int64_t assign   = 0;
  bool atomic           = false;
  std::int64_t threads  = 1;
  bool hand_written     = false;
  bool per_tile         = false;
  // -1 for a kernel without workspace_size_per_elem_c.
  std::int64_t workspace_per_elem = -1;
  bool obstacle                   = false;
  std::int64_t layout             = 0;
  std::int64_t cluster_x          = 1;
  std::int64_t cluster_y          = 1;
  std::int64_t k_unit             = 1;
  bool cluster_tree_only          = false;

  bool operator==(const KernelInputs& o) const {
    return strategy == o.strategy && assign == o.assign && atomic == o.atomic &&
           threads == o.threads && hand_written == o.hand_written && per_tile == o.per_tile &&
           workspace_per_elem == o.workspace_per_elem && obstacle == o.obstacle &&
           layout == o.layout && cluster_x == o.cluster_x && cluster_y == o.cluster_y &&
           k_unit == o.k_unit && cluster_tree_only == o.cluster_tree_only;
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
  in.per_tile     = attribute(config, "per_tile_extra_iters") != 0;
  if (has_attribute(config, "workspace_size_per_elem_c"))
    in.workspace_per_elem = attribute(config, "workspace_size_per_elem_c");
  in.obstacle = attribute(config, "uso_static_obstacle") != 0;
  in.layout   = attribute(config, "persistent_loop_args_version");
  if (has_attribute(config, "cluster_dim_x")) in.cluster_x = attribute(config, "cluster_dim_x");
  if (has_attribute(config, "cluster_dim_y")) in.cluster_y = attribute(config, "cluster_dim_y");
  in.k_unit            = std::max<std::int64_t>(attribute(config, "sk_k_granularity"), 1);
  in.cluster_tree_only = attribute(config, "cluster_tree_only") != 0;
  return in;
}

// spatial_cluster_kernel of hipblaslt_launch.py: a Stream-K kernel of
// TensileLite's spatial-cluster argument layout.
inline bool spatial_cluster(const KernelInputs& in) {
  return in.strategy == 2 && in.layout == kSpatialClusterLayout;
}

// A hand-written Stream-K kernel that can run on a static grid (any
// assignment but DynamicWorkQueue): TensileLite reduces its partial tiles in
// a tree only.
inline bool tree_only(const KernelInputs& in) {
  return in.strategy == 2 && in.assign != 1 && in.hand_written;
}

// A version 3 Stream-K kernel's partial-tile element size; throws for one
// without, as launch_inputs() of hipblaslt_launch.py does.
inline std::int64_t workspace_per_elem(const KernelInputs& in) {
  if (in.workspace_per_elem < 0)
    throw std::invalid_argument(
        "a Stream-K kernel without workspace_size_per_elem_c has no launch under rule version 3");
  return in.workspace_per_elem;
}

// launch_inputs: what plan() reads besides the tile sizes under rule
// `version`, with the values it does not distinguish made equal, and a sixth
// value, 1, for a tree-only kernel. From version 3 on a Stream-K kernel has
// nine: the five, whether it is tree only, its per-tile extras (0 for a
// dynamic queue), its partial-tile element size and whether it has a static
// uniform-summation-order obstacle (its atomic fix-up is one). From version 4
// on a kernel of the spatial-cluster layout has thirteen: the nine, tree only
// also for its cluster_tree_only, then the layout, its cluster's x and y and
// its unit of K.
struct LaunchInputs {
  std::array<std::int64_t, 13> values{};
  std::size_t size = 5;

  const std::int64_t* begin() const { return values.data(); }
  const std::int64_t* end() const { return values.data() + size; }
};

inline LaunchInputs launch_inputs(const KernelInputs& in,
                                  const tilewright::Config& config,
                                  int version = kTrainedRuleVersion) {
  LaunchInputs out;
  if (in.strategy != 1 && in.strategy != 2) return out;
  const bool atomic = in.atomic && in.strategy == 2;
  out.values        = {in.strategy,
                       in.assign,
                atomic ? 1 : 0,
                       in.threads,
                       std::max<std::int64_t>(config.occupancy, 1),
                       1};
  out.size          = tree_only(in) ? 6 : 5;
  if (version >= 3 && in.strategy == 2) {
    const bool cluster = version >= 4 && spatial_cluster(in);
    out.values[5]      = tree_only(in) || (cluster && in.cluster_tree_only) ? 1 : 0;
    out.values[6]      = in.per_tile && in.assign != 1 ? 1 : 0;
    out.values[7]      = workspace_per_elem(in);
    out.values[8]      = atomic || in.obstacle ? 1 : 0;
    out.size           = 9;
    if (cluster) {
      out.values[9]  = kSpatialClusterLayout;
      out.values[10] = in.cluster_x;
      out.values[11] = in.cluster_y;
      out.values[12] = in.k_unit;
      out.size       = 13;
    }
  }
  return out;
}

inline LaunchInputs launch_inputs(const tilewright::Config& config,
                                  int version = kTrainedRuleVersion) {
  return launch_inputs(kernel_inputs(config), config, version);
}

// FNV-1a 64 of the launch inputs as 8-byte little-endian two's complement
// (features.view_key_hash).
inline std::uint64_t launch_key(const KernelInputs& in,
                                const tilewright::Config& config,
                                int version = kTrainedRuleVersion) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const std::int64_t v : launch_inputs(in, config, version)) {
    const auto u = static_cast<std::uint64_t>(v);
    for (int byte = 0; byte < 8; ++byte) {
      h ^= (u >> (8 * byte)) & 0xFFu;
      h *= 0x100000001b3ULL;
    }
  }
  return h;
}

inline std::uint64_t launch_key(const tilewright::Config& config,
                                int version = kTrainedRuleVersion) {
  return launch_key(kernel_inputs(config), config, version);
}

// _uniform_grid of hipblaslt_launch.py: getPersistentGridImpl's snap of a
// static two-tile Stream-K grid under uniform summation order.
inline std::int64_t uniform_grid(std::int64_t g0,
                                 std::int64_t tiles,
                                 std::int64_t iters,
                                 bool per_tile,
                                 bool parallel,
                                 std::int64_t partial_bytes,
                                 std::uint64_t workspace,
                                 bool flag_region_binds) {
  if (tiles == 0 || g0 == 0 || g0 == tiles) return g0;
  if (g0 < tiles) return tiles;
  for (std::int64_t f = std::min(g0 / tiles, iters / kMinItersPerCu); f >= 2; --f) {
    if (!parallel && !per_tile && iters % f != 0) continue;
    if (iters / f < kMinItersPerCu) continue;
    if (static_cast<std::uint64_t>(partial_bytes) * static_cast<std::uint64_t>(tiles * f) >
        workspace)
      continue;
    if (flag_region_binds && tiles * f > kStreamKFlagElements) continue;
    return tiles * f;
  }
  return tiles;
}

// _padded_iters, _static_split and _static_split_uniform of
// hipblaslt_launch.py: streamKStaticSplit and its row-uniformity under uniform
// summation order, with K in units of `k_unit` iterations.
inline std::int64_t padded_iters(std::int64_t iters, std::int64_t k_unit) {
  return k_unit <= 1 || iters == 0 ? iters : ceil_div(iters, k_unit) * k_unit;
}

struct StaticSplit {
  std::int64_t sk_tiles = 0;
  std::int64_t per_wg   = 0;
  std::int64_t extra    = 0;
};

inline StaticSplit static_split(std::int64_t tiles,
                                std::int64_t iters,
                                std::int64_t grid,
                                std::int64_t full_tiles,
                                std::int64_t k_unit) {
  StaticSplit s;
  if (grid == 0) return s;
  s.sk_tiles = grid;
  if (tiles % grid != 0)
    s.sk_tiles = std::min(tiles > grid ? grid * full_tiles + tiles % grid : tiles, tiles);
  const std::int64_t unit     = std::max<std::int64_t>(k_unit, 1);
  const std::int64_t sk_iters = s.sk_tiles * padded_iters(iters, unit);
  s.per_wg                    = unit * (sk_iters / unit / grid);
  s.extra                     = sk_iters - s.per_wg * grid;
  return s;
}

inline bool static_split_uniform(const StaticSplit& s,
                                 std::int64_t tiles,
                                 std::int64_t iters,
                                 std::int64_t grid,
                                 bool per_tile,
                                 std::int64_t k_unit) {
  if (s.sk_tiles == 0) return true;
  if (s.per_wg != 0 && s.extra == 0 && s.sk_tiles == tiles &&
      padded_iters(iters, k_unit) % s.per_wg == 0)
    return true;
  return per_tile && s.sk_tiles == tiles && tiles != 0 && grid % tiles == 0 && s.per_wg != 0;
}

// _spatial_cluster_plan of hipblaslt_launch.py: resolveClusteredStreamKSettings
// for a kernel of the spatial-cluster layout whose reduction is `parallel` or
// tree and whose selected grid is `selected` workgroups over `tiles` tiles.
inline tilewright::LaunchPlan spatial_cluster_plan(const KernelInputs& in,
                                                   const LaunchInputs& li,
                                                   const tilewright::Problem& problem,
                                                   std::int64_t mt_m,
                                                   std::int64_t mt_n,
                                                   std::int64_t iters,
                                                   std::int64_t tiles,
                                                   bool parallel,
                                                   std::int64_t selected,
                                                   std::int64_t n_cu,
                                                   std::int64_t occupancy,
                                                   std::uint64_t workspace,
                                                   bool uso) {
  tilewright::LaunchPlan out;
  const std::int64_t cluster    = in.cluster_x * in.cluster_y;
  const std::int64_t elem_bytes = li.values[7];
  const auto k                  = static_cast<std::int64_t>(problem.size.k);
  const auto batch              = static_cast<std::int64_t>(problem.batch);
  const std::int64_t blocks =
      ceil_div(ceil_div(static_cast<std::int64_t>(problem.size.m), mt_m), in.cluster_x) *
      ceil_div(ceil_div(static_cast<std::int64_t>(problem.size.n), mt_n), in.cluster_y) * batch;
  iters = std::max<std::int64_t>(iters, 1);
  if (blocks == 0) return out;
  const bool whole_tile = selected == tiles;
  std::int64_t budget   = whole_tile ? blocks * cluster : selected;
  if (selected > tiles)
    budget = parallel ? blocks * cluster * std::min(selected / tiles, iters)
                      : std::max(budget, blocks * cluster);
  if (in.threads == 0) throw std::invalid_argument("a spatial cluster of workgroups of no threads");
  const std::int64_t launch_limit = (kU32Items - 1) / in.threads;
  const std::int64_t max_physical =
      whole_tile ? launch_limit : std::min({launch_limit, kStreamKFlagElements, n_cu * occupancy});
  if (max_physical < cluster)
    throw std::invalid_argument("a spatial cluster does not fit the device's grid limits");
  std::int64_t grid =
      std::min({std::max(budget, cluster) / cluster, max_physical / cluster, blocks * iters});
  auto whole_blocks = [&](std::int64_t g) {
    out.grid_wgs = static_cast<std::size_t>(std::min(g, blocks) * cluster);
    return out;
  };
  if (parallel && k != 0) {
    const std::int64_t splits = std::min(grid / blocks, iters);
    if (splits >= 2) {
      grid                    = blocks * splits;
      const auto output_bytes = static_cast<std::uint64_t>(problem.size.m) * problem.size.n *
                                static_cast<std::uint64_t>(batch) *
                                static_cast<std::uint64_t>(elem_bytes);
      if (output_bytes * static_cast<std::uint64_t>(splits) > workspace) return whole_blocks(grid);
      out.grid_wgs          = static_cast<std::size_t>(grid * cluster);
      out.partial_reduction = tilewright::PartialReduction::Parallel;
      return out;
    }
  }
  if (k == 0 || whole_tile) return whole_blocks(grid);
  StaticSplit split =
      static_split(blocks, iters, grid, std::min(kSkFullTiles, blocks / grid), in.k_unit);
  if (uso && !static_split_uniform(split, blocks, iters, grid, li.values[6] != 0, in.k_unit)) {
    std::int64_t splits = grid / blocks;
    while (splits >= 2 && (iters % splits != 0 || iters / splits < kMinItersPerCu)) --splits;
    if (splits < 2) return whole_blocks(grid);
    grid  = blocks * splits;
    split = static_split(blocks, iters, grid, 0, in.k_unit);
  }
  if (split.extra == 0 && split.per_wg % iters == 0) return whole_blocks(grid);
  if (iters >= kMaxTreeIters || split.per_wg >= kMaxTreeIters || blocks * iters >= kMaxTreeDividend)
    return whole_blocks(grid);
  const std::int64_t partial = mt_m * mt_n * elem_bytes;
  if (partial == 0)
    throw std::invalid_argument("the partial tiles of a spatial cluster need its element size");
  if (static_cast<std::uint64_t>(grid * cluster) * static_cast<std::uint64_t>(partial) > workspace)
    return whole_blocks(grid);
  out.grid_wgs          = static_cast<std::size_t>(grid * cluster);
  out.partial_reduction = tilewright::PartialReduction::Tree;
  return out;
}

inline tilewright::LaunchPlan plan(const KernelInputs& in,
                                   const tilewright::Config& config,
                                   const tilewright::Problem& problem,
                                   const tilewright::ExecutionContext& context,
                                   std::int64_t n_cu,
                                   bool runtime_choice,
                                   std::int64_t queues,
                                   int version) {
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
  const LaunchInputs li = launch_inputs(in, config, version);
  bool parallel         = !(li.size > 5 && li.values[5] == 1) && tiles < cu_count &&
                  std::max<std::int64_t>(iters, 1) >= 64 && tiles <= cu_count / 4;
  // From version 3 on, the call's uniform summation order and workspace, and
  // a Stream-K kernel's partial tiles of its own element size.
  const bool uso = version >= 3 && context.uniform_summation_order;
  const std::uint64_t workspace =
      version >= 3 ? static_cast<std::uint64_t>(context.workspace_bytes) : kUnboundedWorkspace;
  const bool v3_stream_k        = version >= 3 && strategy == 2;
  const std::int64_t tile_bytes = v3_stream_k ? mt_m * mt_n * workspace_per_elem(in)
                                              : mt_m * mt_n * partial_bytes(problem.mi_dtype);
  if (v3_stream_k) {
    // Static two-tile packing keeps parallel only for a non-atomic kernel
    // without a static obstacle (getSKReduction).
    if (dynamic || (uso && (atomic || in.obstacle))) parallel = false;
  }

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
                       tile_bytes,
                       batch,
                       version,
                       workspace);
    if (li.size == 13)
      return spatial_cluster_plan(in,
                                  li,
                                  problem,
                                  mt_m,
                                  mt_n,
                                  iters,
                                  tiles,
                                  parallel,
                                  grid,
                                  n_cu,
                                  occupancy,
                                  workspace,
                                  uso);
    if (v3_stream_k && uso)
      grid = uniform_grid(grid,
                          tiles,
                          std::max<std::int64_t>(iters, 1),
                          in.per_tile && assign != 1,
                          parallel,
                          tile_bytes,
                          workspace,
                          !atomic && !parallel);
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
  if (v3_stream_k) {
    // A parallel reduction needs two workgroups per tile
    // (streamKReconcileReduction).
    if (parallel && (tiles == 0 || grid / tiles < 2)) parallel = false;
    // A launch that writes partial tiles the workspace cannot hold runs one
    // workgroup per tile with a tree reduction (resolveStreamKSettings).
    const bool straddles_tiles =
        grid > 0 && std::max<std::int64_t>(iters, 1) > 1 && tiles % grid != 0;
    if (grid > 0 && (parallel || straddles_tiles) &&
        static_cast<std::uint64_t>(tile_bytes) * static_cast<std::uint64_t>(grid) > workspace) {
      parallel = false;
      grid     = tiles;
    }
  }
  out.grid_wgs = static_cast<std::size_t>(grid);
  // With one K iteration per tile no tile has partial results (version 2).
  const bool straddles = iters > 1 || version < 2;
  if (!dynamic && grid > 0 && tiles % grid != 0 && straddles)
    out.partial_reduction = atomic ? PR::Atomic : (parallel ? PR::Parallel : PR::Tree);
  return out;
}

inline tilewright::LaunchPlan plan(const tilewright::Config& config,
                                   const tilewright::Problem& problem,
                                   const tilewright::ExecutionContext& context,
                                   std::int64_t n_cu,
                                   bool runtime_choice,
                                   std::int64_t queues,
                                   int version) {
  return plan(
      kernel_inputs(config), config, problem, context, n_cu, runtime_choice, queues, version);
}

// Given the configs it will see, a Launch looks their attributes up once, by
// Config::index: a config is then taken to be the one of that index it was
// given. Indices it was not given, or given with different attributes, are
// looked up per call. plan() is called for every fully scored config, so an
// integration should likewise not search attributes by name there: hold
// KernelInputs per kernel, or read struct fields as TensileLite's adapter does.
class Launch final : public tilewright::LaunchModel {
 public:
  // Under rule `version`, 0 for the arch's rule_version(); throws for a
  // version plan() does not implement.
  Launch(std::string arch, std::size_t n_cu, int version = 0)
      : arch_(std::move(arch))
      , n_cu_(static_cast<std::int64_t>(n_cu))
      , runtime_choice_(arch_ == "gfx950")
      , queues_(queue_count(arch_))
      , version_(version == 0 ? rule_version(arch_) : version) {
    if (version_ < 1 || version_ > kRuleVersion)
      throw std::invalid_argument("no launch rule version " + std::to_string(version_));
  }

  Launch(std::string arch,
         std::size_t n_cu,
         const std::vector<tilewright::Config>& configs,
         int version = 0)
      : Launch(std::move(arch), n_cu, version) {
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
      return hipblaslt_launch::plan(inputs_[config.index],
                                    config,
                                    problem,
                                    context,
                                    n_cu_,
                                    runtime_choice_,
                                    queues_,
                                    version_);
    return hipblaslt_launch::plan(
        config, problem, context, n_cu_, runtime_choice_, queues_, version_);
  }

  std::uint64_t launch_key(const tilewright::Config& config) const override {
    return held(config) ? hipblaslt_launch::launch_key(inputs_[config.index], config, version_)
                        : hipblaslt_launch::launch_key(config, version_);
  }

  std::string id() const override { return "hipblaslt:" + arch_ + ":" + std::to_string(version_); }

 private:
  enum : unsigned char { kNone, kHeld, kMixed };

  bool held(const tilewright::Config& config) const {
    return config.index < state_.size() && state_[config.index] == kHeld;
  }

  std::string arch_;
  std::int64_t n_cu_;
  bool runtime_choice_;
  std::int64_t queues_;
  int version_;
  std::vector<KernelInputs> inputs_;
  std::vector<unsigned char> state_;
};

}  // namespace hipblaslt_launch
