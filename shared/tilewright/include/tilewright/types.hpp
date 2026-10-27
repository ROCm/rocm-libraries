// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tilewright {

// Element data type. The integer values are stored in model files (MI-latency
// table keys), so members must never be reordered or inserted.
enum class DataType : int {
  Float              = 0,
  Double             = 1,
  ComplexFloat       = 2,
  ComplexDouble      = 3,
  Half               = 4,
  Int8x4             = 5,
  Int32              = 6,
  BFloat16           = 7,
  Int8               = 8,
  Int4               = 9,
  Int64              = 10,
  XFloat32           = 11,
  Float8_fnuz        = 12,
  BFloat8_fnuz       = 13,
  Float8BFloat8_fnuz = 14,
  BFloat8Float8_fnuz = 15,
  Float8             = 16,
  BFloat8            = 17,
  Float8BFloat8      = 18,
  BFloat8Float8      = 19,
  Float6             = 20,
  BFloat6            = 21,
  Float4             = 22,
  Count              = 23,
  None               = Count
};

enum class Transpose : int { T = 0, N = 1, Count = 2 };

// Storage type of a model's weight matrices.
enum class WeightType : std::uint8_t { Fp32 = 0, Bf16 = 1, Int8 = 2, Int4 = 3 };

struct Dim3 {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;

  constexpr std::size_t mn() const noexcept { return m * n; }
  constexpr std::size_t mk() const noexcept { return m * k; }
  constexpr std::size_t nk() const noexcept { return n * k; }
};

// A GEMM problem: D = op(A) * op(B) with A MxK, B KxN, batched `batch` times.
struct Problem {
  Dim3 size{0, 0, 0};
  std::size_t batch = 1;

  Transpose a_transpose = Transpose::N;
  Transpose b_transpose = Transpose::N;

  DataType a_dtype  = DataType::None;
  DataType b_dtype  = DataType::None;
  DataType c_dtype  = DataType::None;
  DataType d_dtype  = DataType::None;
  DataType mi_dtype = DataType::None;
};

// A named integer property of a kernel. Feature catalogs read the names they
// know and ignore the rest.
struct Attribute {
  std::string name;
  std::int64_t value = 0;
};

// A candidate kernel. `index` is the caller's identifier for the kernel and is
// not interpreted by the engine.
struct Config {
  Dim3 mt{0, 0, 0};
  Dim3 mi{0, 0, 0};

  int occupancy = -1;

  int cache_hints_a = 0;
  int cache_hints_b = 0;

  std::size_t grvw_a = 1;
  std::size_t grvw_b = 1;
  std::size_t gwvw_d = 1;

  std::size_t index = 0;

  // Further kernel properties, in any order; names must be non-empty and unique.
  std::vector<Attribute> attributes;
};

// Device properties queried at runtime. Constants that a model was trained
// with (matrix-instruction latencies, bandwidth model) live in the model file.
struct Hardware {
  std::size_t N_CU         = 0;
  std::size_t lds_capacity = 0;  // bytes
  std::size_t L2_capacity  = 0;  // bytes
};

// Tile scheduling the caller will use for the ranked kernels.
enum class Schedule : std::uint8_t { Default = 0, Dynamic = 1, Auto = 2 };

// Where the ranked kernels will run. The defaults describe exclusive use of the
// device.
struct ExecutionContext {
  std::size_t cu_budget = 0;  // CUs the kernels may use; 0, or >= Hardware::N_CU, means every CU
  Schedule schedule     = Schedule::Default;
};

// How ranking orders configs with equal scores: in input order (PoolOrder) or by
// a caller-supplied prior, then input order (Prior); see CandidateSet::rank.
enum class TieBreak : std::uint8_t { PoolOrder = 0, Prior = 1 };

// How the workgroups that split one output tile's K loop combine their partial
// results: by the workgroups themselves (Tree), by a separate reduction pass
// over a workspace (Parallel), or by atomic adds into the output (Atomic).
enum class PartialReduction : std::uint8_t { None = 0, Tree = 1, Parallel = 2, Atomic = 3 };

// How a kernel is launched for one problem, in terms common to GEMM
// frameworks. The default is a static launch of one workgroup per output tile.
struct LaunchPlan {
  bool dynamic_queue                 = false;  // workgroups take output tiles from a shared queue
  std::size_t grid_wgs               = 0;      // workgroups launched; 0 means one per output tile
  PartialReduction partial_reduction = PartialReduction::None;  // None: no split tile
};

// The integration's launch of its kernels on the device it ranks for.
//
// launch_key() names a config's launch class. Contract: configs with equal
// signature (mt, mi, cache hints), equal attribute-spec values and equal
// launch key get the same plan() for every problem and context. The engine
// never interprets the key; it only compares keys, and configs that differ in
// it are separate candidates of an MLREC_v3 model whose serving options ask
// for launch keys (ModelInfo::launch_keys); other models ignore it. The
// default, 0 for every config, declares that every config of a signature
// launches alike.
//
// Ranking asks plan() only for configs that receive the full score, once per
// group of configs that share a candidate (see CandidateSet), and
// launch_key() once per config when a pool of such a model is grouped. Both
// must be thread-safe when the CandidateSet is shared between threads;
// exceptions they throw propagate out of the call that asked. plan() is on
// the ranking's path: what it needs of a config should be held per config
// (for instance by Config::index, prepared when the pool is built) rather
// than searched for in `attributes` by name on each call.
class LaunchModel {
 public:
  virtual ~LaunchModel()                                         = default;
  virtual LaunchPlan plan(const Config& config,
                          const Problem& problem,
                          const ExecutionContext& context) const = 0;
  virtual std::uint64_t launch_key(const Config&) const { return 0; }
  // Names the launch rule and its version, as the training pipeline records
  // it in an MLREC_v3 model's provenance (launch_model); a CandidateSet
  // refuses a launch model whose id differs from the one its model records.
  // Empty for none.
  virtual std::string id() const { return {}; }
  // Asks the memory system for what plan() will read of `config` (its
  // per-config record, for instance) without waiting for it. In an engine
  // built with TW_LEVER_PREFETCH, a ranking whose first tier is a cell
  // whitelist calls it during the query tower for the first config of each
  // key of that tier, before it plans them, so that the misses of their
  // records overlap. It must not change what plan() or launch_key() return,
  // and must not throw. The default asks for nothing.
  virtual void prefetch(const Config&) const noexcept {}
};

// Fixed per CandidateSet. Without `launch`, every config gets LaunchPlan{}
// and launch key 0.
struct PoolOptions {
  TieBreak tie_break = TieBreak::PoolOrder;
  std::shared_ptr<const LaunchModel> launch;
};

// Ranking result for one input config, which is configs[config_index] of the
// ranked list. `score` is meaningful only when `scored` is true; higher is better.
struct Result {
  std::size_t config_index = 0;
  double score             = 0.0;
  bool scored              = false;
};

}  // namespace tilewright
