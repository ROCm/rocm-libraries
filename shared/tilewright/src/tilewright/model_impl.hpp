// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tilewright {
namespace detail {

struct FeatureCatalog;

// Number of base grid labels: 4 M tiers x 4 N tiers x 3 K tiers x 2 batch tiers.
constexpr std::size_t kBaseCells = 96;

// (mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, cache_hints_a, cache_hints_b)
using Signature = std::array<int, 8>;

// A tensor stored in Model::payload: `offset` is the byte offset of its first
// stored byte (the per-tensor scale for int8/int4 weights).
struct TensorRef {
  std::size_t offset = 0;
  std::size_t count  = 0;
};

// One trained cell as stored in the payload. Its tensors stay packed until the
// first rank call that reaches the cell (see Model::weights).
struct Cell {
  std::string label;
  std::uint32_t embed_dim    = 0;
  std::uint32_t hidden_dim   = 0;
  std::uint32_t inter_hidden = 0;
  float temperature          = 1.0f;
  std::vector<Signature> signatures;

  TensorRef q_mean, q_std, i_mean, i_std, x_mean, x_std;
  TensorRef q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  TensorRef i_w0, i_b0, i_w2, i_b2;
  TensorRef x_w0, x_b0, x_w2, x_b2;
};

// fp32 tensors of one cell. A feature whitens to (x - mean) / scale, where
// `*_scale` is the stored std with values below 1e-6 replaced by 1. An item
// feature whose stored std is below 1e-3 (`i_constant`) was constant in
// training: a value further than twice that std from its mean whitens to 0.
// `temperature` is max(|T|, 0.1).
struct CellWeights {
  std::vector<float> q_mean, q_scale, i_mean, i_scale, x_mean, x_scale;
  std::vector<float> i_std;
  std::vector<unsigned char> i_constant;
  std::vector<float> q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  std::vector<float> i_w0, i_b0, i_w2, i_b2;
  std::vector<float> x_w0, x_b0, x_w2;
  float x_b2        = 0.0f;
  float temperature = 1.0f;
};

// A label of the split tree. `split` indexes Model::splits when the label is
// split further; `cell` is the nearest trained cell along the label's
// '#'-separated ancestry (the label itself included), or -1.
struct Node {
  int split = -1;
  int cell  = -1;
};

// Values of `axis` at or below `threshold` route to `lo`.
struct Split {
  char axis              = 'M';
  std::int32_t threshold = 0;
  int lo                 = -1;
  int hi                 = -1;
};

struct MiEntry {
  std::uint32_t m    = 0;
  std::uint32_t n    = 0;
  std::uint32_t k    = 0;
  std::int32_t dtype = 0;
  double cycles      = 0.0;
};

}  // namespace detail

class Model {
 public:
  std::string path;
  std::string arch;
  std::string feature_hash;
  const detail::FeatureCatalog* catalog = nullptr;
  WeightType weight_type                = WeightType::Fp32;
  double parallel_mi_cu                 = 1.0;
  double bw_coef[3]                     = {0.0, 0.0, 0.0};
  double mi_default_cycles              = 32.0;
  std::vector<detail::MiEntry> mi_table;

  std::vector<detail::Cell> cells;
  std::vector<detail::Split> splits;
  std::vector<detail::Node> nodes;
  std::array<int, detail::kBaseCells> base_nodes{};

  // The payload bytes of the model file (split and cell records).
  std::vector<unsigned char> payload;

  Model()                        = default;
  Model(const Model&)            = delete;
  Model& operator=(const Model&) = delete;

  // fp32 tensors of cell `cell`, dequantized on the first call for that cell.
  // Thread-safe. Throws std::bad_alloc when memory runs out.
  const detail::CellWeights& weights(std::size_t cell) const;

  void init_lazy_state();

 private:
  struct LazyCell {
    std::once_flag once;
    std::unique_ptr<const detail::CellWeights> weights;
  };
  std::unique_ptr<LazyCell[]> lazy_;
};

namespace detail {

// Parses an MLREC_v2 image. Returns nullptr and sets *error on any defect.
// May throw std::bad_alloc.
std::unique_ptr<Model> parse_model(const unsigned char* data, std::size_t size, std::string* error);

// CRC-32/ISO-HDLC.
std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept;

// Environment knobs, read once.
struct EnvKnobs {
  bool diag            = false;
  bool pick_log        = false;
  long long force_cell = -1;
};
const EnvKnobs& env_knobs() noexcept;

bool valid_hardware(const Hardware& hardware) noexcept;

// Cell scoring `problem` (honoring TILEWRIGHT_FORCE_CELL), or -1.
int route_cell(const Model& model, const Problem& problem) noexcept;

// A pool's configs grouped by view key (FeatureCatalog::view_key). Keys are
// numbered in order of first appearance; the configs of key k are
// members[member_begin[k] .. member_begin[k + 1]), in input order.
struct PoolKeys {
  std::vector<std::size_t> key_of;  // per config
  std::vector<std::size_t> first;   // per key: its first config
  std::vector<std::size_t> member_begin;
  std::vector<std::size_t> members;
  bool nt_a = false;  // some config has cache_hints_a == 4
  bool nt_b = false;  // some config has cache_hints_b == 4
};

PoolKeys pool_keys(const FeatureCatalog& catalog, const std::vector<Config>& configs);

// rank_configs over the grouping `keys`, scoring each group once. A grouping
// that joins only configs with equal view keys (for example one group per
// config) gives the same result as pool_keys' grouping.
std::vector<Result> rank_keys(const Model& model,
                              const Problem& problem,
                              const Hardware& hardware,
                              const ExecutionContext& context,
                              const std::vector<Config>& configs,
                              const PoolKeys& keys,
                              std::size_t min_scored,
                              TieBreak tie_break,
                              const std::vector<double>* prior);

}  // namespace detail
}  // namespace tilewright
