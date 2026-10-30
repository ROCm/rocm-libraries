// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "kernels.hpp"
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

// The largest drift-fallback share (Model::drift_keep_ppm), in millionths: at
// it a pool must hold every trained whitelisted view key of a cell.
constexpr std::uint32_t kMaxDriftKeepPpm = 1000000;

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
  // The whitelist: configs whose signature is listed, or whose view-key hash
  // (view_key_hash) is listed (MLREC_v3 only), ascending.
  std::vector<Signature> signatures;
  std::vector<std::uint64_t> view_key_hashes;

  TensorRef q_mean, q_std, i_mean, i_std, x_mean, x_std;
  // Raw feature bounds (MLREC_v3 files with clipping; count 0 otherwise).
  TensorRef q_lo, q_hi, i_lo, i_hi, x_lo, x_hi;
  TensorRef q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  TensorRef i_w0, i_b0, i_w2, i_b2;
  TensorRef x_w0, x_b0, x_w2, x_b2;
};

// A weight matrix in the BlockWeights layout (kernels.hpp) at one precision.
struct BlockMatrix {
  BlockWeights::Type type = BlockWeights::Type::F32;
  float scale             = 1.0f;
  std::vector<float> f32;
  std::vector<std::uint16_t> bf16;
  std::vector<std::int8_t> int8;

  BlockWeights view() const {
    const void* data = type == BlockWeights::Type::F32    ? static_cast<const void*>(f32.data())
                       : type == BlockWeights::Type::Bf16 ? static_cast<const void*>(bf16.data())
                                                          : static_cast<const void*>(int8.data());
    return {type, data, scale};
  }
  std::size_t bytes() const {
    return f32.size() * sizeof(float) + bf16.size() * sizeof(std::uint16_t) + int8.size();
  }
};

// Tensors of one cell: fp32, except that the weight matrices of the two towers
// are BlockMatrix at the precision the file stores (int4 as int8). A feature
// whitens to (x - mean) / scale, where `*_scale` is the stored std with values
// below 1e-6 replaced by 1. An item feature whose stored std is below 1e-3
// (`i_constant`) was constant in training: a value further than twice that std
// from its mean whitens to 0. `temperature` is max(|T|, 0.1). With bounds
// (`*_lo` non-empty), a feature is clamped to [lo, hi] before it is whitened.
struct CellWeights {
  std::vector<float> q_mean, q_scale, i_mean, i_scale, x_mean, x_scale;
  std::vector<float> q_lo, q_hi, i_lo, i_hi, x_lo, x_hi;
  std::vector<float> i_std;
  std::vector<unsigned char> i_constant;
  BlockMatrix q_w0, q_w2, q_w4;
  std::vector<float> q_b0, q_b2, q_b4;
  BlockMatrix i_w0, i_w2;
  std::vector<float> i_b0, i_b2;
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

// How an MLREC_v3 item feature is derived from an attribute value x. The
// values are stored in model files.
enum class AttributeTransform : std::uint8_t {
  Identity = 0,  // x
  Log2p1   = 1,  // log2(1 + max(x, 0))
  Nonzero  = 2,  // x != 0
  OneHot   = 3,  // x == values[i], one feature per value
};

// One entry of an MLREC_v3 item-attribute spec. `read` indexes
// Model::attribute_reads.
struct AttributeFeature {
  std::string name;
  std::int64_t default_value   = 0;
  AttributeTransform transform = AttributeTransform::Identity;
  std::vector<std::int64_t> values;
  std::size_t read = 0;
};

// Feasibility rules beyond the LDS fit, which always applies. `leading_free`
// is a validity rule on Config::grvw_a/grvw_b; MLREC_v3 models do not read
// those fields, so they leave it to the integration's own kernel predicates.
// The others are soft. `non_temporal_family` refines `non_temporal` in the
// region in_family_region describes: a kernel family it rejects whole keeps
// one member (Feasibility::family_keeps).
struct SoftRules {
  bool leading_free           = true;
  bool small_batched_one_tile = true;
  bool dot2_small_m           = true;
  bool non_temporal           = true;
  bool non_temporal_family    = false;
};

}  // namespace detail

class Model {
 public:
  std::string path;
  std::string arch;
  std::string feature_hash;
  const detail::FeatureCatalog* catalog = nullptr;
  int format_version                    = 2;
  WeightType weight_type                = WeightType::Fp32;
  double parallel_mi_cu                 = 1.0;
  double bw_coef[3]                     = {0.0, 0.0, 0.0};
  double mi_default_cycles              = 32.0;
  std::vector<detail::MiEntry> mi_table;

  // Feature dims; item_dim includes the attribute features.
  std::size_t query_dim = 0, item_dim = 0, inter_dim = 0;
  std::uint64_t feature_spec_hash = 0;
  // Config attributes the catalog reads, and the value of one a config lacks.
  std::vector<std::string> attribute_reads;
  std::vector<std::int64_t> attribute_defaults;
  std::vector<detail::AttributeFeature> item_attributes;
  std::size_t view_key_size = 0;

  // Serving options stored in the header: the soft feasibility rules that
  // apply, whether cell whitelists form the first tier, how many view keys
  // of a tier with no whitelist get the full score after all are ranked by
  // the embedding term (0: all), and whether an MLREC_v3 pool is grouped by
  // launch key as well.
  detail::SoftRules soft_rules;
  bool whitelist_tier      = true;
  std::size_t cascade_keys = 0;
  bool launch_keys         = false;
  bool clip                = false;
  // MLREC_v3: whether the model was trained on the integration's bank as well
  // as its table and with the integration's kernel guards, neither of which
  // ranking reads, and whether it has the trained-key gate (trained_keys).
  bool bank             = false;
  bool trained_key_gate = false;
  bool kernel_guards    = false;
  // MLREC_v3: the alignment veto the header declares (Feasibility applies it).
  AlignmentVeto alignment_veto;

  // Contexts an MLREC_v3 model supports: usable CUs as a fraction of the
  // device, a bit per Schedule value, and the fewest usable CUs on any
  // device (0: no floor).
  double min_cu_fraction        = 1.0;
  double max_cu_fraction        = 1.0;
  std::uint8_t schedules        = 1;
  std::size_t min_effective_cus = 0;
  // MLREC_v3 drift fallback: the share of a cell's whitelisted view keys the
  // model lists as trained, in millionths, that a pool must hold for the cell
  // to be served from its whitelist alone (rank.cpp, drift_degraded); 0 for a
  // model without it.
  std::uint32_t drift_keep_ppm = 0;
  Provenance provenance;

  std::vector<detail::Cell> cells;
  std::vector<detail::Split> splits;
  std::vector<detail::Node> nodes;
  std::array<int, detail::kBaseCells> base_nodes{};
  // MLREC_v3: per base label, the other base labels in the tree, nearest
  // first; a problem whose own label resolves to no trained cell is scored
  // through the first of them that does.
  std::array<std::vector<std::uint8_t>, detail::kBaseCells> base_fallbacks;

  // The payload bytes of the model file (split and cell records).
  std::vector<unsigned char> payload;
  // With the trained-key gate, the view_key_hash of every view key with
  // training data, ascending: a pool's other keys are scored only for a
  // problem none of these suits.
  std::vector<std::uint64_t> trained_keys;

  Model();
  ~Model();
  Model(const Model&)            = delete;
  Model& operator=(const Model&) = delete;

  // fp32 tensors of cell `cell`, dequantized on first use and kept while the
  // cache limit allows; the pointer keeps them alive. Thread-safe. Throws
  // std::bad_alloc when memory runs out.
  std::shared_ptr<const detail::CellWeights> weights(std::size_t cell) const;

  void init_lazy_state();

 private:
  struct LazyCell;
  std::unique_ptr<LazyCell[]> lazy_;
};

namespace detail {

// Parses an MLREC_v2 or MLREC_v3 image. Returns nullptr and sets *error on any
// defect.
// May throw std::bad_alloc.
std::unique_ptr<Model> parse_model(const unsigned char* data, std::size_t size, std::string* error);

// Bytes of `w`, and of the CellWeights that Model::weights builds for cell
// `cell` (equal once built).
std::size_t weight_bytes(const CellWeights& w) noexcept;
std::size_t cell_weight_bytes(const Model& model, std::size_t cell) noexcept;

// CRC-32/ISO-HDLC.
std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept;

// Environment knobs, read once.
struct EnvKnobs {
  bool diag            = false;
  bool pick_log        = false;
  long long force_cell = -1;
  long long cache_mib  = -1;  // TILEWRIGHT_CACHE_MB; -1 when unset or invalid
};
const EnvKnobs& env_knobs() noexcept;

bool valid_hardware(const Hardware& hardware) noexcept;

// Cell scoring `problem` (honoring TILEWRIGHT_FORCE_CELL), or -1.
int route_cell(const Model& model, const Problem& problem) noexcept;

// Bounds of the problems resolve_cell sends to one cell: the least M, N and K,
// the greatest M, N and K, and whether problems of batch 1 and of a larger
// batch are among them. Exact or wider; `reached` is false for a cell no
// problem reaches.
struct CellBounds {
  std::uint64_t m_lo = ~std::uint64_t{0}, n_lo = ~std::uint64_t{0}, k_lo = ~std::uint64_t{0};
  std::uint64_t m_hi = 0, n_hi = 0, k_hi = 0;
  bool batch_one  = false;
  bool batch_many = false;
  bool reached    = false;
};

// Per cell of `model`, without TILEWRIGHT_FORCE_CELL.
std::vector<CellBounds> cell_bounds(const Model& model);

// Sets the epoch of the calling thread's ranking scratch, so that tests can
// reach its wrap-around.
void set_rank_epoch(std::uint32_t epoch) noexcept;

// What allocate_huge_pages returns where no huge pages can be mapped, and on
// systems other than Linux: zeroed heap memory for `bytes`, 64-byte aligned
// and never at a 2 MiB boundary, which tells free_huge_pages that it is not a
// mapping. Declared here so that tests can reach that path.
void* heap_pages(std::size_t bytes);

// While `refuse` is true, allocate_huge_pages takes every mapping it asks the
// system for as refused, so that tests reach what follows a refusal.
void refuse_huge_page_mappings(bool refuse) noexcept;

// While `refuse` is true, allocate_huge_pages throws std::bad_alloc as if no
// memory could be had at all, so that tests reach what follows: a thread
// without its scratch block, whose arrays all come from the heap, and a
// candidate set without its copy.
void refuse_huge_pages(bool refuse) noexcept;

// The selection-time options (README.md, Building and testing): TW_LEVER_PREFETCH
// and TW_LEVER_MEMO are on unless defined to 0, as the CMake options define them
// when off, and the memo's plan check TW_LEVER_MEMO_CHECK is on wherever
// assertions are unless defined to 0. The defaults are set here, in the header
// the engine and its white-box tests both include, so that a test expects what
// the engine compiled.
#ifndef TW_LEVER_PREFETCH
#define TW_LEVER_PREFETCH 1
#endif
#ifndef TW_LEVER_MEMO
#define TW_LEVER_MEMO 1
#endif

#if TW_LEVER_MEMO
#if !defined(TW_LEVER_MEMO_CHECK) && !defined(NDEBUG)
#define TW_LEVER_MEMO_CHECK 1
#endif

// The calling thread's re-rank memo (TW_LEVER_MEMO): the rankings it answered,
// and those it left to a fresh ranking because the launch model planned a
// remembered key differently (TW_LEVER_MEMO_CHECK).
struct MemoCounts {
  std::uint64_t replays      = 0;
  std::uint64_t plan_changes = 0;
};
MemoCounts memo_counts() noexcept;
#endif

// Throws std::invalid_argument when the model's provenance names a launch
// model and `launch` is null or has another id.
void check_launch(const Model& model, const LaunchModel* launch);

// What the family-wise non-temporal rule reads of a kernel's family, the
// pool's kernels with equal mt and mi: whether a member has no cache hint,
// cache_hints_a == 4 or cache_hints_b == 4, and whether the kernel is the
// member the family keeps (its hint-free member if any, else its lowest
// (cache_hints_a, cache_hints_b)).
constexpr std::uint8_t kFamilyPlain    = 1;
constexpr std::uint8_t kFamilyNtA      = 2;
constexpr std::uint8_t kFamilyNtB      = 4;
constexpr std::uint8_t kFamilyFallback = 8;

// The Config fields the feasibility rules read, and the kernel's family bits
// within its pool (pool_keys sets them for models with the family-wise rule).
struct KernelShape {
  std::uint64_t mt_m = 0, mt_n = 0, mt_k = 0;
  std::uint64_t mk = 0, nk = 0;  // mt.mk(), mt.nk()
  std::uint64_t grvw_a = 0, grvw_b = 0;
  int cache_hints_a = 0, cache_hints_b = 0;
  bool dot2           = false;  // MI 1x1x64
  std::uint8_t family = 0;
};

KernelShape kernel_shape(const Config& c);

// The family bits of every config of `configs` (KernelShape::family).
std::vector<std::uint8_t> family_bits(const std::vector<Config>& configs);

// The values of Model::attribute_reads in `c`, in that order.
void read_attributes(const Model& model, const Config& c, std::int64_t* out);

// A pool's configs grouped by view key (FeatureCatalog::view_key). Keys are
// numbered in order of first appearance; the configs of key k are
// members[member_begin[k] .. member_begin[k + 1]), in input order.
struct PoolKeys {
  std::vector<std::size_t> key_of;  // per config
  std::vector<std::size_t> first;   // per key: its first config
  std::vector<std::size_t> member_begin;
  std::vector<std::size_t> members;
  std::vector<KernelShape> shape;  // per key: kernel_shape of its first config
  // per key: read_attributes of its first config (attribute_count values)
  std::vector<std::int64_t> attribute_values;
  std::size_t attribute_count = 0;
  bool nt_a                   = false;  // some config has cache_hints_a == 4
  bool nt_b                   = false;  // some config has cache_hints_b == 4
  // Keys split by launch key: per key, its group of keys with equal view key
  // (a view), numbered in order of first appearance; the keys of view v are
  // view_keys[view_begin[v] .. view_begin[v + 1]), ascending. All empty when
  // every key is its own view.
  std::vector<std::size_t> view_of;
  std::vector<std::size_t> view_begin;
  std::vector<std::size_t> view_keys;
  // For a model with the trained-key gate: per key, whether the model lists
  // its view key (Model::trained_keys), and the number of keys it does not.
  // Empty for any other model.
  std::vector<unsigned char> trained;
  std::size_t untrained = 0;

  const std::int64_t* attributes(std::size_t key) const {
    return attribute_values.data() + key * attribute_count;
  }
  std::size_t views() const { return view_of.empty() ? first.size() : view_begin.size() - 1; }
  std::size_t view(std::size_t key) const { return view_of.empty() ? key : view_of[key]; }
  std::size_t first_of_view(std::size_t view) const {
    return view_of.empty() ? view : view_keys[view_begin[view]];
  }
};

// Groups configs by view key and, for a catalog that reads launch plans, by
// `launch`'s launch key.
PoolKeys pool_keys(const Model& model,
                   const std::vector<Config>& configs,
                   const LaunchModel* launch = nullptr);

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
                              const std::vector<double>* prior,
                              const LaunchModel* launch = nullptr);

}  // namespace detail
}  // namespace tilewright
