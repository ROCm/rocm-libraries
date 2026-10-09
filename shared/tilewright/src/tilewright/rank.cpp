// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "cache.hpp"
#include "features.hpp"
#include "kernels.hpp"
#include "model_impl.hpp"
#include "stage_timers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tilewright {
namespace detail {

bool valid_hardware(const Hardware& h) noexcept {
  return h.N_CU != 0 && h.lds_capacity != 0 && h.L2_capacity != 0;
}

namespace {

std::size_t tier_mn(std::size_t v) { return v <= 32 ? 0 : v <= 128 ? 1 : v <= 512 ? 2 : 3; }
std::size_t tier_k(std::size_t v) { return v <= 32 ? 0 : v <= 512 ? 1 : 2; }

std::size_t base_index(const Problem& p) {
  return ((tier_mn(p.size.m) * 4 + tier_mn(p.size.n)) * 3 + tier_k(p.size.k)) * 2 +
         (p.batch == 1 ? 0 : 1);
}

std::uint64_t axis_value(char axis, const Problem& p) {
  switch (axis) {
    case 'M': return p.size.m;
    case 'N': return p.size.n;
    case 'K': return p.size.k;
    default: return p.batch;
  }
}

int resolve_from(const Model& model, int node, const Problem& p) noexcept {
  if (node < 0) return -1;
  for (std::size_t step = 0; step <= model.splits.size(); ++step) {
    const Node& nd = model.nodes[static_cast<std::size_t>(node)];
    if (nd.split < 0) return nd.cell;
    const Split& s = model.splits[static_cast<std::size_t>(nd.split)];
    const bool lo =
        s.threshold >= 0 && axis_value(s.axis, p) <= static_cast<std::uint64_t>(s.threshold);
    node = lo ? s.lo : s.hi;
  }
  return -1;
}

int resolve_cell(const Model& model, const Problem& p) noexcept {
  const std::size_t base = base_index(p);
  const int cell         = resolve_from(model, model.base_nodes[base], p);
  if (cell >= 0 || model.format_version < 3) return cell;
  for (std::uint8_t other : model.base_fallbacks[base]) {
    const int c = resolve_from(model, model.base_nodes[other], p);
    if (c >= 0) return c;
  }
  return -1;
}

// Closed ranges of M, N, K and batch (axis order of axis_index); batch 1 is
// excluded when `no_batch_one`.
struct Box {
  std::uint64_t lo[4], hi[4];
  bool no_batch_one;
};

std::size_t axis_index(char axis) {
  return axis == 'M' ? 0 : axis == 'N' ? 1 : axis == 'K' ? 2 : 3;
}

Box base_box(std::size_t base) {
  constexpr std::uint64_t kMax     = ~std::uint64_t{0};
  constexpr std::uint64_t mn_lo[4] = {0, 33, 129, 513}, mn_hi[4] = {32, 128, 512, kMax};
  constexpr std::uint64_t k_lo[3] = {0, 33, 513}, k_hi[3] = {32, 512, kMax};
  const std::size_t batch_tier = base % 2, k = base / 2 % 3, n = base / 6 % 4, m = base / 24;
  Box b;
  b.lo[0]        = mn_lo[m];
  b.hi[0]        = mn_hi[m];
  b.lo[1]        = mn_lo[n];
  b.hi[1]        = mn_hi[n];
  b.lo[2]        = k_lo[k];
  b.hi[2]        = k_hi[k];
  b.lo[3]        = batch_tier == 0 ? 1 : 0;
  b.hi[3]        = batch_tier == 0 ? 1 : kMax;
  b.no_batch_one = batch_tier == 1;
  return b;
}

// Adds to `bounds` the boxes the problems of `box` reach below `node`, as
// resolve_from routes them; returns whether every one of them reaches a cell.
bool walk_bounds(const Model& model,
                 int node,
                 Box box,
                 std::size_t steps,
                 std::vector<CellBounds>& bounds) {
  if (node < 0 || steps > model.splits.size()) return false;
  const Node& nd = model.nodes[static_cast<std::size_t>(node)];
  if (nd.split < 0) {
    if (nd.cell < 0) return false;
    CellBounds& b = bounds[static_cast<std::size_t>(nd.cell)];
    b.reached     = true;
    b.m_lo        = std::min(b.m_lo, box.lo[0]);
    b.n_lo        = std::min(b.n_lo, box.lo[1]);
    b.k_lo        = std::min(b.k_lo, box.lo[2]);
    b.m_hi        = std::max(b.m_hi, box.hi[0]);
    b.n_hi        = std::max(b.n_hi, box.hi[1]);
    b.k_hi        = std::max(b.k_hi, box.hi[2]);
    b.batch_one   = b.batch_one || (!box.no_batch_one && box.lo[3] <= 1 && 1 <= box.hi[3]);
    b.batch_many  = b.batch_many || box.hi[3] > 1;
    return true;
  }
  const Split& s       = model.splits[static_cast<std::size_t>(nd.split)];
  const std::size_t ax = axis_index(s.axis);
  if (s.threshold < 0) return walk_bounds(model, s.hi, box, steps + 1, bounds);
  const std::uint64_t t = static_cast<std::uint64_t>(s.threshold);
  bool all              = true;
  if (box.lo[ax] <= t) {
    Box lo    = box;
    lo.hi[ax] = std::min(lo.hi[ax], t);
    all       = walk_bounds(model, s.lo, lo, steps + 1, bounds) && all;
  }
  if (box.hi[ax] > t) {
    Box hi    = box;
    hi.lo[ax] = std::max(hi.lo[ax], t + 1);
    all       = walk_bounds(model, s.hi, hi, steps + 1, bounds) && all;
  }
  return all;
}

// A key of embedding term `v` (finite or -inf) of candidate `c` that is larger
// exactly when the candidate is better: a larger term, or an equal one and a
// smaller index. -0 and +0 are equal terms.
inline std::uint64_t order_key(float v, std::size_t c) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  const std::uint32_t ordered =
      (bits ^ (static_cast<std::uint32_t>(static_cast<std::int32_t>(bits) >> 31) | 0x80000000u)) +
      (bits == 0x80000000u ? 1u : 0u);
  return (std::uint64_t{ordered} << 32) | (0xFFFFFFFFull - static_cast<std::uint32_t>(c));
}

inline std::size_t order_key_index(std::uint64_t key) {
  return static_cast<std::size_t>(0xFFFFFFFFull - (key & 0xFFFFFFFFull));
}

// Moves the `k` largest of the `n` distinct values at `v` to its first `k`
// (in no order), 0 < k <= n; `spare` holds n values. The partitions do not
// branch on the values, whose order is new to every ranking.
void select_largest(std::uint64_t* v, std::size_t n, std::size_t k, std::uint64_t* spare) {
  std::size_t lo = 0, hi = n;
  while (hi - lo > 32) {
    const std::uint64_t a = v[lo], b = v[lo + (hi - lo) / 2], c = v[hi - 1];
    const std::uint64_t pivot = std::max(std::min(a, b), std::min(std::max(a, b), c));
    std::size_t front = 0, back = hi - lo;
    for (std::size_t i = lo; i < hi; ++i) {
      const std::uint64_t x = v[i];
      const bool above      = x > pivot;
      spare[front]          = x;
      spare[back - 1]       = x;
      front += above ? 1 : 0;
      back -= above ? 0 : 1;
    }
    std::copy(spare, spare + (hi - lo), v + lo);
    const std::size_t mid = lo + front;
    if (mid == k) return;
    if (mid > k)
      hi = mid;
    else
      lo = mid;
  }
  std::nth_element(v + lo, v + (k - 1), v + hi, std::greater<std::uint64_t>());
}

inline void prefetch(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p);
#else
  (void)p;
#endif
}

bool finite(float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  return (bits & 0x7F800000u) != 0x7F800000u;
}

bool finite(double v) {
  std::uint64_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  return (bits & 0x7FF0000000000000ull) != 0x7FF0000000000000ull;
}

// NaN stays NaN.
inline float clamp_to(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

void whiten(const float* f,
            const std::vector<float>& mean,
            const std::vector<float>& scale,
            const std::vector<float>& lo,
            const std::vector<float>& hi,
            std::size_t n,
            float* out) {
  if (lo.empty()) {
    for (std::size_t j = 0; j < n; ++j) out[j] = (f[j] - mean[j]) / scale[j];
    return;
  }
  for (std::size_t j = 0; j < n; ++j) out[j] = (clamp_to(f[j], lo[j], hi[j]) - mean[j]) / scale[j];
}

void whiten_item(const float* f, const CellWeights& w, std::size_t n, float* out) {
  const bool clip = !w.i_lo.empty();
  for (std::size_t j = 0; j < n; ++j) {
    const float d = (clip ? clamp_to(f[j], w.i_lo[j], w.i_hi[j]) : f[j]) - w.i_mean[j];
    out[j] = w.i_constant[j] != 0 && std::fabs(d) > 2.0f * w.i_std[j] ? 0.0f : d / w.i_scale[j];
  }
}

void query_embedding(const Kernels& k,
                     const Cell& cell,
                     const CellWeights& w,
                     const float* q_feat,
                     std::size_t q_dim,
                     float* norm,
                     float* h0,
                     float* h2,
                     float* q_emb) {
  whiten(q_feat, w.q_mean, w.q_scale, w.q_lo, w.q_hi, q_dim, norm);
  const std::size_t h = cell.hidden_dim;
  k.linear_blocks_relu(w.q_w0.view(), w.q_b0.data(), norm, h, q_dim, h0);
  k.linear_blocks_relu(w.q_w2.view(), w.q_b2.data(), h0, h, h, h2);
  k.linear_blocks(w.q_w4.view(), w.q_b4.data(), h2, cell.embed_dim, h, q_emb);
}

void item_embedding(const Kernels& k,
                    const Cell& cell,
                    const CellWeights& w,
                    const float* i_feat,
                    std::size_t i_dim,
                    float* norm,
                    float* hidden,
                    float* i_emb) {
  whiten_item(i_feat, w, i_dim, norm);
  k.linear_blocks_relu(w.i_w0.view(), w.i_b0.data(), norm, cell.hidden_dim, i_dim, hidden);
  k.linear_blocks(w.i_w2.view(), w.i_b2.data(), hidden, cell.embed_dim, cell.hidden_dim, i_emb);
}

bool whitelisted(const Model& model, const Cell& cell, const Config& c, const std::int64_t* attrs) {
  const Signature s = signature_of(c);
  if (std::find(cell.signatures.begin(), cell.signatures.end(), s) != cell.signatures.end())
    return true;
  if (cell.view_key_hashes.empty()) return false;
  std::uint64_t key[kMaxViewKeySize];
  model.catalog->view_key(model, c, attrs, key);
  return std::binary_search(cell.view_key_hashes.begin(),
                            cell.view_key_hashes.end(),
                            view_key_hash(key, model.view_key_size));
}

bool has_whitelist(const Model& model, const Cell& cell) {
  return model.whitelist_tier && (!cell.signatures.empty() || !cell.view_key_hashes.empty());
}

}  // namespace

// What a ranking writes to `results`: its first `limit` scored configs and,
// with `unscored`, then every other config in input order.
struct Output {
  std::vector<Result>& results;
  std::size_t limit;
  bool unscored;
};

namespace {

void write_unscored(std::size_t n, const Output& out) {
  out.results.clear();
  if (!out.unscored) return;
  out.results.reserve(n);
  for (std::size_t j = 0; j < n; ++j) out.results.push_back(Result{j, 0.0, false});
}

void log_pick(const Problem& p,
              const Cell& cell,
              const Config& top,
              float score,
              std::size_t top_index,
              std::size_t n) {
  std::fprintf(stderr,
               "[TILEWRIGHT_PICK] m=%zu n=%zu k=%zu b=%zu tA=%c tB=%c leaf=%s "
               "top1_sig=(mt_m=%zu,mt_n=%zu,mt_k=%zu,mi_m=%zu,mi_n=%zu,"
               "mi_k=%zu,cha=%d,chb=%d) top1_score=%f top1_index=%zu n_configs=%zu\n",
               p.size.m,
               p.size.n,
               p.size.k,
               p.batch,
               (p.a_transpose == Transpose::T ? 'T' : 'N'),
               (p.b_transpose == Transpose::T ? 'T' : 'N'),
               cell.label.c_str(),
               top.mt.m,
               top.mt.n,
               top.mt.k,
               top.mi.m,
               top.mi.n,
               top.mi.k,
               top.cache_hints_a,
               top.cache_hints_b,
               static_cast<double>(score),
               top_index,
               n);
  std::fflush(stderr);
}

// Unrelated odd constants: multipliers in arithmetic progression would make keys
// that differ by small integers such as (1, -2, 1) collide.
constexpr std::uint64_t kKeyMultipliers[16] = {0x9E3779B97F4A7C15ull,
                                               0xC2B2AE3D27D4EB4Full,
                                               0x165667B19E3779F9ull,
                                               0xD6E8FEB86659FD93ull,
                                               0xFF51AFD7ED558CCDull,
                                               0xC4CEB9FE1A85EC53ull,
                                               0x94D049BB133111EBull,
                                               0xBF58476D1CE4E5B9ull,
                                               0x2545F4914F6CDD1Dull,
                                               0x9FB21C651E98DF25ull,
                                               0xA0761D6478BD642Full,
                                               0xE7037ED1A0B428DBull,
                                               0x8EBC6AF09C88C6E3ull,
                                               0x589965CC75374CC3ull,
                                               0x1D8E4E27C47D124Full,
                                               0xD1B54A32D192ED03ull};

std::uint64_t hash_key(const std::uint64_t* key, std::size_t n) {
  std::uint64_t h = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (i > 0 && i % 16 == 0) h = (h ^ (h >> 31)) * 0xBF58476D1CE4E5B9ull;
    h += key[i] * kKeyMultipliers[i % 16];
  }
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ull;
  return h ^ (h >> 33);
}

bool same_names(const std::vector<Attribute>& a, const std::vector<Attribute>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].name != b[i].name) return false;
  return true;
}

void check_attributes(const std::vector<Config>& configs) {
  std::vector<const std::string*> names;
  const std::vector<Attribute>* valid = nullptr;
  for (std::size_t ci = 0; ci < configs.size(); ++ci) {
    const std::vector<Attribute>& attributes = configs[ci].attributes;
    if (attributes.empty() || (valid != nullptr && same_names(*valid, attributes))) continue;
    names.clear();
    for (const Attribute& a : attributes) {
      if (a.name.empty())
        throw std::invalid_argument("tilewright: config " + std::to_string(ci) +
                                    " has an attribute with an empty name");
      names.push_back(&a.name);
    }
    std::sort(names.begin(), names.end(), [](const std::string* a, const std::string* b) {
      return *a < *b;
    });
    const auto dup = std::adjacent_find(
        names.begin(), names.end(), [](const std::string* a, const std::string* b) {
          return *a == *b;
        });
    if (dup != names.end())
      throw std::invalid_argument("tilewright: config " + std::to_string(ci) +
                                  " has more than one attribute named '" + **dup + "'");
    valid = &attributes;
  }
}

// Ascending, with every non-finite value after every finite one.
bool prior_before(double a, double b) {
  const bool fa = finite(a), fb = finite(b);
  if (fa != fb) return fa;
  return fa && a < b;
}

}  // namespace

constexpr std::uint32_t kNoSlot = ~std::uint32_t{0};

// Pool data a CandidateSet keeps for one cell, per view key: the cell's
// whitelisted keys with their item embeddings and, with `all_keys`, the item
// embeddings of the other keys. With a trained-key gate, `whitelist_keys`,
// `possible` and `plain` hold trained keys only.
struct CellCache {
  std::vector<std::size_t> whitelist_keys;  // ascending
  std::vector<std::uint32_t> slot;          // per key: its row of whitelist_emb, or kNoSlot
  std::vector<float> whitelist_emb;         // whitelist_keys x embed_dim
  // PoolKeys::views() x embed_dim in the BlockWeights layout (fp32); rows of
  // the views outside the whitelist, which admits or rejects a view as a
  // whole.
  std::vector<float> other_emb;
  bool all_keys = false;
  // With `pruned`, the keys never_feasible keeps for the cell, ascending:
  // all of them, and those without a cache hint; every other key fails the
  // rules for every problem the cell scores.
  std::vector<std::uint32_t> possible, plain;
  // Per key of `possible`: what the non-temporal rule reads.
  struct Possible {
    std::uint64_t mt_m, mt_n;
    std::uint16_t mt_k_low;  // mt_k % 1024
    std::uint8_t hints;      // hint_class
    std::uint8_t family;     // KernelShape::family
  };
  std::vector<Possible> possible_shape;
  PlainBounds plain_bounds, hinted_bounds;  // of the keys without, with a cache hint
  std::size_t plain_configs = 0;            // configs of the keys of `plain`
  std::size_t hinted_keys   = 0;
  bool pruned               = false;

  // Read only where the family-wise non-temporal rule applies, so they follow
  // the fields every ranking reads. With `pruned`, the kept members of
  // families without a hint-free member, ascending, the union of them and
  // `plain`, and their configs.
  std::vector<std::uint32_t> kept_hinted, plain_and_kept;
  std::size_t kept_configs = 0;
  // With `all_keys`, the rows of other_emb of the views of `kept_hinted` again,
  // contiguous in ascending view order (kept_emb, BlockWeights layout), and
  // per view its row there or kNoSlot; empty without such views. They are
  // scattered over other_emb, so a cascade reads them here.
  std::vector<std::uint32_t> kept_row;
  std::vector<float> kept_emb;
  std::size_t kept_views = 0;
};

inline std::size_t cache_bytes(const CellCache& c) noexcept {
  return sizeof(CellCache) + c.whitelist_keys.size() * sizeof(std::size_t) +
         (c.slot.size() + c.possible.size() + c.plain.size() + c.kept_hinted.size() +
          c.plain_and_kept.size() + c.kept_row.size()) *
             sizeof(std::uint32_t) +
         c.possible_shape.size() * sizeof(CellCache::Possible) +
         (c.whitelist_emb.size() + c.other_emb.size() + c.kept_emb.size()) * sizeof(float);
}

// The item embedding of `key`, or of its view in `row` when outside the
// whitelist.
inline const float* cached_embedding(const CellCache& c,
                                     const PoolKeys& keys,
                                     std::size_t key,
                                     std::size_t embed_dim,
                                     float* row) {
  const std::uint32_t s = c.slot[key];
  if (s != kNoSlot) return &c.whitelist_emb[s * embed_dim];
  const std::size_t v = keys.view(key);
  const float* at     = &c.other_emb[(v / 16 * embed_dim) * 16 + v % 16];
  for (std::size_t j = 0; j < embed_dim; ++j) row[j] = at[j * 16];
  return row;
}

class CellCacheSource {
 public:
  virtual ~CellCacheSource() = default;
  // The cache of `cell`, built with its weights `w` when absent; with
  // `all_keys`, one that holds every key's embedding.
  virtual std::shared_ptr<const CellCache> cache(std::size_t cell,
                                                 const CellWeights& w,
                                                 bool all_keys) const = 0;
  // FeatureCatalog::inter_kernel of every key (keys x inter_kernel_dim) for
  // the data types of `p` and the device `hw`, or nullptr when they are not
  // the ones the pool's values were computed for.
  virtual const float* inter_kernel(const Problem& p, const HwView& hw) const = 0;
};

namespace {

// The batched kernels load whole rows of kBatch floats; a row that straddles
// cache lines costs a split load per weight.
template <class T>
struct CacheLineAllocator {
  using value_type = T;
  static constexpr std::align_val_t kLine{64};
  CacheLineAllocator() = default;
  template <class U>
  CacheLineAllocator(const CacheLineAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), kLine)); }
  void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, kLine); }
  template <class U>
  bool operator==(const CacheLineAllocator<U>&) const noexcept {
    return true;
  }
  template <class U>
  bool operator!=(const CacheLineAllocator<U>&) const noexcept {
    return false;
  }
};
using BatchBuffer = std::vector<float, CacheLineAllocator<float>>;

// Working storage of rank_in_cell, kept per thread so that a ranking call
// does not allocate once the buffers have grown to the pool's size. The
// per-key marks hold values of the call's epoch, so no call clears them:
// feasible[key] is 2 * epoch (infeasible) or 2 * epoch + 1 (feasible) once
// evaluated, in_tier1[key], scored[key] and listed[view] equal epoch when set
// (listed is sized only once a pool splits by launch key).
struct Scratch {
  static constexpr std::uint32_t kMaxEpoch = 0x7FFFFFFFu;
  std::uint32_t epoch                      = 0;
  std::vector<std::uint32_t> feasible, in_tier1, scored, listed;
  std::vector<std::size_t> tier1, tier2, order, ranked, run, views;
  std::vector<std::uint32_t> passing;
  std::vector<float> score, emb, q_feat, i_feat, x_feat, x_problem, norm, q_emb, i_emb, h0, h2;
  std::vector<LaunchPlan> plans;
  std::vector<std::uint64_t> ordered, spare;
  BatchBuffer x_batch, h_batch;

  // The epoch of a new call over `nk` keys.
  std::uint32_t next_epoch(std::size_t nk) {
    if (feasible.size() < nk) {
      feasible.resize(nk, 0);
      in_tier1.resize(nk, 0);
      scored.resize(nk, 0);
    }
    if (epoch >= kMaxEpoch) {
      for (std::vector<std::uint32_t>* v : {&feasible, &in_tier1, &scored, &listed})
        std::fill(v->begin(), v->end(), 0u);
      epoch = 0;
    }
    return ++epoch;
  }
};

Scratch& scratch() {
  thread_local Scratch s;
  return s;
}

// `prior` orders equal scores when non-null.
void rank_in_cell(const Model& model,
                  std::size_t cell_index,
                  const Problem& problem,
                  const Hardware& hardware,
                  const ExecutionContext& context,
                  const std::vector<Config>& configs,
                  const PoolKeys& keys,
                  std::size_t min_scored,
                  const std::vector<double>* prior,
                  const CellCacheSource* source,
                  const LaunchModel* launch,
                  const Output& out) {
  const std::size_t n                              = configs.size();
  const std::size_t nk                             = keys.first.size();
  const FeatureCatalog& cat                        = *model.catalog;
  const std::size_t i_dim                          = model.item_dim;
  const Cell& cell                                 = model.cells[cell_index];
  const std::shared_ptr<const CellWeights> weights = model.weights(cell_index);
  const CellWeights& w                             = *weights;
  std::shared_ptr<const CellCache> held =
      source != nullptr ? source->cache(cell_index, w, false) : nullptr;
  const CellCache* cache = held.get();
  const Kernels& k       = kernels();
  Scratch& s             = scratch();
  const auto size_of     = [&](std::size_t key) {
    return keys.member_begin[key + 1] - keys.member_begin[key];
  };

  const std::uint32_t epoch = s.next_epoch(nk);
  const Feasibility rules(problem, hardware, keys.nt_a, keys.nt_b, model.soft_rules);
  const auto feasible = [&](std::size_t key) {
    std::uint32_t& f = s.feasible[key];
    if (f < 2 * epoch) f = 2 * epoch + (rules(keys.shape[key]) ? 1 : 0);
    return f == 2 * epoch + 1;
  };
  TILEWRIGHT_STAGE_LAP(Setup);

  const bool have_whitelist       = has_whitelist(model, cell);
  std::vector<std::size_t>& tier1 = s.tier1;
  tier1.clear();
  std::size_t tier1_configs = 0;
  const auto add_tier1      = [&](std::size_t key) {
    tier1.push_back(key);
    s.in_tier1[key] = epoch;
    tier1_configs += size_of(key);
  };
  // With a trained-key gate the untrained keys are candidates only when no
  // trained key is feasible (`all_keys`); the cell cache then holds trained
  // keys only.
  bool all_keys        = false;
  const auto candidate = [&](std::size_t key) {
    return all_keys || keys.trained.empty() || keys.trained[key] != 0;
  };
  if (have_whitelist) {
    if (cache != nullptr) {
      for (std::size_t key : cache->whitelist_keys)
        if (feasible(key)) add_tier1(key);
    } else {
      for (std::size_t key = 0; key < nk; ++key)
        if (candidate(key) && feasible(key) &&
            whitelisted(model, cell, configs[keys.first[key]], keys.attributes(key)))
          add_tier1(key);
    }
  }
  const bool pruned     = cache != nullptr && cache->pruned;
  const bool plain_pass = pruned && rules.all_pass(cache->plain_bounds);
  const bool hinted     = pruned && cache->hinted_keys > 0 && rules.hinted_may_pass();
  // Where the rules other than the non-temporal one pass every key of a
  // kind, the non-temporal rule alone decides its keys.
  const bool plain_tile  = pruned && rules.all_but_non_temporal_pass(cache->plain_bounds);
  const bool hinted_tile = hinted && rules.all_but_non_temporal_pass(cache->hinted_bounds);
  std::vector<std::uint32_t>& passing = s.passing;
  std::size_t n_passing               = 0;
  bool listed_passing                 = false;
  const auto plain_and_kept           = [&]() -> const std::vector<std::uint32_t>& {
    return cache->kept_hinted.empty() ? cache->plain : cache->plain_and_kept;
  };
  // Lists in `passing` the feasible keys of a pruned cell, ascending.
  const auto list_passing = [&] {
    if (plain_pass && rules.only_kept_hinted_may_pass()) {
      // The plain keys and the kept members the other rules pass, merged.
      const std::vector<std::uint32_t>& plain = cache->plain;
      if (passing.size() < plain.size() + cache->kept_hinted.size())
        passing.resize(plain.size() + cache->kept_hinted.size());
      std::size_t i = 0;
      for (const std::uint32_t key : cache->kept_hinted) {
        if (!hinted_tile && !feasible(key)) continue;
        for (; i < plain.size() && plain[i] < key; ++i) passing[n_passing++] = plain[i];
        passing[n_passing++] = key;
      }
      for (; i < plain.size(); ++i) passing[n_passing++] = plain[i];
      return;
    }
    const std::vector<std::uint32_t>& possible = cache->possible;
    if (passing.size() < possible.size()) passing.resize(possible.size());
    const std::vector<CellCache::Possible>& shape = cache->possible_shape;
    if (plain_tile && (hinted_tile || !hinted)) {
      // The rule's outcome for each value of its tests.
      unsigned char pass[64];
      for (unsigned t = 0; t < 64; ++t) {
        const bool r = rules.non_temporal_rule(t & 1, (t >> 1) & 1, (t >> 2) & 1, t >> 3);
        pass[t]      = (t >> 3 == 0 ? plain_pass || r : hinted && r) ? 1 : 0;
      }
      if (!rules.family_region()) {
        for (std::size_t i = 0; i < possible.size(); ++i) {
          const CellCache::Possible& k = shape[i];
          const unsigned t             = (rules.tile_aligned(k.mt_k_low) ? 1u : 0u) |
                             (rules.b_branch(k.mt_m) ? 2u : 0u) |
                             (rules.a_branch(k.mt_n) ? 4u : 0u) | (unsigned{k.hints} << 3);
          passing[n_passing] = possible[i];
          n_passing += pass[t];
        }
      } else {
        // Whether a family's kept member passes, for each value of the rule's
        // tests and of the family's members (kFamilyPlain, NtA, NtB).
        unsigned char keep[64];
        rules.family_keep_table(keep);
        for (std::size_t i = 0; i < possible.size(); ++i) {
          const CellCache::Possible& k = shape[i];
          const unsigned tests         = (rules.tile_aligned(k.mt_k_low) ? 1u : 0u) |
                                 (rules.b_branch(k.mt_m) ? 2u : 0u) |
                                 (rules.a_branch(k.mt_n) ? 4u : 0u);
          const unsigned kept = (k.family & kFamilyFallback) != 0 ? 1u : 0u;
          passing[n_passing]  = possible[i];
          n_passing += pass[tests | (unsigned{k.hints} << 3)] |
                       (keep[tests | (unsigned{k.family & 7u} << 3)] & kept);
        }
      }
    } else {
      for (std::size_t i = 0; i < possible.size(); ++i) {
        const std::size_t key        = possible[i];
        const CellCache::Possible& k = shape[i];
        const auto rule              = [&] {
          return rules.non_temporal_rule(rules.tile_aligned(k.mt_k_low),
                                         rules.b_branch(k.mt_m),
                                         rules.a_branch(k.mt_n),
                                         k.hints,
                                         k.family);
        };
        const bool ok      = k.hints == 0 ? plain_pass || (plain_tile ? rule() : feasible(key))
                                          : hinted && (hinted_tile ? rule() : feasible(key));
        passing[n_passing] = static_cast<std::uint32_t>(key);
        n_passing += ok ? 1 : 0;
      }
    }
  };
  // Calls `f` on every feasible candidate key, ascending.
  const auto each_feasible = [&](auto&& f) {
    if (!pruned || all_keys) {
      for (std::size_t key = 0; key < nk; ++key)
        if (candidate(key) && feasible(key)) f(key);
      return;
    }
    if (plain_pass && !hinted) {
      for (const std::uint32_t key : cache->plain) f(key);
      return;
    }
    if (!listed_passing) {
      list_passing();
      listed_passing = true;
    }
    for (std::size_t i = 0; i < n_passing; ++i) f(passing[i]);
  };
  const auto add_feasible = [&](std::size_t key) {
    tier1.push_back(key);
    tier1_configs += size_of(key);
  };
  bool tier1_is_whitelist = have_whitelist;
  if (tier1.empty()) {
    tier1_is_whitelist = false;
    if (plain_pass && !hinted) {
      tier1.assign(cache->plain.begin(), cache->plain.end());
      tier1_configs = cache->plain_configs;
    } else if (plain_pass && hinted_tile && rules.only_kept_hinted_may_pass()) {
      tier1.assign(plain_and_kept().begin(), plain_and_kept().end());
      tier1_configs = cache->plain_configs + cache->kept_configs;
    } else {
      each_feasible(add_feasible);
    }
  }
  if (tier1.empty() && !keys.trained.empty()) {
    all_keys = true;
    each_feasible(add_feasible);
  }
  if (tier1.empty()) return write_unscored(n, out);

  std::vector<std::size_t>& tier2 = s.tier2;
  tier2.clear();
  if (tier1_is_whitelist && min_scored > tier1_configs)
    each_feasible([&](std::size_t key) {
      if (s.in_tier1[key] != epoch) tier2.push_back(key);
    });
  TILEWRIGHT_STAGE_LAP(Feasibility);
  if (cache != nullptr && (!tier1_is_whitelist || !tier2.empty())) {
    held  = source->cache(cell_index, w, true);
    cache = held.get();
  }

  const HwView hw               = hw_view(model, hardware, context);
  const float* x_kernel         = source != nullptr ? source->inter_kernel(problem, hw) : nullptr;
  const std::size_t e           = cell.embed_dim;
  std::vector<float>& q_feat    = s.q_feat;
  std::vector<float>& i_feat    = s.i_feat;
  std::vector<float>& x_feat    = s.x_feat;
  std::vector<float>& norm      = s.norm;
  std::vector<float>& q_emb     = s.q_emb;
  std::vector<float>& i_emb     = s.i_emb;
  std::vector<float>& h0        = s.h0;
  std::vector<float>& h2        = s.h2;
  std::vector<float>& x_problem = s.x_problem;
  q_feat.resize(cat.query_dim);
  i_feat.resize(i_dim);
  x_feat.resize(cat.inter_dim);
  norm.resize(std::max(cat.query_dim, std::max(i_dim, cat.inter_dim)));
  q_emb.resize(e);
  i_emb.resize(e);
  h0.resize(cell.hidden_dim);
  h2.resize(cell.hidden_dim);
  TILEWRIGHT_STAGE_LAP(Setup);
  cat.query(problem, hw, q_feat.data());
  if (x_kernel != nullptr) {
    x_problem.resize(cat.inter_problem_dim);
    cat.inter_problem(problem, hw, x_problem.data());
  }
  query_embedding(
      k, cell, w, q_feat.data(), cat.query_dim, norm.data(), h0.data(), h2.data(), q_emb.data());
  TILEWRIGHT_STAGE_LAP(QueryTower);

  // A cascade model scores a tier that is not a whitelist by the embedding
  // term alone and keeps its best `cascade_keys` candidates, and more while
  // the tiers so far (`before` configs ahead of this one) keep fewer than
  // `min_scored` configs, for the full score. Candidates are keys, or views
  // when launch keys split them: keys of one view share the embedding term,
  // and other_emb has a row per view. `configs_of` counts a candidate's
  // configs, `item_key` names a key whose item features it has. The
  // candidates are ascending; so is what is kept. Returns the configs kept;
  // sets `scored` when s.emb holds the embedding term of every candidate
  // kept, or -inf where it is not finite.
  std::vector<float>& emb = s.emb;
  bool scored             = false;
  const auto cascade_by   = [&](std::vector<std::size_t>& candidates,
                              std::size_t tier_configs,
                              std::size_t before,
                              const auto& configs_of,
                              const auto& item_key) {
    const std::size_t width = model.cascade_keys;
    // Every candidate would be kept.
    if (width == 0 || candidates.size() <= width || before + tier_configs <= min_scored)
      return tier_configs;
    scored = true;
    if (emb.size() < nk) emb.resize(nk);
    const auto better = [&](std::size_t a, std::size_t b) {
      return emb[a] > emb[b] || (!(emb[b] > emb[a]) && a < b);
    };
    if (cache != nullptr) {
      // Candidates of a tier that is not a whitelist are outside the
      // whitelist, so other_emb holds their rows.
      // Every block of 16 rows that holds a candidate, at once; the
      // candidates are ascending.
      const std::size_t views = keys.views();
      float raw[16]           = {};
      std::size_t block       = views;
      if (!rules.family_region() || cache->kept_row.empty()) {
        for (const std::size_t c : candidates) {
          if (c / 16 != block) {
            block = c / 16;
            const BlockWeights rows{
                BlockWeights::Type::F32, &cache->other_emb[block * 16 * e], 1.0f};
            k.linear_blocks(
                rows, nullptr, q_emb.data(), std::min<std::size_t>(16, views - block * 16), e, raw);
          }
          const float v = raw[c % 16] / w.temperature;
          emb[c]        = finite(v) ? v : -std::numeric_limits<float>::infinity();
        }
      } else {
        float kept_raw[16]     = {};
        std::size_t kept_block = cache->kept_views;
        for (const std::size_t c : candidates) {
          const std::uint32_t r = cache->kept_row[c];
          float raw_c           = 0.0f;
          if (r == kNoSlot) {
            if (c / 16 != block) {
              block = c / 16;
              const BlockWeights rows{
                  BlockWeights::Type::F32, &cache->other_emb[block * 16 * e], 1.0f};
              k.linear_blocks(rows,
                              nullptr,
                              q_emb.data(),
                              std::min<std::size_t>(16, views - block * 16),
                              e,
                              raw);
            }
            raw_c = raw[c % 16];
          } else {
            if (r / 16 != kept_block) {
              kept_block = r / 16;
              const BlockWeights rows{
                  BlockWeights::Type::F32, &cache->kept_emb[kept_block * 16 * e], 1.0f};
              k.linear_blocks(rows,
                              nullptr,
                              q_emb.data(),
                              std::min<std::size_t>(16, cache->kept_views - kept_block * 16),
                              e,
                              kept_raw);
            }
            raw_c = kept_raw[r % 16];
          }
          const float v = raw_c / w.temperature;
          emb[c]        = finite(v) ? v : -std::numeric_limits<float>::infinity();
        }
      }
    } else {
      for (std::size_t c : candidates) {
        const std::size_t key = item_key(c);
        cat.item(model, configs[keys.first[key]], keys.attributes(key), i_feat.data());
        item_embedding(k, cell, w, i_feat.data(), i_dim, norm.data(), h0.data(), i_emb.data());
        const float v = k.dot(q_emb.data(), i_emb.data(), e) / w.temperature;
        emb[c]        = finite(v) ? v : -std::numeric_limits<float>::infinity();
      }
    }
    TILEWRIGHT_STAGE_LAP(Stage1);
    // Every candidate has a config, so the best `width` hold at least `width`
    // configs.
    if (before + width >= min_scored && nk <= 0xFFFFFFFFull) {
      std::vector<std::uint64_t>& ordered = s.ordered;
      ordered.resize(candidates.size());
      for (std::size_t i = 0; i < candidates.size(); ++i)
        ordered[i] = order_key(emb[candidates[i]], candidates[i]);
      s.spare.resize(ordered.size());
      select_largest(ordered.data(), ordered.size(), width, s.spare.data());
      candidates.resize(width);
      for (std::size_t i = 0; i < width; ++i) candidates[i] = order_key_index(ordered[i]);
      std::sort(candidates.begin(), candidates.end());
      std::size_t kept = 0;
      for (const std::size_t c : candidates) kept += configs_of(c);
      TILEWRIGHT_STAGE_LAP(TopW);
      return kept;
    }
    std::size_t keep = width;
    std::nth_element(candidates.begin(),
                     candidates.begin() + static_cast<std::ptrdiff_t>(keep - 1),
                     candidates.end(),
                     better);
    std::size_t kept = 0;
    for (std::size_t i = 0; i < keep; ++i) kept += configs_of(candidates[i]);
    if (before + kept < min_scored) {
      std::sort(candidates.begin() + static_cast<std::ptrdiff_t>(keep), candidates.end(), better);
      while (keep < candidates.size() && before + kept < min_scored)
        kept += configs_of(candidates[keep++]);
    }
    candidates.resize(keep);
    std::sort(candidates.begin(), candidates.end());
    TILEWRIGHT_STAGE_LAP(TopW);
    return kept;
  };
  // With launch keys, a tier's views are listed once each (a whitelist admits
  // or rejects a view as a whole, so a view's feasible keys are all in one
  // tier) and a kept view brings all its feasible keys.
  const bool split = keys.views() != nk;
  const auto split_cascade =
      [&](std::vector<std::size_t>& tier, std::size_t tier_configs, std::size_t before) {
        const std::size_t width = model.cascade_keys;
        if (width == 0 || tier.size() <= width || before + tier_configs <= min_scored)
          return tier_configs;
        if (s.listed.size() < nk) s.listed.resize(nk, 0);
        std::vector<std::size_t>& views = s.views;
        views.clear();
        for (std::size_t key : tier) {
          std::uint32_t& mark = s.listed[keys.view_of[key]];
          if (mark != epoch) {
            mark = epoch;
            views.push_back(keys.view_of[key]);
          }
        }
        if (views.size() <= width) return tier_configs;
        const auto view_configs = [&](std::size_t v) {
          std::size_t c = 0;
          for (std::size_t i = keys.view_begin[v]; i < keys.view_begin[v + 1]; ++i)
            if (feasible(keys.view_keys[i])) c += size_of(keys.view_keys[i]);
          return c;
        };
        const std::size_t kept =
            cascade_by(views, tier_configs, before, view_configs, [&](std::size_t v) {
              return keys.first_of_view(v);
            });
        tier.clear();
        for (std::size_t v : views)
          for (std::size_t i = keys.view_begin[v]; i < keys.view_begin[v + 1]; ++i)
            if (feasible(keys.view_keys[i])) tier.push_back(keys.view_keys[i]);
        std::sort(tier.begin(), tier.end());
        return kept;
      };
  const auto by_key = [](std::size_t key) { return key; };
  if (!tier1_is_whitelist)
    tier1_configs = split ? split_cascade(tier1, tier1_configs, 0)
                          : cascade_by(tier1, tier1_configs, 0, size_of, by_key);
  // MLREC_v2 models score every key of tier 2.
  if (!tier2.empty() && model.format_version >= 3) {
    std::size_t tier2_configs = 0;
    for (std::size_t key : tier2) tier2_configs += size_of(key);
    if (split)
      split_cascade(tier2, tier2_configs, tier1_configs);
    else
      cascade_by(tier2, tier2_configs, tier1_configs, size_of, by_key);
  }

  // Read only for keys scored by this call.
  std::vector<float>& score = s.score;
  if (score.size() < nk) score.resize(nk);
  std::vector<std::size_t>& order  = s.order;
  std::vector<std::size_t>& ranked = s.ranked;
  std::vector<std::size_t>& run    = s.run;
  order.clear();
  order.reserve(tier1_configs);
  // Interaction scores of up to kBatch keys at a time: column c of x_batch
  // holds the interaction features of key c of the batch. A batch costs about
  // as much as kMinBatch keys scored one at a time.
  constexpr std::size_t kMinBatch = 6;
  BatchBuffer& x_batch            = s.x_batch;
  BatchBuffer& h_batch            = s.h_batch;
  x_batch.resize(cat.inter_dim * kBatch);
  h_batch.resize(cell.inter_hidden * kBatch);
  if (!cat.reads_launch) launch = nullptr;
  // At most one tier went through the cascade. A non-finite embedding term
  // makes the total non-finite either way, so the cascade's -inf stands for
  // it.
  const auto rank_tier = [&](const std::vector<std::size_t>& tier, bool reuse) {
    ranked.clear();
    // The keys' scattered data is requested at once rather than as each key
    // is scored, and their plans asked for before any is scored.
    for (const std::size_t key : tier) {
      prefetch(&keys.first[key]);
      if (x_kernel != nullptr) prefetch(x_kernel + key * cat.inter_kernel_dim);
    }
    for (const std::size_t key : tier) prefetch(&configs[keys.first[key]]);
    std::vector<LaunchPlan>& plans = s.plans;
    plans.assign(tier.size(), LaunchPlan{});
    if (launch != nullptr)
      for (std::size_t i = 0; i < tier.size(); ++i)
        plans[i] = launch->plan(configs[keys.first[tier[i]]], problem, context);
    TILEWRIGHT_STAGE_LAP(Launch);
    float emb_score[kBatch], inter[kBatch];
    for (std::size_t b0 = 0; b0 < tier.size(); b0 += kBatch) {
      const std::size_t nc = std::min(kBatch, tier.size() - b0);
      const bool batched   = nc >= kMinBatch;
      for (std::size_t c = 0; c < nc; ++c) {
        const std::size_t key = tier[b0 + c];
        const Config& cfg     = configs[keys.first[key]];
        const std::int64_t* a = keys.attributes(key);
        if (reuse) {
          emb_score[c] = emb[keys.view(key)];
        } else {
          const float* ie = nullptr;
          if (cache != nullptr) {
            ie = cached_embedding(*cache, keys, key, e, i_emb.data());
          } else {
            cat.item(model, cfg, a, i_feat.data());
            item_embedding(k, cell, w, i_feat.data(), i_dim, norm.data(), h0.data(), i_emb.data());
            ie = i_emb.data();
          }
          emb_score[c] = k.dot(q_emb.data(), ie, e) / w.temperature;
        }
        TILEWRIGHT_STAGE_LAP(FullScore);
        const LaunchPlan& plan = plans[b0 + c];
        if (x_kernel != nullptr)
          cat.inter_pair(problem,
                         cfg,
                         a,
                         model,
                         hw,
                         x_kernel + key * cat.inter_kernel_dim,
                         x_problem.data(),
                         plan,
                         x_feat.data());
        else
          cat.inter(model, problem, cfg, a, hw, plan, x_feat.data());
        TILEWRIGHT_STAGE_LAP(InterFeatures);
        if (batched) {
          for (std::size_t j = 0; j < cat.inter_dim; ++j) x_batch[j * kBatch + c] = x_feat[j];
        } else {
          whiten(x_feat.data(), w.x_mean, w.x_scale, w.x_lo, w.x_hi, cat.inter_dim, norm.data());
          k.linear_relu(w.x_w0.data(),
                        w.x_b0.data(),
                        norm.data(),
                        cell.inter_hidden,
                        cat.inter_dim,
                        h_batch.data());
          inter[c] = k.dot(w.x_w2.data(), h_batch.data(), cell.inter_hidden);
        }
      }
      if (batched) {
        for (std::size_t j = 0; j < cat.inter_dim; ++j)
          std::fill(&x_batch[j * kBatch] + nc, &x_batch[j * kBatch] + kBatch, 0.0f);
        k.whiten_batch(x_batch.data(),
                       w.x_mean.data(),
                       w.x_scale.data(),
                       w.x_lo.empty() ? nullptr : w.x_lo.data(),
                       w.x_hi.data(),
                       cat.inter_dim);
        k.interaction_batch(w.x_w0.data(),
                            w.x_b0.data(),
                            w.x_w2.data(),
                            x_batch.data(),
                            cell.inter_hidden,
                            cat.inter_dim,
                            nc,
                            h_batch.data(),
                            inter);
      }
      for (std::size_t c = 0; c < nc; ++c) {
        const float total = emb_score[c] + (w.x_b2 + inter[c]);
        if (!finite(total)) continue;
        const std::size_t key = tier[b0 + c];
        score[key]            = total;
        s.scored[key]         = epoch;
        ranked.push_back(key);
      }
    }
    TILEWRIGHT_STAGE_LAP(FullScore);
    const auto better = [&](std::size_t a, std::size_t b) {
      return score[a] > score[b] || (!(score[b] > score[a]) && a < b);
    };
    // Every key has a config, so the best `want` keys, with the keys tied with
    // the last of them, hold the first `want` configs of the tier.
    const std::size_t want = out.limit - order.size();
    std::size_t end        = ranked.size();
    if (want < end) {
      const auto mid = ranked.begin() + static_cast<std::ptrdiff_t>(want);
      std::partial_sort(ranked.begin(), mid, ranked.end(), better);
      const float last = score[*(mid - 1)];
      const auto tied =
          std::partition(mid, ranked.end(), [&](std::size_t key) { return score[key] == last; });
      std::sort(mid, tied);
      end = static_cast<std::size_t>(tied - ranked.begin());
    } else {
      std::sort(ranked.begin(), ranked.end(), better);
    }
    // Configs whose keys score equally interleave in input (or prior) order.
    for (std::size_t i = 0; i < end && order.size() < out.limit;) {
      std::size_t j = i + 1;
      while (j < end && score[ranked[j]] == score[ranked[i]]) ++j;
      if (j == i + 1 && prior == nullptr) {
        const std::size_t key = ranked[i];
        order.insert(
            order.end(),
            keys.members.begin() + static_cast<std::ptrdiff_t>(keys.member_begin[key]),
            keys.members.begin() + static_cast<std::ptrdiff_t>(keys.member_begin[key + 1]));
      } else {
        run.clear();
        for (std::size_t r = i; r < j; ++r)
          for (std::size_t m = keys.member_begin[ranked[r]]; m < keys.member_begin[ranked[r] + 1];
               ++m)
            run.push_back(keys.members[m]);
        if (j > i + 1) std::sort(run.begin(), run.end());
        if (prior != nullptr)
          std::stable_sort(run.begin(), run.end(), [&](std::size_t a, std::size_t b) {
            return prior_before((*prior)[a], (*prior)[b]);
          });
        order.insert(order.end(), run.begin(), run.end());
      }
      i = j;
    }
    TILEWRIGHT_STAGE_LAP(Results);
  };
  rank_tier(tier1, scored && !tier1_is_whitelist);
  if (!tier2.empty() && order.size() < out.limit) rank_tier(tier2, scored && tier1_is_whitelist);
  if (order.size() > out.limit) order.resize(out.limit);

  if (env_knobs().pick_log && !order.empty())
    log_pick(
        problem, cell, configs[order.front()], score[keys.key_of[order.front()]], order.front(), n);

  std::vector<Result>& results = out.results;
  results.clear();
  if (!out.unscored) {
    results.reserve(order.size());
    for (std::size_t ci : order)
      results.push_back(Result{ci, static_cast<double>(score[keys.key_of[ci]]), true});
    TILEWRIGHT_STAGE_END(Results);
    return;
  }
  // Every member of a scored key is in `order`.
  results.resize(n);
  std::size_t pos = 0;
  for (std::size_t ci : order)
    results[pos++] = Result{ci, static_cast<double>(score[keys.key_of[ci]]), true};
  for (std::size_t j = 0; j < n; ++j)
    if (s.scored[keys.key_of[j]] != epoch) results[pos++] = Result{j, 0.0, false};
  TILEWRIGHT_STAGE_END(Results);
}

// `keys` null: built from `configs`.
void rank_pool(const Model& model,
               const Problem& problem,
               const Hardware& hardware,
               const ExecutionContext& context,
               const std::vector<Config>& configs,
               const PoolKeys* keys,
               std::size_t min_scored,
               TieBreak tie_break,
               const std::vector<double>* prior,
               const CellCacheSource* source,
               const LaunchModel* launch,
               const Output& out) {
  TILEWRIGHT_STAGE_BEGIN();
  const std::size_t n = configs.size();
  if (prior != nullptr && prior->size() != n)
    throw std::invalid_argument("tilewright: the prior holds " + std::to_string(prior->size()) +
                                " values for " + std::to_string(n) + " configs");
  if (n == 0 || (out.limit == 0 && !out.unscored)) return out.results.clear();
  if (!supports(model, context, hardware) || !valid_hardware(hardware))
    return write_unscored(n, out);
  const int cell = route_cell(model, problem);
  if (cell < 0) return write_unscored(n, out);
  const std::vector<double>* order_prior = tie_break == TieBreak::Prior ? prior : nullptr;
  const std::size_t ci                   = static_cast<std::size_t>(cell);
  if (keys != nullptr)
    return rank_in_cell(model,
                        ci,
                        problem,
                        hardware,
                        context,
                        configs,
                        *keys,
                        min_scored,
                        order_prior,
                        source,
                        launch,
                        out);
  const PoolKeys built = pool_keys(model, configs, launch);
  rank_in_cell(model,
               ci,
               problem,
               hardware,
               context,
               configs,
               built,
               min_scored,
               order_prior,
               source,
               launch,
               out);
}

std::vector<Result> rank_all(const Model& model,
                             const Problem& problem,
                             const Hardware& hardware,
                             const ExecutionContext& context,
                             const std::vector<Config>& configs,
                             const PoolKeys* keys,
                             std::size_t min_scored,
                             TieBreak tie_break,
                             const std::vector<double>* prior,
                             const CellCacheSource* source,
                             const LaunchModel* launch) {
  std::vector<Result> results;
  rank_pool(model,
            problem,
            hardware,
            context,
            configs,
            keys,
            min_scored,
            tie_break,
            prior,
            source,
            launch,
            Output{results, std::numeric_limits<std::size_t>::max(), true});
  return results;
}

}  // namespace

void set_rank_epoch(std::uint32_t epoch) noexcept {
  Scratch& s = scratch();
  for (std::vector<std::uint32_t>* v : {&s.feasible, &s.in_tier1, &s.scored, &s.listed})
    std::fill(v->begin(), v->end(), 0u);
  s.epoch = std::min(epoch, Scratch::kMaxEpoch);
}

void check_launch(const Model& model, const LaunchModel* launch) {
  const std::string& want = model.provenance.launch_model;
  if (want.empty()) return;
  if (launch == nullptr)
    throw std::invalid_argument("tilewright: the model was trained with launch model '" + want +
                                "' and the candidate set has none");
  const std::string got = launch->id();
  if (got != want)
    throw std::invalid_argument("tilewright: the model was trained with launch model '" + want +
                                "' and the candidate set's launch model is '" + got + "'");
}

int route_cell(const Model& model, const Problem& problem) noexcept {
  const long long forced = env_knobs().force_cell;
  if (forced >= 0 && static_cast<unsigned long long>(forced) < model.cells.size())
    return static_cast<int>(forced);
  return resolve_cell(model, problem);
}

std::vector<CellBounds> cell_bounds(const Model& model) {
  std::vector<CellBounds> bounds(model.cells.size());
  for (std::size_t base = 0; base < kBaseCells; ++base) {
    const Box box = base_box(base);
    if (walk_bounds(model, model.base_nodes[base], box, 0, bounds) || model.format_version < 3)
      continue;
    // Problems the base's own tree leaves without a cell go to the fallbacks
    // in order; every fallback up to the first that serves the whole box may
    // receive some.
    for (std::uint8_t other : model.base_fallbacks[base])
      if (walk_bounds(model, model.base_nodes[other], box, 0, bounds)) break;
  }
  return bounds;
}

PoolKeys pool_keys(const Model& model,
                   const std::vector<Config>& configs,
                   const LaunchModel* launch) {
  const FeatureCatalog& catalog = *model.catalog;
  const std::size_t n           = configs.size();
  if (!catalog.reads_launch || !model.launch_keys) launch = nullptr;
  // The launch key follows the view key in the grouping key only: whitelist
  // hashes cover the view key alone.
  const std::size_t size   = model.view_key_size + (launch != nullptr ? 1 : 0);
  const std::size_t n_attr = model.attribute_reads.size();
  PoolKeys p;
  p.attribute_count = n_attr;
  p.key_of.resize(n);
  p.first.reserve(n);
  std::vector<std::uint64_t> key_hash, first_launch, view_hash;
  std::vector<std::size_t> view_first;
  key_hash.reserve(n);
  std::uint64_t buffers[2][kMaxViewKeySize + 1], other[kMaxViewKeySize + 1];
  std::int64_t attrs[kMaxAttributeReads];
  std::size_t slots = 16;
  while (slots < 2 * n) slots <<= 1;
  const std::size_t empty = static_cast<std::size_t>(-1);
  std::vector<std::size_t> table(slots, empty), view_table;
  if (launch != nullptr) view_table.assign(slots, empty);
  const auto add_view = [&](const std::uint64_t* value, std::size_t key) {
    const std::uint64_t h = hash_key(value, model.view_key_size);
    for (std::size_t s = static_cast<std::size_t>(h) & (slots - 1);; s = (s + 1) & (slots - 1)) {
      const std::size_t v = view_table[s];
      if (v == empty) {
        view_table[s] = view_first.size();
        p.view_of.push_back(view_first.size());
        view_first.push_back(key);
        view_hash.push_back(h);
        return;
      }
      if (view_hash[v] != h) continue;
      const std::size_t vk = view_first[v];
      catalog.view_key(model, configs[p.first[vk]], p.attributes(vk), other);
      if (std::equal(value, value + model.view_key_size, other)) {
        p.view_of.push_back(v);
        return;
      }
    }
  };
  for (std::size_t ci = 0; ci < n; ++ci) {
    const Config& c = configs[ci];
    p.nt_a |= c.cache_hints_a == 4;
    p.nt_b |= c.cache_hints_b == 4;
    std::uint64_t* value = buffers[ci & 1];
    read_attributes(model, c, attrs);
    catalog.view_key(model, c, attrs, value);
    if (launch != nullptr) value[model.view_key_size] = launch->launch_key(c);
    if (ci > 0 && std::equal(value, value + size, buffers[(ci & 1) ^ 1])) {
      p.key_of[ci] = p.key_of[ci - 1];
      continue;
    }
    const std::uint64_t h = hash_key(value, size);
    for (std::size_t s = static_cast<std::size_t>(h) & (slots - 1);; s = (s + 1) & (slots - 1)) {
      const std::size_t key = table[s];
      if (key == empty) {
        table[s]     = p.first.size();
        p.key_of[ci] = p.first.size();
        p.first.push_back(ci);
        p.shape.push_back(kernel_shape(c));
        p.attribute_values.insert(p.attribute_values.end(), attrs, attrs + n_attr);
        key_hash.push_back(h);
        if (model.trained_key_gate) {
          const bool trained = std::binary_search(model.trained_keys.begin(),
                                                  model.trained_keys.end(),
                                                  view_key_hash(value, model.view_key_size));
          p.trained.push_back(trained ? 1 : 0);
          p.untrained += trained ? 0 : 1;
        }
        if (launch != nullptr) {
          first_launch.push_back(value[model.view_key_size]);
          add_view(value, p.first.size() - 1);
        }
        break;
      }
      if (key_hash[key] != h) continue;
      catalog.view_key(model, configs[p.first[key]], p.attributes(key), other);
      if (launch != nullptr) other[model.view_key_size] = first_launch[key];
      if (std::equal(value, value + size, other)) {
        p.key_of[ci] = key;
        break;
      }
    }
  }
  const std::size_t nk = p.first.size();
  // A gate that keeps every key changes nothing.
  if (p.untrained == 0) p.trained.clear();
  if (model.soft_rules.non_temporal && model.soft_rules.non_temporal_family) {
    const std::vector<std::uint8_t> family = family_bits(configs);
    for (std::size_t key = 0; key < nk; ++key) p.shape[key].family = family[p.first[key]];
  }
  if (view_first.size() == nk) {
    p.view_of.clear();
  } else if (!p.view_of.empty()) {
    const std::size_t nv = view_first.size();
    p.view_begin.assign(nv + 1, 0);
    p.view_keys.resize(nk);
    for (std::size_t key = 0; key < nk; ++key) ++p.view_begin[p.view_of[key] + 1];
    for (std::size_t v = 0; v < nv; ++v) p.view_begin[v + 1] += p.view_begin[v];
    std::vector<std::size_t> next(p.view_begin.begin(), p.view_begin.end() - 1);
    for (std::size_t key = 0; key < nk; ++key) p.view_keys[next[p.view_of[key]]++] = key;
  }
  p.member_begin.resize(nk + 1);
  p.members.resize(n);
  if (nk == n) {
    for (std::size_t ci = 0; ci <= n; ++ci) p.member_begin[ci] = ci;
    p.members = p.first;
    return p;
  }
  for (std::size_t ci = 0; ci < n; ++ci) ++p.member_begin[p.key_of[ci] + 1];
  for (std::size_t key = 0; key < nk; ++key) p.member_begin[key + 1] += p.member_begin[key];
  std::vector<std::size_t> next(p.member_begin.begin(), p.member_begin.end() - 1);
  for (std::size_t ci = 0; ci < n; ++ci) p.members[next[p.key_of[ci]]++] = ci;
  return p;
}

std::vector<Result> rank_keys(const Model& model,
                              const Problem& problem,
                              const Hardware& hardware,
                              const ExecutionContext& context,
                              const std::vector<Config>& configs,
                              const PoolKeys& keys,
                              std::size_t min_scored,
                              TieBreak tie_break,
                              const std::vector<double>* prior,
                              const LaunchModel* launch) {
  return rank_all(model,
                  problem,
                  hardware,
                  context,
                  configs,
                  &keys,
                  min_scored,
                  tie_break,
                  prior,
                  nullptr,
                  launch);
}

}  // namespace detail

struct CandidateSet::Impl final : detail::CellCacheSource {
  // The cache of one cell, counted against the cache limit while present.
  struct Slot final : detail::CacheEntry {
    std::mutex mutex;  // guards cache
    std::mutex build;  // held while the cache is built
    std::shared_ptr<const detail::CellCache> cache;

    Slot()                       = default;
    Slot(const Slot&)            = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() { detail::CacheBudget::instance().forget(*this); }

    void drop() noexcept override {
      std::shared_ptr<const detail::CellCache> released;
      std::lock_guard<std::mutex> lock(mutex);
      released.swap(cache);
    }

    std::shared_ptr<const detail::CellCache> current() {
      std::lock_guard<std::mutex> lock(mutex);
      return cache;
    }
  };
  // FeatureCatalog::inter_kernel of every key, for the data types and device
  // of the first ranking that scores a config.
  struct KernelPart {
    std::once_flag once;
    DataType a = DataType::None, b = DataType::None, mi = DataType::None;
    double lds = 0.0;
    std::vector<float> values;
  };

  ModelPtr model;
  std::vector<Config> configs;
  PoolOptions options;
  detail::PoolKeys keys;
  std::vector<float> item_features;  // keys x item_dim
  // (signature, key) of every key, ascending.
  std::vector<std::pair<detail::Signature, std::size_t>> signature_keys;
  // (view_key_hash, key) of every key, ascending (MLREC_v3).
  std::vector<std::pair<std::uint64_t, std::size_t>> view_key_keys;
  std::unique_ptr<Slot[]> slots;  // one per model cell
  std::vector<detail::CellBounds> bounds;
  mutable KernelPart kernel_part;

  void embed(const detail::Cell& cell,
             const detail::CellWeights& w,
             std::size_t key,
             std::vector<float>& hidden,
             std::vector<float>& norm,
             float* out) const {
    const std::size_t i_dim = model->item_dim;
    detail::item_embedding(detail::kernels(),
                           cell,
                           w,
                           &item_features[key * i_dim],
                           i_dim,
                           norm.data(),
                           hidden.data(),
                           out);
  }

  // The whitelisted keys of `cell` and their embeddings, into `c`.
  void build_whitelist(const detail::Cell& cell,
                       const detail::CellWeights& w,
                       detail::CellCache& c) const {
    const std::size_t nk    = keys.first.size();
    const std::size_t e     = cell.embed_dim;
    const auto by_signature = [](const std::pair<detail::Signature, std::size_t>& a,
                                 const std::pair<detail::Signature, std::size_t>& b) {
      return a.first < b.first;
    };
    const auto by_hash = [](const std::pair<std::uint64_t, std::size_t>& a,
                            const std::pair<std::uint64_t, std::size_t>& b) {
      return a.first < b.first;
    };
    std::vector<std::size_t>& wl = c.whitelist_keys;
    if (model->whitelist_tier) {
      for (const detail::Signature& sig : cell.signatures) {
        const auto range = std::equal_range(signature_keys.begin(),
                                            signature_keys.end(),
                                            std::make_pair(sig, std::size_t{0}),
                                            by_signature);
        for (auto it = range.first; it != range.second; ++it) wl.push_back(it->second);
      }
      for (std::uint64_t h : cell.view_key_hashes) {
        const auto range = std::equal_range(
            view_key_keys.begin(), view_key_keys.end(), std::make_pair(h, std::size_t{0}), by_hash);
        for (auto it = range.first; it != range.second; ++it) wl.push_back(it->second);
      }
    }
    std::sort(wl.begin(), wl.end());
    wl.erase(std::unique(wl.begin(), wl.end()), wl.end());
    if (!keys.trained.empty())
      wl.erase(std::remove_if(
                   wl.begin(), wl.end(), [&](std::size_t key) { return keys.trained[key] == 0; }),
               wl.end());
    c.slot.assign(nk, detail::kNoSlot);
    c.whitelist_emb.resize(wl.size() * e);
    std::vector<float> hidden(cell.hidden_dim), norm(model->item_dim);
    for (std::size_t i = 0; i < wl.size(); ++i) {
      c.slot[wl[i]] = static_cast<std::uint32_t>(i);
      embed(cell, w, wl[i], hidden, norm, &c.whitelist_emb[i * e]);
    }
  }

  void build_possible(std::size_t cell_index, detail::CellCache& c) const {
    const long long forced = detail::env_knobs().force_cell;
    if (forced >= 0 && static_cast<unsigned long long>(forced) < model->cells.size()) return;
    const detail::CellBounds& b = bounds[cell_index];
    for (std::size_t key = 0; key < keys.first.size(); ++key) {
      const detail::KernelShape& k = keys.shape[key];
      if (!keys.trained.empty() && keys.trained[key] == 0) continue;
      if (detail::never_feasible(b, k, keys.nt_a, keys.nt_b, model->soft_rules)) continue;
      c.possible.push_back(static_cast<std::uint32_t>(key));
      c.possible_shape.push_back(
          {k.mt_m,
           k.mt_n,
           static_cast<std::uint16_t>(k.mt_k % 1024),
           static_cast<std::uint8_t>(detail::hint_class(k.cache_hints_a, k.cache_hints_b)),
           k.family});
      if (k.cache_hints_a != 0 || k.cache_hints_b != 0) {
        c.hinted_bounds.include(k);
        ++c.hinted_keys;
        if ((k.family & detail::kFamilyFallback) != 0 && (k.family & detail::kFamilyPlain) == 0) {
          c.kept_hinted.push_back(static_cast<std::uint32_t>(key));
          c.kept_configs += keys.member_begin[key + 1] - keys.member_begin[key];
        }
      } else {
        c.plain.push_back(static_cast<std::uint32_t>(key));
        c.plain_bounds.include(k);
        c.plain_configs += keys.member_begin[key + 1] - keys.member_begin[key];
      }
    }
    if (!c.kept_hinted.empty()) {
      c.plain_and_kept.resize(c.plain.size() + c.kept_hinted.size());
      std::merge(c.plain.begin(),
                 c.plain.end(),
                 c.kept_hinted.begin(),
                 c.kept_hinted.end(),
                 c.plain_and_kept.begin());
    }
    c.pruned = true;
  }

  std::shared_ptr<const detail::CellCache> cache(std::size_t cell_index,
                                                 const detail::CellWeights& w,
                                                 bool all_keys) const override {
    Slot& slot                  = slots[cell_index];
    detail::CacheBudget& budget = detail::CacheBudget::instance();
    const auto usable           = [&](const std::shared_ptr<const detail::CellCache>& c) {
      return c != nullptr && (c->all_keys || !all_keys);
    };
    std::shared_ptr<const detail::CellCache> c = slot.current();
    if (usable(c)) {
      budget.touch(slot);
      return c;
    }
    std::lock_guard<std::mutex> building(slot.build);
    c = slot.current();
    if (usable(c)) {
      budget.touch(slot);
      return c;
    }
    const detail::Cell& cell = model->cells[cell_index];
    const std::size_t e      = cell.embed_dim;
    auto built               = std::make_shared<detail::CellCache>();
    if (c != nullptr) {
      built->whitelist_keys = c->whitelist_keys;
      built->slot           = c->slot;
      built->whitelist_emb  = c->whitelist_emb;
      built->possible       = c->possible;
      built->plain          = c->plain;
      built->kept_hinted    = c->kept_hinted;
      built->plain_and_kept = c->plain_and_kept;
      built->kept_configs   = c->kept_configs;
      built->possible_shape = c->possible_shape;
      built->plain_bounds   = c->plain_bounds;
      built->hinted_bounds  = c->hinted_bounds;
      built->plain_configs  = c->plain_configs;
      built->hinted_keys    = c->hinted_keys;
      built->pruned         = c->pruned;
    } else {
      build_whitelist(cell, w, *built);
      build_possible(cell_index, *built);
    }
    if (all_keys) {
      const std::size_t nv = keys.views();
      built->other_emb.assign((nv + 15) / 16 * 16 * e, 0.0f);
      std::vector<float> hidden(cell.hidden_dim), norm(model->item_dim), row(e);
      for (std::size_t v = 0; v < nv; ++v) {
        const std::size_t key = keys.first_of_view(v);
        if (built->slot[key] != detail::kNoSlot) continue;
        embed(cell, w, key, hidden, norm, row.data());
        for (std::size_t j = 0; j < e; ++j)
          built->other_emb[(v / 16 * e + j) * 16 + v % 16] = row[j];
      }
      std::vector<std::size_t> kept;
      for (const std::uint32_t key : built->kept_hinted) {
        const std::size_t v = keys.view(key);
        if (built->slot[keys.first_of_view(v)] == detail::kNoSlot) kept.push_back(v);
      }
      std::sort(kept.begin(), kept.end());
      kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
      if (!kept.empty()) {
        built->kept_row.assign(nv, detail::kNoSlot);
        built->kept_emb.assign((kept.size() + 15) / 16 * 16 * e, 0.0f);
        built->kept_views = kept.size();
        for (std::size_t r = 0; r < kept.size(); ++r) {
          const std::size_t v = kept[r];
          built->kept_row[v]  = static_cast<std::uint32_t>(r);
          for (std::size_t j = 0; j < e; ++j)
            built->kept_emb[(r / 16 * e + j) * 16 + r % 16] =
                built->other_emb[(v / 16 * e + j) * 16 + v % 16];
        }
      }
      built->all_keys = true;
    }
    c = built;
    budget.publish(slot, detail::cache_bytes(*built), [&] {
      std::lock_guard<std::mutex> lock(slot.mutex);
      slot.cache = c;
    });
    return c;
  }

  void prepare_cell(int cell) const {
    if (cell < 0 || static_cast<std::size_t>(cell) >= model->cells.size()) return;
    const std::size_t c                                      = static_cast<std::size_t>(cell);
    const std::shared_ptr<const detail::CellWeights> weights = model->weights(c);
    cache(c, *weights, true);
  }

  const float* inter_kernel(const Problem& p, const detail::HwView& hw) const override {
    KernelPart& part = kernel_part;
    std::call_once(part.once, [&] {
      const detail::FeatureCatalog& cat = *model->catalog;
      const std::size_t d               = cat.inter_kernel_dim;
      part.a                            = p.a_dtype;
      part.b                            = p.b_dtype;
      part.mi                           = p.mi_dtype;
      part.lds                          = hw.LDS;
      part.values.resize(keys.first.size() * d);
      for (std::size_t key = 0; key < keys.first.size(); ++key)
        cat.inter_kernel(*model, p, configs[keys.first[key]], hw, &part.values[key * d]);
    });
    if (p.a_dtype != part.a || p.b_dtype != part.b || p.mi_dtype != part.mi || hw.LDS != part.lds)
      return nullptr;
    return part.values.data();
  }
};

CandidateSet::CandidateSet(ModelPtr model, std::vector<Config> configs, PoolOptions options) {
  if (!model) throw std::invalid_argument("tilewright::CandidateSet requires a model");
  detail::check_attributes(configs);
  detail::check_launch(*model, options.launch.get());
  const detail::FeatureCatalog& cat = *model->catalog;
  impl_                             = std::make_unique<Impl>();
  impl_->options                    = options;
  impl_->keys                       = detail::pool_keys(*model, configs, options.launch.get());
  impl_->configs                    = std::move(configs);
  const std::size_t nk              = impl_->keys.first.size();
  const std::size_t i_dim           = model->item_dim;
  const bool keyed_whitelist        = model->format_version >= 3;
  impl_->item_features.resize(nk * i_dim);
  impl_->signature_keys.reserve(nk);
  std::uint64_t view_key[detail::kMaxViewKeySize];
  for (std::size_t key = 0; key < nk; ++key) {
    const Config& c       = impl_->configs[impl_->keys.first[key]];
    const std::int64_t* a = impl_->keys.attributes(key);
    cat.item(*model, c, a, &impl_->item_features[key * i_dim]);
    impl_->signature_keys.emplace_back(detail::signature_of(c), key);
    if (keyed_whitelist) {
      cat.view_key(*model, c, a, view_key);
      impl_->view_key_keys.emplace_back(detail::view_key_hash(view_key, model->view_key_size), key);
    }
  }
  std::sort(impl_->signature_keys.begin(), impl_->signature_keys.end());
  std::sort(impl_->view_key_keys.begin(), impl_->view_key_keys.end());
  impl_->slots  = std::make_unique<Impl::Slot[]>(model->cells.size());
  impl_->bounds = detail::cell_bounds(*model);
  impl_->model  = std::move(model);
}

CandidateSet::~CandidateSet() = default;

const Model& CandidateSet::model() const noexcept { return *impl_->model; }

const std::vector<Config>& CandidateSet::configs() const noexcept { return impl_->configs; }

std::size_t CandidateSet::untrained_keys() const noexcept { return impl_->keys.untrained; }

std::size_t CandidateSet::untrained_configs() const noexcept {
  const detail::PoolKeys& keys = impl_->keys;
  std::size_t n                = 0;
  for (std::size_t key = 0; key < keys.trained.size(); ++key)
    if (keys.trained[key] == 0) n += keys.member_begin[key + 1] - keys.member_begin[key];
  return n;
}

std::vector<Result> CandidateSet::rank(const Problem& problem,
                                       const Hardware& hardware,
                                       std::size_t min_scored) const {
  return rank(problem, hardware, ExecutionContext{}, min_scored, nullptr);
}

std::vector<Result> CandidateSet::rank(const Problem& problem,
                                       const Hardware& hardware,
                                       const ExecutionContext& context,
                                       std::size_t min_scored,
                                       const std::vector<double>* prior) const {
  return detail::rank_all(*impl_->model,
                          problem,
                          hardware,
                          context,
                          impl_->configs,
                          &impl_->keys,
                          min_scored,
                          impl_->options.tie_break,
                          prior,
                          impl_.get(),
                          impl_->options.launch.get());
}

std::vector<Result> CandidateSet::rank_top(const Problem& problem,
                                           const Hardware& hardware,
                                           const ExecutionContext& context,
                                           std::size_t k,
                                           const std::vector<double>* prior) const {
  std::vector<Result> results;
  rank_top(problem, hardware, context, k, results, prior);
  return results;
}

void CandidateSet::rank_top(const Problem& problem,
                            const Hardware& hardware,
                            const ExecutionContext& context,
                            std::size_t k,
                            std::vector<Result>& out,
                            const std::vector<double>* prior) const {
  detail::rank_pool(*impl_->model,
                    problem,
                    hardware,
                    context,
                    impl_->configs,
                    &impl_->keys,
                    k,
                    impl_->options.tie_break,
                    prior,
                    impl_.get(),
                    impl_->options.launch.get(),
                    detail::Output{out, k, false});
}

void CandidateSet::prepare(const std::vector<int>& cells) const {
  for (int cell : cells) impl_->prepare_cell(cell);
}

void CandidateSet::prepare() const {
  for (std::size_t cell = 0; cell < impl_->model->cells.size(); ++cell)
    impl_->prepare_cell(static_cast<int>(cell));
}

void CandidateSet::prepare(const Problem& problem, const Hardware& hardware) const {
  const Model& model = *impl_->model;
  impl_->prepare_cell(detail::route_cell(model, problem));
  if (detail::valid_hardware(hardware) && !impl_->configs.empty())
    impl_->inter_kernel(problem, detail::hw_view(model, hardware, ExecutionContext{}));
}

std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored) {
  return rank_configs(
      model, problem, hardware, ExecutionContext{}, configs, min_scored, TieBreak::PoolOrder);
}

std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const ExecutionContext& context,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored,
                                 TieBreak tie_break,
                                 const std::vector<double>* prior) {
  detail::check_attributes(configs);
  return detail::rank_all(model,
                          problem,
                          hardware,
                          context,
                          configs,
                          nullptr,
                          min_scored,
                          tie_break,
                          prior,
                          nullptr,
                          nullptr);
}

std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const ExecutionContext& context,
                                 const std::vector<Config>& configs,
                                 const PoolOptions& options,
                                 std::size_t min_scored,
                                 const std::vector<double>* prior) {
  detail::check_attributes(configs);
  detail::check_launch(model, options.launch.get());
  return detail::rank_all(model,
                          problem,
                          hardware,
                          context,
                          configs,
                          nullptr,
                          min_scored,
                          options.tie_break,
                          prior,
                          nullptr,
                          options.launch.get());
}

bool supports(const Model& model,
              const ExecutionContext& context,
              const Hardware& hardware) noexcept {
  return model.catalog->supports(model, context, hardware);
}

std::vector<std::string> attribute_names(const Model& model) {
  std::vector<std::string> names;
  for (const std::string& name : model.attribute_reads)
    if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
  return names;
}

const char* feature_catalog_hash() noexcept { return detail::v2_catalog().hash; }

std::vector<std::string> feature_catalog_hashes() {
  return {detail::v2_catalog().hash, detail::v3_catalog().hash};
}

ModelInfo describe(const Model& model) {
  ModelInfo info;
  info.arch                     = model.arch;
  info.feature_catalog_hash     = model.feature_hash;
  info.weight_type              = model.weight_type;
  info.n_cells                  = model.cells.size();
  info.n_splits                 = model.splits.size();
  info.format_version           = model.format_version;
  info.feature_spec_hash        = model.feature_spec_hash;
  info.query_dim                = model.query_dim;
  info.item_dim                 = model.item_dim;
  info.inter_dim                = model.inter_dim;
  info.min_cu_fraction          = model.min_cu_fraction;
  info.max_cu_fraction          = model.max_cu_fraction;
  info.schedules                = model.schedules;
  info.cascade_keys             = model.cascade_keys;
  info.whitelist_tier           = model.whitelist_tier;
  info.launch_keys              = model.launch_keys;
  info.small_batched_rule       = model.soft_rules.small_batched_one_tile;
  info.dot2_rule                = model.soft_rules.dot2_small_m;
  info.non_temporal_rule        = model.soft_rules.non_temporal;
  info.non_temporal_family_rule = model.soft_rules.non_temporal_family;
  info.clip                     = model.clip;
  info.bank                     = model.bank;
  info.trained_key_gate         = model.trained_key_gate;
  info.trained_keys             = model.trained_keys.size();
  for (std::size_t b = 0; b < detail::kBaseCells; ++b)
    if (model.base_nodes[b] < 0) ++info.uncovered_base_labels;
  info.provenance   = model.provenance;
  info.packed_bytes = model.payload.size();
  for (std::size_t c = 0; c < model.cells.size(); ++c)
    info.expanded_bytes += detail::cell_weight_bytes(model, c);
  return info;
}

int route(const Model& model, const Problem& problem) noexcept {
  return detail::route_cell(model, problem);
}

std::string cell_label(const Model& model, int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= model.cells.size()) return std::string();
  return model.cells[static_cast<std::size_t>(cell)].label;
}

Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware) {
  return compute_features(model, problem, config, hardware, ExecutionContext{});
}

Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware,
                          const ExecutionContext& context) {
  return compute_features(model, problem, config, hardware, context, LaunchPlan{});
}

Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware,
                          const ExecutionContext& context,
                          const LaunchPlan& plan) {
  Features f;
  if (!detail::valid_hardware(hardware)) return f;
  const detail::FeatureCatalog& cat = *model.catalog;
  const detail::HwView hw           = detail::hw_view(model, hardware, context);
  std::int64_t attrs[detail::kMaxAttributeReads];
  detail::read_attributes(model, config, attrs);
  f.query.resize(cat.query_dim);
  f.item.resize(model.item_dim);
  f.interaction.resize(cat.inter_dim);
  cat.query(problem, hw, f.query.data());
  cat.item(model, config, attrs, f.item.data());
  cat.inter(model, problem, config, attrs, hw, plan, f.interaction.data());
  return f;
}

}  // namespace tilewright
