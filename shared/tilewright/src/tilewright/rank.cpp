// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "features.hpp"
#include "kernels.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
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

int resolve_cell(const Model& model, const Problem& p) noexcept {
  int node = model.base_nodes[base_index(p)];
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

void whiten(const float* f,
            const std::vector<float>& mean,
            const std::vector<float>& scale,
            std::size_t n,
            float* out) {
  for (std::size_t j = 0; j < n; ++j) out[j] = (f[j] - mean[j]) / scale[j];
}

void whiten_item(const float* f, const CellWeights& w, std::size_t n, float* out) {
  for (std::size_t j = 0; j < n; ++j) {
    const float d = f[j] - w.i_mean[j];
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
  whiten(q_feat, w.q_mean, w.q_scale, q_dim, norm);
  k.linear_relu(w.q_w0.data(), w.q_b0.data(), norm, cell.hidden_dim, q_dim, h0);
  k.linear_relu(w.q_w2.data(), w.q_b2.data(), h0, cell.hidden_dim, cell.hidden_dim, h2);
  k.linear(w.q_w4.data(), w.q_b4.data(), h2, cell.embed_dim, cell.hidden_dim, q_emb);
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
  k.linear_relu(w.i_w0.data(), w.i_b0.data(), norm, cell.hidden_dim, i_dim, hidden);
  k.linear(w.i_w2.data(), w.i_b2.data(), hidden, cell.embed_dim, cell.hidden_dim, i_emb);
}

float interaction_score(const Kernels& k,
                        const Cell& cell,
                        const CellWeights& w,
                        const float* x_feat,
                        std::size_t x_dim,
                        float* norm,
                        float* hidden) {
  whiten(x_feat, w.x_mean, w.x_scale, x_dim, norm);
  k.linear_relu(w.x_w0.data(), w.x_b0.data(), norm, cell.inter_hidden, x_dim, hidden);
  return w.x_b2 + k.dot(w.x_w2.data(), hidden, cell.inter_hidden);
}

bool whitelisted(const Cell& cell, const Config& c) {
  const Signature s = signature_of(c);
  return std::find(cell.signatures.begin(), cell.signatures.end(), s) != cell.signatures.end();
}

std::vector<Result> all_unscored(std::size_t n) {
  std::vector<Result> result;
  result.reserve(n);
  for (std::size_t j = 0; j < n; ++j) result.push_back(Result{j, 0.0, false});
  return result;
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
constexpr std::uint64_t kKeyMultipliers[kMaxViewKeySize] = {0x9E3779B97F4A7C15ull,
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
  for (std::size_t i = 0; i < n; ++i) h += key[i] * kKeyMultipliers[i];
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

// Pool data a CandidateSet keeps for one cell, per view key.
struct CellCache {
  std::vector<float> item_emb;  // keys x embed_dim
  std::vector<unsigned char> in_whitelist;
};

class CellCacheSource {
 public:
  virtual ~CellCacheSource()                                                   = default;
  virtual const CellCache& cache(std::size_t cell, const CellWeights& w) const = 0;
};

namespace {

// `prior` orders equal scores when non-null.
std::vector<Result> rank_in_cell(const Model& model,
                                 std::size_t cell_index,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const std::vector<Config>& configs,
                                 const PoolKeys& keys,
                                 std::size_t min_scored,
                                 const std::vector<double>* prior,
                                 const CellCacheSource* source) {
  const std::size_t n       = configs.size();
  const std::size_t nk      = keys.first.size();
  const FeatureCatalog& cat = *model.catalog;
  const Cell& cell          = model.cells[cell_index];
  const CellWeights& w      = model.weights(cell_index);
  const CellCache* cache    = source != nullptr ? &source->cache(cell_index, w) : nullptr;
  const Kernels& k          = kernels();
  const auto size_of        = [&](std::size_t key) {
    return keys.member_begin[key + 1] - keys.member_begin[key];
  };

  std::vector<unsigned char> feasible(nk, 0);
  for (std::size_t key = 0; key < nk; ++key) {
    const Config& c = configs[keys.first[key]];
    feasible[key]   = check_lds_capacity(hardware, c.mt, problem.a_dtype, problem.b_dtype) &&
                    is_kernel_feasible(problem, c, keys.nt_a, keys.nt_b);
  }

  const bool have_whitelist = !cell.signatures.empty();
  std::vector<unsigned char> in_tier1(nk, 0);
  std::vector<std::size_t> tier1;
  std::size_t tier1_configs = 0;
  if (have_whitelist)
    for (std::size_t key = 0; key < nk; ++key)
      if (feasible[key] && (cache != nullptr ? cache->in_whitelist[key] != 0
                                             : whitelisted(cell, configs[keys.first[key]]))) {
        tier1.push_back(key);
        in_tier1[key] = 1;
        tier1_configs += size_of(key);
      }
  bool tier1_is_whitelist = have_whitelist;
  if (tier1.empty()) {
    tier1_is_whitelist = false;
    for (std::size_t key = 0; key < nk; ++key)
      if (feasible[key]) {
        tier1.push_back(key);
        tier1_configs += size_of(key);
      }
  }
  if (tier1.empty()) return all_unscored(n);

  std::vector<std::size_t> tier2;
  if (tier1_is_whitelist && min_scored > tier1_configs)
    for (std::size_t key = 0; key < nk; ++key)
      if (feasible[key] && !in_tier1[key]) tier2.push_back(key);

  const HwView hw     = hw_view(model, hardware);
  const std::size_t e = cell.embed_dim;
  std::vector<float> q_feat(cat.query_dim), i_feat(cat.item_dim), x_feat(cat.inter_dim);
  std::vector<float> norm(std::max(cat.query_dim, std::max(cat.item_dim, cat.inter_dim)));
  std::vector<float> q_emb(e), i_emb(e);
  std::vector<float> h0(cell.hidden_dim), h2(std::max(cell.hidden_dim, cell.inter_hidden));
  cat.query(problem, hw, q_feat.data());
  query_embedding(
      k, cell, w, q_feat.data(), cat.query_dim, norm.data(), h0.data(), h2.data(), q_emb.data());

  std::vector<float> score(nk, 0.0f);
  std::vector<std::size_t> order, ranked, run;
  order.reserve(tier1_configs);
  const auto rank_tier = [&](const std::vector<std::size_t>& tier) {
    ranked.clear();
    for (std::size_t key : tier) {
      const Config& c = configs[keys.first[key]];
      const float* ie = nullptr;
      if (cache != nullptr) {
        ie = &cache->item_emb[key * e];
      } else {
        cat.item(c, i_feat.data());
        item_embedding(
            k, cell, w, i_feat.data(), cat.item_dim, norm.data(), h0.data(), i_emb.data());
        ie = i_emb.data();
      }
      const float emb_score = k.dot(q_emb.data(), ie, e) / w.temperature;
      cat.inter(model, problem, c, hw, x_feat.data());
      const float s =
          emb_score +
          interaction_score(k, cell, w, x_feat.data(), cat.inter_dim, norm.data(), h2.data());
      if (!finite(s)) continue;
      score[key] = s;
      ranked.push_back(key);
    }
    std::sort(ranked.begin(), ranked.end(), [&](std::size_t a, std::size_t b) {
      return score[a] > score[b] || (!(score[b] > score[a]) && a < b);
    });
    // Configs whose keys score equally interleave in input (or prior) order.
    for (std::size_t i = 0; i < ranked.size();) {
      std::size_t j = i + 1;
      while (j < ranked.size() && score[ranked[j]] == score[ranked[i]]) ++j;
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
  };
  rank_tier(tier1);
  if (!tier2.empty()) rank_tier(tier2);

  if (env_knobs().pick_log && !order.empty())
    log_pick(
        problem, cell, configs[order.front()], score[keys.key_of[order.front()]], order.front(), n);

  std::vector<Result> result;
  result.reserve(n);
  std::vector<unsigned char> used(n, 0);
  for (std::size_t ci : order) {
    used[ci] = 1;
    result.push_back(Result{ci, static_cast<double>(score[keys.key_of[ci]]), true});
  }
  for (std::size_t j = 0; j < n; ++j)
    if (!used[j]) result.push_back(Result{j, 0.0, false});
  return result;
}

// `keys` null: built from `configs`.
std::vector<Result> rank_pool(const Model& model,
                              const Problem& problem,
                              const Hardware& hardware,
                              const ExecutionContext& context,
                              const std::vector<Config>& configs,
                              const PoolKeys* keys,
                              std::size_t min_scored,
                              TieBreak tie_break,
                              const std::vector<double>* prior,
                              const CellCacheSource* source) {
  const std::size_t n = configs.size();
  if (prior != nullptr && prior->size() != n)
    throw std::invalid_argument("tilewright: the prior holds " + std::to_string(prior->size()) +
                                " values for " + std::to_string(n) + " configs");
  if (n == 0) return {};
  if (!supports(model, context, hardware) || !valid_hardware(hardware)) return all_unscored(n);
  const int cell = route_cell(model, problem);
  if (cell < 0) return all_unscored(n);
  const std::vector<double>* order_prior = tie_break == TieBreak::Prior ? prior : nullptr;
  const std::size_t ci                   = static_cast<std::size_t>(cell);
  if (keys != nullptr)
    return rank_in_cell(
        model, ci, problem, hardware, configs, *keys, min_scored, order_prior, source);
  const PoolKeys built = pool_keys(*model.catalog, configs);
  return rank_in_cell(
      model, ci, problem, hardware, configs, built, min_scored, order_prior, source);
}

}  // namespace

int route_cell(const Model& model, const Problem& problem) noexcept {
  const long long forced = env_knobs().force_cell;
  if (forced >= 0 && static_cast<unsigned long long>(forced) < model.cells.size())
    return static_cast<int>(forced);
  return resolve_cell(model, problem);
}

PoolKeys pool_keys(const FeatureCatalog& catalog, const std::vector<Config>& configs) {
  const std::size_t n    = configs.size();
  const std::size_t size = catalog.view_key_size;
  PoolKeys p;
  p.key_of.resize(n);
  p.first.reserve(n);
  std::vector<std::uint64_t> key_hash;
  key_hash.reserve(n);
  std::uint64_t buffers[2][kMaxViewKeySize], other[kMaxViewKeySize];
  std::size_t slots = 16;
  while (slots < 2 * n) slots <<= 1;
  const std::size_t empty = static_cast<std::size_t>(-1);
  std::vector<std::size_t> table(slots, empty);
  for (std::size_t ci = 0; ci < n; ++ci) {
    const Config& c = configs[ci];
    p.nt_a |= c.cache_hints_a == 4;
    p.nt_b |= c.cache_hints_b == 4;
    std::uint64_t* value = buffers[ci & 1];
    catalog.view_key(c, value);
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
        key_hash.push_back(h);
        break;
      }
      if (key_hash[key] != h) continue;
      catalog.view_key(configs[p.first[key]], other);
      if (std::equal(value, value + size, other)) {
        p.key_of[ci] = key;
        break;
      }
    }
  }
  const std::size_t nk = p.first.size();
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
                              const std::vector<double>* prior) {
  return rank_pool(
      model, problem, hardware, context, configs, &keys, min_scored, tie_break, prior, nullptr);
}

}  // namespace detail

struct CandidateSet::Impl final : detail::CellCacheSource {
  struct Slot {
    std::once_flag once;
    std::unique_ptr<const detail::CellCache> cache;
  };

  ModelPtr model;
  std::vector<Config> configs;
  PoolOptions options;
  detail::PoolKeys keys;
  std::vector<float> item_features;  // keys x item_dim
  std::unique_ptr<Slot[]> slots;     // one per model cell

  const detail::CellCache& cache(std::size_t cell_index,
                                 const detail::CellWeights& w) const override {
    Slot& slot = slots[cell_index];
    std::call_once(slot.once, [&] {
      const detail::Cell& cell = model->cells[cell_index];
      const detail::Kernels& k = detail::kernels();
      const std::size_t nk     = keys.first.size();
      const std::size_t i_dim  = model->catalog->item_dim;
      auto built               = std::make_unique<detail::CellCache>();
      built->item_emb.resize(nk * cell.embed_dim);
      built->in_whitelist.resize(nk);
      std::vector<float> hidden(cell.hidden_dim), norm(i_dim);
      for (std::size_t key = 0; key < nk; ++key) {
        detail::item_embedding(k,
                               cell,
                               w,
                               &item_features[key * i_dim],
                               i_dim,
                               norm.data(),
                               hidden.data(),
                               &built->item_emb[key * cell.embed_dim]);
        built->in_whitelist[key] = detail::whitelisted(cell, configs[keys.first[key]]) ? 1 : 0;
      }
      slot.cache = std::move(built);
    });
    return *slot.cache;
  }
};

CandidateSet::CandidateSet(ModelPtr model, std::vector<Config> configs, PoolOptions options) {
  if (!model) throw std::invalid_argument("tilewright::CandidateSet requires a model");
  detail::check_attributes(configs);
  const detail::FeatureCatalog& cat = *model->catalog;
  impl_                             = std::make_unique<Impl>();
  impl_->options                    = options;
  impl_->keys                       = detail::pool_keys(cat, configs);
  impl_->configs                    = std::move(configs);
  const std::size_t nk              = impl_->keys.first.size();
  impl_->item_features.resize(nk * cat.item_dim);
  for (std::size_t key = 0; key < nk; ++key)
    cat.item(impl_->configs[impl_->keys.first[key]], &impl_->item_features[key * cat.item_dim]);
  impl_->slots = std::make_unique<Impl::Slot[]>(model->cells.size());
  impl_->model = std::move(model);
}

CandidateSet::~CandidateSet() = default;

const Model& CandidateSet::model() const noexcept { return *impl_->model; }

const std::vector<Config>& CandidateSet::configs() const noexcept { return impl_->configs; }

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
  return detail::rank_pool(*impl_->model,
                           problem,
                           hardware,
                           context,
                           impl_->configs,
                           &impl_->keys,
                           min_scored,
                           impl_->options.tie_break,
                           prior,
                           impl_.get());
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
  return detail::rank_pool(
      model, problem, hardware, context, configs, nullptr, min_scored, tie_break, prior, nullptr);
}

bool supports(const Model& model,
              const ExecutionContext& context,
              const Hardware& hardware) noexcept {
  return model.catalog->supports(context, hardware);
}

std::vector<std::string> attribute_names(const Model& model) {
  const detail::FeatureCatalog& catalog = *model.catalog;
  return {catalog.attribute_names, catalog.attribute_names + catalog.attribute_count};
}

const char* feature_catalog_hash() noexcept { return detail::v2_catalog().hash; }

ModelInfo describe(const Model& model) {
  ModelInfo info;
  info.arch                 = model.arch;
  info.feature_catalog_hash = model.feature_hash;
  info.weight_type          = model.weight_type;
  info.n_cells              = model.cells.size();
  info.n_splits             = model.splits.size();
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
  Features f;
  if (!detail::valid_hardware(hardware)) return f;
  const detail::FeatureCatalog& cat = *model.catalog;
  const detail::HwView hw           = detail::hw_view(model, hardware);
  f.query.resize(cat.query_dim);
  f.item.resize(cat.item_dim);
  f.interaction.resize(cat.inter_dim);
  cat.query(problem, hw, f.query.data());
  cat.item(config, f.item.data());
  cat.inter(model, problem, config, hw, f.interaction.data());
  return f;
}

}  // namespace tilewright
