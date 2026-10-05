// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "features.hpp"
#include "kernels.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
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

void whiten(const float* f,
            const std::vector<float>& mean,
            const std::vector<float>& scale,
            std::size_t n,
            float* out) {
  for (std::size_t j = 0; j < n; ++j) out[j] = (f[j] - mean[j]) / scale[j];
}

void query_embedding(const Kernels& k,
                     const Cell& cell,
                     const CellWeights& w,
                     const float* q_feat,
                     float* h0,
                     float* h2,
                     float* q_emb) {
  float norm[kQueryDim];
  whiten(q_feat, w.q_mean, w.q_scale, kQueryDim, norm);
  k.linear_relu(w.q_w0.data(), w.q_b0.data(), norm, cell.hidden_dim, kQueryDim, h0);
  k.linear_relu(w.q_w2.data(), w.q_b2.data(), h0, cell.hidden_dim, cell.hidden_dim, h2);
  k.linear(w.q_w4.data(), w.q_b4.data(), h2, cell.embed_dim, cell.hidden_dim, q_emb);
}

void item_embedding(const Kernels& k,
                    const Cell& cell,
                    const CellWeights& w,
                    const float* i_feat,
                    float* hidden,
                    float* i_emb) {
  float norm[kItemDim];
  whiten(i_feat, w.i_mean, w.i_scale, kItemDim, norm);
  k.linear_relu(w.i_w0.data(), w.i_b0.data(), norm, cell.hidden_dim, kItemDim, hidden);
  k.linear(w.i_w2.data(), w.i_b2.data(), hidden, cell.embed_dim, cell.hidden_dim, i_emb);
}

float interaction_score(const Kernels& k,
                        const Cell& cell,
                        const CellWeights& w,
                        const float* x_feat,
                        float* hidden) {
  float norm[kInterDim];
  whiten(x_feat, w.x_mean, w.x_scale, kInterDim, norm);
  k.linear_relu(w.x_w0.data(), w.x_b0.data(), norm, cell.inter_hidden, kInterDim, hidden);
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

void log_pick(const Problem& p, const Cell& cell, const Config& top, float score, std::size_t n) {
  std::fprintf(stderr,
               "[TILEWRIGHT_PICK] m=%zu n=%zu k=%zu b=%zu tA=%c tB=%c leaf=%s "
               "top1_sig=(mt_m=%zu,mt_n=%zu,mt_k=%zu,mi_m=%zu,mi_n=%zu,"
               "mi_k=%zu,cha=%d,chb=%d) top1_score=%f n_configs=%zu\n",
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
               n);
  std::fflush(stderr);
}

}  // namespace

int route_cell(const Model& model, const Problem& problem) noexcept {
  const long long forced = env_knobs().force_cell;
  if (forced >= 0 && static_cast<unsigned long long>(forced) < model.cells.size())
    return static_cast<int>(forced);
  return resolve_cell(model, problem);
}

struct PoolFlags {
  bool nt_a = false;
  bool nt_b = false;
};

PoolFlags pool_flags(const std::vector<Config>& configs) {
  PoolFlags f;
  for (const Config& c : configs) {
    f.nt_a |= c.cache_hints_a == 4;
    f.nt_b |= c.cache_hints_b == 4;
  }
  return f;
}

// Pool data a CandidateSet keeps for one cell.
struct CellCache {
  std::vector<float> item_emb;  // configs x embed_dim
  std::vector<unsigned char> in_whitelist;
};

class CellCacheSource {
 public:
  virtual ~CellCacheSource()                                                   = default;
  virtual const CellCache& cache(std::size_t cell, const CellWeights& w) const = 0;
};

std::vector<Result> rank_impl(const Model& model,
                              const Problem& problem,
                              const Hardware& hardware,
                              const std::vector<Config>& configs,
                              std::size_t min_scored,
                              PoolFlags pool,
                              const CellCacheSource* source) {
  const std::size_t n = configs.size();
  if (n == 0) return {};
  if (!valid_hardware(hardware)) return all_unscored(n);
  const int cell_index = route_cell(model, problem);
  if (cell_index < 0) return all_unscored(n);

  const std::size_t ci_cell = static_cast<std::size_t>(cell_index);
  const Cell& cell          = model.cells[ci_cell];
  const CellWeights& w      = model.weights(ci_cell);
  const CellCache* cache    = source != nullptr ? &source->cache(ci_cell, w) : nullptr;
  const Kernels& k          = kernels();

  std::vector<unsigned char> feasible(n, 0);
  for (std::size_t ci = 0; ci < n; ++ci) {
    const Config& c = configs[ci];
    feasible[ci]    = check_lds_capacity(hardware, c.mt, problem.a_dtype, problem.b_dtype) &&
                   is_kernel_feasible(problem, c, pool.nt_a, pool.nt_b);
  }

  const bool have_whitelist = !cell.signatures.empty();
  std::vector<std::size_t> tier1;
  if (have_whitelist)
    for (std::size_t ci = 0; ci < n; ++ci)
      if (feasible[ci] &&
          (cache != nullptr ? cache->in_whitelist[ci] != 0 : whitelisted(cell, configs[ci])))
        tier1.push_back(ci);
  bool tier1_is_whitelist = have_whitelist;
  if (tier1.empty()) {
    tier1_is_whitelist = false;
    for (std::size_t ci = 0; ci < n; ++ci)
      if (feasible[ci]) tier1.push_back(ci);
  }
  if (tier1.empty()) return all_unscored(n);

  std::vector<std::size_t> tier2;
  if (tier1_is_whitelist && min_scored > tier1.size()) {
    std::vector<unsigned char> in_tier1(n, 0);
    for (std::size_t ci : tier1) in_tier1[ci] = 1;
    for (std::size_t ci = 0; ci < n; ++ci)
      if (feasible[ci] && !in_tier1[ci]) tier2.push_back(ci);
  }

  const HwView hw     = hw_view(model, hardware);
  const std::size_t e = cell.embed_dim;
  std::vector<float> q_emb(e), i_emb(e);
  std::vector<float> h0(cell.hidden_dim), h2(std::max(cell.hidden_dim, cell.inter_hidden));
  float q_feat[kQueryDim];
  build_query_features(problem, hw, q_feat);
  query_embedding(k, cell, w, q_feat, h0.data(), h2.data(), q_emb.data());

  using Scored          = std::pair<std::size_t, float>;
  const auto score_tier = [&](const std::vector<std::size_t>& tier, std::vector<Scored>* out) {
    out->reserve(tier.size());
    for (std::size_t ci : tier) {
      const Config& c = configs[ci];
      const float* ie = nullptr;
      if (cache != nullptr) {
        ie = &cache->item_emb[ci * e];
      } else {
        float i_feat[kItemDim];
        build_item_features(c, i_feat);
        item_embedding(k, cell, w, i_feat, h0.data(), i_emb.data());
        ie = i_emb.data();
      }
      const float emb_score = k.dot(q_emb.data(), ie, e) / w.temperature;
      float x_feat[kInterDim];
      build_inter_features(model, problem, c, hw, x_feat);
      const float score = emb_score + interaction_score(k, cell, w, x_feat, h2.data());
      if (finite(score)) out->emplace_back(ci, score);
    }
    std::stable_sort(out->begin(), out->end(), [](const Scored& a, const Scored& b) {
      return a.second > b.second;
    });
  };

  std::vector<Scored> scored;
  score_tier(tier1, &scored);
  if (!tier2.empty()) {
    std::vector<Scored> scored2;
    score_tier(tier2, &scored2);
    scored.insert(scored.end(), scored2.begin(), scored2.end());
  }

  if (env_knobs().pick_log && !scored.empty())
    log_pick(problem, cell, configs[scored.front().first], scored.front().second, n);

  std::vector<Result> result;
  result.reserve(n);
  std::vector<unsigned char> used(n, 0);
  for (const Scored& s : scored) {
    used[s.first] = 1;
    result.push_back(Result{s.first, static_cast<double>(s.second), true});
  }
  for (std::size_t j = 0; j < n; ++j)
    if (!used[j]) result.push_back(Result{j, 0.0, false});
  return result;
}

}  // namespace detail

struct CandidateSet::Impl final : detail::CellCacheSource {
  struct Slot {
    std::once_flag once;
    std::unique_ptr<const detail::CellCache> cache;
  };

  ModelPtr model;
  std::vector<Config> configs;
  detail::PoolFlags pool;
  std::vector<float> item_features;  // configs x kItemDim
  std::unique_ptr<Slot[]> slots;     // one per model cell

  const detail::CellCache& cache(std::size_t cell_index,
                                 const detail::CellWeights& w) const override {
    Slot& slot = slots[cell_index];
    std::call_once(slot.once, [&] {
      const detail::Cell& cell = model->cells[cell_index];
      const detail::Kernels& k = detail::kernels();
      const std::size_t n      = configs.size();
      auto built               = std::make_unique<detail::CellCache>();
      built->item_emb.resize(n * cell.embed_dim);
      built->in_whitelist.resize(n);
      std::vector<float> hidden(cell.hidden_dim);
      for (std::size_t ci = 0; ci < n; ++ci) {
        detail::item_embedding(k,
                               cell,
                               w,
                               &item_features[ci * detail::kItemDim],
                               hidden.data(),
                               &built->item_emb[ci * cell.embed_dim]);
        built->in_whitelist[ci] = detail::whitelisted(cell, configs[ci]) ? 1 : 0;
      }
      slot.cache = std::move(built);
    });
    return *slot.cache;
  }
};

CandidateSet::CandidateSet(ModelPtr model, std::vector<Config> configs) {
  if (!model) throw std::invalid_argument("tilewright::CandidateSet requires a model");
  impl_          = std::make_unique<Impl>();
  impl_->pool    = detail::pool_flags(configs);
  impl_->configs = std::move(configs);
  impl_->item_features.resize(impl_->configs.size() * detail::kItemDim);
  for (std::size_t ci = 0; ci < impl_->configs.size(); ++ci)
    detail::build_item_features(impl_->configs[ci], &impl_->item_features[ci * detail::kItemDim]);
  impl_->slots = std::make_unique<Impl::Slot[]>(model->cells.size());
  impl_->model = std::move(model);
}

CandidateSet::~CandidateSet() = default;

const Model& CandidateSet::model() const noexcept { return *impl_->model; }

const std::vector<Config>& CandidateSet::configs() const noexcept { return impl_->configs; }

std::vector<Result> CandidateSet::rank(const Problem& problem,
                                       const Hardware& hardware,
                                       std::size_t min_scored) const {
  return detail::rank_impl(
      *impl_->model, problem, hardware, impl_->configs, min_scored, impl_->pool, impl_.get());
}

std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored) {
  return detail::rank_impl(
      model, problem, hardware, configs, min_scored, detail::pool_flags(configs), nullptr);
}

const char* feature_catalog_hash() noexcept { return detail::kFeatureCatalogHash; }

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
  const detail::HwView hw = detail::hw_view(model, hardware);
  f.query.resize(detail::kQueryDim);
  f.item.resize(detail::kItemDim);
  f.interaction.resize(detail::kInterDim);
  detail::build_query_features(problem, hw, f.query.data());
  detail::build_item_features(config, f.item.data());
  detail::build_inter_features(model, problem, config, hw, f.interaction.data());
  return f;
}

}  // namespace tilewright
