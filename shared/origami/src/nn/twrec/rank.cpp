// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/twrec/rank.hpp"

#include "origami/nn/features/gemm_tilewright.hpp"
#include "origami/nn/filter.hpp"
#include "origami/gemm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami::nn::twrec {
namespace {

using detail::CellModel;
using detail::LoadedModel;
using detail::SplitRule;

float dot(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t j = 0; j < n; ++j) sum += a[j] * b[j];
  return sum;
}

void linear_relu(const float* W,
                 const float* b,
                 const float* x,
                 std::size_t m,
                 std::size_t k,
                 float* out) {
  for (std::size_t i = 0; i < m; ++i) {
    float acc = b[i] + dot(W + i * k, x, k);
    out[i]    = acc > 0.0f ? acc : 0.0f;
  }
}

void linear(const float* W, const float* b, const float* x, std::size_t m, std::size_t k, float* out) {
  for (std::size_t i = 0; i < m; ++i) out[i] = b[i] + dot(W + i * k, x, k);
}

void compute_qe(const CellModel& cm,
                const float* q_feat,
                std::size_t q_dim,
                std::vector<float>& scratch,
                std::vector<float>& q_emb_out) {
  std::vector<float> q_norm(q_dim);
  for (std::size_t j = 0; j < q_dim; ++j) {
    const float s = (cm.q_std[j] < 1e-6f) ? 1.0f : cm.q_std[j];
    q_norm[j]     = (q_feat[j] - cm.q_mean[j]) / s;
  }
  std::vector<float> h0(cm.hidden_dim), h2(cm.hidden_dim);
  scratch.assign(cm.hidden_dim, 0.0f);
  linear_relu(cm.q_w0.data(), cm.q_b0.data(), q_norm.data(), cm.hidden_dim, q_dim, h0.data());
  linear_relu(cm.q_w2.data(), cm.q_b2.data(), h0.data(), cm.hidden_dim, cm.hidden_dim, h2.data());
  q_emb_out.assign(cm.embed_dim, 0.0f);
  linear(cm.q_w4.data(), cm.q_b4.data(), h2.data(), cm.embed_dim, cm.hidden_dim, q_emb_out.data());
}

void compute_ie(const CellModel& cm,
                const float* i_feat,
                std::size_t i_dim,
                std::vector<float>& scratch,
                std::vector<float>& i_emb_out) {
  std::vector<float> i_norm(i_dim);
  for (std::size_t j = 0; j < i_dim; ++j) {
    const float s = (cm.i_std[j] < 1e-6f) ? 1.0f : cm.i_std[j];
    i_norm[j]     = (i_feat[j] - cm.i_mean[j]) / s;
  }
  scratch.assign(cm.hidden_dim, 0.0f);
  linear_relu(cm.i_w0.data(), cm.i_b0.data(), i_norm.data(), cm.hidden_dim, i_dim, scratch.data());
  i_emb_out.assign(cm.embed_dim, 0.0f);
  linear(cm.i_w2.data(), cm.i_b2.data(), scratch.data(), cm.embed_dim, cm.hidden_dim, i_emb_out.data());
}

float score_from_embeds(const CellModel& cm,
                        const float* q_emb,
                        const float* i_emb,
                        std::size_t embed_dim) {
  const float temp = std::max(std::fabs(cm.temperature), 0.1f);
  return dot(q_emb, i_emb, embed_dim) / temp;
}

float compute_inter_score(const CellModel& cm,
                          const float* x_feat,
                          std::size_t x_dim,
                          std::vector<float>& scratch) {
  std::vector<float> x_norm(x_dim);
  for (std::size_t j = 0; j < x_dim; ++j) {
    const float s = (cm.x_std[j] < 1e-6f) ? 1.0f : cm.x_std[j];
    x_norm[j]     = (x_feat[j] - cm.x_mean[j]) / s;
  }
  scratch.assign(cm.inter_hidden, 0.0f);
  linear_relu(cm.x_w0.data(), cm.x_b0.data(), x_norm.data(), cm.inter_hidden, x_dim, scratch.data());
  return cm.x_b2[0] + dot(cm.x_w2.data(), scratch.data(), cm.inter_hidden);
}

std::string m_tier(std::size_t v) {
  if (v <= 32) return "Tiny";
  if (v <= 128) return "Small";
  if (v <= 512) return "Mid";
  return "Large";
}

std::string k_tier(std::size_t v) {
  if (v <= 32) return "TinyK";
  if (v <= 512) return "MidK";
  return "LargeK";
}

std::string b_tier(std::size_t v) { return v == 1 ? "Bnone" : "Bany"; }

std::string base_cell_label(const problem_t& p) {
  return m_tier(p.size.m) + "|" + m_tier(p.size.n) + "|" + k_tier(p.size.k) + "|" +
         b_tier(p.batch);
}

long long axis_value(char axis, const problem_t& p) {
  switch (axis) {
    case 'M': return static_cast<long long>(p.size.m);
    case 'N': return static_cast<long long>(p.size.n);
    case 'K': return static_cast<long long>(p.size.k);
    case 'B': return static_cast<long long>(p.batch);
    default: return 0;
  }
}

std::string assign_subcell(const LoadedModel& mdl, const problem_t& p) {
  std::string label = base_cell_label(p);
  std::unordered_map<std::string, bool> seen;
  for (;;) {
    auto it = mdl.splits.find(label);
    if (it == mdl.splits.end()) break;
    if (seen[label]) break;
    seen[label]        = true;
    const SplitRule& r = it->second;
    const long long v  = axis_value(r.axis, p);
    label              = (v <= r.threshold) ? r.lo_label : r.hi_label;
  }
  return label;
}

int resolve_model_cell_index(const LoadedModel& mdl, const problem_t& p) {
  std::string cur = assign_subcell(mdl, p);
  for (;;) {
    auto it = mdl.cell_index.find(cur);
    if (it != mdl.cell_index.end()) return static_cast<int>(it->second);
    const std::size_t j = cur.rfind('#');
    if (j == std::string::npos) return -1;
    cur = cur.substr(0, j);
  }
}

int resolve_cell_index(const LoadedModel& model,
                       const problem_t& problem,
                       const inference_options_t& options) {
  if (options.force_cell >= 0) {
    const std::size_t idx = static_cast<std::size_t>(options.force_cell);
    if (idx < model.cells.size()) return options.force_cell;
    return -1;
  }
  if (const char* env = std::getenv("ORIGAMI_TW_CELL")) {
    if (env[0] != '\0' && std::strcmp(env, "0") != 0) {
      const int forced = std::atoi(env);
      if (forced >= 0 && static_cast<std::size_t>(forced) < model.cells.size()) return forced;
    }
  }
  return resolve_model_cell_index(model, problem);
}

std::array<int, 8> sig_of(const config_t& c) {
  return {static_cast<int>(c.mt.m),
          static_cast<int>(c.mt.n),
          static_cast<int>(c.mt.k),
          static_cast<int>(c.mi.m),
          static_cast<int>(c.mi.n),
          static_cast<int>(c.mi.k),
          c.cache_hints_a,
          c.cache_hints_b};
}

std::vector<rank_entry_t> fallback_all(const std::vector<config_t>& configs) {
  std::vector<rank_entry_t> result;
  result.reserve(configs.size());
  for (std::size_t j = 0; j < configs.size(); ++j) {
    result.push_back(rank_entry_t{j, 0.0, false});
  }
  return result;
}

std::vector<std::uint32_t> apply_smart_k_filter(const std::vector<config_t>& configs,
                                                const std::vector<std::size_t>& feasible,
                                                bool use_sig_filter,
                                                bool have_sk,
                                                const CellModel& cm) {
  auto sig_in_set = [&](const std::array<int, 8>& s) -> bool {
    for (const auto& w : cm.smart_k_signatures)
      if (w == s) return true;
    return false;
  };

  std::vector<std::uint32_t> out;
  out.reserve(feasible.size());
  for (const std::size_t idx : feasible) {
    if (use_sig_filter && have_sk && !sig_in_set(sig_of(configs[idx]))) continue;
    out.push_back(static_cast<std::uint32_t>(idx));
  }
  return out;
}

}  // namespace

std::vector<rank_entry_t> rank_configs(const LoadedModel& model,
                                       const problem_t& problem,
                                       const hardware_t& hardware,
                                       const std::vector<config_t>& configs,
                                       const inference_options_t& options) {
  if (configs.empty()) return fallback_all(configs);

  const int cell_idx = resolve_cell_index(model, problem, options);
  if (cell_idx < 0) return fallback_all(configs);

  const CellModel& cm = model.cells[static_cast<std::size_t>(cell_idx)];
  const std::size_t q_dim = model.q_dim;
  const std::size_t i_dim = model.i_dim;
  const std::size_t x_dim = model.x_dim;

  const bool have_sk =
      options.use_smart_k_whitelist && !cm.smart_k_signatures.empty();

  const filter::filter_result_t filtered =
      filter::filter_configs(problem, hardware, configs);

  std::vector<std::uint32_t> cand =
      apply_smart_k_filter(configs, filtered.feasible_indices, true, have_sk, cm);
  bool tier1_is_whitelist = have_sk;
  if (cand.empty() && have_sk) {
    cand = apply_smart_k_filter(configs, filtered.feasible_indices, false, have_sk, cm);
    tier1_is_whitelist = false;
  }
  if (cand.empty()) return fallback_all(configs);

  std::vector<std::uint32_t> cand2;
  if (tier1_is_whitelist && options.min_scored > cand.size()) {
    std::vector<char> in_tier1(configs.size(), 0);
    for (std::uint32_t ci : cand) in_tier1[ci] = 1;
    for (std::size_t idx : filtered.feasible_indices) {
      if (!in_tier1[idx]) cand2.push_back(static_cast<std::uint32_t>(idx));
    }
  }

  std::array<float, features::gemm_tilewright::query_dim> q_feat{};
  std::array<float, features::gemm_tilewright::item_dim> item_feat{};
  std::array<float, features::gemm_tilewright::interaction_dim> x_feat{};

  features::gemm_tilewright::build_query(problem, hardware, q_feat.data());
  std::vector<float> scratch;
  std::vector<float> q_emb;
  compute_qe(cm, q_feat.data(), q_dim, scratch, q_emb);

  std::vector<float> i_emb;
  auto score_set = [&](const std::vector<std::uint32_t>& cset,
                       std::vector<std::pair<std::uint32_t, float>>& out) {
    out.reserve(cset.size());
    for (std::uint32_t ci : cset) {
      const config_t& cc = configs[ci];
      features::gemm_tilewright::build_item(cc, item_feat.data());
      compute_ie(cm, item_feat.data(), i_dim, scratch, i_emb);
      const float emb_score = score_from_embeds(cm, q_emb.data(), i_emb.data(), cm.embed_dim);
      features::gemm_tilewright::build_interaction(problem, cc, hardware, x_feat.data());
      const float inter_score = compute_inter_score(cm, x_feat.data(), x_dim, scratch);
      out.emplace_back(ci, emb_score + inter_score);
    }
    std::stable_sort(out.begin(),
                     out.end(),
                     [](const std::pair<std::uint32_t, float>& a,
                        const std::pair<std::uint32_t, float>& b) { return a.second > b.second; });
  };

  std::vector<std::pair<std::uint32_t, float>> scored;
  score_set(cand, scored);
  if (!cand2.empty()) {
    std::vector<std::pair<std::uint32_t, float>> scored2;
    score_set(cand2, scored2);
    scored.insert(scored.end(), scored2.begin(), scored2.end());
  }

  if (std::getenv("ORIGAMI_TW_LOG") != nullptr && !scored.empty()) {
    const config_t& top = configs[scored[0].first];
    std::fprintf(stderr,
                 "[TW_PICK] m=%zu n=%zu k=%zu b=%zu tA=%c tB=%c leaf=%s "
                 "top1_sig=(mt_m=%zu,mt_n=%zu,mt_k=%zu,mi_m=%zu,mi_n=%zu,"
                 "mi_k=%zu,cha=%d,chb=%d) top1_score=%f n_configs=%zu\n",
                 problem.size.m,
                 problem.size.n,
                 problem.size.k,
                 problem.batch,
                 (problem.a_transpose == transpose_t::T ? 'T' : 'N'),
                 (problem.b_transpose == transpose_t::T ? 'T' : 'N'),
                 cm.label.c_str(),
                 top.mt.m,
                 top.mt.n,
                 top.mt.k,
                 top.mi.m,
                 top.mi.n,
                 top.mi.k,
                 top.cache_hints_a,
                 top.cache_hints_b,
                 static_cast<double>(scored[0].second),
                 configs.size());
    std::fflush(stderr);
  }

  std::vector<rank_entry_t> result;
  result.reserve(configs.size());
  std::vector<char> used(configs.size(), 0);
  for (const auto& s : scored) {
    used[s.first] = 1;
    result.push_back(rank_entry_t{s.first, static_cast<double>(s.second), true});
  }
  for (std::size_t j = 0; j < configs.size(); ++j) {
    if (!used[j]) result.push_back(rank_entry_t{j, 0.0, false});
  }
  return result;
}

}  // namespace origami::nn::twrec
