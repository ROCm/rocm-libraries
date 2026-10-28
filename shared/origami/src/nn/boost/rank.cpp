// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/boost/rank.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "origami/nn/features/gemm_boost.hpp"
#include "origami/origami.hpp"

namespace origami::nn::boost {

namespace features = nn::features::gemm_boost;

status_t booster_t::check_problem(const problem_t&  problem,
                                  const hardware_t& hardware,
                                  std::string*      detail) const {
  // The model's dtype class: bf16 = A and B BFloat16; tf32 = A/B/C/D Float with XFloat32
  // math (strict f32 is a different problem and never uses a tf32 model); fp8 = F8BS, A and B
  // Float8 into Float8 math with BFloat16 C and D and no MX blocks. Any other class is refused.
  const std::string& dtype = model_->head_string("dtype");
  const char*        not_in_class = nullptr;
  if (dtype == "bf16") {
    if (!(problem.a_dtype == data_type_t::BFloat16 && problem.b_dtype == data_type_t::BFloat16))
      not_in_class = "problem is not bf16";
  } else if (dtype == "tf32") {
    if (!(problem.a_dtype == data_type_t::Float && problem.b_dtype == data_type_t::Float &&
          problem.c_dtype == data_type_t::Float && problem.d_dtype == data_type_t::Float &&
          problem.mi_dtype == data_type_t::XFloat32))
      not_in_class = "problem is not tf32 (A/B/C/D f32 with xf32 math)";
  } else if (dtype == "fp8") {
    if (!(problem.a_dtype == data_type_t::Float8 && problem.b_dtype == data_type_t::Float8 &&
          problem.c_dtype == data_type_t::BFloat16 && problem.d_dtype == data_type_t::BFloat16 &&
          problem.mi_dtype == data_type_t::Float8 && problem.a_mx_block_size == 0 &&
          problem.b_mx_block_size == 0))
      not_in_class = "problem is not fp8 (F8BS: A/B Float8 with Float8 math, C/D BFloat16, no MX)";
  } else {
    not_in_class = "unknown dtype class";
  }
  if (not_in_class) {
    if (detail) *detail = not_in_class;
    return status_t::out_of_range;
  }
  if (!d_dtypes_.empty() &&
      std::find(d_dtypes_.begin(), d_dtypes_.end(), problem.d_dtype) == d_dtypes_.end()) {
    if (detail) *detail = "d dtype not trained";
    return status_t::out_of_range;
  }
  // The loader admits exactly M, N, K and batch. A CU budget is not a training range: the
  // features carry it (f_cu_util, sk_*).
  for (const auto& rg : model_->ranges()) {
    double v;
    if (rg.name == "M") v = static_cast<double>(problem.size.m);
    else if (rg.name == "N") v = static_cast<double>(problem.size.n);
    else if (rg.name == "K") v = static_cast<double>(problem.size.k);
    else v = static_cast<double>(problem.batch);
    if (v < rg.min || v > rg.max) {
      if (detail) {
        std::ostringstream os;
        os << rg.name << "=" << v << " outside [" << rg.min << ", " << rg.max << "]";
        *detail = os.str();
      }
      return status_t::out_of_range;
    }
  }
  return status_t::ok;
}

bool booster_t::model_features(const problem_t& problem, const hardware_t& hardware,
                               const candidate_t& candidate, double* row, std::string* why) const {
  thread_local std::vector<double>       all;
  thread_local std::vector<std::uint8_t> available;
  all.resize(features::slot_names().size());
  available.resize(all.size());
  if (!features::extract(problem, hardware, candidate, all.data(), why, available.data()))
    return false;
  for (std::size_t j = 0; j < slot_.size(); ++j) {
    if (!available[slot_[j]]) {
      if (why) *why = "caller did not provide the kernel field behind " + model_->features()[j].name;
      return false;
    }
    row[j] = all[slot_[j]];
  }
  return true;
}

boost_result_t booster_t::rank(const problem_t& problem, const hardware_t& hardware,
                               const candidate_t* candidates, std::size_t num_candidates) const {
  boost_result_t res;
  res.order.resize(num_candidates);
  std::iota(res.order.begin(), res.order.end(), std::size_t{0});
  const std::size_t n = std::min(num_candidates, model_->top_n());
  if (n < 2) {
    res.status = status_t::too_few_candidates;
    return res;
  }
  res.status = check_problem(problem, hardware, &res.detail);
  if (res.status != status_t::ok) return res;

  const std::size_t                nf = slot_.size();
  thread_local std::vector<double> rows;
  rows.resize(n * nf);
  for (std::size_t i = 0; i < n; ++i) {
    const candidate_t& c   = candidates[i];
    double*            row = rows.data() + i * nf;
    std::string        why;
    if (c.config == nullptr || !(c.origami_latency > 0.0) ||
        c.origami_latency >= std::numeric_limits<double>::max()) {
      res.status = status_t::missing_feature;
      res.detail = "candidate " + std::to_string(i) + " has no Origami latency";
      return res;
    }
    if (!model_features(problem, hardware, c, row, &why)) {
      res.status = status_t::missing_feature;
      res.detail = "candidate " + std::to_string(i) + ": " + why;
      return res;
    }
    model_->prepare(row);
  }
  res.yhat.resize(n);
  res.corrected.resize(n);
  model_->evaluate_batch(rows.data(), n, res.yhat.data());
  for (std::size_t i = 0; i < n; ++i)
    res.corrected[i] = candidates[i].origami_latency * std::exp(res.yhat[i]);
  // Same key and rounding as training's argmin(log(cycles) + yhat); this orders exactly like
  // origami_latency * exp(yhat) without overflow.
  std::vector<double> key(n);
  for (std::size_t i = 0; i < n; ++i) key[i] = std::log(candidates[i].origami_latency) + res.yhat[i];
  std::stable_sort(res.order.begin(), res.order.begin() + static_cast<std::ptrdiff_t>(n),
                   [&](std::size_t a, std::size_t b) { return key[a] < key[b]; });
  return res;
}

bool rerank_configs(const booster_t&                  booster,
                    const problem_t&                  problem,
                    const hardware_t&                 hardware,
                    std::vector<prediction_result_t>& results,
                    rerank_source_t*                  source) {
  static const std::vector<std::size_t> kNoOrder;
  std::string                           detail;
  const status_t problem_status = booster.check_problem(problem, hardware, &detail);
  if (problem_status != status_t::ok) {
    if (source) source->outcome(problem_status, detail, kNoOrder);
    return false;
  }

  // The first top_n results the caller accepts, in analytical order.
  thread_local std::vector<candidate_t> candidates;
  thread_local std::vector<std::size_t> positions;
  candidates.clear();
  positions.clear();
  const std::size_t top_n = booster.model().top_n();
  for (std::size_t i = 0; i < results.size() && candidates.size() < top_n; ++i) {
    candidate_t c;
    c.config          = &results[i].config;
    c.origami_latency = results[i].latency;
    if (source && !source->candidate(results[i].config, c.fields)) continue;
    candidates.push_back(c);
    positions.push_back(i);
  }
  const boost_result_t res = booster.rank(problem, hardware, candidates.data(), candidates.size());
  if (res.status != status_t::ok) {
    if (source) source->outcome(res.status, res.detail, kNoOrder);
    return false;
  }

  // Candidates to the front in the model's order; the results between them shift back.
  thread_local std::vector<prediction_result_t> ranked;
  ranked.clear();
  for (const std::size_t k : res.order) ranked.push_back(results[positions[k]]);
  std::size_t write = positions.back();
  for (std::size_t read = positions.back() + 1, k = positions.size(); read-- > 0;) {
    if (k > 0 && positions[k - 1] == read) {
      --k;
      continue;
    }
    results[write--] = results[read];
  }
  std::copy(ranked.begin(), ranked.end(), results.begin());
  if (source) source->outcome(status_t::ok, res.detail, res.order);
  return true;
}

}  // namespace origami::nn::boost
