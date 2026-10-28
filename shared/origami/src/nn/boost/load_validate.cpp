// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/boost/load_validate.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "origami/nn/features/gemm_boost.hpp"
#include "origami_source_stamp.hpp"

namespace origami::nn::boost {

namespace features = nn::features::gemm_boost;

const char* to_string(status_t status) {
  switch (status) {
    case status_t::ok: return "ok";
    case status_t::model_missing: return "model_missing";
    case status_t::bad_file: return "bad_file";
    case status_t::unsupported_format: return "unsupported_format";
    case status_t::incompatible: return "incompatible";
    case status_t::out_of_range: return "out_of_range";
    case status_t::missing_feature: return "missing_feature";
    case status_t::too_few_candidates: return "too_few_candidates";
  }
  return "unknown";
}

build_identity_t build_identity() {
  return {ORIGAMI_SOURCE_COMMIT, ORIGAMI_SOURCE_HASH, features::schema().id, features::schema().hash};
}

namespace {

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream        ss(s);
  std::string              item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

bool dtype_from_name(const std::string& name, data_type_t& out) {
  if (name == "bf16") out = data_type_t::BFloat16;
  else if (name == "f16") out = data_type_t::Half;
  else if (name == "f32") out = data_type_t::Float;
  else return false;
  return true;
}

}  // namespace

namespace detail {

bool validate_stamps(const ensemble_t&       m,
                     const deployment_t&     target,
                     const build_identity_t& identity,
                     std::string&            detail) {
  auto mismatch = [&](const char* what, const std::string& model_value, const std::string& want) {
    detail = std::string(what) + " mismatch: model " + model_value + ", runtime " + want;
    return false;
  };
  if (m.head_string("arch") != target.arch)
    return mismatch("arch", m.head_string("arch"), target.arch);
  if (m.head_string("dtype") != target.dtype)
    return mismatch("dtype", m.head_string("dtype"), target.dtype);
  if (m.head_string("feature_schema_hash") != identity.feature_schema_hash)
    return mismatch("feature schema", m.head_string("feature_schema_hash"),
                    identity.feature_schema_hash);
  if (m.head_string("origami_source_hash") != identity.origami_source_hash)
    return mismatch("origami source", m.head_string("origami_source_hash"),
                    identity.origami_source_hash);
  return true;
}

bool validate_library(const ensemble_t&   m,
                      const deployment_t& target,
                      library_diff_t&     diff,
                      std::string&        detail) {
  if (target.library_kernels == nullptr) {
    detail = "kernel library manifest not provided";
    return false;
  }
  diff = diff_library(m, *target.library_kernels);
  if (diff.decision == library_decision_t::retrain_required) {
    detail = "kernel library: " + diff.summary();
    return false;
  }
  return true;
}

bool validate_columns(const ensemble_t& m, std::vector<std::size_t>& slot, std::string& detail) {
  const auto& columns = features::schema().columns;
  for (const auto& f : m.features()) {
    const int ix = features::slot_index(f.name);
    if (ix < 0 || std::find(columns.begin(), columns.end(), f.name) == columns.end()) {
      detail = "model column " + f.name + " is not in schema " + features::schema().id;
      return false;
    }
    slot.push_back(static_cast<std::size_t>(ix));
  }
  return true;
}

bool validate_d_dtypes(const ensemble_t&         m,
                       std::vector<data_type_t>& d_dtypes,
                       std::string&              detail) {
  for (const auto& d : split_csv(m.head_string("d_dtypes"))) {
    data_type_t dt;
    if (!dtype_from_name(d, dt)) {
      detail = "unknown d dtype " + d;
      return false;
    }
    d_dtypes.push_back(dt);
  }
  return true;
}

}  // namespace detail

std::shared_ptr<const booster_t> booster_t::create(const std::string&      path,
                                                   const deployment_t&     target,
                                                   const build_identity_t& identity,
                                                   status_t&               status,
                                                   std::string&            detail) {
  auto m = ensemble_t::load_file(path, status, detail);
  if (!m) return nullptr;
  return create(std::move(m), target, identity, status, detail);
}

std::shared_ptr<const booster_t> booster_t::create(std::shared_ptr<const ensemble_t> model,
                                                   const deployment_t&               target,
                                                   const build_identity_t&           identity,
                                                   status_t&                         status,
                                                   std::string&                      detail) {
  status = status_t::incompatible;
  if (!model) {
    status = status_t::model_missing;
    detail = "no model";
    return nullptr;
  }
  const ensemble_t& m = *model;
  if (!detail::validate_stamps(m, target, identity, detail)) return nullptr;
  library_diff_t diff;
  if (!detail::validate_library(m, target, diff, detail)) return nullptr;

  std::shared_ptr<booster_t> b(new booster_t());
  b->model_ = std::move(model);
  if (!detail::validate_columns(*b->model_, b->slot_, detail)) {
    status = status_t::missing_feature;
    return nullptr;
  }
  if (!detail::validate_d_dtypes(*b->model_, b->d_dtypes_, detail)) {
    status = status_t::bad_file;
    return nullptr;
  }
  status = status_t::ok;
  // The commit is provenance only: the source hash decides, so a build from an uncommitted
  // tree with identical sources is accepted.
  const std::string& trained = m.head_string("origami_commit");
  detail = "origami commit: model " + trained + ", build " + identity.origami_commit;
  if (trained != identity.origami_commit) detail += " (differs; sources identical)";
  detail += "; kernel library " + diff.summary();
  return b;
}

}  // namespace origami::nn::boost
