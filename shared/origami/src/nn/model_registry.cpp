// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/nn.hpp"

#include "origami/nn/boost/load_validate.hpp"
#include "origami/nn/detail/model_store.hpp"
#include "origami/nn/features/gemm_boost.hpp"
#include "origami/nn/features/gemm_tilewright.hpp"
#include "origami/nn/twrec/twrec_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami::nn {
namespace {

std::mutex g_mutex;
std::vector<std::unique_ptr<twrec::detail::LoadedModel>> g_models;
std::unordered_map<std::string, model_handle_t> g_path_to_handle;
std::unordered_map<model_handle_t, model_info_t> g_infos;
// boost_v1 payloads; their g_models slots stay empty.
std::unordered_map<model_handle_t, std::shared_ptr<const boost::booster_t>> g_boost_models;
model_handle_t g_default_tilewright = invalid_handle;
model_handle_t g_default_es         = invalid_handle;
model_handle_t g_default_boost      = invalid_handle;

constexpr const char* kOrigamiNnIndex = "origami_nn_index";

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_yaml_manifest(const std::string& path) { return ends_with(path, ".tilewright.yaml"); }

const char* backend_name(backend_id_t backend) {
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return "tilewright";
    case backend_id_t::embedding_similarity_v1:
      return "embedding_similarity";
    case backend_id_t::boost_v1:
      return "boost";
  }
  return nullptr;
}

std::string resolve_weights_dir(const std::string& hint_dir) {
  if (!hint_dir.empty()) return hint_dir;
  if (const char* env = std::getenv("ORIGAMI_NN_WEIGHTS_DIR")) return env;
#ifdef ORIGAMI_NN_BUNDLED_WEIGHTS_DIR
  return ORIGAMI_NN_BUNDLED_WEIGHTS_DIR;
#else
  return {};
#endif
}

model_info_t make_tilewright_info(const twrec::detail::LoadedModel& model) {
  model_info_t info;
  info.backend                     = backend_id_t::tilewright_v1;
  info.arch                        = model.arch;
  info.features.catalog_id         = features::gemm_tilewright::catalog_id;
  info.features.feature_names_hash = features::gemm_tilewright::feature_names_hash;
  info.features.query_dim          = model.q_dim;
  info.features.item_dim           = model.i_dim;
  info.features.interaction_dim    = model.x_dim;
  info.n_cells                     = static_cast<std::uint32_t>(model.cells.size());
  info.n_splits                    = static_cast<std::uint32_t>(model.splits.size());
  return info;
}

model_info_t make_boost_info(const boost::booster_t& booster) {
  const boost::ensemble_t& model   = booster.model();
  model_info_t             info;
  info.backend                     = backend_id_t::boost_v1;
  info.arch                        = model.head_string("arch");
  info.features.catalog_id         = features::gemm_boost::catalog_id;
  info.features.feature_names_hash = model.head_string("feature_schema_hash");
  info.features.interaction_dim    = static_cast<std::uint32_t>(model.features().size());
  return info;
}

void print_load_diag(model_handle_t handle,
                     const std::string& path,
                     const twrec::detail::LoadedModel& model) {
  if (std::getenv("ORIGAMI_NN_DIAG") == nullptr) {
    return;
  }
  std::fprintf(stderr,
               "[ORIGAMI_NN_DIAG] handle=%d path=%s arch=%s qhash=%s qdim=%u idim=%u xdim=%u "
               "n_cells=%zu n_splits=%zu\n",
               handle,
               path.c_str(),
               model.arch.c_str(),
               model.feature_names_hash.c_str(),
               model.q_dim,
               model.i_dim,
               model.x_dim,
               model.cells.size(),
               model.splits.size());
  std::fflush(stderr);
}

void print_boost_load_diag(model_handle_t handle,
                           const std::string& path,
                           const boost::booster_t& booster) {
  if (std::getenv("ORIGAMI_NN_DIAG") == nullptr) {
    return;
  }
  const boost::ensemble_t& model = booster.model();
  std::fprintf(stderr,
               "[ORIGAMI_NN_DIAG] handle=%d path=%s backend=boost arch=%s dtype=%s "
               "schema=%s columns=%zu trees=%zu top_n=%zu\n",
               handle,
               path.c_str(),
               model.head_string("arch").c_str(),
               model.head_string("dtype").c_str(),
               model.head_string("feature_schema_id").c_str(),
               model.features().size(),
               model.num_trees(),
               model.top_n());
  std::fflush(stderr);
}

model_handle_t load_tilewright_manifest(const std::string& path) {
#if ORIGAMI_ENABLE_NN
  if (!is_yaml_manifest(path)) return invalid_handle;

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_path_to_handle.find(path);
    if (it != g_path_to_handle.end()) return it->second;
  }

  auto model = std::make_unique<twrec::detail::LoadedModel>();
  if (!twrec::load_twrec_yaml(path, model.get())) return invalid_handle;

  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_path_to_handle.find(path);
  if (it != g_path_to_handle.end()) return it->second;

  const model_handle_t handle = static_cast<model_handle_t>(g_models.size());
  g_models.push_back(std::move(model));
  g_path_to_handle[path] = handle;
  const model_info_t info = make_tilewright_info(*g_models[static_cast<std::size_t>(handle)]);
  g_infos[handle]         = info;
  print_load_diag(handle, path, *g_models[static_cast<std::size_t>(handle)]);
  return handle;
#else
  (void)path;
  return invalid_handle;
#endif
}

// Path of the weights the index in the resolved weights dir maps (logic_stem, backend) to.
std::string find_in_origami_nn_index(const std::string& logic_stem,
                                     backend_id_t backend,
                                     const std::string& hint_dir) {
  const std::string dir = resolve_weights_dir(hint_dir);
  if (dir.empty()) return {};

  const char* backend_str = backend_name(backend);
  if (backend_str == nullptr) return {};

  std::ifstream idx(dir + "/" + kOrigamiNnIndex);
  if (!idx) return {};

  std::string line;
  while (std::getline(idx, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);

    std::istringstream ls(line);
    std::string stem;
    std::string indexed_backend;
    std::string weights_file;
    if (!(ls >> stem >> indexed_backend >> weights_file)) continue;
    if (stem != logic_stem || indexed_backend != backend_str) continue;

    return dir + "/" + weights_file;
  }

  return {};
}

model_handle_t load_from_origami_nn_index(const std::string& logic_stem,
                                          backend_id_t backend,
                                          const std::string& hint_dir) {
  const std::string path = find_in_origami_nn_index(logic_stem, backend, hint_dir);
  if (path.empty()) return invalid_handle;

  if (backend == backend_id_t::tilewright_v1) {
    return load_tilewright_manifest(path);
  }
  return invalid_handle;
}

}  // namespace

model_handle_t load_model(const std::string& path) {
  if (const char* override_path = std::getenv("ORIGAMI_NN_WEIGHTS")) {
    return load_tilewright_manifest(override_path);
  }
  return load_tilewright_manifest(path);
}

model_handle_t load_model_by_index(const std::string& logic_stem,
                                   backend_id_t backend,
                                   const std::string& hint_dir) {
  return load_from_origami_nn_index(logic_stem, backend, hint_dir);
}

library_models_t load_models_for_logic(const std::string& logic_stem,
                                       const std::string& hint_dir) {
  library_models_t models;
  models.tilewright =
      load_from_origami_nn_index(logic_stem, backend_id_t::tilewright_v1, hint_dir);
  return models;
}

std::string index_weights_path(const std::string& logic_stem,
                               backend_id_t backend,
                               const std::string& hint_dir) {
  return find_in_origami_nn_index(logic_stem, backend, hint_dir);
}

model_handle_t load_boost_model(const std::string& path,
                                const boost::deployment_t& target,
                                boost::status_t* status,
                                std::string* detail) {
  return load_boost_model(path, target, boost::build_identity(), status, detail);
}

model_handle_t load_boost_model(const std::string& path,
                                const boost::deployment_t& target,
                                const boost::build_identity_t& identity,
                                boost::status_t* status,
                                std::string* detail) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_path_to_handle.find(path);
    if (it != g_path_to_handle.end()) {
      if (status) *status = boost::status_t::ok;
      if (detail) detail->clear();
      return it->second;
    }
  }

  boost::status_t st = boost::status_t::ok;
  std::string why;
  auto booster = boost::booster_t::create(path, target, identity, st, why);
  if (status) *status = st;
  if (!booster) {
    if (detail) *detail = why;
    return invalid_handle;
  }
  if (detail) {
    const boost::ensemble_t& model = booster->model();
    *detail = "loaded " + std::to_string(model.num_trees()) + " trees, " +
              std::to_string(model.features().size()) + " features, schema " +
              model.head_string("feature_schema_id") + ", trained on " +
              model.head_string("training_data_id") + "; " + why;
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_path_to_handle.find(path);
  if (it != g_path_to_handle.end()) return it->second;

  const model_handle_t handle = static_cast<model_handle_t>(g_models.size());
  g_models.push_back(nullptr);
  g_path_to_handle[path] = handle;
  g_infos[handle]        = make_boost_info(*booster);
  print_boost_load_diag(handle, path, *booster);
  g_boost_models[handle] = std::move(booster);
  return handle;
}

void unload_model(model_handle_t handle) {
  if (handle < 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto idx = static_cast<std::size_t>(handle);

  for (auto it = g_path_to_handle.begin(); it != g_path_to_handle.end();) {
    if (it->second == handle) {
      it = g_path_to_handle.erase(it);
    } else {
      ++it;
    }
  }

  if (idx < g_models.size()) {
    g_models[idx].reset();
  }

  g_infos.erase(handle);
  g_boost_models.erase(handle);
  if (g_default_tilewright == handle) g_default_tilewright = invalid_handle;
  if (g_default_es == handle) g_default_es = invalid_handle;
  if (g_default_boost == handle) g_default_boost = invalid_handle;
}

const model_info_t* model_info(model_handle_t handle) {
  if (handle < 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_infos.find(handle);
  if (it == g_infos.end()) return nullptr;
  return &it->second;
}

void set_default_model(model_handle_t handle) {
  if (handle < 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_infos.find(handle);
  if (it == g_infos.end()) return;
  if (it->second.backend == backend_id_t::tilewright_v1) {
    g_default_tilewright = handle;
  } else if (it->second.backend == backend_id_t::embedding_similarity_v1) {
    g_default_es = handle;
  } else if (it->second.backend == backend_id_t::boost_v1) {
    g_default_boost = handle;
  }
}

model_handle_t default_model(backend_id_t backend) {
  std::lock_guard<std::mutex> lock(g_mutex);
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return g_default_tilewright;
    case backend_id_t::embedding_similarity_v1:
      return g_default_es;
    case backend_id_t::boost_v1:
      return g_default_boost;
  }
  return invalid_handle;
}

namespace detail {

const twrec::detail::LoadedModel* model_payload(model_handle_t handle) {
  if (handle < 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto idx = static_cast<std::size_t>(handle);
  if (idx >= g_models.size() || g_models[idx] == nullptr) return nullptr;
  return g_models[idx].get();
}

const boost::booster_t* boost_payload(model_handle_t handle) {
  if (handle < 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_boost_models.find(handle);
  if (it == g_boost_models.end()) return nullptr;
  return it->second.get();
}

}  // namespace detail
}  // namespace origami::nn
