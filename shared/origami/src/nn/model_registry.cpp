// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/nn.hpp"

#include "origami/nn/olc/load_validate.hpp"
#include "origami/nn/detail/model_store.hpp"
#include "origami/nn/features/gemm_olc.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace origami::nn {
namespace {

std::mutex g_mutex;
std::unordered_map<std::string, model_handle_t> g_path_to_handle;
std::unordered_map<model_handle_t, model_info_t> g_infos;
std::unordered_map<model_handle_t, std::shared_ptr<const olc::corrector_t>> g_olc_models;
model_handle_t g_next_handle        = 0;
model_handle_t g_default_tilewright = invalid_handle;
model_handle_t g_default_es         = invalid_handle;
model_handle_t g_default_olc        = invalid_handle;

constexpr const char* kOrigamiNnIndex = "origami_nn_index";

const char* backend_name(backend_id_t backend) {
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return "tilewright";
    case backend_id_t::embedding_similarity_v1:
      return "embedding_similarity";
    case backend_id_t::olc_v1:
      return "olc";
  }
  return nullptr;
}

std::string resolve_weights_dir(const std::string& hint_dir) {
  if (!hint_dir.empty()) return hint_dir;
  if (const char* env = std::getenv("ORIGAMI_NN_WEIGHTS_DIR")) return env;
  return {};
}

model_info_t make_olc_info(const olc::corrector_t& corrector) {
  const olc::ensemble_t& model   = corrector.model();
  model_info_t             info;
  info.backend                     = backend_id_t::olc_v1;
  info.arch                        = model.head_string("arch");
  info.features.catalog_id         = features::gemm_olc::catalog_id;
  info.features.feature_names_hash = model.head_string("feature_schema_hash");
  info.features.interaction_dim    = static_cast<std::uint32_t>(model.features().size());
  return info;
}

void print_olc_load_diag(model_handle_t handle,
                           const std::string& path,
                           const olc::corrector_t& corrector) {
  if (std::getenv("ORIGAMI_NN_DIAG") == nullptr) {
    return;
  }
  const olc::ensemble_t& model = corrector.model();
  std::fprintf(stderr,
               "[ORIGAMI_NN_DIAG] handle=%d path=%s backend=olc arch=%s dtype=%s "
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

// Loader of an index-loadable backend's weights file; none is built in yet.
model_handle_t load_weights_file(const std::string& path, backend_id_t backend) {
  (void)path;
  (void)backend;
  return invalid_handle;
}

model_handle_t load_from_origami_nn_index(const std::string& logic_stem,
                                          backend_id_t backend,
                                          const std::string& hint_dir) {
  const std::string path = find_in_origami_nn_index(logic_stem, backend, hint_dir);
  if (path.empty()) return invalid_handle;
  return load_weights_file(path, backend);
}

}  // namespace

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

model_handle_t load_olc_model(const std::string& path,
                                const olc::deployment_t& target,
                                olc::status_t* status,
                                std::string* detail) {
  return load_olc_model(path, target, olc::build_identity(), status, detail);
}

model_handle_t load_olc_model(const std::string& path,
                                const olc::deployment_t& target,
                                const olc::build_identity_t& identity,
                                olc::status_t* status,
                                std::string* detail) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_path_to_handle.find(path);
    if (it != g_path_to_handle.end()) {
      if (status) *status = olc::status_t::ok;
      if (detail) detail->clear();
      return it->second;
    }
  }

  olc::status_t st = olc::status_t::ok;
  std::string why;
  auto corrector = olc::corrector_t::create(path, target, identity, st, why);
  if (status) *status = st;
  if (!corrector) {
    if (detail) *detail = why;
    return invalid_handle;
  }
  if (detail) {
    const olc::ensemble_t& model = corrector->model();
    *detail = "loaded " + std::to_string(model.num_trees()) + " trees, " +
              std::to_string(model.features().size()) + " features, schema " +
              model.head_string("feature_schema_id") + ", trained on " +
              model.head_string("training_data_id") + "; " + why;
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_path_to_handle.find(path);
  if (it != g_path_to_handle.end()) return it->second;

  const model_handle_t handle = g_next_handle++;
  g_path_to_handle[path] = handle;
  g_infos[handle]        = make_olc_info(*corrector);
  print_olc_load_diag(handle, path, *corrector);
  g_olc_models[handle] = std::move(corrector);
  return handle;
}

void unload_model(model_handle_t handle) {
  if (handle < 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);

  for (auto it = g_path_to_handle.begin(); it != g_path_to_handle.end();) {
    if (it->second == handle) {
      it = g_path_to_handle.erase(it);
    } else {
      ++it;
    }
  }

  g_infos.erase(handle);
  g_olc_models.erase(handle);
  if (g_default_tilewright == handle) g_default_tilewright = invalid_handle;
  if (g_default_es == handle) g_default_es = invalid_handle;
  if (g_default_olc == handle) g_default_olc = invalid_handle;
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
  } else if (it->second.backend == backend_id_t::olc_v1) {
    g_default_olc = handle;
  }
}

model_handle_t default_model(backend_id_t backend) {
  std::lock_guard<std::mutex> lock(g_mutex);
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return g_default_tilewright;
    case backend_id_t::embedding_similarity_v1:
      return g_default_es;
    case backend_id_t::olc_v1:
      return g_default_olc;
  }
  return invalid_handle;
}

namespace detail {

const olc::corrector_t* olc_payload(model_handle_t handle) {
  if (handle < 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_olc_models.find(handle);
  if (it == g_olc_models.end()) return nullptr;
  return it->second.get();
}

}  // namespace detail
}  // namespace origami::nn
