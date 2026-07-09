// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/nn.hpp"

#include "origami/nn/features/gemm_tilewright.hpp"

#if defined(ORIGAMI_ENABLE_NN_TILEWRIGHT) && ORIGAMI_ENABLE_NN_TILEWRIGHT

#  include "tilewright/twrec.hpp"

#  include <fstream>
#  include <mutex>
#  include <sstream>
#  include <unordered_map>

namespace origami::nn {
namespace {

std::mutex g_mutex;
std::unordered_map<model_handle_t, model_info_t> g_infos;
model_handle_t g_default_tilewright = invalid_handle;
model_handle_t g_default_es         = invalid_handle;

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
  }
  return nullptr;
}

model_info_t make_tilewright_info() {
  model_info_t info;
  info.backend                     = backend_id_t::tilewright_v1;
  info.arch                        = "gfx950";
  info.features.catalog_id         = features::gemm_tilewright_v1::catalog_id;
  info.features.feature_names_hash = features::gemm_tilewright_v1::feature_names_hash;
  info.features.query_dim          = static_cast<std::uint32_t>(features::gemm_tilewright_v1::query_dim);
  info.features.item_dim           = static_cast<std::uint32_t>(features::gemm_tilewright_v1::item_dim);
  info.features.interaction_dim    = static_cast<std::uint32_t>(features::gemm_tilewright_v1::interaction_dim);
  return info;
}

void register_handle(model_handle_t handle, backend_id_t backend) {
  if (handle < 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_infos[handle] = (backend == backend_id_t::tilewright_v1) ? make_tilewright_info()
                                                               : model_info_t{};
  g_infos[handle].backend = backend;
}

model_handle_t load_tilewright_manifest(const std::string& path) {
  if (!is_yaml_manifest(path)) return invalid_handle;
  const int handle = tilewright::load_model_yaml(path);
  if (handle < 0) return invalid_handle;
  register_handle(handle, backend_id_t::tilewright_v1);
  return handle;
}

model_handle_t load_from_origami_nn_index(const std::string& logic_stem,
                                          backend_id_t backend,
                                          const std::string& hint_dir) {
  if (hint_dir.empty()) return invalid_handle;

  const char* backend_str = backend_name(backend);
  if (backend_str == nullptr) return invalid_handle;

  std::ifstream idx(hint_dir + "/" + kOrigamiNnIndex);
  if (!idx) return invalid_handle;

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

    if (backend == backend_id_t::tilewright_v1) {
      return load_tilewright_manifest(hint_dir + "/" + weights_file);
    }
    return invalid_handle;
  }

  return invalid_handle;
}

}  // namespace

model_handle_t load_model(const std::string& path) { return load_tilewright_manifest(path); }

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

void unload_model(model_handle_t handle) {
  if (handle < 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_infos.erase(handle);
  if (g_default_tilewright == handle) g_default_tilewright = invalid_handle;
  if (g_default_es == handle) g_default_es = invalid_handle;
}

const model_info_t* model_info(model_handle_t handle) {
  if (handle < 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_infos.find(handle);
  if (it == g_infos.end()) return nullptr;
  return &it->second;
}

void set_default_model(model_handle_t handle) {
  const model_info_t* info = model_info(handle);
  if (info == nullptr) return;
  if (info->backend == backend_id_t::tilewright_v1) {
    g_default_tilewright = handle;
  } else if (info->backend == backend_id_t::embedding_similarity_v1) {
    g_default_es = handle;
  }
}

model_handle_t default_model(backend_id_t backend) {
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return g_default_tilewright;
    case backend_id_t::embedding_similarity_v1:
      return g_default_es;
  }
  return invalid_handle;
}

}  // namespace origami::nn

#else

namespace origami::nn {

model_handle_t load_model(const std::string&) { return invalid_handle; }

model_handle_t load_model_by_index(const std::string&, backend_id_t, const std::string&) {
  return invalid_handle;
}

library_models_t load_models_for_logic(const std::string&, const std::string&) {
  return {};
}

void unload_model(model_handle_t) {}

const model_info_t* model_info(model_handle_t) { return nullptr; }

void set_default_model(model_handle_t) {}

model_handle_t default_model(backend_id_t) { return invalid_handle; }

}  // namespace origami::nn

#endif
