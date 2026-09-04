// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/nn.hpp"

#include "origami/nn/detail/model_store.hpp"
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

model_handle_t load_tilewright_manifest(const std::string& path) {
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
}

model_handle_t load_from_origami_nn_index(const std::string& logic_stem,
                                          backend_id_t backend,
                                          const std::string& hint_dir) {
  const std::string dir = resolve_weights_dir(hint_dir);
  if (dir.empty()) return invalid_handle;

  const char* backend_str = backend_name(backend);
  if (backend_str == nullptr) return invalid_handle;

  std::ifstream idx(dir + "/" + kOrigamiNnIndex);
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
      return load_tilewright_manifest(dir + "/" + weights_file);
    }
    return invalid_handle;
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
  if (handle < 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_infos.find(handle);
  if (it == g_infos.end()) return;
  if (it->second.backend == backend_id_t::tilewright_v1) {
    g_default_tilewright = handle;
  } else if (it->second.backend == backend_id_t::embedding_similarity_v1) {
    g_default_es = handle;
  }
}

model_handle_t default_model(backend_id_t backend) {
  std::lock_guard<std::mutex> lock(g_mutex);
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return g_default_tilewright;
    case backend_id_t::embedding_similarity_v1:
      return g_default_es;
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

}  // namespace detail
}  // namespace origami::nn
