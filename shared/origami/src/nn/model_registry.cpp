// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/nn.hpp"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace origami::nn {
namespace {

std::mutex g_mutex;
std::unordered_map<std::string, model_handle_t> g_path_to_handle;
std::unordered_map<model_handle_t, model_info_t> g_infos;
model_handle_t g_default_tilewright = invalid_handle;
model_handle_t g_default_es         = invalid_handle;

constexpr const char* kOrigamiNnIndex = "origami_nn_index";

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
  return {};
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

}  // namespace origami::nn
