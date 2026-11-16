// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/nn.hpp"

#include "origami/nn/detail/param_store.hpp"
#include "origami/nn/param/param_model.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
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
model_handle_t g_default_tilewright = invalid_handle;
model_handle_t g_default_es         = invalid_handle;
model_handle_t g_default_param      = invalid_handle;
model_handle_t g_next_handle        = 0;
std::unordered_map<model_handle_t, std::shared_ptr<const param::model_t>> g_param_models;

constexpr const char* kOrigamiNnIndex = "origami_nn_index";

const char* backend_name(backend_id_t backend) {
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return "tilewright";
    case backend_id_t::embedding_similarity_v1:
      return "embedding_similarity";
    case backend_id_t::param_v1:
      return "param";
  }
  return nullptr;
}

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && std::string(v) != "0";
}

// param_v1: parse + validate + self-test (the model's embedded test vectors) once per file.
model_handle_t load_param_file(const std::string& path) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_path_to_handle.find(path);
    if (it != g_path_to_handle.end()) return it->second;
  }
  std::string error;
  auto model = param::load(path, &error);
  if (model && !param::self_test(*model, &error)) model = nullptr, error = "self-test failed: " + error;
  if (!model) {
    std::cerr << "[origami::nn] param model " << path << " refused: " << error << std::endl;
    return invalid_handle;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_path_to_handle.find(path);
  if (it != g_path_to_handle.end()) return it->second;
  const model_handle_t handle = g_next_handle++;
  model_info_t info;
  info.backend = backend_id_t::param_v1;
  info.features.catalog_id = param::problem_key(*model);
  g_infos.emplace(handle, info);
  g_param_models.emplace(handle, std::move(model));
  g_path_to_handle.emplace(path, handle);
  if (env_flag("ORIGAMI_NN_DIAG"))
    std::cerr << "[origami::nn] param model " << path << " loaded as handle " << handle << ": "
              << param::summary(*g_param_models[handle]) << ", self-test passed" << std::endl;
  return handle;
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

// Loader of an index-loadable backend's weights file.
model_handle_t load_weights_file(const std::string& path, backend_id_t backend) {
  switch (backend) {
    case backend_id_t::param_v1:
      return load_param_file(path);
    case backend_id_t::tilewright_v1:
    case backend_id_t::embedding_similarity_v1:
      break;
  }
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
  models.param = load_from_origami_nn_index(logic_stem, backend_id_t::param_v1, hint_dir);
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
  g_param_models.erase(handle);
  if (g_default_tilewright == handle) g_default_tilewright = invalid_handle;
  if (g_default_es == handle) g_default_es = invalid_handle;
  if (g_default_param == handle) g_default_param = invalid_handle;
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
  } else if (it->second.backend == backend_id_t::param_v1) {
    g_default_param = handle;
  }
}

model_handle_t default_model(backend_id_t backend) {
  std::lock_guard<std::mutex> lock(g_mutex);
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return g_default_tilewright;
    case backend_id_t::embedding_similarity_v1:
      return g_default_es;
    case backend_id_t::param_v1:
      return g_default_param;
  }
  return invalid_handle;
}

std::string param_problem_key(model_handle_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_param_models.find(handle);
  return it == g_param_models.end() ? std::string() : param::problem_key(*it->second);
}

namespace detail {

std::shared_ptr<const param::model_t> param_payload(model_handle_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_param_models.find(handle);
  return it == g_param_models.end() ? nullptr : it->second;
}

}  // namespace detail

}  // namespace origami::nn
