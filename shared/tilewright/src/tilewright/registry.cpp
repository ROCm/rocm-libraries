// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "features.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tilewright {
namespace detail {
namespace {

bool flag_set(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

long long cell_override() {
  const char* v = std::getenv("TILEWRIGHT_FORCE_CELL");
  if (v == nullptr || v[0] == '\0') return -1;
  errno               = 0;
  char* end           = nullptr;
  const long long val = std::strtoll(v, &end, 10);
  if (errno != 0 || end == v || *end != '\0' || val < 0) return -1;
  return val;
}

const char* weight_type_name(WeightType wt) {
  switch (wt) {
    case WeightType::Fp32: return "fp32";
    case WeightType::Bf16: return "bf16";
    case WeightType::Int8: return "int8";
    case WeightType::Int4: return "int4";
  }
  return "unknown";
}

void diag_loaded(const Model& m) {
  if (!env_knobs().diag) return;
  std::fprintf(stderr,
               "[TILEWRIGHT_DIAG FILE] path=%s arch=%s qhash=%s qdim=%zu idim=%zu xdim=%zu "
               "n_cells=%zu n_splits=%zu weights=%s\n",
               m.path.c_str(),
               m.arch.c_str(),
               m.feature_hash.c_str(),
               m.catalog->query_dim,
               m.catalog->item_dim,
               m.catalog->inter_dim,
               m.cells.size(),
               m.splits.size(),
               weight_type_name(m.weight_type));
  std::fflush(stderr);
}

void diag_failed(const std::string& message) {
  if (!env_knobs().diag) return;
  std::fprintf(stderr, "[TILEWRIGHT_DIAG FAIL] %s\n", message.c_str());
  std::fflush(stderr);
}

void set_error(std::string* error, const std::string& message) {
  if (error == nullptr) return;
  try {
    *error = message;
  } catch (...) { error->clear(); }
}

// For catch handlers of noexcept functions: builds the message inside the try.
void set_error(std::string* error,
               const char* origin,
               const char* suffix,
               const char* why) noexcept {
  if (error == nullptr) return;
  try {
    *error = std::string(origin) + suffix + ": " + why;
  } catch (...) { error->clear(); }
}

std::shared_ptr<const Model> parse_image(const unsigned char* data,
                                         std::size_t size,
                                         const std::string& origin,
                                         std::string* error) {
  std::string why;
  std::unique_ptr<Model> model = parse_model(data, size, &why);
  if (!model) {
    *error = origin + ": " + why;
    return nullptr;
  }
  model->path = origin;
  return std::shared_ptr<const Model>(std::move(model));
}

// Reads the fixed header first so that a file that is not a model, or whose
// declared sizes disagree with its length, is rejected without reading it.
std::shared_ptr<const Model> read_model_file(const std::string& path, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *error = path + ": cannot open file";
    return nullptr;
  }
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  f.seekg(0, std::ios::beg);
  if (end < 0) {
    *error = path + ": cannot determine file size";
    return nullptr;
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(end);

  unsigned char head[52] = {};
  const std::size_t head_size =
      static_cast<std::size_t>(std::min<std::uint64_t>(file_size, sizeof head));
  f.read(reinterpret_cast<char*>(head), static_cast<std::streamsize>(head_size));
  if (f.gcount() != static_cast<std::streamsize>(head_size)) {
    *error = path + ": read error";
    return nullptr;
  }
  if (head_size < sizeof head || std::memcmp(head, "MLREC_v2", 8) != 0)
    return parse_image(head, head_size, path, error);
  std::uint64_t header_size = 0, payload_size = 0;
  for (int b = 3; b >= 0; --b) header_size = (header_size << 8) | head[12 + b];
  for (int b = 7; b >= 0; --b) payload_size = (payload_size << 8) | head[16 + b];
  if (payload_size > file_size || header_size > file_size - payload_size ||
      file_size - payload_size - header_size != 8) {
    *error = path + ": file size " + std::to_string(file_size) +
             " does not match header_size + payload_size + 8";
    return nullptr;
  }

  std::vector<unsigned char> image(static_cast<std::size_t>(file_size));
  std::memcpy(image.data(), head, head_size);
  f.read(reinterpret_cast<char*>(image.data() + head_size),
         static_cast<std::streamsize>(file_size - head_size));
  if (f.gcount() != static_cast<std::streamsize>(file_size - head_size)) {
    *error = path + ": read error";
    return nullptr;
  }
  return parse_image(image.data(), image.size(), path, error);
}

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::weak_ptr<const Model>>& registry() {
  static std::unordered_map<std::string, std::weak_ptr<const Model>> r;
  return r;
}

}  // namespace

const EnvKnobs& env_knobs() noexcept {
  static const EnvKnobs knobs = [] {
    EnvKnobs k;
    k.diag       = flag_set("TILEWRIGHT_DIAG");
    k.pick_log   = flag_set("TILEWRIGHT_PICK_LOG");
    k.force_cell = cell_override();
    return k;
  }();
  return knobs;
}

}  // namespace detail

ModelPtr load_model(const std::string& path, std::string* error) noexcept {
  try {
    if (error != nullptr) error->clear();
    std::error_code ec;
    const std::filesystem::path canonical = std::filesystem::canonical(path, ec);
    if (ec) {
      detail::set_error(error, path + ": " + ec.message());
      detail::diag_failed(path + ": " + ec.message());
      return nullptr;
    }
    if (!std::filesystem::is_regular_file(canonical, ec)) {
      detail::set_error(error, path + ": not a regular file");
      detail::diag_failed(path + ": not a regular file");
      return nullptr;
    }
    const std::string key = canonical.string();
    {
      std::lock_guard<std::mutex> lock(detail::registry_mutex());
      auto it = detail::registry().find(key);
      if (it != detail::registry().end())
        if (ModelPtr existing = it->second.lock()) return existing;
    }

    std::string why;
    std::shared_ptr<const Model> model = detail::read_model_file(key, &why);
    if (!model) {
      detail::set_error(error, why);
      detail::diag_failed(why);
      return nullptr;
    }

    {
      std::lock_guard<std::mutex> lock(detail::registry_mutex());
      auto& reg = detail::registry();
      for (auto it = reg.begin(); it != reg.end();)
        it = it->second.expired() ? reg.erase(it) : ++it;
      std::weak_ptr<const Model>& slot = reg[key];
      if (ModelPtr existing = slot.lock()) return existing;
      slot = model;
    }
    detail::diag_loaded(*model);
    return model;
  } catch (const std::bad_alloc&) {
    detail::set_error(error, path.c_str(), "", "out of memory");
  } catch (const std::exception& e) {
    detail::set_error(error, path.c_str(), "", e.what());
  } catch (...) { detail::set_error(error, path.c_str(), "", "unknown error"); }
  return nullptr;
}

ModelPtr load_model_from_memory(const void* data, std::size_t size, std::string* error) noexcept {
  try {
    if (error != nullptr) error->clear();
    if (data == nullptr && size != 0) {
      detail::set_error(error, "<memory>: null data");
      return nullptr;
    }
    static const unsigned char kEmpty = 0;
    const unsigned char* bytes = size == 0 ? &kEmpty : static_cast<const unsigned char*>(data);
    std::string why;
    std::shared_ptr<const Model> model = detail::parse_image(bytes, size, "<memory>", &why);
    if (!model) {
      detail::set_error(error, why);
      detail::diag_failed(why);
      return nullptr;
    }
    detail::diag_loaded(*model);
    return model;
  } catch (const std::bad_alloc&) {
    detail::set_error(error, "<memory>", "", "out of memory");
  } catch (const std::exception& e) {
    detail::set_error(error, "<memory>", "", e.what());
  } catch (...) { detail::set_error(error, "<memory>", "", "unknown error"); }
  return nullptr;
}

ModelPtr load_model_by_index(const std::string& logic_stem,
                             const std::string& dir,
                             std::string* error) noexcept {
  try {
    if (error != nullptr) error->clear();
    if (dir.empty()) return nullptr;
    const std::filesystem::path index_path = std::filesystem::path(dir) / "tilewright_index";
    std::error_code ec;
    if (!std::filesystem::exists(index_path, ec)) return nullptr;
    std::ifstream index(index_path);
    if (!index) {
      detail::set_error(error, index_path.string() + ": cannot open index");
      return nullptr;
    }
    std::string line;
    while (std::getline(index, line)) {
      const std::size_t hash = line.find('#');
      if (hash != std::string::npos) line.resize(hash);
      std::istringstream fields(line);
      std::string stem, file;
      if (!(fields >> stem >> file)) continue;
      if (stem == logic_stem)
        return load_model((std::filesystem::path(dir) / file).string(), error);
    }
    return nullptr;
  } catch (const std::bad_alloc&) {
    detail::set_error(error, dir.c_str(), "/tilewright_index", "out of memory");
  } catch (const std::exception& e) {
    detail::set_error(error, dir.c_str(), "/tilewright_index", e.what());
  } catch (...) { detail::set_error(error, dir.c_str(), "/tilewright_index", "unknown error"); }
  return nullptr;
}

}  // namespace tilewright
