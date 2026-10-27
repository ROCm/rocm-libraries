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
#include <limits>
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

// A whole non-negative decimal number, else -1.
long long count_knob(const char* name) {
  const char* v = std::getenv(name);
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
               m.query_dim,
               m.item_dim,
               m.inter_dim,
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
  if (head_size < sizeof head ||
      (std::memcmp(head, "MLREC_v2", 8) != 0 && std::memcmp(head, "MLREC_v3", 8) != 0))
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

// The PCI chip ID of an index row's optional third field, `chip=0x` and one
// to four hex digits that are not all zero; 0 for any other field.
std::uint32_t index_chip_id(const std::string& field) {
  static const char kPrefix[] = "chip=0x";
  const std::size_t n         = sizeof kPrefix - 1;
  if (field.size() <= n || field.size() > n + 4 || field.compare(0, n, kPrefix) != 0) return 0;
  std::uint32_t id = 0;
  for (std::size_t i = n; i < field.size(); ++i) {
    const char c        = field[i];
    std::uint32_t digit = 0;
    if (c >= '0' && c <= '9')
      digit = static_cast<std::uint32_t>(c - '0');
    else if (c >= 'a' && c <= 'f')
      digit = static_cast<std::uint32_t>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F')
      digit = static_cast<std::uint32_t>(c - 'A' + 10);
    else
      return 0;
    id = id * 16 + digit;
  }
  return id;
}

// Up to 64 characters of `text`, with every byte that is not graphic ASCII
// shown as '?'.
std::string printable(const std::string& text) {
  std::string out = text.substr(0, 64);
  for (char& c : out)
    if (c < 0x21 || c > 0x7E) c = '?';
  return out;
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
    k.force_cell = count_knob("TILEWRIGHT_FORCE_CELL");
    // Values whose bytes do not fit a size_t count as unset.
    const long long mib = count_knob("TILEWRIGHT_CACHE_MB");
    if (mib >= 0 &&
        static_cast<unsigned long long>(mib) <= (std::numeric_limits<std::size_t>::max() >> 20))
      k.cache_mib = mib;
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
  return load_model_by_index(logic_stem, dir, std::uint32_t{0}, error);
}

// Every row of the stem is read, so that a row naming the chip ID wins
// wherever it stands and a malformed row of the stem is never passed over.
ModelPtr load_model_by_index(const std::string& logic_stem,
                             const std::string& dir,
                             std::uint32_t pci_chip_id,
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
    std::string line, any_device, this_device;
    bool have_any = false, have_this = false;
    std::size_t line_no = 0;
    while (std::getline(index, line)) {
      ++line_no;
      const std::size_t hash = line.find('#');
      if (hash != std::string::npos) line.resize(hash);
      std::istringstream fields(line);
      std::string stem, file, chip, extra;
      if (!(fields >> stem >> file) || stem != logic_stem) continue;
      if (!(fields >> chip)) {
        if (!have_any) any_device = file;
        have_any = true;
        continue;
      }
      const std::uint32_t id = detail::index_chip_id(chip);
      if (id == 0 || (fields >> extra)) {
        const std::string why = index_path.string() + ":" + std::to_string(line_no) +
                                ": malformed row for " + logic_stem + ": '" +
                                detail::printable(id == 0 ? chip : extra) +
                                "' where chip=0x<1 to 4 hex digits> or the end of the row belongs";
        detail::set_error(error, why);
        detail::diag_failed(why);
        return nullptr;
      }
      if (id == pci_chip_id && !have_this) {
        this_device = file;
        have_this   = true;
      }
    }
    if (index.bad()) {
      detail::set_error(error, index_path.string() + ": cannot read index");
      return nullptr;
    }
    if (!have_this && !have_any) return nullptr;
    return load_model(
        (std::filesystem::path(dir) / (have_this ? this_device : any_device)).string(), error);
  } catch (const std::bad_alloc&) {
    detail::set_error(error, dir.c_str(), "/tilewright_index", "out of memory");
  } catch (const std::exception& e) {
    detail::set_error(error, dir.c_str(), "/tilewright_index", e.what());
  } catch (...) { detail::set_error(error, dir.c_str(), "/tilewright_index", "unknown error"); }
  return nullptr;
}

}  // namespace tilewright
