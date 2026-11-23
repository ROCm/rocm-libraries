// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// standins_flag IN OUT [KERNELS] writes the model file IN with stand-ins to OUT,
// whether or not IN has them, for the ctest entry tilewright-levers-parity
// (levers_parity.cmake): records chosen without data (synthetic_standins,
// model_writer.hpp), preferring the view keys of the kernels the KERNELS file
// lists (the `^[0-9]` lines of tests/data/sample_kernels.tsv), so that they
// join whitelists on that pool. It exits 3 without writing for a file that is
// not MLREC_v3 or that cannot take stand-ins (no trained-key gate or whitelist
// tier, launch keys, a cell that lists signatures, nothing to list), and 1 on
// any other error.

#include "features.hpp"
#include "model_writer.hpp"

#include <tilewright/model.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace tw = tilewright;

// The view-key hashes under `model` of the kernels the file `path` lists.
std::vector<std::uint64_t> kernel_hashes(const tw::Model& model, const char* path) {
  std::vector<std::uint64_t> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] < '0' || line[0] > '9') continue;
    std::istringstream s(line);
    long v[12];
    for (long& x : v) s >> x;
    if (!s) continue;
    tw::Config c;
    c.mt            = {static_cast<std::size_t>(v[0]),
                       static_cast<std::size_t>(v[1]),
                       static_cast<std::size_t>(v[2])};
    c.mi            = {static_cast<std::size_t>(v[3]),
                       static_cast<std::size_t>(v[4]),
                       static_cast<std::size_t>(v[5])};
    c.occupancy     = static_cast<int>(v[6]);
    c.cache_hints_a = static_cast<int>(v[7]);
    c.cache_hints_b = static_cast<int>(v[8]);
    std::int64_t attrs[tw::detail::kMaxAttributeReads];
    std::uint64_t key[tw::detail::kMaxViewKeySize];
    tw::detail::read_attributes(model, c, attrs);
    model.catalog->view_key(model, c, attrs, key);
    out.push_back(tw::detail::view_key_hash(key, model.view_key_size));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: standins_flag IN OUT [KERNELS]\n");
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                         std::istreambuf_iterator<char>());
  if (bytes.size() < 8 || std::memcmp(bytes.data(), "MLREC_v3", 8) != 0) return 3;
  std::vector<unsigned char> image;
  try {
    const std::vector<unsigned char> base = tilewright_test::without_standins(bytes);
    std::string err;
    const tw::ModelPtr model = tw::load_model_from_memory(base.data(), base.size(), &err);
    if (!model) {
      std::fprintf(stderr, "%s: %s\n", argv[1], err.c_str());
      return 1;
    }
    const tw::ModelInfo info = tw::describe(*model);
    if (!info.trained_key_gate || !info.whitelist_tier || info.launch_keys) return 3;
    for (const tw::detail::Cell& cell : model->cells)
      if (!cell.signatures.empty()) return 3;
    const std::vector<std::uint64_t> prefer =
        argc == 4 ? kernel_hashes(*model, argv[3]) : std::vector<std::uint64_t>{};
    const std::vector<tilewright_test::StandInRecord> records =
        tilewright_test::synthetic_standins(*model, prefer);
    if (records.empty()) return 3;
    image = tilewright_test::with_standins(base, records);
    if (!tw::load_model_from_memory(image.data(), image.size(), &err)) {
      std::fprintf(stderr, "%s: %s\n", argv[1], err.c_str());
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", argv[1], e.what());
    return 1;
  }
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(image.data()),
            static_cast<std::streamsize>(image.size()));
  out.close();
  return out ? 0 : 1;
}
