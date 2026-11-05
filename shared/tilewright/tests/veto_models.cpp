// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// For the ctest entry tilewright-levers-parity (levers_parity.cmake): writes
// two copies of every given model that the alignment veto can serve (an
// MLREC_v3 model without the non-temporal rule), under the model's file
// name: into OUT/plain the model without the veto (without_alignment_veto,
// test_util.hpp: the model as it is when it declares none), into OUT that
// model with the veto of K above K_MIN and ALIGN_BITS-bit lines
// (with_alignment_veto), whether or not the model ships with a veto; it
// prints "copy <plain copy> <copy>" for each and "skip <model>" for the
// others.
//
// usage: tilewright-veto-models OUT K_MIN ALIGN_BITS MODEL...

#include "test_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s OUT K_MIN ALIGN_BITS MODEL...\n", argv[0]);
    return 2;
  }
  namespace fs = std::filesystem;
  const fs::path out(argv[1]);
  const unsigned long k_min = std::strtoul(argv[2], nullptr, 10);
  const unsigned long bits  = std::strtoul(argv[3], nullptr, 10);
  std::error_code ec;
  fs::create_directories(out, ec);
  if (ec) {
    std::fprintf(stderr, "cannot create %s: %s\n", out.string().c_str(), ec.message().c_str());
    return 1;
  }
  fs::create_directories(out / "plain", ec);
  if (ec) {
    std::fprintf(
        stderr, "cannot create %s: %s\n", (out / "plain").string().c_str(), ec.message().c_str());
    return 1;
  }
  for (int i = 4; i < argc; ++i) {
    const fs::path model(argv[i]);
    std::ifstream in(model, std::ios::binary);
    const std::vector<unsigned char> shipped((std::istreambuf_iterator<char>(in)),
                                             std::istreambuf_iterator<char>());
    const std::vector<unsigned char> stripped = tilewright_test::without_alignment_veto(shipped);
    const std::vector<unsigned char>& bytes   = stripped.empty() ? shipped : stripped;
    std::string err;
    const tilewright::ModelPtr plain =
        tilewright::load_model_from_memory(bytes.data(), bytes.size(), &err);
    if (!plain) {
      std::fprintf(stderr, "cannot load %s: %s\n", argv[i], err.c_str());
      return 1;
    }
    const tilewright::ModelInfo info = tilewright::describe(*plain);
    if (info.format_version < 3 || info.non_temporal_rule) {
      std::printf("skip %s\n", argv[i]);
      continue;
    }
    const std::vector<unsigned char> veto = tilewright_test::with_alignment_veto(
        bytes, static_cast<std::uint32_t>(k_min), static_cast<std::uint32_t>(bits));
    const tilewright::ModelPtr with =
        veto.empty() ? nullptr : tilewright::load_model_from_memory(veto.data(), veto.size(), &err);
    if (!with || !tilewright::describe(*with).alignment_veto.on) {
      std::fprintf(stderr, "cannot add the alignment veto to %s: %s\n", argv[i], err.c_str());
      return 1;
    }
    const fs::path plain_copy = out / "plain" / model.filename();
    const fs::path copy       = out / model.filename();
    tilewright_test::write_file(plain_copy, bytes);
    tilewright_test::write_file(copy, veto);
    std::printf("copy %s %s\n", plain_copy.string().c_str(), copy.string().c_str());
  }
  return 0;
}
