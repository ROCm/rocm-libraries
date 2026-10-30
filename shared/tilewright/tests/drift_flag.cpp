// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// drift_flag IN OUT KEEP_PPM writes the model file IN served with the drift
// fallback at KEEP_PPM to OUT, for the ctest entry tilewright-levers-parity
// (levers_parity.cmake). It exits 3 without writing for a file that is not
// MLREC_v3 or that the engine refuses with the fallback (no whitelist tier,
// launch keys, a cell that lists signatures), and 1 on any other error.

#include "model_writer.hpp"

#include <tilewright/model.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: drift_flag IN OUT KEEP_PPM\n");
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
  char* end               = nullptr;
  const unsigned long ppm = std::strtoul(argv[3], &end, 10);
  if (end == argv[3] || *end != '\0' || ppm > 0xFFFFFFFFul) {
    std::fprintf(stderr, "bad KEEP_PPM %s\n", argv[3]);
    return 1;
  }
  std::vector<unsigned char> image;
  try {
    image = tilewright_test::with_drift_fallback(bytes, static_cast<std::uint32_t>(ppm));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", argv[1], e.what());
    return 1;
  }
  std::string err;
  if (!tilewright::load_model_from_memory(image.data(), image.size(), &err)) {
    std::fprintf(stderr, "%s: %s\n", argv[1], err.c_str());
    return 3;
  }
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(image.data()),
            static_cast<std::streamsize>(image.size()));
  out.close();
  return out ? 0 : 1;
}
