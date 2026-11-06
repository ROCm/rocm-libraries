// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/twrec/b64_decode.hpp"

#include <cmath>
#include <cstring>

namespace origami::nn::twrec::detail {
namespace {

int b64_value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

}  // namespace

bool b64_decode(const std::string& in, std::vector<std::uint8_t>* out) {
  if (!out) return false;
  out->clear();
  if (in.empty()) return true;

  std::size_t olen = 0;
  int val          = 0;
  int valb         = -8;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    const int d = b64_value(c);
    if (d < 0) return false;
    val   = (val << 6) + d;
    valb += 6;
    if (valb >= 0) {
      if (olen >= kMaxB64Decoded) return false;
      out->push_back(static_cast<std::uint8_t>((val >> valb) & 0xFF));
      ++olen;
      valb -= 8;
    }
  }
  return true;
}

bool decode_int4_tensor(const std::vector<std::uint8_t>& raw,
                        std::size_t expected_count,
                        std::vector<float>* out) {
  if (!out || expected_count == 0) return false;
  if (raw.size() < sizeof(float)) return false;
  const std::size_t nbytes = (expected_count + 1) / 2;
  if (raw.size() != sizeof(float) + nbytes) return false;

  float scale = 0.0f;
  std::memcpy(&scale, raw.data(), sizeof(float));
  if (!std::isfinite(scale)) return false;

  out->resize(expected_count);
  const std::uint8_t* packed = raw.data() + sizeof(float);
  for (std::size_t j = 0; j < expected_count; ++j) {
    const std::uint8_t byte = packed[j >> 1];
    const int nib           = (j & 1) ? (byte >> 4) : (byte & 0x0F);
    (*out)[j]               = scale * static_cast<float>(nib - 8);
  }
  return true;
}

bool decode_fp32_tensor(const std::vector<std::uint8_t>& raw,
                        std::size_t expected_count,
                        std::vector<float>* out) {
  if (!out) return false;
  if (raw.size() != expected_count * sizeof(float)) return false;
  out->resize(expected_count);
  for (std::size_t i = 0; i < expected_count; ++i) {
    float v = 0.0f;
    std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
    if (!std::isfinite(v)) return false;
    (*out)[i] = v;
  }
  return true;
}

}  // namespace origami::nn::twrec::detail
