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

// Strict RFC 4648 decoding. The payload is machine-generated, so anything a generator would
// not emit is an error: padding anywhere but the end, more than two padding characters, a
// length that is not a multiple of 4, and non-zero bits left over in the final group.
// Whitespace is skipped and does not count toward the length.
bool b64_decode(const std::string& in, std::vector<std::uint8_t>* out) {
  if (!out) return false;
  out->clear();
  if (in.empty()) return true;

  const auto fail = [out]() {
    out->clear();
    return false;
  };

  std::size_t olen = 0;
  std::size_t data = 0;  // characters from the alphabet
  std::size_t pad  = 0;  // trailing '=' characters
  std::uint32_t acc = 0;
  int bits          = 0;  // bits in acc not yet emitted: 0 to 7
  for (char c : in) {
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    if (c == '=') {
      if (++pad > 2) return fail();
      continue;
    }
    if (pad != 0) return fail();  // data after padding
    const int d = b64_value(c);
    if (d < 0) return fail();
    ++data;
    acc = (acc << 6) | static_cast<std::uint32_t>(d);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      if (olen >= kMaxB64Decoded) return fail();
      out->push_back(static_cast<std::uint8_t>((acc >> bits) & 0xFF));
      ++olen;
      acc &= (1u << bits) - 1u;
    }
  }

  if ((data + pad) % 4 != 0) return fail();
  if (acc != 0) return fail();  // the unused low bits of the last character must be zero
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
