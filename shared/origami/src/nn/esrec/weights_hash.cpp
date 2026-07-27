// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/esrec/weights_hash.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace origami::nn::esrec::detail {
namespace {

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
  return (value >> bits) | (value << (32 - bits));
}

struct Sha256Context {
  std::array<std::uint32_t, 8> state{};
  std::array<std::uint8_t, 64>  buffer{};
  std::uint64_t                 total_bytes = 0;
  std::size_t                   buffer_len  = 0;
};

void sha256_transform(Sha256Context* ctx, const std::uint8_t block[64]) {
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
           (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<std::uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i]                     = w[i - 16] + s0 + w[i - 7] + s1;
  }

  std::uint32_t a = ctx->state[0];
  std::uint32_t b = ctx->state[1];
  std::uint32_t c = ctx->state[2];
  std::uint32_t d = ctx->state[3];
  std::uint32_t e = ctx->state[4];
  std::uint32_t f = ctx->state[5];
  std::uint32_t g = ctx->state[6];
  std::uint32_t h = ctx->state[7];

  for (int i = 0; i < 64; ++i) {
    const std::uint32_t s1    = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch    = (e & f) ^ (~e & g);
    const std::uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
    const std::uint32_t s0    = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = s0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

void sha256_init(Sha256Context* ctx) {
  ctx->state     = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  ctx->total_bytes = 0;
  ctx->buffer_len  = 0;
}

void sha256_update(Sha256Context* ctx, const std::uint8_t* data, std::size_t len) {
  ctx->total_bytes += len;

  while (len > 0) {
    const std::size_t space = 64 - ctx->buffer_len;
    const std::size_t chunk = len < space ? len : space;
    std::memcpy(ctx->buffer.data() + ctx->buffer_len, data, chunk);
    ctx->buffer_len += chunk;
    data += chunk;
    len -= chunk;

    if (ctx->buffer_len == 64) {
      sha256_transform(ctx, ctx->buffer.data());
      ctx->buffer_len = 0;
    }
  }
}

std::array<std::uint8_t, 32> sha256_final(Sha256Context* ctx) {
  const std::uint64_t bit_len = ctx->total_bytes * 8;

  ctx->buffer[ctx->buffer_len++] = 0x80;
  if (ctx->buffer_len > 56) {
    while (ctx->buffer_len < 64) ctx->buffer[ctx->buffer_len++] = 0;
    sha256_transform(ctx, ctx->buffer.data());
    ctx->buffer_len = 0;
  }
  while (ctx->buffer_len < 56) ctx->buffer[ctx->buffer_len++] = 0;

  for (int i = 7; i >= 0; --i) {
    ctx->buffer[ctx->buffer_len++] = static_cast<std::uint8_t>((bit_len >> (i * 8)) & 0xff);
  }
  sha256_transform(ctx, ctx->buffer.data());

  std::array<std::uint8_t, 32> digest{};
  for (int i = 0; i < 8; ++i) {
    digest[i * 4]     = static_cast<std::uint8_t>((ctx->state[i] >> 24) & 0xff);
    digest[i * 4 + 1] = static_cast<std::uint8_t>((ctx->state[i] >> 16) & 0xff);
    digest[i * 4 + 2] = static_cast<std::uint8_t>((ctx->state[i] >> 8) & 0xff);
    digest[i * 4 + 3] = static_cast<std::uint8_t>(ctx->state[i] & 0xff);
  }
  return digest;
}

std::string sha256_hex_prefix(const std::string& data, std::size_t hex_chars) {
  Sha256Context ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
  const auto digest = sha256_final(&ctx);

  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(hex_chars);
  for (std::size_t i = 0; i < hex_chars / 2; ++i) {
    out.push_back(kHex[digest[i] >> 4]);
    out.push_back(kHex[digest[i] & 0x0f]);
  }
  return out;
}

bool looks_like_json_int(const std::string& scalar) {
  if (scalar.empty()) return false;
  std::size_t i = 0;
  if (scalar[0] == '-' || scalar[0] == '+') {
    if (scalar.size() == 1) return false;
    ++i;
  }
  for (; i < scalar.size(); ++i) {
    const char c = scalar[i];
    if (c < '0' || c > '9') return false;
  }
  return true;
}

bool looks_like_json_float(const std::string& scalar) {
  if (scalar.empty() || looks_like_json_int(scalar)) return false;
  bool has_digit = false;
  for (char c : scalar) {
    if (c >= '0' && c <= '9') {
      has_digit = true;
      continue;
    }
    if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') continue;
    return false;
  }
  return has_digit &&
         (scalar.find('.') != std::string::npos || scalar.find('e') != std::string::npos ||
          scalar.find('E') != std::string::npos);
}

bool looks_like_json_number(const std::string& scalar) {
  return looks_like_json_int(scalar) || looks_like_json_float(scalar);
}

void append_json_string(const std::string& value, std::string* out) {
  out->push_back('"');
  for (char c : value) {
    switch (c) {
      case '"': out->append("\\\""); break;
      case '\\': out->append("\\\\"); break;
      case '\b': out->append("\\b"); break;
      case '\f': out->append("\\f"); break;
      case '\n': out->append("\\n"); break;
      case '\r': out->append("\\r"); break;
      case '\t': out->append("\\t"); break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char hex[7];
          std::snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned char>(c));
          out->append(hex);
        } else {
          out->push_back(c);
        }
        break;
    }
  }
  out->push_back('"');
}

void append_canonical_json(const YAML::Node& node, std::string* out);

void append_json_map(const YAML::Node& node, std::string* out) {
  out->push_back('{');
  std::vector<std::string> keys;
  keys.reserve(node.size());
  for (auto it = node.begin(); it != node.end(); ++it) {
    keys.push_back(it->first.Scalar());
  }
  std::sort(keys.begin(), keys.end());

  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (i > 0) out->push_back(',');
    append_json_string(keys[i], out);
    out->push_back(':');
    append_canonical_json(node[keys[i]], out);
  }
  out->push_back('}');
}

void append_json_sequence(const YAML::Node& node, std::string* out) {
  out->push_back('[');
  for (std::size_t i = 0; i < node.size(); ++i) {
    if (i > 0) out->push_back(',');
    append_canonical_json(node[i], out);
  }
  out->push_back(']');
}

void append_canonical_json(const YAML::Node& node, std::string* out) {
  if (!node || node.IsNull()) {
    out->append("null");
    return;
  }
  if (node.IsMap()) {
    append_json_map(node, out);
    return;
  }
  if (node.IsSequence()) {
    append_json_sequence(node, out);
    return;
  }
  if (!node.IsScalar()) {
    out->append("null");
    return;
  }

  const std::string scalar = node.Scalar();
  if (scalar == "true" || scalar == "false") {
    out->append(scalar);
    return;
  }
  if (scalar == "null" || scalar == "~") {
    out->append("null");
    return;
  }
  if (looks_like_json_number(scalar)) {
    out->append(scalar);
    return;
  }

  append_json_string(scalar, out);
}

}  // namespace

std::string compute_weights_hash(const YAML::Node& sidecar_root) {
  std::string canonical;
  canonical.reserve(1 << 20);
  append_canonical_json(sidecar_root, &canonical);
  return sha256_hex_prefix(canonical, 16);
}

}  // namespace origami::nn::esrec::detail
