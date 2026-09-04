// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace origami::nn::twrec::detail {

inline constexpr std::uint32_t kTwrecSchemaVersion = 1;

inline constexpr std::uint32_t kMaxCells    = 8192;
inline constexpr std::uint32_t kMaxSplits   = 8192;
inline constexpr std::uint32_t kMaxStrLen   = 512;
inline constexpr std::uint32_t kMaxSmartK   = 65536;
inline constexpr std::uint32_t kMaxQDim     = 128;
inline constexpr std::uint32_t kMaxIDim     = 64;
inline constexpr std::uint32_t kMaxXDim     = 128;
inline constexpr std::uint32_t kMaxHidden   = 512;
inline constexpr std::uint32_t kMaxEmbed    = 256;
inline constexpr std::uint32_t kMaxInterH   = 256;
inline constexpr std::size_t kMaxB64Decoded = 64 * 1024 * 1024;

inline bool valid_split_axis(char axis) {
  return axis == 'M' || axis == 'N' || axis == 'K' || axis == 'B';
}

inline bool valid_hyperparam_tier(std::uint32_t embed_dim,
                                  std::uint32_t hidden_dim,
                                  std::uint32_t inter_hidden) {
  if (hidden_dim == 0 || embed_dim == 0 || inter_hidden == 0) return false;
  if (hidden_dim > kMaxHidden || embed_dim > kMaxEmbed || inter_hidden > kMaxInterH)
    return false;
  switch (embed_dim) {
    case 16:
      return hidden_dim == 48 && inter_hidden == 24;
    case 24:
      return hidden_dim == 80 && inter_hidden == 40;
    case 40:
      return hidden_dim == 112 && inter_hidden == 56;
    case 64:
      return hidden_dim == 160 && inter_hidden == 96;
    case 80:
      return hidden_dim == 192 && inter_hidden == 112;
    default:
      return false;
  }
}

}  // namespace origami::nn::twrec::detail
