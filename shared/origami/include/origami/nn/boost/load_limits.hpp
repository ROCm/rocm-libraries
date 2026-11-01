// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

// ORIBOOST (.orr): "ORIBOOST", u16 format major, u16 minor, u32 reserved, u64 body length, the
// body (tagged sections HEAD, FEAT, RNGE, TREE, MANI; unknown tags are skipped), and the sha256 of
// the body.
namespace origami::nn::boost::detail {

inline constexpr char          kMagic[8]      = {'O', 'R', 'I', 'B', 'O', 'O', 'S', 'T'};
inline constexpr std::uint16_t kFormatMajor   = 1;
inline constexpr std::size_t   kHeaderBytes   = 24;
inline constexpr std::size_t   kChecksumBytes = 32;
inline constexpr std::size_t   kNumSections   = 5;

inline constexpr std::uint32_t kMaxFeatures = 65535;
// The quantized evaluator stores a feature index and a threshold bin in one byte each (255 marks
// a leaf and NaN), and a child as a 16-bit offset from its tree's root.
inline constexpr std::size_t kMaxQuantizedFeatures    = 254;
inline constexpr std::size_t kMaxThresholdsPerFeature = 254;
inline constexpr std::size_t kMaxNodesPerTree         = 65536;

// LightGBM node decision byte: bit 0 categorical, bits 2-3 missing type (0 None, 1 Zero, 2 NaN).
inline constexpr std::uint8_t kDecCategorical = 1;
inline constexpr int          kMissingZero = 1, kMissingNaN = 2;

inline constexpr const char* kManifestVersion = "1";

inline bool known_section(const char* tag) {
  const auto is = [&](const char* t) {
    return tag[0] == t[0] && tag[1] == t[1] && tag[2] == t[2] && tag[3] == t[3];
  };
  return is("HEAD") || is("FEAT") || is("RNGE") || is("TREE") || is("MANI");
}

}  // namespace origami::nn::boost::detail
