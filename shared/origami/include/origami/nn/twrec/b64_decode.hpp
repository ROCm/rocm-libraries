// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/twrec/load_limits.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace origami::nn::twrec::detail {

bool b64_decode(const std::string& in, std::vector<std::uint8_t>* out);

bool decode_int4_tensor(const std::vector<std::uint8_t>& raw,
                        std::size_t expected_count,
                        std::vector<float>* out);

bool decode_fp32_tensor(const std::vector<std::uint8_t>& raw,
                        std::size_t expected_count,
                        std::vector<float>* out);

}  // namespace origami::nn::twrec::detail
