// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/esrec/loaded_model.hpp"

#include <vector>

namespace origami::nn::esrec {

std::vector<float> encode_query(const detail::EncoderModel& encoder,
                                const std::vector<float>& features);

}  // namespace origami::nn::esrec
