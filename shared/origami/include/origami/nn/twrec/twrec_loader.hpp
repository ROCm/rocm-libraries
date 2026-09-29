// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/twrec/loaded_model.hpp"

#include <string>

namespace origami::nn::twrec {

bool load_twrec_yaml(const std::string& manifest_path, detail::LoadedModel* out);

}  // namespace origami::nn::twrec
