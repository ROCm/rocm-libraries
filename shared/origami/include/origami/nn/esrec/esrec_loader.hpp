// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/esrec/loaded_model.hpp"

#include <string>

namespace origami::nn::esrec {

bool load_esrec_yaml(const std::string& manifest_path, detail::LoadedModel* out);

}  // namespace origami::nn::esrec
