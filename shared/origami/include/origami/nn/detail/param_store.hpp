// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/param/param_model.hpp"
#include "origami/nn/types.hpp"

#include <memory>

namespace origami::nn::detail {

/// The loaded param_v1 model behind @p handle, or nullptr.
std::shared_ptr<const param::model_t> param_payload(model_handle_t handle);

}  // namespace origami::nn::detail
