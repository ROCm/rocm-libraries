// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/twrec/loaded_model.hpp"
#include "origami/nn/types.hpp"

#include <memory>

namespace origami::nn::detail {

/// Returns a shared reference so an in-flight rank call keeps the model alive even if
/// unload_model runs concurrently. Null when the handle is invalid or unloaded.
std::shared_ptr<const twrec::detail::LoadedModel> model_payload(model_handle_t handle);

}  // namespace origami::nn::detail
