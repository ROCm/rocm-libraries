// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/esrec/loaded_model.hpp"
#include "origami/nn/twrec/loaded_model.hpp"
#include "origami/nn/types.hpp"

namespace origami::nn::detail {

const twrec::detail::LoadedModel* twrec_model_payload(model_handle_t handle);
const esrec::detail::LoadedModel* esrec_model_payload(model_handle_t handle);

}  // namespace origami::nn::detail
