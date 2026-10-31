// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/boost/loaded_model.hpp"
#include "origami/nn/twrec/loaded_model.hpp"
#include "origami/nn/types.hpp"

namespace origami::nn::detail {

const twrec::detail::LoadedModel* model_payload(model_handle_t handle);

const boost::booster_t* boost_payload(model_handle_t handle);

}  // namespace origami::nn::detail
