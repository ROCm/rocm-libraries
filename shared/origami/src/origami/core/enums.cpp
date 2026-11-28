// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>

#include "origami/core/enums.hpp"

namespace origami {

std::string hybrid_mode_to_string(hybrid_mode_t mode) {
  switch (mode) {
    case hybrid_mode_t::static_: return "static";
    case hybrid_mode_t::dynamic: return "dynamic";
    case hybrid_mode_t::none: return "none";
    default: return "Invalid";
  }
}

}  // namespace origami
