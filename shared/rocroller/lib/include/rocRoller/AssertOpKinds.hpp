// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <rocRoller/AssertOpKinds_fwd.hpp>

namespace rocRoller
{

    std::string   toString(const AssertOpKind& assertOpKind);
    std::ostream& operator<<(std::ostream&, AssertOpKind const);

}
