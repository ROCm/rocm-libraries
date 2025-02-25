#pragma once

#include <string>

#include <rocRoller/AssertOpKinds_fwd.hpp>

namespace rocRoller
{

    std::string   toString(const AssertOpKind& assertOpKind);
    std::ostream& operator<<(std::ostream&, AssertOpKind const);

}
