/**
 */

#pragma once

#include <iostream>
#include <memory>

namespace rocRoller
{
    class Command;
    using CommandPtr = std::shared_ptr<Command>;

    std::ostream& operator<<(std::ostream& stream, Command const& command);
}
