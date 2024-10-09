#pragma once

#include <memory>

namespace rocRoller
{
    class CommandParameters;
    class CommandLaunchParameters;
    class CommandKernel;
    class CommandSolution;

    using CommandParametersPtr       = std::shared_ptr<CommandParameters>;
    using CommandKernelPtr           = std::shared_ptr<CommandKernel>;
    using CommandLaunchParametersPtr = std::shared_ptr<CommandLaunchParameters>;
}
