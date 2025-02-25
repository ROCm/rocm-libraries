#pragma once

#include <memory>

namespace rocRoller
{
    class CommandKernel;
    class CommandLaunchParameters;
    class CommandParameters;
    class CommandSolution;

    using CommandKernelPtr           = std::shared_ptr<CommandKernel>;
    using CommandLaunchParametersPtr = std::shared_ptr<CommandLaunchParameters>;
    using CommandParametersPtr       = std::shared_ptr<CommandParameters>;
}
