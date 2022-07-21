/**
 */

#pragma once

#include <memory>

namespace rocRoller
{
    class AssemblyKernel;
    struct AssemblyKernelArgument;

    using AssemblyKernelArgumentPtr = std::shared_ptr<AssemblyKernelArgument>;

}
