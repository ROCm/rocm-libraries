#pragma once

#include "Utilities/Utils.hpp"

namespace rocRoller
{
    struct KernelOptions
    {
        KernelOptions();

        LogLevel logLevel;
        bool     alwaysWaitAfterLoad;
        bool     alwaysWaitAfterStore;
        bool     preloadKernelArguments;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
