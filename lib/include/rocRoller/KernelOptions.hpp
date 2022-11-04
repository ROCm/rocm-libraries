#pragma once

#include "Utilities/Settings.hpp"

namespace rocRoller
{
    struct KernelOptions
    {
        KernelOptions();

        Settings::LogLevel logLevel;
        bool               alwaysWaitAfterLoad;
        bool               alwaysWaitAfterStore;
        bool               alwaysWaitBeforeBranch;
        bool               preloadKernelArguments;

        unsigned int loadLocalWidth;
        unsigned int loadGlobalWidth;
        unsigned int storeLocalWidth;
        unsigned int storeGlobalWidth;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
