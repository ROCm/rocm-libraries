#pragma once

#include <ostream>
#include <string>

#include "Utilities/Settings_fwd.hpp"

namespace rocRoller
{
    struct KernelOptions
    {
        KernelOptions();

        LogLevel logLevel;
        bool     alwaysWaitAfterLoad;
        bool     alwaysWaitAfterStore;
        bool     alwaysWaitBeforeBranch;
        bool     preloadKernelArguments;

        unsigned int loadLocalWidth;
        unsigned int loadGlobalWidth;
        unsigned int storeLocalWidth;
        unsigned int storeGlobalWidth;
        unsigned int unrollX;
        unsigned int unrollY;

        bool fuseLoops;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
