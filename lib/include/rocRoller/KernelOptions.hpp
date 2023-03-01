#pragma once

#include <ostream>
#include <string>

#include "Utilities/Settings_fwd.hpp"

namespace rocRoller
{
    const std::string XLOOP = "XLoop";
    const std::string YLOOP = "YLoop";
    const std::string KLOOP = "KLoop";

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
        unsigned int unrollK;

        bool fuseLoops;

        bool transposeMemoryAccessA;
        bool transposeMemoryAccessB;
        bool transposeMemoryAccessOther;

        bool assertWaitCntState;

        bool packMultipleElementsInto1VGPR;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
