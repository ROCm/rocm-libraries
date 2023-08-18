#pragma once

#include <ostream>
#include <string>

#include <rocRoller/DataTypes/DataTypes.hpp>

#include "Utilities/EnumBitset.hpp"
#include "Utilities/Settings_fwd.hpp"

namespace rocRoller
{
    const std::string XLOOP = "XLoop";
    const std::string YLOOP = "YLoop";
    const std::string KLOOP = "KLoop";

    const std::string SCRATCH = "SCRATCH";

    struct KernelOptions
    {
        KernelOptions();

        LogLevel logLevel;

        bool alwaysWaitAfterLoad;
        bool alwaysWaitAfterStore;
        bool alwaysWaitBeforeBranch;
        bool alwaysWaitZeroBeforeBarrier;

        bool preloadKernelArguments;

        unsigned int maxACCVGPRs;
        unsigned int maxSGPRs;
        unsigned int maxVGPRs;
        unsigned int loadLocalWidth;
        unsigned int loadGlobalWidth;
        unsigned int storeLocalWidth;
        unsigned int storeGlobalWidth;
        unsigned int unrollX;
        unsigned int unrollY;
        unsigned int unrollK;

        bool fuseLoops;
        bool allowAmbiguousMemoryNodes;

        bool prefetch;
        int  prefetchInFlight;
        int  prefetchLDSFactor;
        bool prefetchMixMemOps;

        bool streamK;

        std::vector<int>  loopOverOutputTilesDimensions;
        std::string       loopOverOutputTilesTopLoop;
        std::vector<uint> loopOverOutputTilesCoordSizes;
        uint              loopOverOutputTilesIteratedTiles;

        bool enableScratch;
        uint numScratchTiles;

        EnumBitset<LayoutType> transposeMemoryAccess;

        bool assertWaitCntState;

        bool packMultipleElementsInto1VGPR;
        bool enableLongDwordInstructions;
        bool setNextFreeVGPRToMax;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
