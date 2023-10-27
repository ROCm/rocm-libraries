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
        LogLevel logLevel = LogLevel::Verbose;

        bool alwaysWaitAfterLoad         = false;
        bool alwaysWaitAfterStore        = false;
        bool alwaysWaitBeforeBranch      = false;
        bool alwaysWaitZeroBeforeBarrier = false;

        bool preloadKernelArguments = true;

        unsigned int maxACCVGPRs      = 256;
        unsigned int maxSGPRs         = 102;
        unsigned int maxVGPRs         = 256;
        unsigned int loadLocalWidth   = 4;
        unsigned int loadGlobalWidth  = 8;
        unsigned int storeLocalWidth  = 4;
        unsigned int storeGlobalWidth = 4;
        unsigned int unrollX          = 0;
        unsigned int unrollY          = 0;
        unsigned int unrollK          = 0;

        bool fuseLoops                 = true;
        bool allowAmbiguousMemoryNodes = false;

        bool prefetch          = false;
        int  prefetchInFlight  = 1;
        int  prefetchLDSFactor = 0;
        bool prefetchMixMemOps = false;

        bool streamK        = false;
        bool streamKTwoTile = false;

        std::vector<int>  loopOverOutputTilesDimensions = {};
        std::string       loopOverOutputTilesTopLoop    = XLOOP;
        std::vector<uint> loopOverOutputTilesCoordSizes = {};
        uint              loopOverOutputTilesIteratedTiles;

        uint numScratchTiles = 0;

        EnumBitset<LayoutType> transposeMemoryAccess = {};

        bool assertWaitCntState = true;

        bool packMultipleElementsInto1VGPR = true;
        bool enableLongDwordInstructions   = true;
        bool setNextFreeVGPRToMax          = false;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
