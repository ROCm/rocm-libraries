#pragma once

#include <ostream>
#include <string>

#include <rocRoller/DataTypes/DataTypes.hpp>

#include <rocRoller/AssertOpKinds_fwd.hpp>
#include <rocRoller/Utilities/EnumBitset.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

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

        bool assertWaitCntState = true;

        bool setNextFreeVGPRToMax = false;

        /**
         * These two are expected to become permanently enabled;
         */

        /**
         * If enabled, when adding a kernel argument, we will check all currently existing
         * arguments for one with an equivalent expression. If one exists, no new argument is
         * added and we will return the existing one instead.
         */
        bool deduplicateArguments = true;

        /**
         * If enabled, command arguments are not necessarily added as kernel arguments.  We
         * instead depend on the CleanArguments and other passes to add all necessary kernel
         * arguments.
         */
        bool lazyAddArguments = true;

        /**
         * The minimum complexity of an expression before we will add a kernel argument to
         * calculate its value on the CPU before launch.  This is a very rough heuristic for
         * now, and doesn't (yet) take into account different datatypes or different
         * architectures.
         *
         * Magic division includes a subexpression of complexity 8, so if this number is <= 8,
         * there will be a fourth kernel argument for every magic division denominator.
         *
         * Increasing this value could decrease SGPR pressure; decreasing it could speed up a
         * kernel if there are enough available SGPRs.
         */
        int minLaunchTimeExpressionComplexity = 10;

        AssertOpKind assertOpKind = AssertOpKind::NoOp;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);
    };
}
