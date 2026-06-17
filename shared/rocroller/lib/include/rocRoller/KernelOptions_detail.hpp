// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "KernelOptions.hpp"
#include <rocRoller/Parameters/Solution/LDSBankSwizzleMode.hpp>
#include <rocRoller/Parameters/Solution/ScaleSkipPermlaneMode.hpp>

namespace rocRoller
{
    struct KernelOptionValues
    {
        LogLevel logLevel = LogLevel::Verbose;

        bool alwaysWaitAfterLoad         = false;
        bool alwaysWaitAfterStore        = false;
        bool alwaysWaitBeforeBranch      = false;
        bool alwaysWaitZeroBeforeBarrier = false;

        /**
         * Maximum limit on the number of kernel arguments we will allow to be preloaded into
         * SGPRs by the GPU at the beginning of the kernel. If it is set to -1, this is
         * unlimited and we will use the maximum defined by the architecture.
         */
        int systemPreloadedKernelArguments = -1;

        /**
         * If enabled, kernel arguments that are not preloaded by the system will be lazily
         * loaded as they are needed in the kernel. Note that this does not currently work well
         * for complex kernels as we cannot efficiently overlap kernel argument loading with
         * other memory traffic.
         */
        bool lazyLoadKernelArguments = false;

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
        int minLaunchTimeExpressionComplexity = 20;

        /**
         * The maximum number of concurrent subexpressions given at once to the scheduler when
         * generating code for an expression using CSE. A higher number may reveal more ILP
         * opportunities to the scheduler but may also result in higher register usage.
         *
         * This is counted per result type of subexpression. So 2 means 2 SGPR subexpressions
         * AND 2 VGPR subexpressions at once.
         *
         * If you are running out of registers (particularly SGPRs), reducing this number
         * might help.
         */
        int maxConcurrentSubExpressions = 1;

        /**
         * The maximum number of concurrent control operations given at once to the scheduler
         * when generating code in LowerFromKernelGraph.
         *
         * This is counted per node type.
         *
         * If this is empty, it will revert to the original algorithm where nodes are not
         * separated into categories.
         */
        std::optional<int> maxConcurrentControlOps;

        /**
         * Choose the LDS Memory observer to use for scheduling.
         *
         * Affects cycle count predictions for LDS instructions.
         */
        DSObserverType dsObserver = DSObserverType::DSMEMObserver;

        /**
         * By default, we no longer allow full integer division or modulo in
         * kernels.  If this is needed for testing or some other reason, it can
         * be enabled via this option.
         */
        bool enableFullDivision = false;

        /**
         * Skip generation of permlane instructions when loading scale data.
         * This is experimental and requires that the input be specifically
         * modified, but will show better performance.
         */
        ScaleSkipPermlaneMode scaleSkipPermlane = ScaleSkipPermlaneMode::None;

        /**
         * Which method to use to crash the kernel if an assertion fails.
         */
        AssertOpKind assertOpKind = AssertOpKind::NoOp;

        /**
         * Enable/Disable the RemoveSetCoordinate transformation
         */
        bool removeSetCoordinate = false;

        /**
         * LDS bank conflict elimination via intra-wave column rotation + pair-swap swizzle.
         * When set to Swizzle, remaps K-column indices on
         * both the LoadTiled (write) and LoadLDSTile (read) sides to eliminate ds_read_b128 bank conflicts.
         */
        LDSBankSwizzleMode ldsSwizzleMode = LDSBankSwizzleMode::None;

        bool coexecutionEnabled = true;

        std::optional<std::array<unsigned int, 3>> workgroupClusterSize;

        /**
         * By default, v_(mfma|wmma)_*_f8f6f4 instructions are used for F8 datatypes
         * with compatible wavetile sizes. Setting this option to false generates
         * v_(mfma|wmma)_*_(fp8|bf8)_(fp8|bf8) instead when available.
         */
        bool favourF8F6F4OverF8MatrixInstruction = true;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, KernelOptionValues const& values);
    std::string   toString(KernelOptionValues const& values);

}
