/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "KernelOptions.hpp"

namespace rocRoller
{
    struct KernelOptionValues
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
        int maxConcurrentSubExpressions = 2;

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
         * By default, we no longer allow full integer division or modulo in
         * kernels.  If this is needed for testing or some other reason, it can
         * be enabled via this option.
         */
        bool enableFullDivision = false;

        /**
         * Skip generation of permlane instructions when loading scale data.
         * This is experimental and will cause the validation to fail, but will
         * show better performance.
         */
        bool scaleSkipPermlane = false;

        /**
         * Which method to use to crash the kernel if an assertion fails.
         */
        AssertOpKind assertOpKind = AssertOpKind::NoOp;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, KernelOptionValues const& values);
    std::string   toString(KernelOptionValues const& values);

}
