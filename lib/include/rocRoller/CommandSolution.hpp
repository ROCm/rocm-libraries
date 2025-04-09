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

#include <memory>

#include <rocRoller/CommandSolution_fwd.hpp>

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/ExecutableKernel_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Utilities/EnumBitset.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>

namespace rocRoller
{
    KernelArguments  getKernelArguments(AssemblyKernelPtr kernel, RuntimeArguments const& args);
    KernelInvocation getKernelInvocation(AssemblyKernelPtr kernel, RuntimeArguments const& args);

    /**
     * CommandParameters - tunable command parameters.
     */
    class CommandParameters
    {
    public:
        CommandParameters();

        /**
         * Set (reset) a dimensions properties.
         */
        void setDimensionInfo(Operations::OperationTag                tag,
                              KernelGraph::CoordinateGraph::Dimension dim);
        std::map<Operations::OperationTag, KernelGraph::CoordinateGraph::Dimension>
            getDimensionInfo() const;

        /**
         * Manually override kernel launch dimensions.
         *
         * TODO remove this.
         */
        void setManualKernelDimension(int dim);
        int  getManualKernelDimension() const;

        /**
         * Manually override workgroup sizes.
         *
         * TODO remove this.
         */
        void setManualWorkgroupSize(std::array<unsigned int, 3> const&);
        std::optional<std::array<unsigned int, 3>> getManualWorkgroupSize() const;

        void setManualWavefrontCount(std::pair<uint, uint> wavefrontCounts);
        std::optional<std::pair<uint, uint>> getManualWavefrontCounts() const;

        /**
         * @brief Set the number of wave tiles to execute within a workgroup
         *
         * @param x Number of wave tiles to execute in the x dimension
         * @param y Number of wave tiles to execute in the y dimension
         */
        void setWaveTilesPerWavefront(unsigned int x = 1, unsigned int y = 1);

        /**
         * @brief Get the number of wave tiles to execute within a workgroup
         *
         * @return std::vector<unsigned int>
         */
        std::vector<unsigned int> getWaveTilesPerWavefront() const;

        /**
         * @brief Enable/disable wave-storage strategy.
         *
         * When storing through LDS; the LDS traffic is done following
         * the MFMA accumulator layout intead of straight threads.
         */
        void setSplitStoreTileIntoWaveBlocks(bool);
        bool getSplitStoreTileIntoWaveBlocks() const;

        std::string toString() const;

        /**
         * Lowering parameters.
         */
        bool allowAmbiguousMemoryNodes   = false;
        bool enableLongDwordInstructions = true;

        EnumBitset<LayoutType> transposeMemoryAccess = {};

        bool packMultipleElementsInto1VGPR = true;

        unsigned int unrollX   = 0;
        unsigned int unrollY   = 0;
        unsigned int unrollK   = 0;
        bool         fuseLoops = true;
        bool         tailLoops = true;

        bool swizzleScale = false;

        bool prefetch          = false;
        int  prefetchInFlight  = 1;
        int  prefetchLDSFactor = 0;
        bool prefetchMixMemOps = false;

        bool streamK        = false;
        bool streamKTwoTile = false;

        std::vector<int>  loopOverOutputTilesDimensions    = {};
        std::string       loopOverOutputTilesTopLoop       = XLOOP;
        std::vector<uint> loopOverOutputTilesCoordSizes    = {};
        uint              loopOverOutputTilesIteratedTiles = 0;

    private:
        std::map<Operations::OperationTag, KernelGraph::CoordinateGraph::Dimension> m_dimInfo;
        std::optional<std::array<unsigned int, 3>>                                  m_workgroupSize;
        std::optional<std::pair<uint, uint>> m_wavefrontCounts;

        int m_kernelDimension = 0;

        std::vector<unsigned int> m_waveTilesPerWavefront;

        bool m_splitStoreTileIntoWaveBlocks = true;
    };

    /**
     * CommandLaunchParameters - manual kernel launch parameters.
     *
     * TODO: Remove this!
     */
    class CommandLaunchParameters
    {
    public:
        CommandLaunchParameters() = default;

        /**
         * Manually override work item counts.
         */
        void setManualWorkitemCount(std::array<Expression::ExpressionPtr, 3> const&);
        std::optional<std::array<Expression::ExpressionPtr, 3>> getManualWorkitemCount() const;

    private:
        std::optional<std::array<Expression::ExpressionPtr, 3>> m_workitemCount;
    };

    class CommandKernel
    {
    public:
        CommandKernel() = default;

        /**
         * @brief Create a CommandKernel based on a Command object.
         *
         * Callers should set the context, parameters, and then call
         * `generateKernel`.
         */
        CommandKernel(CommandPtr command, std::string kernelName);

        /**
         * @brief Set context.
         *
         * This must be called before `generateKernel`.
         */
        void setContext(ContextPtr);

        /**
         * @brief CommandParameters applied to the kernel graph after
         * translation from a command to a graph but before all
         * lowering stages.
         */
        void                 setCommandParameters(CommandParametersPtr commandParams);
        CommandParametersPtr getCommandParameters() const;

        /**
         * @brief Generates the kernel (does graph lowering and
         * code-generation).
         */
        void generateKernel();

        /**
         * @brief Assembles a generated kernel.  Does not try to load
         * it.
         */
        std::vector<char> assembleKernel();

        /**
         * @brief Assembles and loads a generated kernel onto the GPU.
         */
        void loadKernel();

        /**
         * @brief Add an expression predicate to be evaluated using the runtime arguments.
         *
         * @param expression The expression predicate to add.
         */
        void addPredicate(Expression::ExpressionPtr expression);

        /**
         * @brief Evaluate all added predicates, return AND of all results.
         *
         * @param args The runtime arguments which will be passed to the kernel.
         * @param level The log level to log any predicate mismatch messages to (defaults to debug)
         */
        bool matchesPredicates(RuntimeArguments const& args,
                               LogLevel                level = LogLevel::Debug) const;

        /**
         * @brief Set (manual) launch parameters.
         */
        void setLaunchParameters(CommandLaunchParametersPtr);

        /**
         * Load (and compile) a kernel from the assembly source file `fileName`.
         */
        void loadKernelFromAssembly(std::string const& fileName, std::string const& kernelName);

        /**
         * Load a kernel from the code-object file `fileName`.
         */
        AssemblyKernelPtr loadKernelFromCodeObject(std::string const& fileName,
                                                   std::string const& kernelName);

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel.
         *
         * @param args The runtime arguments being passed to the kernel
         * @param timer HIPTimer that will record how long the kernel took to execute
         * @param iteration Iteration number within the timer
         *
         * Assembles and loads a generated kernel if this has not been
         * done already.
         */
        void launchKernel(RuntimeArguments const&   args,
                          std::shared_ptr<HIPTimer> timer,
                          int                       iteration);

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel on
         *        the given stream..
         *
         * @param args The runtime arguments being passed to the kernel
         * @param stream The stream that the kernel to run on
         *
         * Assembles and loads a generated kernel if this has not been
         * done already.
         */
        void launchKernel(RuntimeArguments const& args, hipStream_t stream = 0);

        KernelGraph::KernelGraphPtr getKernelGraph() const;

        std::string getInstructions() const;

        ContextPtr getContext();

        /**
         * @brief Get the Command object
         *
         * @return CommandPtr
         */
        CommandPtr getCommand() const;

        /**
         * Determines kernel arguments for a particular invocation.
         */
        KernelArguments getKernelArguments(RuntimeArguments const& args);

        /**
         * Determines launch bounds for a particular invocation.
         */
        KernelInvocation getKernelInvocation(RuntimeArguments const& args);

        /**
         * @brief Returns the total number of bytes required for scratch space
         *
         * If this value is greather than 0, the user is required to allocate this
         * amount of device memory and pass it into the kernel.
         *
         * @return size_t
         */
        size_t scratchSpaceRequired(RuntimeArguments const& args) const;

        /**
         * @brief Returns the workgroup size
         */
        std::array<unsigned int, 3> const& getWorkgroupSize() const;

        /**
         * @brief Returns the hipFunction for the kernel
         */
        hipFunction_t getHipFunction() const;

    private:
        CommandPtr  m_command;
        std::string m_name;

        std::vector<Expression::ExpressionPtr> m_predicates;

        KernelGraph::KernelGraphPtr       m_kernelGraph;
        ContextPtr                        m_context;
        std::shared_ptr<ExecutableKernel> m_executableKernel;
        CommandParametersPtr              m_commandParameters;
        CommandLaunchParametersPtr        m_launchParameters;

        void generateKernelGraph(std::string name);
        void generateKernelSource();

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel with
         *        optional timer and stream
         *
         * @param args The runtime arguments being passed to the kernel
         * @param timer HIPTimer that will record how long the kernel took to execute
         * @param iteration Iteration number within the timer
         * @param stream The stream that the kernel to run on
         *
         * Assembles and loads a generated kernel if this has not been
         * done already.
         */
        void launchKernel(RuntimeArguments const&   args,
                          std::shared_ptr<HIPTimer> timer,
                          int                       iteration,
                          hipStream_t               stream);
    };

    class CommandSolution
    {
    public:
        explicit CommandSolution(CommandPtr command);

        void                                 appendKernel(CommandKernelPtr kernel);
        std::vector<CommandKernelPtr> const& kernels() const;

        void generateKernels();

        void launchKernels();
    };
}
