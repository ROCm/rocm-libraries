#pragma once

#include <memory>

#include "AssemblyKernel.hpp"
#include "ExecutableKernel_fwd.hpp"
#include "KernelArguments.hpp"
#include "Operations/Command_fwd.hpp"
#include "Utilities/HIPTimer.hpp"

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
        void setDimensionInfo(int tag, KernelGraph::CoordinateGraph::Dimension dim);
        std::map<int, KernelGraph::CoordinateGraph::Dimension> getDimensionInfo() const;

        /**
         * Manually override kernel launch dimensions.
         *
         * TODO remove this.
         */
        void setManualKernelDimension(int dim);
        int  getManualKernelDimension() const;

        /**
         * Manually overrite workgroup sizes.
         *
         * TODO remove this.
         */
        void setManualWorkgroupSize(std::array<unsigned int, 3> const&);
        std::optional<std::array<unsigned int, 3>> getManualWorkgroupSize() const;

        /**
         * Manually override work item counts.
         *
         * TODO remove this.
         */
        void setManualWorkitemCount(std::array<Expression::ExpressionPtr, 3> const&);
        std::optional<std::array<Expression::ExpressionPtr, 3>> getManualWorkitemCount() const;

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

    private:
        std::map<int, KernelGraph::CoordinateGraph::Dimension>  m_dimInfo;
        std::optional<std::array<unsigned int, 3>>              m_workgroupSize;
        std::optional<std::array<Expression::ExpressionPtr, 3>> m_workitemCount;
        std::optional<std::pair<uint, uint>>                    m_wavefrontCounts;

        int m_kernelDimension = 0;

        std::vector<unsigned int> m_waveTilesPerWavefront;

        bool m_splitStoreTileIntoWaveBlocks = true;
    };

    class CommandKernel
    {
    public:
        /**
         * Initialize a CommandKernel with a Context. When this is done, instructions
         * should have already been scheduled within the context. This is probably
         * only useful for creating unit tests.
         */
        CommandKernel(ContextPtr);

        /**
         * Create a CommandKernel based on a Command object. This will generate
         * a kernel and allow the launchKernel method to be called.
         */
        CommandKernel(CommandPtr command, std::string name);

        /**
         * Create a CommandKernel based on a Command object. This will
         * generate a kernel and allow the launchKernel method to be
         * called.
         *
         * @param preParams CommandParameters applied to the kernel
         * graph after translation from a command to a graph but
         * before all lowering stages.
         *
         * @param postParams CommandParameters applied to the kernel
         * graph after all lowering stages but before code-generation.
         */
        // TODO: When we have a more robust way of parameterizing the
        // graph, the two sets of command parameters might be
        // unnecessary.  Currently we need two sets because:
        // 1. before lowering you need to specify, eg, tile sizes
        // 2. graph ids/indexing change during lowering.
        // 3. during lowering new nodes might be created that didn't before
        // 4. before code-generating you need to specify, eg, how many wavefronts
        CommandKernel(CommandPtr                         command,
                      std::string                        name,
                      std::shared_ptr<CommandParameters> preParams,
                      std::shared_ptr<CommandParameters> postParams    = nullptr,
                      std::shared_ptr<KernelOptions>     kernelOptions = nullptr);

        CommandKernel(CommandPtr                      command,
                      ContextPtr                      ctx,
                      KernelGraph::KernelGraph const& kernelGraph);

        void addPredicate(Expression::ExpressionPtr expression);
        bool matchesPredicates(/* args */) const;

        /**
         * Load (and compile) a kernel from the assembly source file `fileName`.
         */
        void loadKernelFromAssembly(std::string const& fileName, std::string const& kernelName);

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel.
         *
         * @param args The runtime arguments being passed to the kernel
        */
        void launchKernel(RuntimeArguments const& args);

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel.
         *
         * @param args The runtime arguments being passed to the kernel
         * @param timer HIPTimer that will record how long the kernel took to execute
         * @param iteration Iteration number within the timer
        */
        void launchKernel(RuntimeArguments const&   args,
                          std::shared_ptr<HIPTimer> timer,
                          int                       iteration);

        KernelGraph::KernelGraph getKernelGraph() const;

        std::string getInstructions() const;

        ContextPtr getContext();

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
        size_t scratchSpaceRequired() const;

        /**
         * @brief Returns the hipFunction for the kernel
        */
        hipFunction_t getHipFunction() const;

    private:
        CommandPtr m_command;

        std::vector<Expression::ExpressionPtr> m_predicates;

        KernelGraph::KernelGraph           m_kernelGraph;
        ContextPtr                         m_context;
        std::shared_ptr<ExecutableKernel>  m_executableKernel;
        std::shared_ptr<CommandParameters> m_preParameters, m_postParameters;
        std::shared_ptr<KernelOptions>     m_kernelOptions;

        Generator<Instruction> commandComments();

        /**
         * Generates the whole kernel, assembles it, and records the
         * info needed to launch the kernel, given command arguments.
         */
        void generateKernel(std::string);

        void generateKernelGraph(std::string name);
        void generateKernelSource();

        Generator<Instruction> kernelInstructions();

        void assembleKernel();
    };

    class CommandSolution
    {
    public:
        CommandSolution(CommandPtr command);

        void appendKernel(std::shared_ptr<CommandKernel> kernel);
        std::vector<std::shared_ptr<CommandKernel>> const& kernels() const;

        void generateKernels();

        void launchKernels();
    };
}
