#pragma once

#include <memory>

#include "AssemblyKernel.hpp"
#include "ExecutableKernel_fwd.hpp"
#include "KernelArguments.hpp"
#include "KernelGraph/CoordinateTransform/Dimension.hpp"
#include "Operations/Command_fwd.hpp"

namespace rocRoller
{
    KernelArguments  getKernelArguments(std::shared_ptr<AssemblyKernel> kernel,
                                        RuntimeArguments const&         args);
    KernelInvocation getKernelInvocation(std::shared_ptr<AssemblyKernel> kernel,
                                         RuntimeArguments const&         args);

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
        void setDimensionInfo(KernelGraph::CoordinateTransform::Dimension dim);
        std::vector<KernelGraph::CoordinateTransform::Dimension> getDimensionInfo() const;

        // TODO Delete above when graph rearch complete
        void setDimensionInfo(int tag, KernelGraph::CoordGraph::Dimension dim);
        // TODO Rename this when graph rearch complete
        std::map<int, KernelGraph::CoordGraph::Dimension> getDimensionInfo2() const;

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
        std::vector<unsigned int> getWaveTilesPerWorkgroup() const;

    private:
        std::vector<KernelGraph::CoordinateTransform::Dimension> m_dimInfo;
        // TODO Delete above when graph rearch complete
        // TODO Rename this when graph rearch complete
        std::map<int, KernelGraph::CoordGraph::Dimension>       m_dimInfo2;
        std::optional<std::array<unsigned int, 3>>              m_workgroupSize;
        std::optional<std::array<Expression::ExpressionPtr, 3>> m_workitemCount;

        int m_kernelDimension = 0;

        std::vector<unsigned int> m_waveTilesPerWorkgroup;
    };

    class CommandKernel
    {
    public:
        /**
         * Initialize a CommandKernel with a Context. When this is done, instructions
         * should have already been scheduled within the context. This is probably
         * only useful for creating unit tests.
         */
        CommandKernel(std::shared_ptr<Context>);

        /**
         * Create a CommandKernel based on a Command object. This will generate
         * a kernel and allow the launchKernel method to be called.
         */
        CommandKernel(std::shared_ptr<Command> command, std::string name);

        CommandKernel(std::shared_ptr<Command>           command,
                      std::string                        name,
                      std::shared_ptr<CommandParameters> params);

        CommandKernel(std::shared_ptr<Command>        command,
                      std::shared_ptr<Context>        ctx,
                      KernelGraph::KernelGraph const& kernelGraph);

        void addPredicate(Expression::ExpressionPtr expression);
        bool matchesPredicates(/* args */) const;

        /**
         * Load (and compile) a kernel from the assembly source file `fileName`.
         */
        void loadKernelFromAssembly(std::string const& fileName, std::string const& kernelName);

        /**
         * Determines launch bounds and arguments, and launches the kernel.
         */
        void launchKernel(RuntimeArguments const& args);

        KernelGraph::KernelGraph getKernelGraph() const;

        std::shared_ptr<Context> getContext();

    private:
        std::shared_ptr<Command> m_command;

        std::vector<Expression::ExpressionPtr> m_predicates;

        KernelGraph::KernelGraph           m_kernelGraph;
        std::shared_ptr<Context>           m_context;
        std::shared_ptr<ExecutableKernel>  m_executableKernel;
        std::shared_ptr<CommandParameters> m_parameters;

        KernelArguments  getKernelArguments(RuntimeArguments const& args);
        KernelInvocation getKernelInvocation(RuntimeArguments const& args);

        Generator<Instruction> commandComments();

        /**
         * Generates the whole kernel, assembles it, and records the
         * info needed to launch the kernel, given command arguments.
         */
        void generateKernel(std::string);

        void generateKernelGraph(std::string name);
        void generateKernelSource();
        void assembleKernel();
    };

    class CommandSolution
    {
    public:
        CommandSolution(std::shared_ptr<Command> command);

        void appendKernel(std::shared_ptr<CommandKernel> kernel);
        std::vector<std::shared_ptr<CommandKernel>> const& kernels() const;

        void generateKernels();

        void launchKernels();
    };
}

#include "CommandSolution_impl.hpp"
