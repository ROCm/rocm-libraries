#pragma once

#include <memory>

#include "CommandSolution.hpp"

#include "ExecutableKernel.hpp"

#include "AssemblyKernel.hpp"
#include "KernelArguments.hpp"
#include "Operations/Command.hpp"
#include "Scheduling/Scheduler.hpp"
#include "Utilities/Settings_fwd.hpp"
#include "Utilities/Timer.hpp"

namespace rocRoller
{
    inline CommandParameters::CommandParameters()
        : m_waveTilesPerWorkgroup({1, 1})
    {
    }

    // TODO Delete this when graph rearch complete
    inline void CommandParameters::setDimensionInfo(KernelGraph::CoordinateTransform::Dimension dim)
    {
        m_dimInfo.push_back(dim);
    }

    // TODO Delete this when graph rearch complete
    inline std::vector<KernelGraph::CoordinateTransform::Dimension>
        CommandParameters::getDimensionInfo() const
    {
        return m_dimInfo;
    }

    // TODO Rename this when graph rearch complete
    inline void CommandParameters::setDimensionInfo(int tag, KernelGraph::CoordGraph::Dimension dim)
    {
        m_dimInfo2[tag] = dim;
    }

    // TODO Rename this when graph rearch complete
    inline std::map<int, KernelGraph::CoordGraph::Dimension>
        CommandParameters::getDimensionInfo2() const
    {
        return m_dimInfo2;
    }

    inline int CommandParameters::getManualKernelDimension() const
    {
        return m_kernelDimension;
    }

    inline void CommandParameters::setManualKernelDimension(int dim)
    {
        m_kernelDimension = dim;
    }

    inline void CommandParameters::setManualWorkgroupSize(std::array<unsigned int, 3> const& v)
    {
        m_workgroupSize = v;
    }

    inline std::optional<std::array<unsigned int, 3>>
        CommandParameters::getManualWorkgroupSize() const
    {
        return m_workgroupSize;
    }

    inline void
        CommandParameters::setManualWorkitemCount(std::array<Expression::ExpressionPtr, 3> const& v)
    {
        m_workitemCount = v;
    }

    inline std::optional<std::array<Expression::ExpressionPtr, 3>>
        CommandParameters::getManualWorkitemCount() const
    {
        return m_workitemCount;
    }

    inline void CommandParameters::setWaveTilesPerWavefront(unsigned int x, unsigned int y)
    {
        m_waveTilesPerWorkgroup[0] = x;
        m_waveTilesPerWorkgroup[1] = y;
    }

    inline std::vector<unsigned int> CommandParameters::getWaveTilesPerWorkgroup() const
    {
        return m_waveTilesPerWorkgroup;
    }

    inline KernelArguments CommandKernel::getKernelArguments(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::getKernelArguments");

        bool            log = m_context->kernelOptions().logLevel >= LogLevel::Debug;
        KernelArguments rv(log);

        auto const& argStructs = m_context->kernel()->arguments();

        rv.reserve(m_context->kernel()->argumentSize(), argStructs.size());

        for(auto& arg : argStructs)
        {
            auto value = Expression::evaluate(arg.expression, args);

            if(variableType(value) != arg.variableType)
            {
                throw std::runtime_error(concatenate("Evaluated argument type ",
                                                     variableType(value),
                                                     " doesn't match expected type ",
                                                     arg.variableType));
            }

            rv.append(arg.name, value);
        }

        return rv;
    }

    inline KernelInvocation CommandKernel::getKernelInvocation(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::getKernelInvocation");

        KernelInvocation rv;

        rv.workgroupSize = m_context->kernel()->workgroupSize();

        auto const& workitems = m_context->kernel()->workitemCount();
        if(workitems[0])
            rv.workitemCount[0] = getUnsignedInt(evaluate(workitems[0], args));
        if(workitems[1])
            rv.workitemCount[1] = getUnsignedInt(evaluate(workitems[1], args));
        if(workitems[2])
            rv.workitemCount[2] = getUnsignedInt(evaluate(workitems[2], args));

        auto const& sharedMem = m_context->kernel()->dynamicSharedMemBytes();
        if(sharedMem)
            rv.sharedMemBytes = getUnsignedInt(evaluate(sharedMem, args));

        return rv;
    }

    inline CommandKernel::CommandKernel(std::shared_ptr<Context> context)
        : m_context(context)
    {
    }

    inline CommandKernel::CommandKernel(std::shared_ptr<Command> command, std::string name)
        : m_command(command)
        , m_parameters(std::make_shared<CommandParameters>())
    {
        generateKernel(name);
    }

    inline CommandKernel::CommandKernel(std::shared_ptr<Command>           command,
                                        std::string                        name,
                                        std::shared_ptr<CommandParameters> parameters)
        : m_command(command)
        , m_parameters(parameters)
    {
        AssertFatal(m_parameters);

        generateKernel(name);
    }

    inline CommandKernel::CommandKernel(std::shared_ptr<Command>        command,
                                        std::shared_ptr<Context>        context,
                                        KernelGraph::KernelGraph const& kernelGraph)
        : m_command(command)
        , m_context(context)
        , m_kernelGraph(kernelGraph)
        , m_parameters(std::make_shared<CommandParameters>())
    {
        generateKernelSource();
        assembleKernel();
    }

    inline KernelGraph::KernelGraph CommandKernel::getKernelGraph() const
    {
        return m_kernelGraph;
    }

    inline Generator<Instruction> CommandKernel::commandComments()
    {
        co_yield Instruction::Comment(m_command->toString());
        co_yield Instruction::Comment(m_command->argInfo());
    }

    inline void CommandKernel::generateKernelGraph(std::string name)
    {
        TIMER(t, "CommandKernel::generateKernelGraph");

        auto logger = rocRoller::Log::getLogger();

        m_context = Context::ForDefaultHipDevice(name);

        // TODO: Determine the correct kernel dimensions
        if(m_parameters->getManualKernelDimension() > 0)
            m_context->kernel()->setKernelDimensions(m_parameters->getManualKernelDimension());
        else
            m_context->kernel()->setKernelDimensions(1);

        // TODO: Determine the correct work group size
        if(m_parameters->getManualWorkgroupSize())
            m_context->kernel()->setWorkgroupSize(*m_parameters->getManualWorkgroupSize());
        else
            m_context->kernel()->setWorkgroupSize({64, 1, 1});

        if(m_parameters->getManualWorkitemCount())
            m_context->kernel()->setWorkitemCount(*m_parameters->getManualWorkitemCount());
        else
            m_context->kernel()->setWorkitemCount(m_command->createWorkItemCount());

        auto zero = std::make_shared<Expression::Expression>(0u);
        m_context->kernel()->setDynamicSharedMemBytes(zero);

        m_context->kernel()->addCommandArguments(m_command->getArguments());

        m_kernelGraph = KernelGraph::translate(m_command);
        logger->debug("CommandKernel::generateKernel: post translate: {}", m_kernelGraph.toDOT());
        m_kernelGraph = updateParameters(m_kernelGraph, m_parameters);

        m_kernelGraph = KernelGraph::lowerLinear(m_kernelGraph, m_context);

        m_kernelGraph = KernelGraph::lowerTile(m_kernelGraph, m_parameters, m_context);

        m_kernelGraph = KernelGraph::cleanArguments(m_kernelGraph, m_context->kernel());

        m_kernelGraph = updateParameters(m_kernelGraph, m_parameters);
    }

    inline void CommandKernel::generateKernelSource()
    {
        TIMER(t, "CommandKernel::generateKernelSource");
        m_context->kernel()->setKernelGraphMeta(
            std::make_shared<KernelGraph::KernelGraph>(m_kernelGraph));

        // A sequential scheduler should always be used here to ensure
        // the parts of the kernel are yielded in the correct order.
        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, m_context);

        std::vector<Generator<Instruction>> generators;

        generators.push_back(commandComments());
        generators.push_back(m_context->kernel()->preamble());
        generators.push_back(m_context->kernel()->prolog());

        generators.push_back(KernelGraph::generate(m_kernelGraph, m_command, m_context->kernel()));

        generators.push_back(m_context->kernel()->postamble());
        generators.push_back(m_context->kernel()->amdgpu_metadata());

        m_context->schedule((*scheduler)(generators));
    }

    inline void CommandKernel::assembleKernel()
    {
        TIMER(t, "CommandKernel::assembleKernel");

        m_executableKernel = m_context->instructions()->getExecutableKernel();
    }

    inline void CommandKernel::generateKernel(std::string name)
    {
        TIMER(t, "CommandKernel::generateKernel");

        generateKernelGraph(name);
        generateKernelSource();
        assembleKernel();
    }

    inline bool CommandKernel::matchesPredicates(/* args */) const
    {
        // TODO: CommandKernel predicates.
        return true;
    }

    inline void CommandKernel::launchKernel(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::launchKernel");

        AssertFatal(m_context);
        AssertFatal(m_context->kernel());

        if(!m_executableKernel)
            m_executableKernel = m_context->instructions()->getExecutableKernel();

        auto kargs = getKernelArguments(args);
        auto inv   = getKernelInvocation(args);

        m_executableKernel->executeKernel(kargs, inv);
    }

    inline void CommandKernel::loadKernelFromAssembly(const std::string& fileName,
                                                      const std::string& kernelName)
    {
        AssertFatal(m_context);
        AssertFatal(m_context->kernel());

        m_executableKernel = std::make_shared<ExecutableKernel>();
        m_executableKernel->loadKernelFromFile(
            fileName, kernelName, m_context->targetArchitecture().target());
    }

    inline std::shared_ptr<Context> CommandKernel::getContext()
    {
        return m_context;
    }

}

#include "CommandSolution_impl.hpp"
