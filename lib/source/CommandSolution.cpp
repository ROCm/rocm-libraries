
#include <memory>

#include "CommandSolution.hpp"

#include "ExecutableKernel.hpp"

#include "AssemblyKernel.hpp"
#include "KernelArguments.hpp"
#include "KernelGraph/KernelGraph.hpp"
#include "KernelGraph/Transforms/AddComputeIndex.hpp"
#include "KernelGraph/Transforms/AddConvert.hpp"
#include "KernelGraph/Transforms/AddDeallocate.hpp"
#include "KernelGraph/Transforms/AddLDS.hpp"
#include "KernelGraph/Transforms/CleanArguments.hpp"
#include "KernelGraph/Transforms/CleanLoops.hpp"
#include "KernelGraph/Transforms/ConnectWorkgroups.hpp"
#include "KernelGraph/Transforms/FuseExpressions.hpp"
#include "KernelGraph/Transforms/FuseLoops.hpp"
#include "KernelGraph/Transforms/GraphTransform.hpp"
#include "KernelGraph/Transforms/InlineIncrements.hpp"
#include "KernelGraph/Transforms/LowerLinear.hpp"
#include "KernelGraph/Transforms/LowerTensorContraction.hpp"
#include "KernelGraph/Transforms/LowerTile.hpp"
#include "KernelGraph/Transforms/UnrollLoops.hpp"
#include "KernelGraph/Transforms/UpdateParameters.hpp"
#include "Operations/Command.hpp"
#include "Scheduling/Costs/Cost.hpp"
#include "Scheduling/Scheduler.hpp"
#include "Utilities/Settings_fwd.hpp"
#include "Utilities/Timer.hpp"

namespace rocRoller
{
    CommandParameters::CommandParameters()
        : m_waveTilesPerWorkgroup({1, 1})
    {
    }

    void CommandParameters::setDimensionInfo(int tag, KernelGraph::CoordinateGraph::Dimension dim)
    {
        m_dimInfo[tag] = dim;
    }

    std::map<int, KernelGraph::CoordinateGraph::Dimension>
        CommandParameters::getDimensionInfo() const
    {
        return m_dimInfo;
    }

    int CommandParameters::getManualKernelDimension() const
    {
        return m_kernelDimension;
    }

    void CommandParameters::setManualKernelDimension(int dim)
    {
        m_kernelDimension = dim;
    }

    void CommandParameters::setManualWorkgroupSize(std::array<unsigned int, 3> const& v)
    {
        m_workgroupSize = v;
    }

    std::optional<std::array<unsigned int, 3>> CommandParameters::getManualWorkgroupSize() const
    {
        return m_workgroupSize;
    }

    void CommandParameters::setManualWavefrontCount(std::pair<uint, uint> wavefrontCounts)
    {
        m_wavefrontCounts = wavefrontCounts;
    }

    std::optional<std::pair<uint, uint>> CommandParameters::getManualWavefrontCounts() const
    {
        return m_wavefrontCounts;
    }

    void
        CommandParameters::setManualWorkitemCount(std::array<Expression::ExpressionPtr, 3> const& v)
    {
        m_workitemCount = v;
    }

    std::optional<std::array<Expression::ExpressionPtr, 3>>
        CommandParameters::getManualWorkitemCount() const
    {
        return m_workitemCount;
    }

    void CommandParameters::setWaveTilesPerWavefront(unsigned int x, unsigned int y)
    {
        m_waveTilesPerWorkgroup[0] = x;
        m_waveTilesPerWorkgroup[1] = y;
    }

    std::vector<unsigned int> CommandParameters::getWaveTilesPerWorkgroup() const
    {
        return m_waveTilesPerWorkgroup;
    }

    KernelArguments CommandKernel::getKernelArguments(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::getKernelArguments");

        bool            log = Log::getLogger()->should_log(spdlog::level::debug);
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

    KernelInvocation CommandKernel::getKernelInvocation(RuntimeArguments const& args)
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

    CommandKernel::CommandKernel(ContextPtr context)
        : m_context(context)
    {
    }

    CommandKernel::CommandKernel(std::shared_ptr<Command> command, std::string name)
        : m_command(command)
        , m_preParameters(std::make_shared<CommandParameters>())
    {
        generateKernel(name);
    }

    CommandKernel::CommandKernel(std::shared_ptr<Command>           command,
                                 std::string                        name,
                                 std::shared_ptr<CommandParameters> preParameters,
                                 std::shared_ptr<CommandParameters> postParameters,
                                 std::shared_ptr<KernelOptions>     kernelOptions)
        : m_command(command)
        , m_preParameters(preParameters)
        , m_postParameters(postParameters)
        , m_kernelOptions(kernelOptions)
    {
        AssertFatal(m_preParameters);

        generateKernel(name);
    }

    CommandKernel::CommandKernel(std::shared_ptr<Command>        command,
                                 ContextPtr                      context,
                                 KernelGraph::KernelGraph const& kernelGraph)
        : m_command(command)
        , m_context(context)
        , m_kernelGraph(kernelGraph)
        , m_preParameters(std::make_shared<CommandParameters>())
    {
        generateKernelSource();
        assembleKernel();
    }

    KernelGraph::KernelGraph CommandKernel::getKernelGraph() const
    {
        return m_kernelGraph;
    }

    std::string CommandKernel::getInstructions() const
    {
        return m_context->instructions()->toString();
    }

    Generator<Instruction> CommandKernel::commandComments()
    {
        co_yield Instruction::Comment(m_command->toString());
        co_yield Instruction::Comment(m_command->argInfo());
    }

    void CommandKernel::generateKernelGraph(std::string name)
    {
        TIMER(t, "CommandKernel::generateKernelGraph");

        auto logger = rocRoller::Log::getLogger();

        KernelGraph::ConstraintStatus check;

        if(!m_kernelOptions)
        {
            m_kernelOptions = std::make_shared<KernelOptions>();
        }

        m_context = Context::ForDefaultHipDevice(name, *m_kernelOptions);

        // TODO: Determine the correct kernel dimensions
        if(m_preParameters->getManualKernelDimension() > 0)
            m_context->kernel()->setKernelDimensions(m_preParameters->getManualKernelDimension());
        else
            m_context->kernel()->setKernelDimensions(1);

        // TODO: Determine the correct work group size
        if(m_preParameters->getManualWorkgroupSize())
            m_context->kernel()->setWorkgroupSize(*m_preParameters->getManualWorkgroupSize());
        else
            m_context->kernel()->setWorkgroupSize({64, 1, 1});

        if(m_preParameters->getManualWorkitemCount())
            m_context->kernel()->setWorkitemCount(*m_preParameters->getManualWorkitemCount());
        else
            m_context->kernel()->setWorkitemCount(m_command->createWorkItemCount());

        auto zero = std::make_shared<Expression::Expression>(0u);
        m_context->kernel()->setDynamicSharedMemBytes(zero);

        m_context->kernel()->addCommandArguments(m_command->getArguments());

        m_kernelGraph = KernelGraph::translate(m_command);
        logger->debug("CommandKernel::generateKernel: post translate: {}",
                      m_kernelGraph.toDOT(false, "CommandKernel::generateKernel: post translate"));

        if(Settings::getInstance()->get(Settings::EnforceGraphConstraints))
        {
            check = m_kernelGraph.checkConstraints();
            AssertFatal(
                check.satisfied,
                concatenate("CommandKernel::generateKernel: post translate:\n", check.explanation));
        }

        std::vector<KernelGraph::GraphTransformPtr> transforms;
        transforms.push_back(std::make_shared<KernelGraph::UpdateParameters>(m_preParameters));
        transforms.push_back(std::make_shared<KernelGraph::LowerLinear>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::LowerTile>(m_preParameters, m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::LowerTensorContraction>(m_preParameters, m_context));
        transforms.push_back(std::make_shared<KernelGraph::ConnectWorkgroups>());
        transforms.push_back(std::make_shared<KernelGraph::FuseExpressions>());
        transforms.push_back(std::make_shared<KernelGraph::UnrollLoops>(m_context));
        if(m_context->kernelOptions().fuseLoops)
        {
            transforms.push_back(std::make_shared<KernelGraph::FuseLoops>());
        }
        transforms.push_back(std::make_shared<KernelGraph::AddLDS>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::CleanLoops>());
        transforms.push_back(std::make_shared<KernelGraph::AddComputeIndex>());
        transforms.push_back(std::make_shared<KernelGraph::AddConvert>());
        transforms.push_back(std::make_shared<KernelGraph::AddDeallocate>());
        transforms.push_back(std::make_shared<KernelGraph::InlineIncrements>());
        transforms.push_back(std::make_shared<KernelGraph::CleanArguments>(m_context->kernel()));
        if(m_postParameters)
        {
            transforms.push_back(std::make_shared<KernelGraph::UpdateParameters>(m_postParameters));
        }

        for(auto& t : transforms)
        {
            m_kernelGraph = m_kernelGraph.transform(t);
        }

        logger->debug("CommandKernel::generateKernelGraph: end: {}",
                      m_kernelGraph.toDOT(false, "CommandKernel::generateKernelGraph: end"));
    }

    Generator<Instruction> CommandKernel::kernelInstructions()
    {
        co_yield commandComments();
        co_yield m_context->kernel()->preamble();
        co_yield m_context->kernel()->prolog();

        co_yield KernelGraph::generate(m_kernelGraph, m_context->kernel());

        co_yield m_context->kernel()->postamble();
        co_yield m_context->kernel()->amdgpu_metadata();
    }

    void CommandKernel::generateKernelSource()
    {
        TIMER(t, "CommandKernel::generateKernelSource");
        m_context->kernel()->setKernelGraphMeta(
            std::make_shared<KernelGraph::KernelGraph>(m_kernelGraph));

        m_context->schedule(kernelInstructions());
    }

    void CommandKernel::assembleKernel()
    {
        TIMER(t, "CommandKernel::assembleKernel");

        m_executableKernel = m_context->instructions()->getExecutableKernel();
    }

    void CommandKernel::generateKernel(std::string name)
    {
        TIMER(t, "CommandKernel::generateKernel");

        generateKernelGraph(name);
        generateKernelSource();
        assembleKernel();
    }

    bool CommandKernel::matchesPredicates(/* args */) const
    {
        // TODO: CommandKernel predicates.
        return true;
    }

    void CommandKernel::launchKernel(RuntimeArguments const& args)
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

    void CommandKernel::loadKernelFromAssembly(const std::string& fileName,
                                               const std::string& kernelName)
    {
        AssertFatal(m_context);
        AssertFatal(m_context->kernel());

        m_executableKernel = std::make_shared<ExecutableKernel>();
        m_executableKernel->loadKernelFromFile(
            fileName, kernelName, m_context->targetArchitecture().target());
    }

    ContextPtr CommandKernel::getContext()
    {
        return m_context;
    }
}
