// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>

#include <rocRoller/CommandSolution.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{

    std::string CommandParameters::toString() const
    {
        std::ostringstream msg;

        msg << ShowValue(allowAmbiguousMemoryNodes);
        msg << ShowValue(enableLongDwordInstructions);

        msg << ShowValue(transposeMemoryAccess);
        msg << ShowValue(packMultipleElementsInto1VGPR);

        msg << ShowValue(unrollK);
        msg << ShowValue(fuseLoops);
        msg << ShowValue(tailLoops);

        msg << ShowValue(prefetch);
        msg << ShowValue(prefetchInFlight);
        msg << ShowValue(prefetchLDSFactor);
        msg << ShowValue(prefetchMixMemOps);

        msg << ShowValue(streamK);

        msg << "loopOverOutputTilesDimensions: ";
        streamJoin(msg, loopOverOutputTilesDimensions, ", ");
        msg << std::endl;

        msg << ShowValue(loopOverOutputTilesTopLoop);

        msg << "loopOverOutputTilesCoordSizes: ";
        streamJoin(msg, loopOverOutputTilesCoordSizes, ", ");
        msg << std::endl;

        msg << ShowValue(loopOverOutputTilesIteratedTiles);

        msg << "dimInfo: " << std::endl;
        for(auto const& [tag, dim] : m_dimInfo)
            msg << "  " << tag << ": " << dim << std::endl;

        if(m_workgroupSize)
        {
            msg << "workgroupSize: ";
            streamJoin(msg, *m_workgroupSize, ", ");
            msg << std::endl;
        }

        if(m_wavefrontCounts)
        {
            auto const& [a, b] = *m_wavefrontCounts;
            msg << "wavefrontCounts: [" << a << ", " << b << "]" << std::endl;
        }

        msg << ShowValue(m_kernelDimension);

        msg << "waveTilesPerWavefront: [";
        streamJoin(msg, m_waveTilesPerWavefront, ", ");
        msg << "]" << std::endl;

        msg << ShowValue(m_splitStoreTileIntoWaveBlocks);

        return msg.str();
    }

    CommandParameters::CommandParameters()
        : m_waveTilesPerWavefront({1, 1})
    {
        transposeMemoryAccess.set(LayoutType::ROW_MAJOR, true);
        transposeMemoryAccess.set(LayoutType::COLUMN_MAJOR, false);
    }

    void CommandParameters::setDimensionInfo(Operations::OperationTag                tag,
                                             KernelGraph::CoordinateGraph::Dimension dim)
    {
        m_dimInfo[tag] = dim;
    }

    std::map<Operations::OperationTag, KernelGraph::CoordinateGraph::Dimension>
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

    void CommandParameters::setManualWorkgroupClusterSize(std::array<unsigned int, 3> const& v)
    {
        m_workgroupClusterSize = v;
    }

    std::optional<std::array<unsigned int, 3>> CommandParameters::getManualWorkgroupSize() const
    {
        return m_workgroupSize;
    }

    std::optional<std::array<unsigned int, 3>>
        CommandParameters::getManualWorkgroupClusterSize() const
    {
        return m_workgroupClusterSize;
    }

    void CommandParameters::setManualWavefrontCount(std::pair<uint, uint> wavefrontCounts)
    {
        m_wavefrontCounts = wavefrontCounts;
    }

    std::optional<std::pair<uint, uint>> CommandParameters::getManualWavefrontCounts() const
    {
        return m_wavefrontCounts;
    }

    void CommandLaunchParameters::setManualWorkitemCount(
        std::array<Expression::ExpressionPtr, 3> const& v)
    {
        m_workitemCount = v;
    }

    std::optional<std::array<Expression::ExpressionPtr, 3>>
        CommandLaunchParameters::getManualWorkitemCount() const
    {
        return m_workitemCount;
    }

    void CommandParameters::setWaveTilesPerWavefront(unsigned int x, unsigned int y)
    {
        m_waveTilesPerWavefront[0] = x;
        m_waveTilesPerWavefront[1] = y;
    }

    std::vector<unsigned int> CommandParameters::getWaveTilesPerWavefront() const
    {
        return m_waveTilesPerWavefront;
    }

    void CommandParameters::setSplitStoreTileIntoWaveBlocks(bool x)
    {
        m_splitStoreTileIntoWaveBlocks = x;
    }

    bool CommandParameters::getSplitStoreTileIntoWaveBlocks() const
    {
        return m_splitStoreTileIntoWaveBlocks;
    }

    KernelArguments CommandKernel::getKernelArguments(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::getKernelArguments");

        bool            log = Log::getLogger()->should_log(LogLevel::Debug);
        KernelArguments rv(log);

        auto const& argStructs = m_context->kernel()->arguments();

        rv.reserve(m_context->kernel()->argumentSize(), argStructs.size());

        for(auto& arg : argStructs)
        {
            auto value = Expression::evaluate(arg.getExpression(), args);

            if(variableType(value) != arg.getVariableType())
            {
                throw std::runtime_error(concatenate("Evaluated argument type ",
                                                     variableType(value),
                                                     " doesn't match expected type ",
                                                     arg.getVariableType(),
                                                     ", Expression: ",
                                                     toString(arg.getExpression()),
                                                     ", name: ",
                                                     arg.getName()));
            }

            rv.append(arg.getName(), value);
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

        rv.workgroupClusterSize = m_context->kernel()->workgroupClusterSize();

        return rv;
    }

    void CommandKernel::setContext(ContextPtr context)
    {
        m_context = context;
    }

    CommandKernel::CommandKernel(CommandPtr command, std::string kernelName)
        : m_command(command)
        , m_name(kernelName)
    {
    }

    KernelGraph::KernelGraphPtr CommandKernel::getKernelGraph() const
    {
        return m_kernelGraph;
    }

    CommandPtr CommandKernel::getCommand() const
    {
        return m_command;
    }

    hipFunction_t CommandKernel::getHipFunction() const
    {
        return m_executableKernel->getHipFunction();
    }

    std::string CommandKernel::getInstructions() const
    {
        return m_context->instructions()->toString();
    }

    std::string CommandKernel::getKernelName() const
    {
        return m_name;
    }

    Generator<Instruction> commandComments(CommandPtr command)
    {
        co_yield Instruction::Comment(command->toString());
        co_yield Instruction::Comment(command->argInfo());
    }

    CommandArgumentPtr findArgumentByName(CommandPtr const command, std::string const& argName)
    {
        auto const& arguments = command->getArguments();
        auto it = std::ranges::find_if(arguments, [&](auto x) { return x->name() == argName; });

        if(it == arguments.end())
            return nullptr;
        return *it;
    }

    void CommandKernel::generateKernelGraph()
    {
        TIMER(t, "CommandKernel::generateKernelGraph");

        AssertFatal(m_context);

        KernelGraph::ConstraintStatus check;

        if(!m_commandParameters)
            m_commandParameters = std::make_shared<CommandParameters>();

        // TODO: Determine the correct kernel dimensions
        if(m_commandParameters->getManualKernelDimension() > 0)
            m_context->kernel()->setKernelDimensions(
                m_commandParameters->getManualKernelDimension());
        else
            m_context->kernel()->setKernelDimensions(1);

        // TODO: Determine the correct work group size
        if(m_commandParameters->getManualWorkgroupSize())
            m_context->kernel()->setWorkgroupSize(*m_commandParameters->getManualWorkgroupSize());
        else
        {
            unsigned int wfs = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            m_context->kernel()->setWorkgroupSize({wfs, 1, 1});
        }

        if(m_commandParameters->getManualWorkgroupClusterSize())
        {
            m_context->kernel()->setWorkgroupClusterSize(
                *m_commandParameters->getManualWorkgroupClusterSize());
        }

        auto zero = std::make_shared<Expression::Expression>(0u);
        m_context->kernel()->setDynamicSharedMemBytes(zero);

        auto kernelGraph = KernelGraph::translate(m_command, m_commandParameters);

        if(Settings::getInstance()->get(Settings::LogGraphs))
            Log::debug("CommandKernel::generateKernel: post translate: {}",
                       kernelGraph.toDOT(false, "CommandKernel::generateKernel: post translate"));

        if(Settings::getInstance()->get(Settings::EnforceGraphConstraints))
        {
            check = kernelGraph.checkConstraints();
            AssertFatal(
                check.satisfied,
                concatenate("CommandKernel::generateKernel: post translate:\n", check.explanation));
        }

        std::vector<KernelGraph::GraphTransformPtr> transforms;

        bool applyScheduleMultiplyAndLDS = m_commandParameters->prefetch
                                           && m_commandParameters->prefetchMixMemOps
                                           && m_commandParameters->prefetchLDSFactor == 1;

        // UpdateParameters should go before IdentifyParallelDimensions so the User.size expression
        // that it sets is updated in IdentifyParallelDimensions
        transforms.push_back(std::make_shared<KernelGraph::UpdateParameters>(m_commandParameters));
        transforms.push_back(std::make_shared<KernelGraph::IdentifyParallelDimensions>());

        transforms.push_back(std::make_shared<KernelGraph::OrderMemory>(
            !m_commandParameters->allowAmbiguousMemoryNodes));
        transforms.push_back(std::make_shared<KernelGraph::AddLDS>(m_commandParameters, m_context));
        transforms.push_back(std::make_shared<KernelGraph::LowerLinear>(m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::LowerTile>(m_commandParameters, m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::LowerTensorContraction>(m_commandParameters, m_context));
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        transforms.push_back(
            std::make_shared<KernelGraph::AddLDSPadding>(m_context, m_commandParameters));
        transforms.push_back(std::make_shared<KernelGraph::ConstantPropagation>());
        transforms.push_back(std::make_shared<KernelGraph::FuseExpressions>());

        if(m_commandParameters->workgroupMappingDim.has_value())
        {
            auto wgmArg = findArgumentByName(m_command, rocRoller::WGM);
            AssertFatal(wgmArg != nullptr,
                        "Can not find WGM Command argument required for workgroup mapping.");

            transforms.push_back(std::make_shared<KernelGraph::RemapOutputTiles>(
                m_commandParameters->workgroupMappingDim,
                std::make_shared<Expression::Expression>(wgmArg)));
        }

        if(m_commandParameters->streamK)
        {
            auto numWGsArg = findArgumentByName(m_command, rocRoller::NUMWGS);
            AssertFatal(numWGsArg != nullptr,
                        "Can not find numWGs Command argument required for StreamK kernels.");

            transforms.push_back(std::make_shared<KernelGraph::AddStreamK>(
                m_context,
                m_commandParameters,
                rocRoller::XLOOP,
                rocRoller::KLOOP,
                std::make_shared<Expression::Expression>(numWGsArg)));
        }
        else if(!m_commandParameters->loopOverOutputTilesDimensions.empty())
        {
            transforms.push_back(std::make_shared<KernelGraph::LoopOverTileNumbers>(
                m_commandParameters->loopOverOutputTilesDimensions,
                m_commandParameters->loopOverOutputTilesCoordSizes,
                m_commandParameters->loopOverOutputTilesIteratedTiles,
                m_commandParameters->loopOverOutputTilesTopLoop,
                m_context));
        }

        transforms.push_back(std::make_shared<KernelGraph::ConnectWorkgroups>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::WorkgroupRemapXCC>(
            m_context, m_commandParameters->workgroupRemapXCC));

        transforms.push_back(
            std::make_shared<KernelGraph::UnrollLoops>(m_commandParameters, m_context));
        if(m_commandParameters->fuseLoops)
        {
            transforms.push_back(std::make_shared<KernelGraph::FuseLoops>());
        }
        transforms.push_back(std::make_shared<KernelGraph::RemoveDuplicates>());
        transforms.push_back(std::make_shared<KernelGraph::OrderEpilogueBlocks>());
        // TODO: Investigate why Simplify cannot be called BEFORE RemoveDuplicates and OrderEpilogueBlocks
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        transforms.push_back(std::make_shared<KernelGraph::CleanLoops>());
        transforms.push_back(
            std::make_shared<KernelGraph::SwizzleScale>(m_commandParameters, m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::AddPrefetch>(m_commandParameters, m_context));
        if(m_commandParameters->prefetch && m_commandParameters->prefetchScale)
        {
            transforms.push_back(
                std::make_shared<KernelGraph::PrefetchScale>(m_commandParameters, m_context));
        }
        transforms.push_back(std::make_shared<KernelGraph::AddF6LDSPadding>(m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::AddDirect2LDS>(m_context, m_commandParameters));
        transforms.push_back(std::make_shared<KernelGraph::AddTDMToLDS>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        transforms.push_back(std::make_shared<KernelGraph::AddPRNG>(m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::UpdateWavefrontParameters>(m_commandParameters));
        transforms.push_back(
            std::make_shared<KernelGraph::AssignIndexExpressions>(m_context, m_command));
        transforms.push_back(std::make_shared<KernelGraph::LoadPacked>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::AddConvert>(m_context));

        //
        // TODO: Turn on this transformation by default when SGPR issue gets resolved
        //
        if(m_context->kernelOptions()->removeSetCoordinate)
        {
            transforms.push_back(std::make_shared<KernelGraph::RemoveSetCoordinate>());
            transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        }
        transforms.push_back(std::make_shared<KernelGraph::NopExtraScopes>());
        transforms.push_back(std::make_shared<KernelGraph::InlineInits>());
        transforms.push_back(std::make_shared<KernelGraph::InlineIncrements>());
        if(applyScheduleMultiplyAndLDS)
            transforms.push_back(std::make_shared<KernelGraph::RemoveImplicitScheduling>());
        transforms.push_back(std::make_shared<KernelGraph::OrderMultiplyNodes>());
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        if(applyScheduleMultiplyAndLDS)
            transforms.push_back(std::make_shared<KernelGraph::ScheduleMultiplyAndLDS>());
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        transforms.push_back(std::make_shared<KernelGraph::OrderExchangeNodes>());
        transforms.push_back(std::make_shared<KernelGraph::AliasDataFlowTags>());
        transforms.push_back(std::make_shared<KernelGraph::AddLDSBarriers>());
        transforms.push_back(std::make_shared<KernelGraph::AddDeallocateDataFlow>());
        transforms.push_back(std::make_shared<KernelGraph::CleanArguments>(m_context, m_command));
        transforms.push_back(std::make_shared<KernelGraph::AddDeallocateArguments>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::MergeAdjacentDeallocates>());
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        transforms.push_back(std::make_shared<KernelGraph::SortArguments>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::SetWorkitemCount>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::ModelAddresses>(m_context));

        for(auto const& t : transforms)
        {
            kernelGraph = kernelGraph.transform(t);
        }

        m_kernelGraph = std::make_shared<KernelGraph::KernelGraph>(kernelGraph);
    }

    Generator<Instruction> kernelInstructions(ContextPtr                  context,
                                              CommandPtr                  command,
                                              KernelGraph::KernelGraphPtr kernelGraph)
    {
        co_yield commandComments(command);
        co_yield context->kernel()->preamble();
        co_yield context->kernel()->prolog();

        co_yield KernelGraph::generate(*kernelGraph, context->kernel());

        co_yield context->kernel()->postamble();
        co_yield context->kernel()->amdgpu_metadata();
    }

    void CommandKernel::generateKernelSource()
    {
        TIMER(t, "CommandKernel::generateKernelSource");
        m_context->kernel()->setKernelGraphMeta(m_kernelGraph);
        m_context->kernel()->setCommandMeta(m_command);

        for(auto inst : kernelInstructions(m_context, m_command, m_kernelGraph))
            m_context->schedule(inst);
    }

    std::vector<char> CommandKernel::assembleKernel()
    {
        TIMER(t, "CommandKernel::assembleKernel");

        return m_context->instructions()->assemble();
    }

    void CommandKernel::loadKernel()
    {
        TIMER(t, "CommandKernel::loadKernel");

        if(!m_executableKernel)
            m_executableKernel = m_context->instructions()->getExecutableKernel();
    }

    void CommandKernel::generateKernel()
    {
        TIMER(t, "CommandKernel::generateKernel");

        if(m_command)
        {
            generateKernelGraph();
            generateKernelSource();
        }
        else
        {
            // Probably from a unit test.  The context should contain
            // scheduled instructions already.
        }
    }

    void CommandKernel::addPredicate(Expression::ExpressionPtr expression)
    {
        m_predicates.push_back(expression);
    }

    bool CommandKernel::matchesPredicates(RuntimeArguments const& args, LogLevel level) const
    {
        bool retVal = true;
        for(auto predicate : m_predicates)
        {
            try
            {
                if(!std::get<bool>(Expression::evaluate(predicate, args)))
                {
                    retVal = false;
                    std::string comment;
                    if(!(comment = Expression::getComment(predicate)).empty())
                    {
                        Log::log(level, "Predicate mismatch for {}: {}", m_name, comment);
                    }
                    else
                    {
                        Log::log(level,
                                 "Predicate {} for {} is false.",
                                 Expression::toString(predicate),
                                 m_name);
                    }
                }
            }
            catch(std::bad_variant_access err)
            {
                Throw<FatalError>(
                    "Predicate ", Expression::toString(predicate), " does not evaluate to a bool");
            }
        }

        return retVal;
    }

    void CommandKernel::setCommandParameters(CommandParametersPtr commandParams)
    {
        m_commandParameters = commandParams;
    }

    CommandParametersPtr CommandKernel::getCommandParameters() const
    {
        return m_commandParameters;
    }

    //
    // 2024-11-05: This is only used in a few tests.  Please see the
    // note in Command::createWorkItemCount.
    //
    void CommandKernel::setLaunchParameters(CommandLaunchParametersPtr launch)
    {
        m_launchParameters = launch;
    }

    void CommandKernel::launchKernel(RuntimeArguments const&   args,
                                     std::shared_ptr<HIPTimer> timer,
                                     int                       iteration)
    {
        launchKernel(args, timer, iteration, timer ? timer->stream() : 0);
    }

    void CommandKernel::launchKernel(RuntimeArguments const& args, hipStream_t stream)
    {
        launchKernel(args, nullptr, 0, stream);
    }

    void CommandKernel::launchKernel(RuntimeArguments const&   args,
                                     std::shared_ptr<HIPTimer> timer,
                                     int                       iteration,
                                     hipStream_t               stream)
    {
        TIMER(t, "CommandKernel::launchKernel");

        AssertFatal(m_context, "Unable to launch kernel: CommandKernel must have a Context.");
        AssertFatal(m_context->kernel(),
                    "Unable to launch kernel: Context must have an AssemblyKernel.");
        AssertFatal(matchesPredicates(args, LogLevel::Error),
                    "Unable to launch kernel: all CommandKernel predicates must match.");

        if(m_launchParameters)
        {
            if(m_launchParameters->getManualWorkitemCount())
                m_context->kernel()->setWorkitemCount(
                    *m_launchParameters->getManualWorkitemCount());
        }

        auto kargs = getKernelArguments(args);
        auto inv   = getKernelInvocation(args);

        loadKernel();

        if(timer)
            m_executableKernel->executeKernel(kargs, inv, timer, iteration);
        else
            m_executableKernel->executeKernel(kargs, inv, stream);
    }

    void CommandKernel::loadKernelFromAssembly(const std::string& fileName,
                                               const std::string& kernelName)
    {
        AssertFatal(m_context);
        AssertFatal(m_context->kernel());

        m_executableKernel
            = std::make_shared<ExecutableKernel>(m_context->targetArchitecture().target());
        m_executableKernel->loadKernelFromFile(
            fileName, kernelName, m_context->targetArchitecture().target());
    }

    AssemblyKernelPtr CommandKernel::loadKernelFromCodeObject(const std::string& fileName,
                                                              const std::string& kernelName)
    {
        AssertFatal(m_context);

        m_executableKernel
            = std::make_shared<ExecutableKernel>(m_context->targetArchitecture().target());
        m_executableKernel->loadKernelFromCodeObjectFile(
            fileName, kernelName, m_context->targetArchitecture().target());

        // XXX Instead of adding `setKernel`, should the context load from a code object?
        auto kernels   = AssemblyKernels::fromELF(fileName).kernels;
        auto kernel    = kernels.at(0);
        auto kernelPtr = std::make_shared<AssemblyKernel>(kernel);
        m_context->setKernel(kernelPtr);
        return kernelPtr;
    }

    ContextPtr CommandKernel::getContext()
    {
        return m_context;
    }

    size_t CommandKernel::scratchSpaceRequired(Operations::ScratchPolicy policy,
                                               RuntimeArguments const&   args) const
    {
        auto amount = m_context->getScratchAmount(policy);

        auto times = evaluationTimes(amount);
        AssertFatal(times[Expression::EvaluationTime::Translate]
                        || times[Expression::EvaluationTime::KernelLaunch],
                    "Unable to evaluate the scratch space required",
                    ShowValue(toString(amount)));

        return getUnsignedInt(evaluate(amount, args));
    }

    std::array<unsigned int, 3> const& CommandKernel::getWorkgroupSize() const
    {
        return m_context->kernel()->workgroupSize();
    }
}
