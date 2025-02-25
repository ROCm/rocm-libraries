#pragma once

#include "GEMMParameters.hpp"

#include <rocRoller/Operations/CommandArgument_fwd.hpp>

using namespace rocRoller;

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            class GEMMSolution
            {
            public:
                GEMMSolution() = delete;
                GEMMSolution(rocRoller::ContextPtr context)
                    : m_context(context)
                {
                }

                virtual CommandPtr           makeCommand(SolutionParameters const&)           = 0;
                virtual CommandParametersPtr makeCommandParameters(SolutionParameters const&) = 0;
                virtual CommandLaunchParametersPtr makeLaunchParameters(ProblemParameters const&,
                                                                        SolutionParameters const&,
                                                                        RunParameters const&)
                    = 0;

                virtual CommandArguments commandArguments(ProblemParameters const& problemParams,
                                                          RunParameters const& runParams) const = 0;

                virtual void setPredicates(CommandKernelPtr          commandKernel,
                                           SolutionParameters const& solutionParams)
                {
                }

                virtual Operations::OperationTag getScratchTag() const
                {
                    return {};
                }

                virtual void generateSolution(SolutionParameters const& solutionParams)
                {
                    this->m_command       = this->makeCommand(solutionParams);
                    this->m_commandKernel = std::make_shared<CommandKernel>(
                        this->m_command, solutionParams.generateKernelName());
                    this->m_commandKernel->setContext(m_context);
                    this->m_commandKernel->setCommandParameters(
                        this->makeCommandParameters(solutionParams));
                    this->setPredicates(this->m_commandKernel, solutionParams);
                    this->m_commandKernel->generateKernel();
                }

                virtual void validateRunParameters(SolutionParameters const& solutionParams,
                                                   ProblemParameters const&  problemParams,
                                                   RunParameters const&      runParams,
                                                   CommandKernelPtr          commandKernel)
                {
                    auto args = this->commandArguments(problemParams, runParams).runtimeArguments();
                    AssertFatal(this->m_commandKernel->matchesPredicates(args, spdlog::level::err),
                                "Invalid run parameters: all predicates must match.");
                }

                using ABCDTags                       = std::tuple<Operations::OperationTag,
                                            Operations::OperationTag,
                                            Operations::OperationTag,
                                            Operations::OperationTag>;
                virtual ABCDTags getABCDTags() const = 0;

                ContextPtr context() const
                {
                    return this->m_context;
                }

                CommandPtr command() const
                {
                    return this->m_command;
                }

                CommandKernelPtr commandKernel() const
                {
                    return this->m_commandKernel;
                }

            protected:
                ContextPtr       m_context;
                CommandPtr       m_command;
                CommandKernelPtr m_commandKernel;
            };
        }
    }
}
