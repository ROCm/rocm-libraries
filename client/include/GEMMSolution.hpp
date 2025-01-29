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

                virtual CommandPtr makeCommand(SolutionParameters const&) = 0;

                virtual CommandParametersPtr makeCommandParameters(CommandPtr,
                                                                   SolutionParameters const&)
                    = 0;

                virtual CommandArguments commandArguments(CommandPtr,
                                                          ProblemParameters const& problemParams,
                                                          RunParameters const& runParams) const = 0;

                virtual void setPredicates(CommandPtr, CommandKernelPtr, SolutionParameters const&)
                {
                }

                virtual Operations::OperationTag getScratchTag() const
                {
                    return {};
                }

                virtual CommandKernelPtr
                    generateCommandKernel(CommandPtr                command,
                                          SolutionParameters const& solutionParams)
                {
                    auto commandKernel = std::make_shared<CommandKernel>(
                        command, solutionParams.generateKernelName());
                    commandKernel->setContext(m_context);
                    commandKernel->setCommandParameters(
                        this->makeCommandParameters(command, solutionParams));
                    this->setPredicates(command, commandKernel, solutionParams);
                    commandKernel->generateKernel();
                    return commandKernel;
                }

                virtual void validateRunParameters(CommandPtr               command,
                                                   ProblemParameters const& problemParams,
                                                   RunParameters const&     runParams,
                                                   CommandKernelPtr         commandKernel)
                {
                    auto args = this->commandArguments(command, problemParams, runParams)
                                    .runtimeArguments();
                    AssertFatal(commandKernel->matchesPredicates(args, LogLevel::Error),
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

            protected:
                ContextPtr m_context;
            };
        }
    }
}
