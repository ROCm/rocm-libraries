// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "GEMMParameters.hpp"

#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/Scratch_fwd.hpp>

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

                virtual Operations::OperationTag
                    getScratchTag(Operations::ScratchPolicy scratchPolicy) const
                {
                    return {};
                }

                virtual CommandKernelPtr
                    generateCommandKernel(CommandPtr                command,
                                          SolutionParameters const& solutionParams)
                {
                    auto commandKernel = std::make_shared<CommandKernel>(
                        command, solutionParams.generateKernelName().shortName);
                    commandKernel->setContext(m_context);
                    commandKernel->setCommandParameters(
                        this->makeCommandParameters(command, solutionParams));
                    this->setPredicates(command, commandKernel, solutionParams);
                    commandKernel->generateKernel();
                    return commandKernel;
                }

                virtual void validateRunParameters(CommandPtr                 command,
                                                   ProblemParameters const&   problemParams,
                                                   RunParameters const&       runParams,
                                                   BenchmarkParameters const& benchmarkParams,
                                                   CommandKernelPtr           commandKernel)
                {
                    auto commandArgs = this->commandArguments(command, problemParams, runParams);
                    AssertFatal(commandKernel->matchesPredicates(commandArgs.runtimeArguments(),
                                                                 LogLevel::Error),
                                "Invalid run parameters: all predicates must match.");
                }

                using ABCDTags = std::tuple<Operations::OperationTag,
                                            Operations::OperationTag,
                                            Operations::OperationTag,
                                            Operations::OperationTag>;

                virtual ABCDTags getABCDTags() const = 0;

                using ABScaleTags = std::tuple<std::optional<Operations::OperationTag>,
                                               std::optional<Operations::OperationTag>>;

                virtual ABScaleTags getABScaleTags() const = 0;

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
