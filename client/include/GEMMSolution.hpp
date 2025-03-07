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

                using ABCDTags = std::tuple<Operations::OperationTag,
                                            Operations::OperationTag,
                                            Operations::OperationTag,
                                            Operations::OperationTag>;

                virtual ABCDTags getABCDTags() const = 0;

                using ABScaleTags = std::tuple<Operations::OperationTag, Operations::OperationTag>;

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
