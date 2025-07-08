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
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "DataParallelGEMMSolution.hpp"
#include "GEMMParameters.hpp"
#include "GEMMSolution.hpp"
#include "visualize.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            class StreamKGEMMSolution : public DataParallelGEMMSolution
            {
                Operations::OperationTag m_scratchTag, m_numWGsTag;

            public:
                using DataParallelGEMMSolution::DataParallelGEMMSolution;

                Operations::OperationTag getScratchTag() const override
                {
                    return m_scratchTag;
                }

            protected:
                CommandPtr makeCommand(SolutionParameters const& solutionParams) override
                {
                    auto command = DataParallelGEMMSolution::makeCommand(solutionParams);

                    m_numWGsTag    = command->allocateTag();
                    auto numWGsArg = command->allocateArgument(DataType::UInt32,
                                                               m_numWGsTag,
                                                               ArgumentType::Value,
                                                               DataDirection::ReadOnly,
                                                               rocRoller::NUMWGS);

                    m_scratchTag = command->allocateTag();
                    command->allocateArgument(
                        VariableType(DataType::UInt32, PointerType::PointerGlobal),
                        m_scratchTag,
                        ArgumentType::Value,
                        DataDirection::ReadWrite,
                        rocRoller::SCRATCH);

                    return command;
                }

                virtual CommandParametersPtr
                    makeCommandParameters(CommandPtr                command,
                                          SolutionParameters const& solutionParams) override
                {
                    auto params
                        = DataParallelGEMMSolution::makeCommandParameters(command, solutionParams);

                    params->loopOverOutputTilesDimensions = {0, 1};
                    params->streamK                       = true;
                    params->streamKTwoTile                = solutionParams.streamKTwoTile;

                    return params;
                }

                virtual CommandArguments
                    commandArguments(CommandPtr               command,
                                     ProblemParameters const& problemParams,
                                     RunParameters const&     runParams) const override
                {
                    auto commandArgs = DataParallelGEMMSolution::commandArguments(
                        command, problemParams, runParams);

                    commandArgs.setArgument(m_numWGsTag, ArgumentType::Value, runParams.numWGs);

                    return commandArgs;
                }

                virtual void validateRunParameters(CommandPtr               command,
                                                   ProblemParameters const& problemParams,
                                                   RunParameters const&     runParams,
                                                   CommandKernelPtr         commandKernel) override
                {
                    DataParallelGEMMSolution::validateRunParameters(
                        command, problemParams, runParams, commandKernel);

                    // Determine the number of WGs on the device
                    hipDeviceProp_t deviceProperties;
                    AssertFatal(hipGetDeviceProperties(&deviceProperties, runParams.device)
                                == (hipError_t)HIP_SUCCESS);
                    auto numWGs = deviceProperties.multiProcessorCount;

                    auto flatWorkgroupSize = product(commandKernel->getWorkgroupSize());

                    // Determine the occupancy for the kernel
                    int occupancy;
                    AssertFatal(
                        hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                            &occupancy, commandKernel->getHipFunction(), flatWorkgroupSize, 0)
                        == (hipError_t)HIP_SUCCESS);

                    AssertFatal(runParams.numWGs <= numWGs * occupancy,
                                "StreamK kernel requires that the number of workgroups is not "
                                "greater than the number of compute units * occupancy.");
                }
            };
        }
    }
}
