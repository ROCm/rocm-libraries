// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "client/DataParallelGEMMSolution.hpp"
#include "client/GEMMParameters.hpp"
#include "client/GEMMSolution.hpp"
#include "client/visualize.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            class StreamKGEMMSolution : public DataParallelGEMMSolution
            {
                std::map<Operations::ScratchPolicy, Operations::OperationTag> m_scratchTags;
                Operations::OperationTag                                      m_numWGsTag;

            public:
                using DataParallelGEMMSolution::DataParallelGEMMSolution;

                Operations::OperationTag
                    getScratchTag(Operations::ScratchPolicy scratchPolicy) const override
                {
                    return m_scratchTags.at(scratchPolicy);
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

                    // Create a scratch operation for tile data
                    m_scratchTags[Operations::ScratchPolicy::None] = command->allocateTag();
                    command->addOperation(
                        Operations::Scratch(m_scratchTags[Operations::ScratchPolicy::None],
                                            Operations::ScratchPolicy::None));
                    command->allocateArgument(
                        VariableType(DataType::UInt32, PointerType::PointerGlobal),
                        m_scratchTags[Operations::ScratchPolicy::None],
                        ArgumentType::Value,
                        DataDirection::ReadWrite,
                        getScratchName(Operations::ScratchPolicy::None));

                    // Create a scratch operation for flags
                    m_scratchTags[Operations::ScratchPolicy::ZeroedBeforeAndAfter]
                        = command->allocateTag();
                    command->addOperation(Operations::Scratch(
                        m_scratchTags[Operations::ScratchPolicy::ZeroedBeforeAndAfter],
                        Operations::ScratchPolicy::ZeroedBeforeAndAfter));
                    command->allocateArgument(
                        VariableType(DataType::UInt32, PointerType::PointerGlobal),
                        m_scratchTags[Operations::ScratchPolicy::ZeroedBeforeAndAfter],
                        ArgumentType::Value,
                        DataDirection::ReadWrite,
                        getScratchName(Operations::ScratchPolicy::ZeroedBeforeAndAfter));

                    return command;
                }

                CommandParametersPtr
                    makeCommandParameters(CommandPtr                command,
                                          SolutionParameters const& solutionParams) override
                {
                    auto params
                        = DataParallelGEMMSolution::makeCommandParameters(command, solutionParams);

                    params->loopOverOutputTilesDimensions = {0, 1};

                    params->streamK = solutionParams.streamK;

                    return params;
                }

                CommandArguments commandArguments(CommandPtr               command,
                                                  ProblemParameters const& problemParams,
                                                  RunParameters const&     runParams) const override
                {
                    auto commandArgs = DataParallelGEMMSolution::commandArguments(
                        command, problemParams, runParams);

                    commandArgs.setArgument(m_numWGsTag, ArgumentType::Value, runParams.numWGs);

                    return commandArgs;
                }

                void validateRunParameters(CommandPtr                 command,
                                           ProblemParameters const&   problemParams,
                                           RunParameters const&       runParams,
                                           BenchmarkParameters const& benchmarkParams,
                                           CommandKernelPtr           commandKernel) override
                {
                    DataParallelGEMMSolution::validateRunParameters(
                        command, problemParams, runParams, benchmarkParams, commandKernel);

                    // Determine the number of WGs on the device
                    hipDeviceProp_t deviceProperties;
                    AssertFatal(hipGetDeviceProperties(&deviceProperties, benchmarkParams.device)
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
