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

                virtual Operations::OperationTag getScratchTag() const override
                {
                    return m_scratchTag;
                }

            protected:
                virtual CommandPtr makeCommand(SolutionParameters const& solutionParams) override
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

                    auto flatWorkgroupSize
                        = product(commandKernel->getContext()->kernel()->workgroupSize());

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
