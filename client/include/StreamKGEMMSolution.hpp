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
                Operations::OperationTag m_scratchTag;

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
                    makeCommandParameters(SolutionParameters const& solutionParams) override
                {
                    auto params = DataParallelGEMMSolution::makeCommandParameters(solutionParams);

                    params->loopOverOutputTilesDimensions = {0, 1};
                    params->streamK                       = true;
                    params->streamKTwoTile                = solutionParams.streamKTwoTile;

                    return params;
                }

                virtual CommandArguments
                    commandArguments(ProblemParameters const& problemParams,
                                     RunParameters const&     runParams) const override
                {
                    auto commandArgs
                        = DataParallelGEMMSolution::commandArguments(problemParams, runParams);

                    commandArgs.setArgument(
                        command()->getNextTag(), ArgumentType::Value, runParams.numWGs);

                    return commandArgs;
                }

                virtual void validateRunParameters(SolutionParameters const& solutionParams,
                                                   ProblemParameters const&  problemParams,
                                                   RunParameters const&      runParams,
                                                   CommandKernelPtr          commandKernel) override
                {
                    // Determine the number of CUs on the device
                    hipDeviceProp_t deviceProperties;
                    AssertFatal(hipGetDeviceProperties(&deviceProperties, runParams.device)
                                == (hipError_t)HIP_SUCCESS);
                    auto numWGs = deviceProperties.multiProcessorCount;

                    // Determine the occupancy for the kernel
                    int occupancy;
                    AssertFatal(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                                    &occupancy,
                                    commandKernel->getHipFunction(),
                                    solutionParams.workgroupSizeX * solutionParams.workgroupSizeY,
                                    0)
                                == (hipError_t)HIP_SUCCESS);

                    AssertFatal(runParams.numWGs <= numWGs * occupancy,
                                "StreamK kernel requires that the number of workgroups is not "
                                "greater than the number of compute units * occupancy.");
                }

                virtual CommandLaunchParametersPtr
                    makeLaunchParameters(ProblemParameters const&  problemParams,
                                         SolutionParameters const& solutionParams,
                                         RunParameters const&      runParams) override
                {
                    uint num_workgroup_x = runParams.numWGs;
                    uint num_workgroup_y = 1;

                    uint workgroup_size_x
                        = solutionParams.workgroupSizeX * solutionParams.workgroupSizeY;
                    uint workgroup_size_y = 1;

                    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x
                                                                       * workgroup_size_x);
                    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y
                                                                       * workgroup_size_y);
                    auto NZ = std::make_shared<Expression::Expression>(1u);

                    auto launch = std::make_shared<CommandLaunchParameters>();
                    launch->setManualWorkitemCount({NX, NY, NZ});
                    return launch;
                }
            };
        }
    }
}
