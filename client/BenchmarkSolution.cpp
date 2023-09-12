#include "include/BenchmarkSolution.hpp"

namespace rocRoller
{
    namespace Client
    {
        rocRoller::CommandPtr BenchmarkSolution::getCommand()
        {
            return m_command;
        }

        std::shared_ptr<rocRoller::CommandKernel> BenchmarkSolution::getKernel()
        {
            return m_kernel;
        }

        BenchmarkResults BenchmarkSolution::benchmark(RunParameters const&       runParams,
                                                      rocRoller::KernelArguments runtimeArgs)
        {
            BenchmarkResults result;
            result.runParams = runParams;

            // Warmup runs
            for(int i = 0; i < runParams.numWarmUp; ++i)
            {
                m_kernel->launchKernel(runtimeArgs.runtimeArguments());
            }

            // Benchmark runs
            for(int outer = 0; outer < runParams.numOuter; ++outer)
            {
                HIP_TIMER(t_kernel, "BENCH");
                HIP_TIC(t_kernel);
                for(int inner = 0; inner < runParams.numInner; ++inner)
                {
                    m_kernel->launchKernel(runtimeArgs.runtimeArguments());
                }
                HIP_TOC(t_kernel);
                HIP_SYNC(t_kernel);
                result.kernelExecute.push_back(t_kernel.nanoseconds());
            }

            double totalTime = 0;
            for(auto ke : result.kernelExecute)
                totalTime += static_cast<double>(ke) / 1.e9;
            double averageTime = totalTime / (runParams.numInner * runParams.numOuter);

            std::cout << "Average runtime (s): " << averageTime << std::endl;

            result.kernelAssemble = TimerPool::nanoseconds("CommandKernel::assembleKernel");
            result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

            return result;
        }
    }
}
