#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>

namespace rocRoller
{
    namespace Client
    {
        struct RunParameters
        {
            int device;

            int numWarmUp;
            int numOuter;
            int numInner;
        };

        struct BenchmarkResults
        {
            RunParameters runParams;

            size_t              kernelGenerate;
            size_t              kernelAssemble;
            std::vector<size_t> kernelExecute;
            bool                checked = false;
            bool                correct = true;
            double              rnorm   = 1.e12;
        };

        class BenchmarkSolution
        {
        public:
            rocRoller::CommandPtr       getCommand();
            rocRoller::CommandKernelPtr getKernel();
            BenchmarkResults            benchmark(RunParameters const& runParams,
                                                  CommandArguments     runtimeArgs);

            virtual void setContext(ContextPtr context)
            {
                m_kernel->setContext(context);
            }

        protected:
            rocRoller::CommandPtr                     m_command;
            std::shared_ptr<rocRoller::CommandKernel> m_kernel;
        };
    }
}
