#include "include/TensileGEMMSolution.hpp"

#include "../../test/unit/Utilities.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            template <>
            TensileGEMMSolution<Half, Half, Half, Half>::TensileGEMMSolution(
                SolutionParameters const& solutionParams)
                : GEMMSolution<Half, Half, Half, Half>(solutionParams.problemParams)
            {
                AssertFatal(solutionParams.problemParams.m == 7680);
                AssertFatal(solutionParams.problemParams.n == 8448);
                AssertFatal(solutionParams.problemParams.k == 8448
                            || solutionParams.problemParams.k == 8192);
                m_solutionParams = solutionParams;
                rocRoller::KernelOptions ko;
                this->m_context = rocRoller::Context::ForDefaultHipDevice(
                    "Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1", ko);
                this->m_command = makeCommand();
                this->m_kernel  = std::make_shared<CommandKernel>(makeKernel());
            }

            template <>
            TensileGEMMSolution<float, float, float, float>::TensileGEMMSolution(
                SolutionParameters const& solutionParams)
                : GEMMSolution<float, float, float, float>(solutionParams.problemParams)
            {
                AssertFatal(false, "No Tensile GEMM Solution for Floats.");
            }
        }
    }
}
