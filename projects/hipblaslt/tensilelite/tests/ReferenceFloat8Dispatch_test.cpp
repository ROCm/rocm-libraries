// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <type_traits>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    enum class Storage
    {
        HalfHalf,
        HalfFloat8,
        Float8Half
    };

    template <typename F8A, typename F8B, typename Output>
    void checkSlowDispatch(Storage storage, bool scaled)
    {
        constexpr auto f8A   = TypeInfo<F8A>::Enum;
        constexpr auto f8B   = TypeInfo<F8B>::Enum;
        const bool     halfA = storage != Storage::Float8Half;
        const bool     halfB = storage != Storage::HalfFloat8;
        SCOPED_TRACE(::testing::Message()
                     << "compute=" << ToString(f8A) << "/" << ToString(f8B)
                     << " storage=" << int(storage)
                     << " output=" << ToString(TypeInfo<Output>::Enum) << " scaled=" << scaled);

        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            halfA ? rocisa::DataType::Half : f8A,
                                                            halfB ? rocisa::DataType::Half : f8B,
                                                            TypeInfo<Output>::Enum,
                                                            TypeInfo<Output>::Enum,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            scaled ? 0.25 : 0.0);
        problem.setComputeInputTypeA(f8A);
        problem.setComputeInputTypeB(f8B);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setUseScaleAB("Scalar");
        problem.setScaleA(rocisa::DataType::Float, 1);
        problem.setScaleB(rocisa::DataType::Float, 1);

        // Put 0.9 and scale 3 on a Half operand in either mixed direction.
        // If both operands are Half, the other is 1 with scale 0.5.
        Half              hA = Half(scaled && halfA ? 0.9f : 1.0f);
        Half              hB = Half(scaled && !halfA ? 0.9f : 1.0f);
        F8A               fA = F8A(1.0f);
        F8B               fB = F8B(1.0f);
        Output            c(2.0f), d(-99.0f);
        float             scaleA = scaled ? (halfA ? 3.0f : 0.5f) : 1.0f;
        float             scaleB = scaled ? (halfA ? 0.5f : 3.0f) : 1.0f;
        ContractionInputs inputs(halfA ? static_cast<void const*>(&hA) : &fA,
                                 halfB ? static_cast<void const*>(&hB) : &fB,
                                 &c,
                                 &d,
                                 scaled ? 1.5f : 1.0f,
                                 scaled ? 0.25f : 0.0f);
        inputs.scaleA = &scaleA;
        inputs.scaleB = &scaleB;

        // Slow execution is the prerequisite for the Scalar fallback in the
        // following scaling PR. Unit scales isolate missing dispatch from
        // rounding order. The scaled profile exercises the float epilogue:
        // Half(0.9)*3 narrows to 2.75 in F8 and 2.5 in B8, then
        // D = narrowed*0.5*1.5 + 2*0.25. These literals are exactly
        // representable in every output type; the fast route is not an oracle.
        const bool  isB8     = halfA
                                   ? (std::is_same_v<F8A, BFloat8> || std::is_same_v<F8A, BFloat8_fnuz>)
                                   : (std::is_same_v<F8B, BFloat8> || std::is_same_v<F8B, BFloat8_fnuz>);
        const float expected = scaled ? (isB8 ? 2.375f : 2.5625f) : 1.0f;
        ASSERT_NO_THROW(
            SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false));
        EXPECT_EQ(float(d), expected);
    }

    template <typename F8A, typename F8B = F8A>
    void checkAllStorageAndOutputs(bool scaled)
    {
        for(auto storage : {Storage::HalfHalf, Storage::HalfFloat8, Storage::Float8Half})
        {
            checkSlowDispatch<F8A, F8B, Half>(storage, scaled);
            checkSlowDispatch<F8A, F8B, float>(storage, scaled);
            checkSlowDispatch<F8A, F8B, BFloat16>(storage, scaled);
        }
    }
}

TEST(ReferenceFloat8Dispatch, ScalarUnitScalesExecuteEverySupportedWideOutput)
{
    checkAllStorageAndOutputs<Float8>(false);
    checkAllStorageAndOutputs<BFloat8>(false);
    checkAllStorageAndOutputs<Float8_fnuz>(false);
    checkAllStorageAndOutputs<BFloat8_fnuz>(false);
    checkAllStorageAndOutputs<Float8, BFloat8>(false);
    checkAllStorageAndOutputs<BFloat8, Float8>(false);
    checkAllStorageAndOutputs<Float8_fnuz, BFloat8_fnuz>(false);
    checkAllStorageAndOutputs<BFloat8_fnuz, Float8_fnuz>(false);
}

TEST(ReferenceFloat8Dispatch, ScalarNarrowingAndEpilogueMatchIndependentLiterals)
{
    checkAllStorageAndOutputs<Float8>(true);
    checkAllStorageAndOutputs<BFloat8>(true);
    checkAllStorageAndOutputs<Float8_fnuz>(true);
    checkAllStorageAndOutputs<BFloat8_fnuz>(true);
    checkAllStorageAndOutputs<Float8, BFloat8>(true);
    checkAllStorageAndOutputs<BFloat8, Float8>(true);
    checkAllStorageAndOutputs<Float8_fnuz, BFloat8_fnuz>(true);
    checkAllStorageAndOutputs<BFloat8_fnuz, Float8_fnuz>(true);
}
