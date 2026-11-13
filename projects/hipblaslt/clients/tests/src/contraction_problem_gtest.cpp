// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The TensileLite problem factories give the compute input types, which are part of the
// solution-cache key, a defined value.

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem_Detail.hpp>

#include <functional>
#include <unordered_map>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm halfGemm()
    {
        return ContractionProblemGemm::GEMM_Strides(false,
                                                    false,
                                                    rocisa::DataType::Half,
                                                    rocisa::DataType::Half,
                                                    rocisa::DataType::Float,
                                                    rocisa::DataType::Float,
                                                    128,
                                                    256,
                                                    64,
                                                    1,
                                                    128,
                                                    128 * 64,
                                                    64,
                                                    64 * 256,
                                                    128,
                                                    128 * 256,
                                                    128,
                                                    128 * 256,
                                                    1.0);
    }

    TEST(ContractionProblemFactory, smoke_ComputeInputTypesFollowInputTypes)
    {
        auto half = halfGemm();
        EXPECT_EQ(half.computeInputTypeA(), rocisa::DataType::Half);
        EXPECT_EQ(half.computeInputTypeB(), rocisa::DataType::Half);

        auto sgemm
            = ContractionProblemGemm::GEMM(false, false, 128, 256, 64, 128, 64, 128, 1.0, false, 1);
        EXPECT_EQ(sgemm.computeInputTypeA(), rocisa::DataType::Float);
        EXPECT_EQ(sgemm.computeInputTypeB(), rocisa::DataType::Float);

        ContractionProblemGemm empty;
        EXPECT_EQ(empty.computeInputTypeA(), empty.a().dataType());
        EXPECT_EQ(empty.computeInputTypeB(), empty.b().dataType());
    }

    TEST(ContractionProblemFactory, smoke_IdenticalProblemsShareCacheKey)
    {
        auto first  = halfGemm();
        auto second = halfGemm();
        EXPECT_TRUE(first == second);
        EXPECT_EQ(std::hash<ContractionProblemGemm>{}(first),
                  std::hash<ContractionProblemGemm>{}(second));
    }

    ContractionProblemGemm gemm(bool transA, bool transB, size_t m, size_t n, size_t k)
    {
        size_t lda = transA ? k : m;
        size_t ldb = transB ? n : k;
        return ContractionProblemGemm::GEMM_Strides(transA,
                                                    transB,
                                                    rocisa::DataType::Half,
                                                    rocisa::DataType::Half,
                                                    rocisa::DataType::Float,
                                                    rocisa::DataType::Float,
                                                    m,
                                                    n,
                                                    k,
                                                    1,
                                                    lda,
                                                    lda * (transA ? m : k),
                                                    ldb,
                                                    ldb * (transB ? k : n),
                                                    m,
                                                    m * n,
                                                    m,
                                                    m * n,
                                                    1.0);
    }

    // transA()/transB() are cached in normalize(); they must still reflect the layout.
    TEST(ContractionProblemFactory, smoke_TransposeFlagsMatchLayout)
    {
        for(bool tA : {false, true})
            for(bool tB : {false, true})
            {
                auto p = gemm(tA, tB, 128, 256, 64);
                EXPECT_EQ(p.transA(), tA);
                EXPECT_EQ(p.transB(), tB);
            }
    }

    // The hash covers only a subset of the key. Problems that differ only in fields outside
    // that subset must collide in the hash yet still compare unequal, so the caches keep
    // them apart; equal problems must always hash equally.
    TEST(ContractionProblemFactory, smoke_HashIsSubsetOfEquality)
    {
        std::hash<ContractionProblemGemm> h;
        auto                              base = halfGemm();

        auto gsu = halfGemm();
        gsu.setParams().setGSU(4);
        auto ws = halfGemm();
        ws.setWorkspaceSize(1 << 20);
        auto act = halfGemm();
        act.setActivationType(ActivationType::Relu);
        auto sm = halfGemm();
        sm.setParams().setSmCountTarget(32);

        for(auto const* p : {&gsu, &ws, &act, &sm})
        {
            EXPECT_FALSE(base == *p);
            EXPECT_EQ(h(base), h(*p));

            // The same change applied to both sides keeps them equal and hash-equal.
            auto copy = *p;
            EXPECT_TRUE(copy == *p);
            EXPECT_EQ(h(copy), h(*p));
        }

        // Cache maps store the full key, so hash-colliding problems stay distinct.
        std::unordered_map<ContractionProblemGemm, int> map;
        map.emplace(base, 0);
        map.emplace(gsu, 1);
        map.emplace(ws, 2);
        map.emplace(act, 3);
        map.emplace(sm, 4);
        EXPECT_EQ(map.size(), 5u);
        EXPECT_EQ(map.at(base), 0);
        EXPECT_EQ(map.at(gsu), 1);
        EXPECT_EQ(map.at(ws), 2);
        EXPECT_EQ(map.at(act), 3);
        EXPECT_EQ(map.at(sm), 4);
    }

    // The fast operator== must accept exactly the pairs that the full compare() accepts.
    TEST(ContractionProblemFactory, smoke_EqualityAgreesWithCompare)
    {
        auto base = halfGemm();
        auto gsu  = halfGemm();
        gsu.setParams().setGSU(4);
        auto ws = halfGemm();
        ws.setWorkspaceSize(1 << 20);
        auto act = halfGemm();
        act.setActivationType(ActivationType::Relu);
        auto tb = gemm(false, true, 128, 256, 64);
        auto nn = gemm(false, false, 128, 256, 64);

        // Fields that were added to compare() after the equality operator was written; a
        // drift between compare() and operator== for any such field fails this test.
        auto dsa = halfGemm();
        dsa.setParams().setDeviceScalarAlpha(true);
        auto world = halfGemm();
        world.setWorld(4);

        EXPECT_FALSE(base == dsa);
        EXPECT_FALSE(base == world);

        std::vector<ContractionProblemGemm> probs{base, gsu, ws, act, tb, nn, dsa, world, halfGemm()};
        for(auto const& x : probs)
            for(auto const& y : probs)
                EXPECT_EQ(x == y, Comparison<ContractionProblemGemm>::compare(x, y) == 0);
    }

    // Sanity check on bucket spread: the hashed fields do separate common variants.
    TEST(ContractionProblemFactory, smoke_HashSeparatesShapeAndTranspose)
    {
        std::hash<ContractionProblemGemm> h;
        auto                              nn = gemm(false, false, 128, 256, 64);
        EXPECT_NE(h(nn), h(gemm(true, false, 128, 256, 64)));
        EXPECT_NE(h(nn), h(gemm(false, true, 128, 256, 64)));
        EXPECT_NE(h(nn), h(gemm(false, false, 256, 128, 64)));
        EXPECT_NE(h(nn), h(gemm(false, false, 128, 256, 128)));
    }
} // namespace
