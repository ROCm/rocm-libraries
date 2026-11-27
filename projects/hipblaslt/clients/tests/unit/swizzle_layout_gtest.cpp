// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the swizzle layout that keeps each HIPBLASLT_ORDER_COL256_R128_*
// order in its own solution rows: the order-to-layout map, the SwizzleLayoutA and
// SwizzleLayoutB predicates, selection from a serialized library holding two orders at
// one key, and the solution cache key.

#include "../../../library/src/amd_detail/rocblaslt/src/include/swizzle_layout.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/PropertyMatching.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/Writing.hpp>

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <initializer_list>
#include <memory>
#include <set>
#include <utility>

namespace
{
    constexpr int kLayouts = 5;

    TensileLite::ContractionProblemGemm swizzledProblem(int layoutA, int layoutB)
    {
        auto problem = TensileLite::ContractionProblemGemm::GEMM_Strides(true,
                                                                         false,
                                                                         rocisa::DataType::Float6,
                                                                         rocisa::DataType::Float6,
                                                                         rocisa::DataType::BFloat16,
                                                                         rocisa::DataType::BFloat16,
                                                                         4096,
                                                                         4096,
                                                                         65536,
                                                                         1,
                                                                         65536,
                                                                         65536 * 4096,
                                                                         65536,
                                                                         65536 * 4096,
                                                                         4096,
                                                                         4096 * 4096,
                                                                         4096,
                                                                         4096 * 4096,
                                                                         0.0);
        problem.setSwizzleTensorA(true);
        problem.setSwizzleTensorB(true);
        problem.setMXScaleFormat(3);
        problem.setSwizzleLayoutA(layoutA);
        problem.setSwizzleLayoutB(layoutB);
        return problem;
    }

    using Gemm         = TensileLite::ContractionProblemGemm;
    using Solution     = TensileLite::ContractionSolution;
    using Library      = TensileLite::SolutionLibrary<Gemm, Solution>;
    using Master       = TensileLite::MasterSolutionLibrary<Gemm, Solution>;
    using PredicatePtr = std::shared_ptr<TensileLite::Predicates::Predicate<Gemm>>;

    constexpr std::array<int64_t, 4> kKey = {4096, 4096, 1, 65536};

    // One swizzled problem-type row, as TensileCreateLibrary writes it for a logic file
    // whose ProblemType sets SwizzleTensorA/B and, with withLayout, SwizzleLayoutA/B.
    PredicatePtr problemTypeRow(int layout, bool withLayout)
    {
        namespace C = TensileLite::Predicates::Contraction;
        std::vector<PredicatePtr> preds{std::make_shared<C::SwizzleTensorA>(true),
                                        std::make_shared<C::SwizzleTensorB>(true)};
        if(withLayout)
        {
            preds.push_back(std::make_shared<C::SwizzleLayoutA>(layout));
            preds.push_back(std::make_shared<C::SwizzleLayoutB>(layout));
        }
        return std::make_shared<TensileLite::Predicates::And<Gemm>>(preds);
    }

    // An Equality table holding solution `index` at kKey.
    std::shared_ptr<Library> equalityTable(Master& master, int index)
    {
        using Key   = std::array<int64_t, 4>;
        using Table = TensileLite::Matching::DistanceMatchingTable<
            Key,
            Gemm,
            TensileLite::ContractionProblemMatchingLibrary::Element,
            std::shared_ptr<Solution>,
            TensileLite::Matching::Equality<Key>>;
        namespace C  = TensileLite::Contraction;
        auto freeA   = std::make_shared<C::FreeSizeA>();
        auto freeB   = std::make_shared<C::FreeSizeB>();
        auto batch   = std::make_shared<C::BatchSize>();
        auto bound   = std::make_shared<C::BoundSize>();
        freeA->index = freeB->index = batch->index = bound->index = 0;

        auto table        = std::make_shared<Table>();
        table->properties = {freeA, freeB, batch, bound};
        table->table.push_back(
            {kKey,
             std::make_shared<TensileLite::SingleContractionLibrary>(master.solutions.at(index)),
             1.0});
        auto lib   = std::make_shared<TensileLite::ContractionProblemMatchingLibrary>();
        lib->table = table;
        return lib;
    }

    // The order-104 (Default) and order-105 (P48) logic files merged into one library, each
    // with one solution at kKey, written to msgpack and loaded back.
    std::shared_ptr<Master> mergedLibrary(bool withLayout)
    {
        auto master = std::make_shared<Master>();
        auto rows   = std::make_shared<TensileLite::ContractionProblemSelectionLibrary>();
        for(int layout : {0, 1})
        {
            auto s                        = std::make_shared<Solution>();
            s->index                      = layout;
            s->solutionName               = layout == 0 ? "c0c1" : "c0c1p48";
            s->kernelName                 = s->solutionName;
            s->problemType.swizzleLayoutA = layout;
            s->problemType.swizzleLayoutB = layout;
            // Selection evaluates the launch settings, which read the tile shape.
            auto& m                   = s->sizeMapping;
            m.waveNum                 = 4;
            m.workGroupSize           = {256, 1, 1};
            m.threadTile              = {1, 1, 1};
            m.macroTile               = {256, 256, 1};
            m.matrixInstruction       = {16, 16, 128, 1};
            m.waveGroup               = {2, 2};
            m.depthU                  = 128;
            m.globalSplitU            = 1;
            m.workGroupMapping        = 8;
            master->solutions[layout] = s;
            rows->rows.emplace_back(
                TensileLite::ContractionProblemPredicate(problemTypeRow(layout, withLayout)),
                equalityTable(*master, layout));
        }
        master->library = rows;
        return std::dynamic_pointer_cast<Master>(TensileLite::LoadLibraryData<Gemm, Solution>(
            TensileLite::MessagePackWriteLibraryData(*master)));
    }

    std::shared_ptr<Solution> select(Master const& library, hipblasLtOrder_t order)
    {
        TensileLite::AMDGPU gpu(TensileLite::AMDGPU::Processor::gfx950, 256, "AMD Instinct MI355X");
        int const           layout = rocblaslt_swizzle_layout(order);
        return library.findBestSolution(swizzledProblem(layout, layout), gpu);
    }
}

// Each new order has a layout of its own; HIPBLASLT_ORDER_COL256_R128_EXT and the plain
// orders keep the layout every existing swizzled solution has.
TEST(HostUnitSwizzleLayout, smoke_NewOrdersMapToDistinctLayouts)
{
    using Layout = TensileLite::SwizzleLayout;
    EXPECT_EQ(rocblaslt_swizzle_layout(HIPBLASLT_ORDER_COL256_R128_P48_EXT),
              static_cast<int>(Layout::P48));
    EXPECT_EQ(rocblaslt_swizzle_layout(HIPBLASLT_ORDER_COL256_R128_P16_EXT),
              static_cast<int>(Layout::P16));
    EXPECT_EQ(rocblaslt_swizzle_layout(HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT),
              static_cast<int>(Layout::P48Page24));
    EXPECT_EQ(rocblaslt_swizzle_layout(HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT),
              static_cast<int>(Layout::P48Page24Split));

    std::set<int> layouts;
    for(auto order : {HIPBLASLT_ORDER_COL256_R128_EXT,
                      HIPBLASLT_ORDER_COL256_R128_P48_EXT,
                      HIPBLASLT_ORDER_COL256_R128_P16_EXT,
                      HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT,
                      HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT})
        EXPECT_TRUE(layouts.insert(rocblaslt_swizzle_layout(order)).second) << int(order);

    for(auto order : {HIPBLASLT_ORDER_COL,
                      HIPBLASLT_ORDER_ROW,
                      HIPBLASLT_ORDER_COL16_4R32,
                      HIPBLASLT_ORDER_COL16_4R16,
                      HIPBLASLT_ORDER_COL16_4R8,
                      HIPBLASLT_ORDER_COL256_R128_EXT})
        EXPECT_EQ(rocblaslt_swizzle_layout(order), static_cast<int>(Layout::Default)) << int(order);
}

TEST(HostUnitSwizzleLayout, smoke_PredicatesMatchOnlyTheirLayout)
{
    using TensileLite::Predicates::Contraction::SwizzleLayoutA;
    using TensileLite::Predicates::Contraction::SwizzleLayoutB;
    for(int sol = 0; sol < kLayouts; ++sol)
        for(int prob = 0; prob < kLayouts; ++prob)
        {
            EXPECT_EQ(SwizzleLayoutA(sol)(swizzledProblem(prob, 0)), sol == prob)
                << "solution layout " << sol << ", problem layout " << prob;
            EXPECT_EQ(SwizzleLayoutB(sol)(swizzledProblem(0, prob)), sol == prob)
                << "solution layout " << sol << ", problem layout " << prob;
        }
}

TEST(HostUnitSwizzleLayout, smoke_ProblemsStartAtTheDefaultLayout)
{
    auto problem = TensileLite::ContractionProblemGemm::GEMM(
        true, false, 256, 256, 128, 128, 128, 256, 1.0, false, 1);
    EXPECT_EQ(problem.swizzleLayoutA(), 0);
    EXPECT_EQ(problem.swizzleLayoutB(), 0);
}

// A library holding an order-104 and an order-105 solution at one Equality key, after a
// msgpack round trip, gives each order its own solution and gives an order with no
// solution none. Without the layout predicates the two rows are indistinguishable and the
// order-105 problem takes the order-104 kernel, which the second half pins.
TEST(HostUnitSwizzleLayout, smoke_MergedLibrarySelectsEachOrdersRow)
{
    auto const library = mergedLibrary(true);
    ASSERT_NE(library, nullptr);
    ASSERT_EQ(library->solutions.size(), 2u);
    for(auto const& [index, solution] : library->solutions)
    {
        EXPECT_EQ(solution->problemType.swizzleLayoutA, index);
        EXPECT_EQ(solution->problemType.swizzleLayoutB, index);
    }

    auto const c0c1 = select(*library, HIPBLASLT_ORDER_COL256_R128_EXT);
    auto const p48  = select(*library, HIPBLASLT_ORDER_COL256_R128_P48_EXT);
    ASSERT_NE(c0c1, nullptr);
    ASSERT_NE(p48, nullptr);
    EXPECT_EQ(c0c1->index, 0);
    EXPECT_EQ(p48->index, 1);
    EXPECT_EQ(select(*library, HIPBLASLT_ORDER_COL256_R128_P16_EXT), nullptr);

    auto const collapsed = mergedLibrary(false);
    ASSERT_NE(collapsed, nullptr);
    auto const collapsedP48 = select(*collapsed, HIPBLASLT_ORDER_COL256_R128_P48_EXT);
    ASSERT_NE(collapsedP48, nullptr);
    EXPECT_EQ(collapsedP48->index, 0);
}

// The solution cache keys on problem equality, so an order-104 pick cannot be reused for
// another layout of the same size.
TEST(HostUnitSwizzleLayout, smoke_LayoutDistinguishesCachedProblems)
{
    auto const c0c1 = swizzledProblem(0, 0);
    EXPECT_TRUE(c0c1 == swizzledProblem(0, 0));
    for(auto const [a, b] : {std::pair<int, int>{1, 1}, {2, 0}, {3, 4}, {0, 1}})
    {
        auto const other = swizzledProblem(a, b);
        EXPECT_FALSE(c0c1 == other) << "A layout " << a << ", B layout " << b;
        EXPECT_NE(std::hash<TensileLite::ContractionProblemGemm>{}(c0c1),
                  std::hash<TensileLite::ContractionProblemGemm>{}(other))
            << "A layout " << a << ", B layout " << b;
    }
}
