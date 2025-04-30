/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/DecisionTree.hpp>
#include <Tensile/DecisionTreeLibrary.hpp>
#include <Tensile/ExactLogicLibrary.hpp>

using namespace Tensile;
using namespace DecisionTree;

using Key   = std::array<float, 3>;
using DTree = Tree<Key, std::shared_ptr<ContractionLibrary>, std::shared_ptr<ContractionSolution>>;

/*
 * Tests for invalid tree structures
 * e.g. trees that try go out of bounds, trees with circular paths
 */
TEST(DecisionTree, ValidTree)
{
    DTree test_tree{{
        /* start */ {0, 700.f, 1, IDX_RETURN_FALSE},
        /* - 1 - */ {1, 300.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE},
    }};

    /*
    Makes the following tree:
        start:(Key[0] <= 700?)
            |
            |
            |------------------------
            |                       |
           YES                      NO
            |                       |
            |                       |
        1:(Key[1] <= 300?)     2:(RETURN false)
            |
            |
            |------------------------
            |                       |
           YES                      NO
            |                       |
            |                       |
        3:(RETURN false)        4:(RETURN true)
    */

    EXPECT_TRUE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeEmpty)
{
    DTree test_tree{{}};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeIdxOOB)
{
    DTree test_tree{{
        /* start */ {0, 700.f, IDX_RETURN_TRUE, 7}, // right idx OOB
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeIdxOOBNeg)
{
    DTree test_tree{{
        /* start */ {0, 700.f, -3, IDX_RETURN_TRUE}, // left idx OOB
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeCircularShort)
{
    DTree test_tree{{
        /* start */ {0, 700.f, IDX_RETURN_TRUE, 0}, // right idx circular
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeCircularLong)
{
    DTree test_tree{{
        /* start */ {0, 700.f, 1, IDX_RETURN_TRUE},
        /* - 1 - */ {0, 700.f, 0, IDX_RETURN_TRUE}, // left idx circular
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidNoTrueNodes)
{
    DTree test_tree{{
        /* start */ {0, 700.f, 1, IDX_RETURN_FALSE},
        /* - 1 - */ {0, 700.f, IDX_RETURN_FALSE, IDX_RETURN_FALSE},
    }};

    EXPECT_FALSE(test_tree.valid());
}

/*
 * Tests for correct predictions.
 */
TEST(DecisionTree, SimplePrediction)
{
    DTree test_tree{{
        /* start */ {0, 700.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE},
    }};

    Key test_input0{{800.f, 200.f, 200.f}};
    Key test_input1{{600.f, 200.f, 200.f}};

    EXPECT_EQ(test_tree.predict(test_input0), true); // Expected: start->true
    EXPECT_EQ(test_tree.predict(test_input1), false); // Expected: start->false
}

TEST(DecisionTree, MultiStepPrediction)
{
    DTree test_tree{{
        /* start */ {0, 700.f, 1, IDX_RETURN_FALSE},
        /* - 1 - */ {1, 300.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE},
    }};

    Key test_input0{{800.f, 400.f, 200.f}};
    Key test_input1{{600.f, 400.f, 200.f}};
    Key test_input2{{600.f, 200.f, 200.f}};

    EXPECT_EQ(test_tree.predict(test_input0), false); // Expected: start->false
    EXPECT_EQ(test_tree.predict(test_input1), true); // Expected: start->1->true
    EXPECT_EQ(test_tree.predict(test_input2), false); // Expected: start->1->false
}

/*
 * Tests for libraries.
 */
TEST(DecisionTree, DecisionTreeLibrary)
{
    // Solutions
    auto Solution0 = std::make_shared<ContractionSolution>();
    auto Solution1 = std::make_shared<ContractionSolution>();
    auto Solution2 = std::make_shared<ContractionSolution>();
    auto Solution3 = std::make_shared<ContractionSolution>();

    Solution0->index = 0;
    Solution1->index = 1;
    Solution2->index = 2;
    Solution3->index = 3;

    auto Library0        = std::make_shared<SingleContractionLibrary>(Solution0);
    auto Library1        = std::make_shared<SingleContractionLibrary>(Solution1);
    auto Library2        = std::make_shared<SingleContractionLibrary>(Solution2);
    auto LibraryFallback = std::make_shared<SingleContractionLibrary>(Solution3);

    std::vector<std::shared_ptr<MLFeatures::MLFeature<ContractionProblem>>> features;
    auto freeSizeA   = std::make_shared<MLFeatures::FreeSizeA>();
    freeSizeA->index = 0;
    features.push_back(freeSizeA);
    auto freeSizeB   = std::make_shared<MLFeatures::FreeSizeB>();
    freeSizeB->index = 0;
    features.push_back(freeSizeB);
    auto boundSize   = std::make_shared<MLFeatures::BoundSize>();
    boundSize->index = 0;
    features.push_back(boundSize);

    // Make trees library
    std::vector<DTree> trees;

    DTree tree0{{
        {0, 700.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE}, // YES for freeSizeA>7000
    }};
    tree0.value = Library0;
    trees.push_back(tree0);

    DTree tree1{{
        {1, 700.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE}, // YES for freeSizeB>700
    }};
    tree1.value = Library1;
    trees.push_back(tree1);

    DTree tree2{{
        {0, 300.f, IDX_RETURN_TRUE, IDX_RETURN_FALSE}, // YES for freeSizeA<300
    }};
    tree2.value = Library2;
    trees.push_back(tree2);

    // Forest and full library - Note: Solution 3 as fallback
    using BForest = BasicForest<Key,
                                ContractionProblem,
                                std::shared_ptr<ContractionLibrary>,
                                std::shared_ptr<ContractionSolution>>;

    // this change in the constructor enables the template magic to
    // handle null values as a default when serializing the fallback
    // solutions if it is optionally not present.
    auto forest       = std::make_shared<BForest>(features);
    forest->nullValue = LibraryFallback;
    forest->trees     = trees;

    auto dtreelib    = std::make_shared<DecisionTreeLibrary<ContractionProblem>>();
    dtreelib->forest = forest;

    // Problems
    auto Problem0
        = ContractionProblem::GEMM(false, false, 800, 800, 800, 800, 800, 800, 1.0, false, 1);
    auto Problem1
        = ContractionProblem::GEMM(false, false, 500, 800, 800, 500, 800, 500, 1.0, false, 1);
    auto Problem2
        = ContractionProblem::GEMM(false, false, 500, 500, 500, 500, 500, 500, 1.0, false, 1);

    // Tests
    AMDGPU gpu;
    EXPECT_EQ(dtreelib->findBestSolution(Problem0, gpu), Solution0);
    EXPECT_EQ(dtreelib->findBestSolution(Problem1, gpu), Solution1);
    EXPECT_EQ(dtreelib->findBestSolution(Problem2, gpu), Solution3); // No match, goes to fallback
}

TEST(DecisionTree, DecisionTreeMultiLibrary)
{

    using Predicate   = Predicates::Predicate<ContractionProblem>;
    using SizeInRange = Predicates::Contraction::SizeInRange;
    using SizeEqual   = Predicates::Contraction::SizeEqual;
    using Range       = Predicates::Contraction::Range;
    using And         = Predicates::And<ContractionProblem>;
    using BForest     = BasicForest<Key,
                                ContractionProblem,
                                std::shared_ptr<ContractionLibrary>,
                                std::shared_ptr<ContractionSolution>>;

    // This will test the behavior of the dtree logic can handle multiple regions correctly.
    // The two regions that are constructed have the opposite branching logic.
    auto region1Solution0 = std::make_shared<ContractionSolution>();
    auto region1Solution1 = std::make_shared<ContractionSolution>();

    region1Solution0->index = 0;
    region1Solution1->index = 1;

    auto region1Library0        = std::make_shared<SingleContractionLibrary>(region1Solution0);
    auto region1LibraryFallback = std::make_shared<SingleContractionLibrary>(region1Solution1);

    auto region2Solution0 = std::make_shared<ContractionSolution>();
    auto region2Solution1 = std::make_shared<ContractionSolution>();

    region2Solution0->index = 0;
    region2Solution1->index = 1;

    auto region2Library0        = std::make_shared<SingleContractionLibrary>(region2Solution0);
    auto region2LibraryFallback = std::make_shared<SingleContractionLibrary>(region2Solution1);

    // Features (generic)
    std::vector<std::shared_ptr<MLFeatures::MLFeature<ContractionProblem>>> features;
    auto freeSizeA   = std::make_shared<MLFeatures::FreeSizeA>();
    freeSizeA->index = 0;
    features.push_back(freeSizeA);
    auto freeSizeB   = std::make_shared<MLFeatures::FreeSizeB>();
    freeSizeB->index = 0;
    features.push_back(freeSizeB);
    auto boundSize   = std::make_shared<MLFeatures::BoundSize>();
    boundSize->index = 0;
    features.push_back(boundSize);

    // Make trees library
    std::vector<DTree> region1trees;

    DTree region1tree0{{
        {0, 5000.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE}, // YES for freeSizeA > 5000
    }};

    region1tree0.value = region1Library0;
    region1trees.push_back(region1tree0);

    auto region1forest       = std::make_shared<BForest>(features);
    region1forest->nullValue = region1LibraryFallback;
    region1forest->trees     = region1trees;

    auto region1dtreelib    = std::make_shared<DecisionTreeLibrary<ContractionProblem>>();
    region1dtreelib->forest = region1forest;

    // Make trees library
    std::vector<DTree> region2trees;

    DTree region2tree0{{
        {0, 5000.f, IDX_RETURN_TRUE, IDX_RETURN_FALSE}, // YES for freeSizeA <= 5000
    }};
    region2tree0.value = region2Library0;
    region2trees.push_back(region2tree0);

    auto region2forest       = std::make_shared<BForest>(features);
    region2forest->nullValue = region2LibraryFallback;
    region2forest->trees     = region2trees;

    auto region2dtreelib    = std::make_shared<DecisionTreeLibrary<ContractionProblem>>();
    region2dtreelib->forest = region2forest;

    /// region library
    size_t max_size = std::numeric_limits<size_t>::max();

    std::shared_ptr<Predicate> regionM  = std::make_shared<SizeInRange>(0, Range{0, 40000});
    std::shared_ptr<Predicate> regionB  = std::make_shared<SizeEqual>(2, 1);
    std::shared_ptr<Predicate> regionN1 = std::make_shared<SizeInRange>(1, Range{0, 8000});
    std::shared_ptr<Predicate> regionN2 = std::make_shared<SizeInRange>(1, Range{8000, max_size});

    // Create region predicate for (0 <= M < 40000), (0 <= N < 8000)
    auto preds1    = {regionM, regionB, regionN1};
    auto isRegion1 = std::make_shared<And>(preds1);

    ContractionProblemSelectionLibrary::Row Region1Row_dtreelib(isRegion1, region1dtreelib);

    // Create region predicate for (0 <= M < 40000), (8000 <= N < max)
    auto preds2    = {regionM, regionB, regionN2};
    auto isRegion2 = std::make_shared<And>(preds2);

    ContractionProblemSelectionLibrary::Row Region2Row_dtreelib(isRegion2, region2dtreelib);

    ContractionProblemSelectionLibrary lib({Region1Row_dtreelib, Region2Row_dtreelib});

    // Problems
    auto Region1Problem1
        = ContractionProblem::GEMM(false, false, 7000, 6500, 1000, 7000, 1000, 7000, 1.0, false, 1);
    auto Region1Problem2
        = ContractionProblem::GEMM(false, false, 4000, 6500, 1000, 4000, 1000, 4000, 1.0, false, 1);
    auto Region2Problem1 = ContractionProblem::GEMM(
        false, false, 7000, 16500, 1000, 7000, 1000, 7000, 1.0, false, 1);
    auto Region2Problem2 = ContractionProblem::GEMM(
        false, false, 4000, 16500, 1000, 4000, 1000, 4000, 1.0, false, 1);

    AMDGPU gpu;
    EXPECT_EQ(lib.findBestSolution(Region1Problem1, gpu), region1Solution0);
    EXPECT_EQ(lib.findBestSolution(Region1Problem2, gpu), region1Solution1);
    EXPECT_EQ(lib.findBestSolution(Region2Problem1, gpu), region2Solution1);
    EXPECT_EQ(lib.findBestSolution(Region2Problem2, gpu), region2Solution0);
}

TEST(DecisionTree, DecisionTreeBatch)
{

    using Predicate   = Predicates::Predicate<ContractionProblem>;
    using SizeInRange = Predicates::Contraction::SizeInRange;
    using Range       = Predicates::Contraction::Range;
    using And         = Predicates::And<ContractionProblem>;
    // Create a Key4 aliase specific to this test because this test uses four features which
    // requires a size four Key type; whereas previous tests only use three features.
    using Key4 = std::array<float, 4>;
    using DTree4
        = Tree<Key4, std::shared_ptr<ContractionLibrary>, std::shared_ptr<ContractionSolution>>;
    using BForest = BasicForest<Key4,
                                ContractionProblem,
                                std::shared_ptr<ContractionLibrary>,
                                std::shared_ptr<ContractionSolution>>;

    // This will test the behavior of the dtree logic can handle multiple regions correctly.
    // The two regions that are constructed have the opposite branching logic.
    auto region1Solution0 = std::make_shared<ContractionSolution>();
    auto region1Solution1 = std::make_shared<ContractionSolution>();

    region1Solution0->index = 0;
    region1Solution1->index = 1;

    auto region1Library0        = std::make_shared<SingleContractionLibrary>(region1Solution0);
    auto region1LibraryFallback = std::make_shared<SingleContractionLibrary>(region1Solution1);

    auto region2Solution0 = std::make_shared<ContractionSolution>();
    auto region2Solution1 = std::make_shared<ContractionSolution>();

    region2Solution0->index = 0;
    region2Solution1->index = 1;

    auto region2Library0        = std::make_shared<SingleContractionLibrary>(region2Solution0);
    auto region2LibraryFallback = std::make_shared<SingleContractionLibrary>(region2Solution1);

    // Features (generic)
    std::vector<std::shared_ptr<MLFeatures::MLFeature<ContractionProblem>>> features;
    auto freeSizeA   = std::make_shared<MLFeatures::FreeSizeA>();
    freeSizeA->index = 0;
    features.push_back(freeSizeA);
    auto freeSizeB   = std::make_shared<MLFeatures::FreeSizeB>();
    freeSizeB->index = 0;
    features.push_back(freeSizeB);
    auto batchSize   = std::make_shared<MLFeatures::BatchSize>();
    batchSize->index = 0;
    features.push_back(batchSize);
    auto boundSize   = std::make_shared<MLFeatures::BoundSize>();
    boundSize->index = 0;
    features.push_back(boundSize);

    // Make trees library
    std::vector<DTree4> region1trees;

    DTree4 region1tree0{{
        {0, 5000.f, IDX_RETURN_FALSE, IDX_RETURN_TRUE}, // YES for freeSizeA > 5000
    }};
    region1tree0.value = region1Library0;
    region1trees.push_back(region1tree0);

    auto region1forest       = std::make_shared<BForest>(features);
    region1forest->nullValue = region1LibraryFallback;
    region1forest->trees     = region1trees;

    auto region1dtreelib    = std::make_shared<DecisionTreeLibrary<ContractionProblem>>();
    region1dtreelib->forest = region1forest;

    // Make trees library
    std::vector<DTree4> region2trees;

    DTree4 region2tree0{{
        {0, 5000.f, IDX_RETURN_TRUE, IDX_RETURN_FALSE}, // YES for freeSizeA <= 5000
    }};
    region2tree0.value = region2Library0;
    region2trees.push_back(region2tree0);

    auto region2forest       = std::make_shared<BForest>(features);
    region2forest->nullValue = region2LibraryFallback;
    region2forest->trees     = region2trees;

    auto region2dtreelib    = std::make_shared<DecisionTreeLibrary<ContractionProblem>>();
    region2dtreelib->forest = region2forest;

    /// region library
    size_t max_size = std::numeric_limits<size_t>::max();

    std::shared_ptr<Predicate> regionM = std::make_shared<SizeInRange>(0, Range{0, max_size});
    std::shared_ptr<Predicate> regionN = std::make_shared<SizeInRange>(1, Range{0, max_size});
    std::shared_ptr<Predicate> regionK = std::make_shared<SizeInRange>(3, Range{0, max_size});

    std::shared_ptr<Predicate> regionBa = std::make_shared<SizeInRange>(2, Range{0, 64});
    std::shared_ptr<Predicate> regionBb = std::make_shared<SizeInRange>(2, Range{64, max_size});

    // Create region predicate for (0 <= M,N,K < in_max), (0 <= B < 64)
    auto preds1    = {regionM, regionN, regionBa, regionK};
    auto isRegion1 = std::make_shared<And>(preds1);

    ContractionProblemSelectionLibrary::Row Region1Row_dtreelib(isRegion1, region1dtreelib);

    // Create region predicate for (0 <= M,N,K < intmax), (64 <= B < intmax)
    auto preds2    = {regionM, regionN, regionBb, regionK};
    auto isRegion2 = std::make_shared<And>(preds2);

    ContractionProblemSelectionLibrary::Row Region2Row_dtreelib(isRegion2, region2dtreelib);
    ContractionProblemSelectionLibrary      lib({Region1Row_dtreelib, Region2Row_dtreelib});

    // Problems
    auto Region1Problem1 = ContractionProblem::GEMM(
        false, false, 7000, 6500, 1000, 7000, 1000, 7000, 1.0, false, 10);
    auto Region1Problem2 = ContractionProblem::GEMM(
        false, false, 4000, 6500, 1000, 4000, 1000, 4000, 1.0, false, 10);
    auto Region2Problem1 = ContractionProblem::GEMM(
        false, false, 7000, 16500, 1000, 7000, 1000, 7000, 1.0, false, 100);
    auto Region2Problem2 = ContractionProblem::GEMM(
        false, false, 4000, 16500, 1000, 4000, 1000, 4000, 1.0, false, 100);

    AMDGPU gpu;
    EXPECT_EQ(lib.findBestSolution(Region1Problem1, gpu), region1Solution0);
    EXPECT_EQ(lib.findBestSolution(Region1Problem2, gpu), region1Solution1);
    EXPECT_EQ(lib.findBestSolution(Region2Problem1, gpu), region2Solution1);
    EXPECT_EQ(lib.findBestSolution(Region2Problem2, gpu), region2Solution0);
}
