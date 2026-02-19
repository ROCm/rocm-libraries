// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

using namespace ck;

// =============================================================================
// ThreadwiseTransferHelper_Base tests
// =============================================================================

TEST(ThreadwiseTransferHelperBase, CompileTimeConstants)
{
    EXPECT_EQ(ThreadwiseTransferHelper_Base::I0.value, 0);
    EXPECT_EQ(ThreadwiseTransferHelper_Base::I1.value, 1);
    EXPECT_EQ(ThreadwiseTransferHelper_Base::I2.value, 2);
    EXPECT_EQ(ThreadwiseTransferHelper_Base::I4.value, 4);
    EXPECT_EQ(ThreadwiseTransferHelper_Base::I8.value, 8);
    EXPECT_EQ(ThreadwiseTransferHelper_Base::I16.value, 16);
}

TEST(ThreadwiseTransferHelperBase, ConstantsInheritedBySerpentine)
{
    // Serpentine inherits all constants from Base via public inheritance.
    EXPECT_EQ(ThreadwiseTransferHelper_Serpentine::I0.value, 0);
    EXPECT_EQ(ThreadwiseTransferHelper_Serpentine::I16.value, 16);
}

TEST(ThreadwiseTransferHelperBase, ConstantsInheritedBySFC)
{
    // SFC inherits all constants from Base via public inheritance.
    EXPECT_EQ(ThreadwiseTransferHelper_SFC::I0.value, 0);
    EXPECT_EQ(ThreadwiseTransferHelper_SFC::I16.value, 16);
}

// =============================================================================
// ThreadwiseTransferHelper_Serpentine::ComputeForwardSweep tests
// =============================================================================

TEST(ThreadwiseTransferHelperSerpentine, ComputeForwardSweep_2D_EvenRow)
{
    /*
     * 2D serpentine traversal on a 4x4 grid:
     *
     *   dim1 ->
     *   0   1   2   3
     *   +-->-->-->--+   row 0: forward (dim0=0 is even)
     *   +--<--<--<--+   row 1: backward (dim0=1 is odd)
     *   +-->-->-->--+   row 2: forward (dim0=2 is even)
     *   +--<--<--<--+   row 3: backward (dim0=3 is odd)
     *   dim0
     *
     * At position (0, *): dim0 is even -> dim1 sweeps FORWARD
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto idx     = make_tuple(Number<0>{}, Number<0>{});
    constexpr auto lengths = make_tuple(Number<4>{}, Number<4>{});
    constexpr auto sweep   = Helper::ComputeForwardSweep<2>(idx, lengths);

    EXPECT_TRUE(sweep[Number<0>{}]); // dim 0: always forward (outermost)
    EXPECT_TRUE(sweep[Number<1>{}]); // dim 1: forward because dim0 position (0) is even
}

TEST(ThreadwiseTransferHelperSerpentine, ComputeForwardSweep_2D_OddRow)
{
    /*
     * Same 4x4 grid, but at row 1:
     *
     *   +-->-->-->--+   row 0
     *   +--<--<--<--+   row 1: dim0=1 is odd -> dim1 sweeps BACKWARD
     *
     * At position (1, *): dim0 is odd -> dim1 sweeps BACKWARD
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto idx     = make_tuple(Number<1>{}, Number<0>{});
    constexpr auto lengths = make_tuple(Number<4>{}, Number<4>{});
    constexpr auto sweep   = Helper::ComputeForwardSweep<2>(idx, lengths);

    EXPECT_TRUE(sweep[Number<0>{}]);  // dim 0: always forward
    EXPECT_FALSE(sweep[Number<1>{}]); // dim 1: backward (dim0 position 1 is odd)
}

TEST(ThreadwiseTransferHelperSerpentine, ComputeForwardSweep_1D)
{
    /*
     * 1D traversal: always forward regardless of position.
     *
     *   0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto idx     = make_tuple(Number<3>{});
    constexpr auto lengths = make_tuple(Number<8>{});
    constexpr auto sweep   = Helper::ComputeForwardSweep<1>(idx, lengths);

    EXPECT_TRUE(sweep[Number<0>{}]); // 1D: only dimension, always forward
}

// =============================================================================
// ThreadwiseTransferHelper_Serpentine::ComputeMoveOnDim tests
// =============================================================================

TEST(ThreadwiseTransferHelperSerpentine, ComputeMoveOnDim_InnerNotComplete)
{
    /*
     * 2D grid with ordered_access_lengths = (3, 2):
     *
     *   dim1: 0   1
     *   dim0:
     *     0   [*]  .    <-- at (0,0): dim1 hasn't reached end yet
     *     1    .   .
     *     2    .   .
     *
     * Rule: a dimension moves only when all faster-varying (higher-index)
     * dimensions have completed their range.
     *
     * At (0, 0):
     *   dim0: dim1 is at 0, not at end (1). -> dim0 does NOT move.
     *   dim1: no higher dims to check, and 0 < 1. -> dim1 MOVES.
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto idx     = make_tuple(Number<0>{}, Number<0>{});
    constexpr auto lengths = make_tuple(Number<3>{}, Number<2>{});
    constexpr auto move    = Helper::ComputeMoveOnDim<2>(idx, lengths);

    EXPECT_FALSE(move[Number<0>{}]); // dim 0: inner dim NOT at end
    EXPECT_TRUE(move[Number<1>{}]);  // dim 1: can advance
}

TEST(ThreadwiseTransferHelperSerpentine, ComputeMoveOnDim_InnerComplete)
{
    /*
     * Same grid, at position (0, 1):
     *
     *   dim1: 0   1
     *   dim0:
     *     0    .  [*]   <-- at (0,1): dim1 at its end (1 == 2-1)
     *     1    .   .
     *     2    .   .
     *
     * At (0, 1):
     *   dim0: dim1 is at end (1 == 1). dim0 < 2. -> dim0 MOVES.
     *   dim1: at end. -> dim1 does NOT move.
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto idx     = make_tuple(Number<0>{}, Number<1>{});
    constexpr auto lengths = make_tuple(Number<3>{}, Number<2>{});
    constexpr auto move    = Helper::ComputeMoveOnDim<2>(idx, lengths);

    EXPECT_TRUE(move[Number<0>{}]);  // dim 0: inner dim at end, can advance
    EXPECT_FALSE(move[Number<1>{}]); // dim 1: at its limit, cannot advance
}

// =============================================================================
// ThreadwiseTransferHelper_Serpentine::ComputeDataIndex tests
// =============================================================================

TEST(ThreadwiseTransferHelperSerpentine, ComputeDataIndex_ForwardSweep)
{
    /*
     * 2D grid (4x3), both dims sweeping forward, identity order, scale=1:
     *
     *   ordered_access_idx = (2, 1)
     *   forward_sweep      = (true, true)
     *   dim_access_order   = (0, 1)  <-- identity
     *   scalar_per_access  = (1, 1)  <-- no scaling
     *
     * Forward: data_idx = ordered_idx = (2, 1)
     * Reorder: identity -> (2, 1)
     * Scale:   * (1,1) -> (2, 1)
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto idx     = make_tuple(Number<2>{}, Number<1>{});
    constexpr auto lengths = make_tuple(Number<4>{}, Number<3>{});
    constexpr auto sweep   = Helper::ComputeForwardSweep<2>(idx, lengths);
    constexpr auto order   = Sequence<0, 1>{};
    constexpr auto spa     = Sequence<1, 1>{};

    constexpr auto data_idx = Helper::ComputeDataIndex<2>(idx, lengths, sweep, order, spa);

    EXPECT_EQ(data_idx[Number<0>{}], 2);
    EXPECT_EQ(data_idx[Number<1>{}], 1);
}

// =============================================================================
// ThreadwiseTransferHelper_Serpentine::ComputeCoordinateResetStep tests
// =============================================================================

TEST(ThreadwiseTransferHelperSerpentine, ComputeCoordinateResetStep_2D)
{
    /*
     * SliceLengths = (4, 2), VectorDim = 1, ScalarPerVector = 2
     * DimAccessOrder = (0, 1)
     *
     * scalar_per_access = (1, 2)  [only dim 1 is vectorized with width 2]
     * access_lengths    = (4, 1)  [4/1=4, 2/2=1]
     *
     * The traversal visits 4 positions along dim 0, each accessing 2 elements:
     *
     *   dim0=0: access [0,0..1]
     *   dim0=1: access [1,0..1]  (backward sweep, but only 1 step on dim1)
     *   dim0=2: access [2,0..1]
     *   dim0=3: access [3,0..1]
     *
     * Final position: data_idx = (3, 0) * scalar_per_access = (3, 0)
     * Reset step:     -(3, 0) = (-3, 0)
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    constexpr auto reset =
        Helper::ComputeCoordinateResetStep<2, Sequence<4, 2>, 1, 2, Sequence<0, 1>>();

    EXPECT_EQ(reset[Number<0>{}], -3);
    EXPECT_EQ(reset[Number<1>{}], 0);
}

// =============================================================================
// VectorSizeLookupTable / VectorOffsetsLookupTable tests
// =============================================================================

TEST(ThreadwiseTransferHelperSerpentine, VectorSizeLookupTable)
{
    /*
     * Binary decomposition of vector widths into power-of-2 sub-loads:
     *
     * Width 0:  (empty)          -- no loads
     * Width 1:  {1}              -- single 1-wide load
     * Width 7:  {4, 2, 1}       -- 4+2+1 = 7
     * Width 8:  {8}              -- single 8-wide load
     * Width 16: {16}             -- single 16-wide load
     */
    using Helper = ThreadwiseTransferHelper_Serpentine;

    using VecSize0  = tuple_element_t<0, Helper::VectorSizeLookupTable>;
    using VecSize1  = tuple_element_t<1, Helper::VectorSizeLookupTable>;
    using VecSize7  = tuple_element_t<7, Helper::VectorSizeLookupTable>;
    using VecSize8  = tuple_element_t<8, Helper::VectorSizeLookupTable>;
    using VecSize16 = tuple_element_t<16, Helper::VectorSizeLookupTable>;

    EXPECT_EQ(VecSize0::Size(), 0);

    EXPECT_EQ(VecSize1::Size(), 1);
    EXPECT_EQ(VecSize1::At(0), 1);

    EXPECT_EQ(VecSize7::Size(), 3);
    EXPECT_EQ(VecSize7::At(0), 4); // first sub-load: 4 elements
    EXPECT_EQ(VecSize7::At(1), 2); // second sub-load: 2 elements
    EXPECT_EQ(VecSize7::At(2), 1); // third sub-load: 1 element

    EXPECT_EQ(VecSize8::Size(), 1);
    EXPECT_EQ(VecSize8::At(0), 8);

    EXPECT_EQ(VecSize16::Size(), 1);
    EXPECT_EQ(VecSize16::At(0), 16);
}

TEST(ThreadwiseTransferHelperSerpentine, VectorOffsetsLookupTable)
{
    /*
     * Starting byte offsets for each sub-load in the decomposition:
     *
     * Width 7 = {4, 2, 1}:
     *   |<--- 4 --->|<- 2 ->|1|
     *   offset 0     offset 4  offset 6
     *
     * So offsets = {0, 4, 6}
     */
    using Helper  = ThreadwiseTransferHelper_Serpentine;
    using VecOff7 = tuple_element_t<7, Helper::VectorOffsetsLookupTable>;

    EXPECT_EQ(VecOff7::Size(), 3);
    EXPECT_EQ(VecOff7::At(0), 0); // first sub-load starts at offset 0
    EXPECT_EQ(VecOff7::At(1), 4); // second sub-load starts at offset 4
    EXPECT_EQ(VecOff7::At(2), 6); // third sub-load starts at offset 6
}

// =============================================================================
// ThreadwiseTransferHelper_SFC tests
// =============================================================================

TEST(ThreadwiseTransferHelperSFC, ComputeSFCCoordinateResetStep_SingleAccess)
{
    /*
     * SliceLengths = (1, 1), ScalarPerAccess = (1, 1)
     * Only 1 access position total -> already at origin, reset = (0, 0)
     *
     *   [*]  <-- single element, no movement needed
     */
    using SFCHelper = ThreadwiseTransferHelper_SFC;

    constexpr auto scalar_per_access = Sequence<1, 1>{};
    constexpr auto reset             = SFCHelper::ComputeSFCCoordinateResetStep<Sequence<1, 1>,
                                                                                Sequence<0, 1>,
                                                                                decltype(scalar_per_access)>();

    EXPECT_EQ(reset[Number<0>{}], 0);
    EXPECT_EQ(reset[Number<1>{}], 0);
}

// =============================================================================
// Inheritance structure tests
// =============================================================================

TEST(ThreadwiseTransferHelperInheritance, SerpentineIsDerivedFromBase)
{
    /*
     * ThreadwiseTransferHelper_Base
     *       |
     *       +-- ThreadwiseTransferHelper_Serpentine  <-- this relationship
     *       |
     *       +-- ThreadwiseTransferHelper_SFC
     */
    static_assert(
        std::is_base_of_v<ThreadwiseTransferHelper_Base, ThreadwiseTransferHelper_Serpentine>);
}

TEST(ThreadwiseTransferHelperInheritance, SFCIsDerivedFromBase)
{
    /*
     * ThreadwiseTransferHelper_Base
     *       |
     *       +-- ThreadwiseTransferHelper_Serpentine
     *       |
     *       +-- ThreadwiseTransferHelper_SFC  <-- this relationship
     */
    static_assert(std::is_base_of_v<ThreadwiseTransferHelper_Base, ThreadwiseTransferHelper_SFC>);
}

TEST(ThreadwiseTransferHelperInheritance, SerpentineAndSFCAreNotRelated)
{
    /*
     * Serpentine and SFC are siblings -- neither inherits from the other.
     *
     * ThreadwiseTransferHelper_Base
     *       |
     *       +-- Serpentine  (NOT parent of SFC)
     *       |
     *       +-- SFC         (NOT parent of Serpentine)
     */
    static_assert(
        !std::is_base_of_v<ThreadwiseTransferHelper_Serpentine, ThreadwiseTransferHelper_SFC>);
    static_assert(
        !std::is_base_of_v<ThreadwiseTransferHelper_SFC, ThreadwiseTransferHelper_Serpentine>);
}

// =============================================================================
// detail:: functor tests
// =============================================================================

TEST(DetailFunctors, LambdaScalarPerAccess)
{
    /*
     * For VectorDim=1 and ScalarPerVector=8:
     *
     *   dim:    0   1   2
     *   result: 1   8   1
     *           ^   ^   ^
     *           |   |   +-- not the vector dim
     *           |   +------ THE vector dim (returns ScalarPerVector)
     *           +---------- not the vector dim
     */
    constexpr auto f = detail::lambda_scalar_per_access<1, 8>{};

    EXPECT_EQ(f(0), 1);
    EXPECT_EQ(f(1), 8);
    EXPECT_EQ(f(2), 1);
}

TEST(DetailFunctors, LambdaScalarStepInVector)
{
    /*
     * For VectorDim=2:
     *
     *   dim:    0   1   2   3
     *   result: 0   0   1   0
     *                   ^
     *                   +-- THE vector dim (step = 1)
     */
    constexpr auto f = detail::lambda_scalar_step_in_vector<2>{};

    EXPECT_EQ(f(0), 0);
    EXPECT_EQ(f(1), 0);
    EXPECT_EQ(f(2), 1);
    EXPECT_EQ(f(3), 0);
}

TEST(DetailFunctors, LambdaScalarPerAccessForSrcAndDst_SameDim)
{
    /*
     * Src and Dst both vectorize dim 1:
     *   SrcVectorDim=1, SrcScalarPerVector=4
     *   DstVectorDim=1, DstScalarPerVector=8
     *
     *   dim:    0    1         2
     *   result: 1    lcm(4,8) 1
     *                = 8
     */
    constexpr auto f = detail::lambda_scalar_per_access_for_src_and_dst<1, 4, 1, 8>{};

    EXPECT_EQ(f(0), 1);
    EXPECT_EQ(f(1), 8); // lcm(4, 8) = 8
    EXPECT_EQ(f(2), 1);
}

TEST(DetailFunctors, LambdaScalarPerAccessForSrcAndDst_DifferentDims)
{
    /*
     * Src vectorizes dim 0 (width 4), Dst vectorizes dim 2 (width 8):
     *
     *   dim:    0         1    2
     *   result: 4(src)    1    8(dst)
     */
    constexpr auto f = detail::lambda_scalar_per_access_for_src_and_dst<0, 4, 2, 8>{};

    EXPECT_EQ(f(0), 4); // src vector dim
    EXPECT_EQ(f(1), 1); // neither
    EXPECT_EQ(f(2), 8); // dst vector dim
}
