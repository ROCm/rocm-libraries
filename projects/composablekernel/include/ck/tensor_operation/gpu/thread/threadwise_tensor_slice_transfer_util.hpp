// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

namespace detail {

template <index_t VectorDim, index_t ScalarPerVector>
struct lambda_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? ScalarPerVector : 1;
    }
};

template <index_t VectorDim>
struct lambda_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? 1 : 0;
    }
};

template <index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t DstVectorDim,
          index_t DstScalarPerVector>
struct lambda_scalar_per_access_for_src_and_dst
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if(i == SrcVectorDim && i == DstVectorDim)
        {
            return math::lcm(SrcScalarPerVector, DstScalarPerVector);
        }
        else if(i == SrcVectorDim)
        {
            return SrcScalarPerVector;
        }
        else if(i == DstVectorDim)
        {
            return DstScalarPerVector;
        }
        else
        {
            return 1;
        }
    }
};

template <index_t WaveNum, index_t nDim>
struct lambda_wave_cluster_dimension
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if((nDim - i) == 3)
            return WaveNum;
        else
            return 1;
    }
};

} // namespace detail

// =====================================================================
// Base helper with methods shared by all threadwise transfer variants.
// Both serpentine and SpaceFillingCurve helpers inherit from this.
// =====================================================================
struct ThreadwiseTransferHelper_Base
{
    // Move slice window with optional coordinate reset fusion.
    template <typename Desc,
              typename Coord,
              bool ResetCoordinateAfterRun,
              typename StepIdx,
              typename GetCoordinateResetStepFunc>
    __device__ static void MoveSliceWindow(const Desc& desc,
                                           Coord& coord,
                                           const StepIdx& slice_origin_step_idx,
                                           GetCoordinateResetStepFunc get_reset_step)
    {
        // if coord was not reset by RunRead/RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx = ResetCoordinateAfterRun
                                           ? slice_origin_step_idx
                                           : slice_origin_step_idx + get_reset_step();

        const auto adjusted_step = make_tensor_coordinate_step(desc, adjusted_step_idx);

        move_tensor_coordinate(desc, coord, adjusted_step);
    }

    // Compute thread scratch descriptor for a given vector dimension and scalar per vector.
    // Builds a transformed tensor descriptor with merge on the vector dimension.
    template <index_t nDim, typename SliceLengths, index_t VectorDim, index_t ScalarPerVector_>
    __device__ static constexpr auto ComputeThreadScratchDescriptor()
    {
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector_>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        constexpr auto access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(access_lengths), Number<ScalarPerVector_>{});

        // 1st stage of transforms
        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(access_lengths_and_vector_length);

        // 2nd stage of transforms
        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == VectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(access_lengths_and_vector_length[i],
                                   access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == VectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }
};

// =====================================================================
// Serpentine (snake/zigzag) traversal helper.
// Used by: v3r1, v3r2, v3r1_gather, v3r1_dequant
// =====================================================================
struct ThreadwiseTransferHelper_Serpentine : ThreadwiseTransferHelper_Base
{
    // Compile-time index constants
    static constexpr auto I0  = Number<0>{};
    static constexpr auto I1  = Number<1>{};
    static constexpr auto I2  = Number<2>{};
    static constexpr auto I3  = Number<3>{};
    static constexpr auto I4  = Number<4>{};
    static constexpr auto I5  = Number<5>{};
    static constexpr auto I6  = Number<6>{};
    static constexpr auto I7  = Number<7>{};
    static constexpr auto I8  = Number<8>{};
    static constexpr auto I10 = Number<10>{};
    static constexpr auto I12 = Number<12>{};
    static constexpr auto I13 = Number<13>{};
    static constexpr auto I14 = Number<14>{};
    static constexpr auto I16 = Number<16>{};

    // Binary decomposition of vector widths 0-16 into power-of-2 sub-load sizes.
    // Index N gives the sequence of sub-load widths whose sum equals N.
    // E.g. index 7 -> Sequence<I4, I2, I1> means loads of width 4, 2, 1.
    using VectorSizeLookupTable = Tuple<Sequence<>,
                                        Sequence<I1>,
                                        Sequence<I2>,
                                        Sequence<I2, I1>,
                                        Sequence<I4>,
                                        Sequence<I4, I1>,
                                        Sequence<I4, I2>,
                                        Sequence<I4, I2, I1>,
                                        Sequence<I8>,
                                        Sequence<I8, I1>,
                                        Sequence<I8, I2>,
                                        Sequence<I8, I2, I1>,
                                        Sequence<I8, I4>,
                                        Sequence<I8, I4, I1>,
                                        Sequence<I8, I4, I2>,
                                        Sequence<I8, I4, I2, I1>,
                                        Sequence<I16>>;

    // Starting offsets for each sub-load in VectorSizeLookupTable.
    // E.g. index 7 -> Sequence<I0, I4, I6> means offsets 0, 4, 6.
    using VectorOffsetsLookupTable = Tuple<Sequence<>,
                                           Sequence<I0>,
                                           Sequence<I0>,
                                           Sequence<I0, I2>,
                                           Sequence<I0>,
                                           Sequence<I0, I4>,
                                           Sequence<I0, I4>,
                                           Sequence<I0, I4, I6>,
                                           Sequence<I0>,
                                           Sequence<I0, I8>,
                                           Sequence<I0, I8>,
                                           Sequence<I0, I8, I10>,
                                           Sequence<I0, I8>,
                                           Sequence<I0, I8, I12>,
                                           Sequence<I0, I8, I12>,
                                           Sequence<I0, I8, I12, I14>,
                                           Sequence<I0>>;

    // Compute serpentine (snake/zigzag) sweep direction for each dimension.
    // Returns a StaticallyIndexedArray_v2<bool, nDim> where true = forward, false = backward.
    template <index_t nDim, typename OrderedAccessIdx, typename OrderedAccessLengths>
    __device__ static constexpr auto
    ComputeForwardSweep(const OrderedAccessIdx& ordered_access_idx,
                        const OrderedAccessLengths& ordered_access_lengths)
    {
        StaticallyIndexedArray_v2<bool, nDim> forward_sweep_;

        forward_sweep_(I0) = true;

        static_for<1, nDim, 1>{}([&](auto i) {
            index_t tmp = ordered_access_idx[I0];

            static_for<1, i, 1>{}(
                [&](auto j) { tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j]; });

            forward_sweep_(i) = tmp % 2 == 0;
        });

        return forward_sweep_;
    }

    // Compute which dimensions need coordinate movement at a given iteration point.
    // A dimension moves when it hasn't reached its end and all higher-priority dimensions
    // have completed their ranges.
    template <index_t nDim, typename OrderedAccessIdx, typename OrderedAccessLengths>
    __device__ static constexpr auto
    ComputeMoveOnDim(const OrderedAccessIdx& ordered_access_idx,
                     const OrderedAccessLengths& ordered_access_lengths)
    {
        StaticallyIndexedArray_v2<bool, nDim> move_on_dim_;

        static_for<0, nDim, 1>{}([&](auto i) {
            move_on_dim_(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;

            static_for<i + 1, nDim, 1>{}([&](auto j) {
                move_on_dim_(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
            });
        });

        return move_on_dim_;
    }

    // Compute data index from ordered access index, converting back to natural dimension order.
    template <index_t nDim,
              typename OrderedAccessIdx,
              typename OrderedAccessLengths,
              typename ForwardSweep,
              typename DimAccessOrder,
              typename ScalarPerAccess>
    __device__ static constexpr auto
    ComputeDataIndex(const OrderedAccessIdx& ordered_access_idx,
                     const OrderedAccessLengths& ordered_access_lengths,
                     const ForwardSweep& forward_sweep,
                     const DimAccessOrder& dim_access_order,
                     const ScalarPerAccess& scalar_per_access)
    {
        MultiIndex<nDim> ordered_idx;

        static_for<0, nDim, 1>{}([&](auto i) {
            ordered_idx(i) = forward_sweep[i]
                                 ? ordered_access_idx[i]
                                 : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
        });

        return container_reorder_given_old2new(ordered_idx, dim_access_order) * scalar_per_access;
    }

    // Compute forward coordinate steps for each dimension.
    template <index_t nDim, typename Desc, typename ScalarPerAccess>
    __device__ static constexpr auto ComputeForwardSteps(const Desc& desc,
                                                         const ScalarPerAccess& scalar_per_access)
    {
        return generate_tuple(
            [&](auto i) {
                MultiIndex<nDim> forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(desc, forward_step_idx);
            },
            Number<nDim>{});
    }

    // Compute backward coordinate steps for each dimension.
    template <index_t nDim, typename Desc, typename ScalarPerAccess>
    __device__ static constexpr auto ComputeBackwardSteps(const Desc& desc,
                                                          const ScalarPerAccess& scalar_per_access)
    {
        return generate_tuple(
            [&](auto i) {
                MultiIndex<nDim> backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(desc, backward_step_idx);
            },
            Number<nDim>{});
    }

    // Compute coordinate reset step (to return to origin after serpentine traversal).
    template <index_t nDim,
              typename SliceLengths,
              index_t VectorDim,
              index_t ScalarPerVector_,
              typename DimAccessOrder>
    __device__ static constexpr auto ComputeCoordinateResetStep()
    {
        // scalar per access on each dim
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector_>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto ordered_access_lengths_minus_1 = generate_tuple(
            [&](auto i) { return Number<ordered_access_lengths.At(i) - 1>{}; }, Number<nDim>{});
        constexpr auto forward_sweep =
            ComputeForwardSweep<nDim>(ordered_access_lengths_minus_1, ordered_access_lengths);

        // calculate data index after last iteration, if it has not being reset
        constexpr auto data_idx = [&]() {
            MultiIndex<nDim> ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                   scalar_per_access;
        }();

        constexpr auto reset_data_step = [&]() {
            MultiIndex<nDim> reset_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_data_step_(i) = -data_idx[i]; });

            return reset_data_step_;
        }();

        return reset_data_step;
    }
};

// =====================================================================
// SpaceFillingCurve traversal helper.
// Used by: v6r1, v6r1r2, v6r2, v6r3, v7r2, v7r3, v7r3_scatter
// =====================================================================
struct ThreadwiseTransferHelper_SFC : ThreadwiseTransferHelper_Base
{
    // Compute coordinate reset step using SpaceFillingCurve traversal.
    // Returns the step needed to move from the last access position back to the origin.
    template <typename SliceLengths, typename DimAccessOrder, typename ScalarPerAccess>
    __device__ static constexpr auto ComputeSFCCoordinateResetStep()
    {
        using SFC = SpaceFillingCurve<SliceLengths, DimAccessOrder, remove_cv_t<ScalarPerAccess>>;

        constexpr auto num_access = SFC::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SFC::Index{};
        }
        else
        {
            return SFC::GetStepBetween(Number<num_access - 1>{}, Number<0>{});
        }
    }

    // Generate a tuple of vector types from a data type tuple.
    template <typename DataTypes, index_t ScalarPerVector>
    __device__ static auto GenerateVectors()
    {
        auto data_types = DataTypes{};

        constexpr index_t num = data_types.Size();

        return generate_tuple(
            [&](auto i) {
                using DataType = remove_cvref_t<decltype(data_types[i])>;

                return vector_type_maker_t<DataType, ScalarPerVector>{};
            },
            Number<num>{});
    }
};

} // namespace ck
