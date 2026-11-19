// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-time checks that every cuDNN C enum <-> compatibility layer FE enum table maps both ways.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>

namespace
{

namespace compat = hipdnn_frontend::compatibility::cudnn_frontend::detail;
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// Each table row maps both ways: its FE value to its cuDNN value, and its cuDNN
// value to an FE value that maps back to it.
template <typename Table>
constexpr bool rowsRoundTrip(const Table& table)
{
    for(const auto& row : table)
    {
        const auto fe = compat::toFe(row.cudnn);
        if(compat::toCudnn(row.fe) != row.cudnn || !fe || compat::toCudnn(*fe) != row.cudnn)
        {
            return false;
        }
    }
    return true;
}

// Rows whose FE value aliases another row's cuDNN value, so toFe returns the other FE value.
template <typename Table>
constexpr int feAliasRows(const Table& table)
{
    int aliases = 0;
    for(const auto& row : table)
    {
        aliases += compat::toFe(row.cudnn) == row.fe ? 0 : 1;
    }
    return aliases;
}

static_assert(rowsRoundTrip(compat::DATA_TYPE_MAP) && feAliasRows(compat::DATA_TYPE_MAP) == 0);
static_assert(rowsRoundTrip(compat::POINTWISE_MODE_MAP)
              && feAliasRows(compat::POINTWISE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::CONVOLUTION_MODE_MAP)
              && feAliasRows(compat::CONVOLUTION_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::REDUCTION_MODE_MAP)
              && feAliasRows(compat::REDUCTION_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::RESAMPLE_MODE_MAP)
              && feAliasRows(compat::RESAMPLE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::PADDING_MODE_MAP)
              && feAliasRows(compat::PADDING_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::NORM_FWD_PHASE_MAP)
              && feAliasRows(compat::NORM_FWD_PHASE_MAP) == 0);
static_assert(rowsRoundTrip(compat::NORM_MODE_MAP) && feAliasRows(compat::NORM_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::BEHAVIOR_NOTE_MAP)
              && feAliasRows(compat::BEHAVIOR_NOTE_MAP) == 0);
static_assert(rowsRoundTrip(compat::NUMERICAL_NOTE_MAP)
              && feAliasRows(compat::NUMERICAL_NOTE_MAP) == 0);
static_assert(rowsRoundTrip(compat::TENSOR_REORDERING_MAP)
              && feAliasRows(compat::TENSOR_REORDERING_MAP) == 0);
static_assert(rowsRoundTrip(compat::RESHAPE_MODE_MAP)
              && feAliasRows(compat::RESHAPE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(compat::KNOB_TYPE_MAP) && feAliasRows(compat::KNOB_TYPE_MAP) == 0);
static_assert(rowsRoundTrip(compat::RNG_DISTRIBUTION_MAP)
              && feAliasRows(compat::RNG_DISTRIBUTION_MAP) == 0);
// OPERATION_CONCATENATE_DESCRIPTOR aliases OPERATION_CONCAT_DESCRIPTOR.
static_assert(rowsRoundTrip(compat::DESCRIPTOR_TYPE_MAP)
              && feAliasRows(compat::DESCRIPTOR_TYPE_MAP) == 1);
static_assert(compat::toFe(CUDNN_RESAMPLE_AVGPOOL) == fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING);
static_assert(!compat::toCudnn(fe::DataType_t::NOT_SET).has_value());

} // namespace
