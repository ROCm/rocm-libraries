// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-time checks that every cuDNN C enum <-> shim FE enum table maps both ways.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>

namespace
{

namespace shim = hipdnn_frontend::compatibility::cudnn_frontend::detail;
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// Each table row maps both ways: its FE value to its cuDNN value, and its cuDNN
// value to an FE value that maps back to it.
template <typename Table>
constexpr bool rowsRoundTrip(const Table& table)
{
    for(const auto& row : table)
    {
        const auto fe = shim::toFe(row.cudnn);
        if(shim::toCudnn(row.fe) != row.cudnn || !fe || shim::toCudnn(*fe) != row.cudnn)
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
        aliases += shim::toFe(row.cudnn) == row.fe ? 0 : 1;
    }
    return aliases;
}

static_assert(rowsRoundTrip(shim::DATA_TYPE_MAP) && feAliasRows(shim::DATA_TYPE_MAP) == 0);
static_assert(rowsRoundTrip(shim::POINTWISE_MODE_MAP)
              && feAliasRows(shim::POINTWISE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::CONVOLUTION_MODE_MAP)
              && feAliasRows(shim::CONVOLUTION_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::REDUCTION_MODE_MAP)
              && feAliasRows(shim::REDUCTION_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::RESAMPLE_MODE_MAP) && feAliasRows(shim::RESAMPLE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::PADDING_MODE_MAP) && feAliasRows(shim::PADDING_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::NORM_FWD_PHASE_MAP)
              && feAliasRows(shim::NORM_FWD_PHASE_MAP) == 0);
static_assert(rowsRoundTrip(shim::NORM_MODE_MAP) && feAliasRows(shim::NORM_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::BEHAVIOR_NOTE_MAP) && feAliasRows(shim::BEHAVIOR_NOTE_MAP) == 0);
static_assert(rowsRoundTrip(shim::NUMERICAL_NOTE_MAP)
              && feAliasRows(shim::NUMERICAL_NOTE_MAP) == 0);
static_assert(rowsRoundTrip(shim::TENSOR_REORDERING_MAP)
              && feAliasRows(shim::TENSOR_REORDERING_MAP) == 0);
static_assert(rowsRoundTrip(shim::RESHAPE_MODE_MAP) && feAliasRows(shim::RESHAPE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::KNOB_TYPE_MAP) && feAliasRows(shim::KNOB_TYPE_MAP) == 0);
static_assert(rowsRoundTrip(shim::RNG_DISTRIBUTION_MAP)
              && feAliasRows(shim::RNG_DISTRIBUTION_MAP) == 0);
// OPERATION_CONCATENATE_DESCRIPTOR aliases OPERATION_CONCAT_DESCRIPTOR.
static_assert(rowsRoundTrip(shim::DESCRIPTOR_TYPE_MAP)
              && feAliasRows(shim::DESCRIPTOR_TYPE_MAP) == 1);
static_assert(shim::toFe(CUDNN_RESAMPLE_AVGPOOL) == fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING);
static_assert(!shim::toCudnn(fe::DataType_t::NOT_SET).has_value());

} // namespace
