// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_types.hpp"

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace fe = hipdnn_frontend;

bool ToHipdnnDataType(miopenDataType_t type, fe::DataType& out)
{
    switch(type)
    {
    case miopenHalf: out = fe::DataType::HALF; return true;
    case miopenFloat: out = fe::DataType::FLOAT; return true;
    case miopenBFloat16: out = fe::DataType::BFLOAT16; return true;
    case miopenDouble: out = fe::DataType::DOUBLE; return true;
    case miopenInt8: out = fe::DataType::INT8; return true;
    case miopenInt32: out = fe::DataType::INT32; return true;
    case miopenInt64: out = fe::DataType::INT64; return true;
    case miopenFloat8_fnuz: out = fe::DataType::FP8_E4M3_FNUZ; return true;
    case miopenBFloat8_fnuz: out = fe::DataType::FP8_E5M2_FNUZ; return true;
    }
    return false;
}

// Accumulate wider than you store: half and bfloat16 both compute in fp32.
bool ComputeTypeFor(miopenDataType_t type, fe::DataType& out)
{
    switch(type)
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz: out = fe::DataType::FLOAT; return true;
    case miopenDouble: out = fe::DataType::DOUBLE; return true;
    case miopenInt8:
    case miopenInt32:
    case miopenInt64: out = fe::DataType::INT32; return true;
    }
    return false;
}

bool ToPointwiseActivation(miopenActivationMode_t mode, fe::PointwiseMode& out)
{
    // Deliberately narrow. The remaining MIOpen activation modes carry alpha,
    // beta or gamma coefficients that the plain pointwise node has nowhere to
    // put, so they are declined rather than silently approximated.
    if(mode == miopenActivationRELU)
    {
        out = fe::PointwiseMode::RELU_FWD;
        return true;
    }
    return false;
}

miopenStatus_t TranslateHipdnnError(const fe::Error& error)
{
    const auto code = error.get_code();

    // Not a switch: hipDNN owns this enum and MIOpen compiles with -Wswitch-enum
    // -Werror, so any value it gains later would break the build here instead of
    // landing on the catch-all, which is already the right answer for it.
    if(code == fe::ErrorCode::GRAPH_NOT_SUPPORTED ||
       code == fe::ErrorCode::UNSUPPORTED_GRAPH_FORMAT)
        return miopenStatusUnsupportedOp;

    if(code == fe::ErrorCode::INVALID_VALUE || code == fe::ErrorCode::ATTRIBUTE_NOT_SET ||
       code == fe::ErrorCode::SHAPE_DEDUCTION_FAILED ||
       code == fe::ErrorCode::INVALID_TENSOR_NAME || code == fe::ErrorCode::INVALID_VARIANT_PACK)
        return miopenStatusBadParm;

    return miopenStatusInternalError;
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
