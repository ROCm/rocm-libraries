// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <ck_tile/builder/types.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

namespace ckb = ck_tile::builder;

// Data types
[[maybe_unused]] constexpr auto BF16 = ckb::DataType::BF16;
[[maybe_unused]] constexpr auto F16  = ckb::DataType::FP16;
[[maybe_unused]] constexpr auto F32  = ckb::DataType::FP32;
[[maybe_unused]] constexpr auto I8   = ckb::DataType::I8;
[[maybe_unused]] constexpr auto I32  = ckb::DataType::I32;
[[maybe_unused]] constexpr auto F8   = ckb::DataType::FP8;
[[maybe_unused]] constexpr auto BF8  = ckb::DataType::BF8;

// Elementwise operations
[[maybe_unused]] constexpr auto PassThrough = ckb::ElementwiseOperation::PASS_THROUGH;
[[maybe_unused]] constexpr auto Bilinear    = ckb::ElementwiseOperation::BILINEAR;

// GEMM specializations
[[maybe_unused]] constexpr auto GemmDefault    = ckb::GemmSpecialization::Default;
[[maybe_unused]] constexpr auto GemmMNKPadding = ckb::GemmSpecialization::MNKPadding;

// Pipeline schedulers
[[maybe_unused]] constexpr auto Intrawave = ckb::PipelineScheduler::INTRAWAVE;
[[maybe_unused]] constexpr auto Interwave = ckb::PipelineScheduler::INTERWAVE;

// Pipeline versions
[[maybe_unused]] constexpr auto PipeV1 = ckb::PipelineVersion::V1;
[[maybe_unused]] constexpr auto PipeV2 = ckb::PipelineVersion::V2;
[[maybe_unused]] constexpr auto PipeV3 = ckb::PipelineVersion::V3;
[[maybe_unused]] constexpr auto PipeV4 = ckb::PipelineVersion::V4;
[[maybe_unused]] constexpr auto PipeV5 = ckb::PipelineVersion::V5;

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
