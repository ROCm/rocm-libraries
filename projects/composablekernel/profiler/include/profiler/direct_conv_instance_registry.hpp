// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include "grouped_convolution_signatures.hpp"
#include "ck_tile/builder/testing/conv/args.hpp"

namespace ck_tile::builder::profiling {

namespace ckt = ck_tile::builder::test;

/// Function pointer type matching run_direct_conv<Kernel, SIGNATURE>.
template <auto SIGNATURE>
using AlgFuncPtr = std::tuple<bool, float, std::string> (*)(
    const ckt::Args<SIGNATURE>&,
    const ckt::Inputs<SIGNATURE>&,
    const ckt::Outputs<SIGNATURE>&,
    const ck_tile::stream_config&);

// Forward — per-channel variants
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_FWD>> get_fwd_direct_instances_nhwgc_fp16_4c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_FWD>> get_fwd_direct_instances_nhwgc_fp16_16c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_FWD>> get_fwd_direct_instances_nhwgc_fp16_8c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_FWD>> get_fwd_direct_instances_nhwgc_fp16_32c();

std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_FWD>> get_fwd_direct_instances_nhwgc_bf16_4c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_FWD>> get_fwd_direct_instances_nhwgc_bf16_16c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_FWD>> get_fwd_direct_instances_nhwgc_bf16_8c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_FWD>> get_fwd_direct_instances_nhwgc_bf16_32c();

// Backward data — per-channel variants
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_fp16_4c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_fp16_16c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_fp16_8c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_FP16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_fp16_32c();

std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_bf16_4c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_bf16_16c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_bf16_8c();
std::vector<AlgFuncPtr<SIGNATURE_NHWGC_BF16_BWD_DATA>> get_bwd_data_direct_instances_nhwgc_bf16_32c();

} // namespace ck_tile::builder::profiling
