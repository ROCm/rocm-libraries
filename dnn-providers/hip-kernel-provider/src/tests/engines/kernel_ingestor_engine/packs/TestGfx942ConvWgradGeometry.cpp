// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/Gfx942ConvWgradGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx942ConvWgradTestCatalog.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <exception>
#include <utility>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#endif

/**
 * @file TestGfx942ConvWgradGeometry.cpp
 * @brief The hipkernel:Gfx942ConvWgradAot launch plan against the rocKE adapter's own
 *        (rocke/library/builders/common/convolution_backward_filter.py): every expected
 *        value below -- digests, serialized plans, split-K degrees -- was produced once by
 *        that adapter (gfx942_conv_wgrad_aot_plan, Plan.serialize/digest) and by
 *        kernels.common.conv_split_k.resolve_conv_split_k under ROCKE_LLVM_FLAVOR=llvm22,
 *        over hand-built instances of the shipped gfx950 wgrad catalog, and pasted here as
 *        literals; nothing runs Python at test time. Then the rules the plan encodes one at
 *        a time, the metadata decoding, and the native symbols' match / workspace / prepare
 *        lockstep. Host-only: no kernel is loaded or launched.
 */
namespace
{

namespace wgrad = hip_kernel_provider::kernel_ingestor_engine::gfx942_conv_wgrad;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::wgrad_test_vectors;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::wgrad_test_catalog;
using wgrad::AotInstance;
using wgrad::DirectCapabilities;
using wgrad::DirectKnobs;
using wgrad::DirectVariant;
using wgrad::ElementType;
using wgrad::Epilogue;
using wgrad::ImplicitInstance;
using wgrad::Problem;

constexpr auto FP16 = ElementType::FP16;
constexpr auto BF16 = ElementType::BF16;

AotInstance implicitInstance(const ImplicitInstance& instance)
{
    return AotInstance{wgrad::KernelFamily::IMPLICIT_GEMM, FP16, instance, std::nullopt};
}

AotInstance directInstance(DirectVariant variant,
                           const DirectCapabilities& compiled,
                           const DirectKnobs& knobs)
{
    return AotInstance{wgrad::KernelFamily::DIRECT,
                       FP16,
                       std::nullopt,
                       wgrad::DirectInstance{variant, compiled, knobs}};
}

std::string describe(const Problem& p)
{
    return "{N" + std::to_string(p.n) + " C" + std::to_string(p.c) + " K" + std::to_string(p.k)
           + " " + std::to_string(p.hi) + "x" + std::to_string(p.wi) + " f" + std::to_string(p.y)
           + "x" + std::to_string(p.x) + " s" + std::to_string(p.strideH) + " p"
           + std::to_string(p.padH) + " d" + std::to_string(p.dilationH) + " g"
           + std::to_string(p.groups) + (p.dtype == BF16 ? " bf16}" : " fp16}");
}

// ---------------------------------------------------------------------------------------
// Golden plans
// ---------------------------------------------------------------------------------------

#if 0 // superseded by PlansMatchTheAdapterOverTheCatalog/FullPlansMatchTheAdapter below, \
    // which are generated from the real shipped catalog and stay correct across a        \
    // split-K fix like the one that made these hand-pasted digests stale.
/// One (problem, instance) of the adapter: whether it plans, and if so its split-K degree
/// and Plan.digest(). Up to ten served instances of the shipped catalog per problem
/// (plus one of each family the ten miss) and four refused ones (two-stage ones among
/// them), seeded-random.
struct GoldenRow
{
    const char* shape;
    Problem problem;
    AotInstance instance;
    ElementType dtype;
    bool served;
    int64_t splitK;
    uint64_t digest;
};

// clang-format off
const std::vector<GoldenRow>& goldenRows()
{
    static const std::vector<GoldenRow> s_rows{
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 15, 512, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x3db30e3b41e13354ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x6883646788e3ad37ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0xf46590ce830ff8b8ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x1bccab38a14c094eULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 4, 0, 2, 0, 30, 256, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0xd4572d69ee27bcd3ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x6883646788e3ad37ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 2, 0x4485be573dd7880fULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x013070143a4847b3ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x25357309f48ba512ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 64, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0x68d92d0f8ba14937ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 64, 2, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 256, 64, 4, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::GATHER, {3, 3, 1, 1, 1, 1, 1, 1, 3, 0}, {0, 0, 0, 2, 0, 1, 0, 8, 1024, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense3x3", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 32, 32, 2, 1, 8, 1, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0xc5b5921aa359157dULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 128, 64, 1, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0x0a291ec698ca7c28ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x062c0e201067dee8ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 32, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x78f9e2a766d8cee4ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 256, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0x44384ba6e1547259ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0xe5f2d91840b60d5dULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 128, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x240cc43eacd35175ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x220c7bf4e91c692cULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0xe5f2d91840b60d5dULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({16, 128, 64, 1, 8, 4, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0x003ca37d1ba877e9ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 256, 16, 1, 4, 0, 0, 0, Epilogue::DEFAULT, 4, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, implicitInstance({16, 256, 32, 1, 8, 0, 0, 0, Epilogue::DEFAULT, 4, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 14, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_stride2", {2, 64, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::DW, {3, 3, 2, 2, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 64, 1024, 0, 0, 0, 8, 16, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 2, 0}), ElementType::FP16, true, 1, 0x5c6b4b2fbad2b364ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 2, 0}), ElementType::FP16, true, 1, 0x5c6b4b2fbad2b364ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 2, 3, 0, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 1, 0}), ElementType::FP16, true, 1, 0x5fa023d58b5a3fe8ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, implicitInstance({32, 256, 16, 1, 4, 0, 0, 0, Epilogue::DEFAULT, 4, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, implicitInstance({128, 256, 32, 1, 8, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg4", {2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16}, implicitInstance({256, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 32, 2048, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x7729e8b4a559786cULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::FP16, true, 1, 0x9edb801beba29632ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::FP16, true, 1, 0x9edb801beba29632ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 8, 28, 0, 0}), ElementType::FP16, true, 1, 0xf9136a4f766c52a8ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0}), ElementType::FP16, true, 1, 0x99604e5f906db730ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 8, 28, 0, 0}), ElementType::FP16, true, 1, 0xf9136a4f766c52a8ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 14, 0, 0}), ElementType::FP16, true, 1, 0xdc1cd7541308849aULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, implicitInstance({256, 64, 64, 8, 1, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 5, 6, 30, 512, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, implicitInstance({32, 128, 16, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise3x3", {4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 15, 256, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"stem_c3_11x11", {4, 3, 3, 64, 64, 11, 11, 1, 1, 5, 5, 1, 1, 3, FP16}, directInstance(DirectVariant::DW_COLS, {11, 11, 1, 1, 5, 5, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 4096, 0, 0, 0, 0, 0, 3, 8, 256, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x789796c418c9dd72ULL},
    {"stem_c3_11x11", {4, 3, 3, 64, 64, 11, 11, 1, 1, 5, 5, 1, 1, 3, FP16}, implicitInstance({128, 128, 64, 2, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"stem_c3_11x11", {4, 3, 3, 64, 64, 11, 11, 1, 1, 5, 5, 1, 1, 3, FP16}, implicitInstance({32, 32, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"stem_c3_11x11", {4, 3, 3, 64, 64, 11, 11, 1, 1, 5, 5, 1, 1, 3, FP16}, implicitInstance({32, 256, 64, 2, 8, 0, 0, 0, Epilogue::DEFAULT, 8, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"stem_c3_11x11", {4, 3, 3, 64, 64, 11, 11, 1, 1, 5, 5, 1, 1, 3, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 2, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 1, 4, 8, 4, 4, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 8, 0xc2c281f1c4dfa6e6ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 1, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xc2c281f1c4dfa6e6ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 256, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 11, 0x075e1c09c9ab105eULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({16, 128, 32, 1, 8, 4, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 8, 0xdc1568fb9097b737ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 11, 0x26e1cfb5792d922eULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xb1471c44c8993bc7ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 12, 0xccf923c8825bc229ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x701c2198f60b0956ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 11, 0x26e1cfb5792d922eULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0x508f474bf0825427ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 15, 512, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x6844ed40da88ff8cULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::GATHER, {3, 3, 1, 1, 1, 1, 1, 1, 24, 0}, {0, 0, 0, 1, 0, 2, 0, 16, 1024, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0xedab2b76da6f11bcULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 8, 8}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 2, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 128, 64, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::DW_MFMA, {5, 5, 1, 1, 2, 2, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"gather_c24", {4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 256, 64, 4, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 2, 0}), ElementType::FP16, true, 1, 0xccffcc973629fbd4ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 2, 3, 0, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 1, 0}), ElementType::FP16, true, 1, 0xfc17faaa9f3ee2d2ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 2, 0}), ElementType::FP16, true, 1, 0x0bfebe284c869238ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, implicitInstance({32, 128, 64, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, directInstance(DirectVariant::GATHER, {3, 3, 2, 2, 1, 1, 1, 1, 3, 0}, {0, 0, 0, 2, 0, 1, 0, 16, 1024, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped_mfma_cpg4", {4, 16, 16, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 4, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 15, 512, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 3, 0x0fb9ba9159208523ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({192, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x710fd1dea2a831bdULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 5, 6, 30, 256, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x6a98dedf1e4d2a31ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 30, 512, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0xd09a4878414ba488ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 128, 64, 1, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 3, 0x67092dc615e512abULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xca57456734f8c8a1ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({16, 128, 32, 1, 8, 4, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 3, 0x52a3bbd986af33c6ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xb038d7d95c33af20ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 32, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 3, 0xa230a2b0035bb163ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 32, 1, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xca57456734f8c8a1ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x01bf5f7f4eb580b7ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 256, 64, 1, 8, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({16, 256, 32, 1, 8, 0, 0, 0, Epilogue::DEFAULT, 4, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 192, 16, 1, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_big", {32, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 192, 16, 1, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 16, 0x56a6066f750437b0ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({256, 256, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 16, 0xe1d6b232c3ab43d5ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({128, 16, 64, 8, 1, 8, 1, 1, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0x4662256d3cd88399ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({128, 256, 32, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 16, 0xe1d6b232c3ab43d5ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 2, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 8, 0x8cf2b7bc6f9dae07ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xf24f3cfba15e2f84ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 16, 0x58ddbbaf341c363dULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({128, 256, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 16, 0xe1d6b232c3ab43d5ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 16, 0x58ddbbaf341c363dULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 16, 0x56a6066f750437b0ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, directInstance(DirectVariant::DW_MFMA, {5, 5, 1, 1, 2, 2, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({16, 192, 32, 1, 4, 1, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, implicitInstance({64, 192, 32, 2, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_1x1", {64, 512, 128, 28, 28, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16}, directInstance(DirectVariant::GATHER, {3, 3, 2, 2, 1, 1, 1, 1, 32, 0}, {0, 0, 0, 3, 0, 2, 0, 8, 256, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({64, 64, 64, 1, 4, 8, 4, 4, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 4, 0xe069dd8c20cba622ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({128, 256, 32, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 8, 0xac0fa6eaafc97f03ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xac0fa6eaafc97f03ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({64, 128, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xcc4cede84d2676e3ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({256, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xcc4cede84d2676e3ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({64, 64, 32, 2, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0xe1a6f1875ee944d1ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0xe069dd8c20cba622ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({32, 32, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x20b7548fa5c3b046ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({128, 192, 32, 1, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xf708b2f123340f62ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({192, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 8, 0xcc4cede84d2676e3ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 8, 8}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 2, 0}), ElementType::FP16, true, 1, 0x166aa80b8b5a1adbULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({32, 192, 32, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({16, 256, 64, 1, 4, 0, 0, 0, Epilogue::DEFAULT, 4, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({128, 256, 32, 2, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"grouped32", {8, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32, FP16}, implicitInstance({128, 128, 64, 2, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({64, 64, 32, 1, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 7, 0xf433826df6721f34ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({128, 256, 64, 1, 8, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::BF16, true, 8, 0x7958afeeb007f299ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({128, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 7, 0x1038c8fdb551fa6eULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 8, 0x6254b9a69a51e408ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({192, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 7, 0x1038c8fdb551fa6eULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({256, 256, 64, 2, 8, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 10, 0x94933afbe0995ad9ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({32, 32, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 4, 0xb63abb694eb47e52ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({128, 16, 64, 8, 1, 8, 1, 1, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 5, 0xec1d5338f758af7dULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({64, 128, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 4, 0x2857d166070ef9ebULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({64, 128, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 4, 0x2857d166070ef9ebULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, directInstance(DirectVariant::DW, {3, 3, 2, 2, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 64, 1024, 0, 0, 0, 8, 16, 0, 0, 0, 0, 0, 0, 0}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, implicitInstance({16, 192, 32, 1, 4, 0, 0, 0, Epilogue::DEFAULT, 16, 2, true, true, false}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, directInstance(DirectVariant::DW_MFMA, {5, 5, 1, 1, 2, 2, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dense_s2_bf16", {16, 128, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF16}, directInstance(DirectVariant::DW, {3, 3, 2, 2, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 32, 1024, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {30, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x8caece5e82e06cd5ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 16, 64, 8, 1, 8, 1, 1, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0xf4e06d76e7869222ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x8caece5e82e06cd5ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x90f7973a37cbbd87ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({192, 192, 32, 6, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::DW_MFMA, {7, 7, 1, 1, 3, 3, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 8, 28, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"odd_cpg", {2, 33, 48, 17, 19, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 64, 2, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 8, 28, 0, 0}), ElementType::FP16, true, 1, 0x31e46a55707deb1dULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::FP16, true, 1, 0xa2186d7fce93bad8ULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0}), ElementType::FP16, true, 1, 0x31e46a55707deb1dULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 8, 28, 0, 0}), ElementType::FP16, true, 1, 0x31e46a55707deb1dULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 14, 0, 0}), ElementType::FP16, true, 1, 0xa2186d7fce93bad8ULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::FP16, true, 1, 0xca9b0edc8a09b23eULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::DW, {3, 3, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 32, 2048, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0xb847ce6cc7dbe68cULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::GATHER, {3, 3, 2, 2, 1, 1, 1, 1, 24, 0}, {0, 0, 0, 6, 0, 2, 0, 16, 256, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, implicitInstance({256, 256, 64, 2, 8, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, implicitInstance({32, 256, 64, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"depthwise_big", {16, 256, 256, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 256, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 4, 0, 2, 0, 15, 512, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dw5x5_bf16", {8, 128, 128, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 128, BF16}, directInstance(DirectVariant::DW_MFMA, {5, 5, 1, 1, 2, 2, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 1024, 1, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0}), ElementType::BF16, true, 1, 0x6d7698a2c6fc724cULL},
    {"dw5x5_bf16", {8, 128, 128, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 128, BF16}, directInstance(DirectVariant::DW_MFMA, {5, 5, 1, 1, 2, 2, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 2048, 1, 0, 0, 0, 0, 0, 0, 0, 4, 28, 0, 0}), ElementType::BF16, true, 1, 0x5a4428b2625a8ea4ULL},
    {"dw5x5_bf16", {8, 128, 128, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 128, BF16}, implicitInstance({256, 192, 64, 4, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dw5x5_bf16", {8, 128, 128, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 128, BF16}, implicitInstance({32, 128, 64, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dw5x5_bf16", {8, 128, 128, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 128, BF16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dw5x5_bf16", {8, 128, 128, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 128, BF16}, implicitInstance({32, 144, 64, 2, 1, 0, 0, 0, Epilogue::DEFAULT, 16, 2, true, true, false}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({128, 64, 32, 4, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 16, 0xd48720dd328ebfb0ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({256, 256, 64, 2, 8, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 8, 0x5d12e720be06a2fdULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 8, 8}, {0, 0, 0, 0, 4, 3, 0, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 1, 0}), ElementType::BF16, true, 1, 0xaef4e6519ecc92aeULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 16, 0x85516cad5970665cULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({64, 128, 64, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 16, 0x9a7e78ec13be5421ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 8, 8}, {0, 0, 0, 0, 4, 2, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 2, 0}), ElementType::BF16, true, 1, 0xa956921b9067ba2cULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 16, 0x73cf4be45a77d810ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::BF16, true, 16, 0x85516cad5970665cULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({256, 128, 32, 2, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::BF16, true, 16, 0x39c6a3f33ba76c81ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({256, 128, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::BF16, true, 16, 0x9a7e78ec13be5421ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, directInstance(DirectVariant::GATHER, {3, 3, 2, 2, 1, 1, 1, 1, 24, 0}, {0, 0, 0, 6, 0, 2, 0, 16, 256, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 4, 0, 2, 0, 30, 512, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"grouped_cpg8_bf16", {4, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 32, BF16}, implicitInstance({256, 128, 64, 2, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::BF16, false, 0, 0x0000000000000000ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x556c0a0bbf95c230ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0xbe9d3a96672221adULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 30, 512, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x78d2018661091c41ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 64, 32, 4, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x2465135ef5c47a35ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 256, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x556c0a0bbf95c230ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 192, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x8c09e4ad49bde551ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x04028b4ffe1828c8ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 256, 64, 1, 8, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0x556c0a0bbf95c230ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x5de5659d8f0b3a0dULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({64, 64, 64, 2, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 1, 0x5ef10c04d2fb744eULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, directInstance(DirectVariant::WGRAD, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {60, 32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, true, 1, 0x7facc51e05c48ef8ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({32, 256, 64, 1, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 4, 4, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({128, 128, 64, 4, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_tiny_k", {1, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16}, implicitInstance({192, 192, 64, 2, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({256, 128, 32, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x3055982deb034989ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({16, 128, 64, 1, 8, 4, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 2, 0xf1b2e70e37314569ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({128, 64, 32, 4, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 3, 0x9b6d952d566be7aeULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({64, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 2, false, true, false}), ElementType::FP16, true, 2, 0xacdb55f4bd538db0ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({64, 64, 32, 2, 1, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 2, 0xa5188d95a099b1c7ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({192, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0xff0ad4d227995698ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({128, 192, 32, 1, 4, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0x4cb71cbe2a09c4e0ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0xff0ad4d227995698ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({32, 64, 64, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 1, 0x52ef1d1a61fba57eULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({256, 256, 64, 4, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}), ElementType::FP16, true, 4, 0xcee37992494e247cULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({256, 256, 64, 4, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, directInstance(DirectVariant::DW, {3, 3, 2, 2, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 64, 2048, 0, 0, 0, 8, 32, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, implicitInstance({64, 64, 64, 2, 4, 8, 4, 1, Epilogue::DEFAULT, 1, 2, true, true, false}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    {"dense_dil2", {2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1, FP16}, directInstance(DirectVariant::TILED, {3, 3, 1, 1, 1, 1, 1, 1, 0, 0}, {0, 0, 2, 2, 0, 2, 0, 15, 512, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}), ElementType::FP16, false, 0, 0x0000000000000000ULL},
    };
    return s_rows;
}
// clang-format on

TEST(TestGfx942ConvWgradGeometry, PlansMatchTheAdapterOverTheGoldenTable)
{
    size_t served = 0;
    std::map<std::string, size_t> servedByFamily;
    for(const auto& row : goldenRows())
    {
        auto instance = row.instance;
        instance.dtype = row.dtype;
        const auto plan = wgrad::instancePlan(row.problem, instance);
        const auto label = std::string(row.shape) + " " + describe(row.problem);
        ASSERT_EQ(plan.has_value(), row.served) << label;
        if(!plan)
        {
            continue;
        }
        ++served;
        EXPECT_EQ(plan->splitK, row.splitK) << label;
        EXPECT_EQ(wgrad::planDigest(*plan), row.digest) << label;
        servedByFamily[instance.implicit
                           ? "implicit"
                           : std::to_string(static_cast<int>(instance.direct->variant))]++;
    }
    EXPECT_EQ(served, 132U);
    // Implicit GEMM and all seven direct pipelines are among the served rows.
    EXPECT_EQ(servedByFamily.size(), 8U);
}
#endif // superseded golden table

/// One family's full plan on one problem, Plan.serialize() verbatim.
struct SerializedCase
{
    const char* family;
    Problem problem;
    AotInstance instance;
    std::vector<int64_t> serialized;
    uint64_t digest;
};

std::vector<SerializedCase> serializedCases()
{
    const Problem dense3x3{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    const Problem groupedCpg4{2, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, FP16};
    const Problem depthwise3x3{4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16};
    const Problem stemC3{4, 3, 3, 64, 64, 11, 11, 1, 1, 5, 5, 1, 1, 3, FP16};
    const Problem gatherC24{4, 24, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    const DirectCapabilities f3{3, 3, 1, 1, 1, 1, 1, 1, 0, 0};
    const DirectCapabilities f3c1{3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
    DirectKnobs wgradKnobs{};
    wgradKnobs.hoPerBlock = 30;
    wgradKnobs.mfmaK = 32;
    wgradKnobs.wavesC = wgradKnobs.wavesK = wgradKnobs.wavesQ = 1;
    DirectKnobs tiledKnobs{};
    tiledKnobs.wavesC = tiledKnobs.wavesK = 2;
    tiledKnobs.prefetch = 2;
    tiledKnobs.rowsPerSeg = 15;
    tiledKnobs.targetBlocks = 256;
    tiledKnobs.unfoldN = 1;
    tiledKnobs.waveC16 = 1;
    tiledKnobs.waveK16 = 2;
    DirectKnobs groupedKnobs{};
    groupedKnobs.wavesQ = 4;
    groupedKnobs.prefetch = 2;
    groupedKnobs.targetBlocks = 1024;
    groupedKnobs.rowsPerChunk = 28;
    groupedKnobs.waveQ4 = 2;
    DirectKnobs dwKnobs{};
    dwKnobs.lanesC = dwKnobs.lanesW = 8;
    dwKnobs.prefetch = 2;
    dwKnobs.rowsPerSeg = 32;
    dwKnobs.targetBlocks = 2048;
    DirectKnobs dwMfmaKnobs{};
    dwMfmaKnobs.prefetch = 2;
    dwMfmaKnobs.targetBlocks = 1024;
    dwMfmaKnobs.qTiles = 4;
    dwMfmaKnobs.rowsPerChunk = 14;
    dwMfmaKnobs.unfoldN = 1;
    DirectKnobs colsKnobs{};
    colsKnobs.targetBlocks = 4096;
    colsKnobs.channels = 3;
    colsKnobs.strip = 8;
    colsKnobs.threads = 256;
    DirectKnobs gatherKnobs{};
    gatherKnobs.wavesK = 1;
    gatherKnobs.prefetch = 2;
    gatherKnobs.rowsPerSeg = 16;
    gatherKnobs.targetBlocks = 1024;
    gatherKnobs.waveK16 = 2;
    // clang-format off
    std::vector<SerializedCase> cases{
        {"direct_wgrad", dense3x3, directInstance(DirectVariant::WGRAD, f3, wgradKnobs),
         {1, 0, 1, 0, 0, 1, 0, 1, 16, 1, 2, 64, 0, 1, 2, 14, 2, 28, 28, 28, 28, 1, 64, 64, 50176, 1792, 64, 50176, 1792, 64},
         0x3694f6228f5b3538ULL},
        {"direct_wgrad_tiled", dense3x3, directInstance(DirectVariant::TILED, f3, tiledKnobs),
         {1, 0, 1, 0, 0, 1, 0, 1, 8, 1, 1, 256, 0, 1, 2, 14, 2, 28, 28, 28, 28, 1, 64, 64, 50176, 1792, 64, 50176, 1792, 64},
         0x1259e70a434173dcULL},
        {"implicit_gemm", dense3x3,
         implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true, false}),
         {1, 0, 4, 0, 0, 1, 0, 0, 5, 1, 4, 256, 0, 1, 2, 42, 2, 28, 28, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 28, 28, 64, 64, 64,
          576, 1568, 50176, 1792, 64, 50176, 1792, 64, 576, 192, 64, 613566757, 5, 613566757, 5, 1431655766, 2, 1, 6, 1, 5, 416, 4},
         0x04e78421cf5e4433ULL},
        {"direct_wgrad_grouped_mfma", groupedCpg4,
         directInstance(DirectVariant::GROUPED_MFMA, {3, 3, 1, 1, 1, 1, 1, 1, 4, 4}, groupedKnobs),
         {0, 0, 1, 147456, 1, 3, 0, 147456, 2, 0, 1, 8, 1, 2, 256, 0, 1, 3, 14, 2, 28, 28, 28, 28, 128, 512, 512, 401408, 14336,
          512, 401408, 14336, 512, 1, 2, 288, 1, 1, 1024, 3, -1, 2, 2, 2, 512},
         0xad4b687810b9f3c4ULL},
        {"direct_wgrad_dw", depthwise3x3, directInstance(DirectVariant::DW, f3c1, dwKnobs),
         {0, 0, 1, 129024, 1, 3, 0, 129024, 2, 0, 1, 1, 56, 1, 64, 0, 1, 3, 14, 4, 56, 56, 56, 56, 64, 64, 64, 200704, 3584, 64,
          200704, 3584, 64, 1, 2, 9, 1, 1, 1024, 3, -1, 2, 2, 56, 64},
         0x77ab07b0e6c07c0cULL},
        {"direct_wgrad_dw_mfma", depthwise3x3, directInstance(DirectVariant::DW_MFMA, f3c1, dwMfmaKnobs),
         {0, 0, 1, 147456, 1, 3, 0, 147456, 2, 0, 1, 1, 4, 16, 256, 0, 1, 3, 14, 4, 56, 56, 56, 56, 64, 64, 64, 200704, 3584, 64,
          200704, 3584, 64, 1, 2, 9, 1, 1, 1024, 3, -1, 2, 2, 64, 64},
         0xba912ebef5185fbaULL},
        {"direct_wgrad_dw_cols", stemC3,
         directInstance(DirectVariant::DW_COLS, {11, 11, 1, 1, 5, 5, 1, 1, 1, 1}, colsKnobs),
         {0, 0, 1, 11616, 1, 3, 0, 11616, 2, 0, 1, 11, 8, 1, 256, 0, 1, 3, 14, 4, 64, 64, 64, 64, 3, 3, 3, 12288, 192, 3, 12288,
          192, 3, 1, 2, 6, 1, 1, 1024, 3, -1, 2, 2, 8, 3},
         0x95489fe640d49792ULL},
        {"direct_wgrad_gather", gatherC24,
         directInstance(DirectVariant::GATHER, {3, 3, 1, 1, 1, 1, 1, 1, 24, 0}, gatherKnobs),
         {0, 0, 1, 2654208, 1, 3, 0, 2654208, 2, 0, 1, 4, 24, 1, 64, 0, 1, 3, 14, 4, 56, 56, 56, 56, 1, 24, 128, 401408, 7168,
          128, 75264, 1344, 24, 1, 2, 432, 1, 1, 1024, 3, -1, 2, 2, 24, 128},
         0x586b6f8dfc26c11cULL},
    };
    // clang-format on
    // The direct literals are the adapter's plans on an fp32 graph dW, which no cast step
    // follows.
    for(auto& c : cases)
    {
        c.problem.dwFloat = c.instance.direct.has_value();
    }
    return cases;
}

// OnePlanPerFamilySerializesAsTheAdapterDoes (hand-pasted serialize()/digest() literals,
// including an implicit_gemm one whose split-K degree the fix below changed) is superseded
// by FullPlansMatchTheAdapter, generated from the real shipped catalog.

TEST(TestGfx942ConvWgradGeometry, FoldedPipelinesFoldThroughTheWorkspaceAndAtomicOnesZeroDw)
{
    for(const auto& c : serializedCases())
    {
        const auto plan = wgrad::instancePlan(c.problem, c.instance);
        ASSERT_TRUE(plan) << c.family;
        if(c.instance.implicit)
        {
            ASSERT_EQ(plan->steps.size(), 1U);
            EXPECT_EQ(plan->steps[0].abi, wgrad::Abi::IMPLICIT);
            EXPECT_EQ(plan->steps[0].values.size(), wgrad::IMPLICIT_KERNARG_NAMES.size());
            // The split slices atomic-add into dW; ks_count is the last kernarg.
            EXPECT_EQ(plan->zeroDw, plan->splitK > 1);
            EXPECT_EQ(plan->steps[0].values.back(), plan->splitK);
            continue;
        }
        const bool folded = wgrad::isFoldedVariant(c.instance.direct->variant);
        EXPECT_EQ(plan->steps.size(), folded ? 2U : 1U) << c.family;
        EXPECT_EQ(plan->zeroDw, !folded) << c.family;
        EXPECT_EQ(plan->splitK, 1) << c.family;
        EXPECT_EQ(plan->steps[0].values.size(), wgrad::DIRECT_WGRAD_KERNARG_NAMES.size());
        if(folded)
        {
            const auto ws = plan->workspace(wgrad::Buffer::WS);
            ASSERT_TRUE(ws) << c.family;
            EXPECT_EQ(ws->offset, 0);
            EXPECT_EQ(plan->steps[0].buffers[2], wgrad::Buffer::WS);
            EXPECT_EQ(plan->steps[1].kernel, wgrad::PlanKernel::FOLD);
            EXPECT_EQ(plan->steps[1].buffers,
                      (std::array<wgrad::Buffer, 3>{
                          wgrad::Buffer::WS, wgrad::Buffer::NONE, wgrad::Buffer::DW}));
            EXPECT_EQ(plan->steps[1].block, 1024);
        }
        else
        {
            EXPECT_TRUE(plan->workspaces.empty());
            EXPECT_EQ(plan->steps[0].buffers[2], wgrad::Buffer::DW);
        }
    }
}

// ---------------------------------------------------------------------------------------
// The problem
// ---------------------------------------------------------------------------------------

TEST(TestGfx942ConvWgradGeometry, DeriveGeometryOnRepresentativeShapes)
{
    // Dense, padded 3x3: Ho = Wo = 28; dY = X = 2 * 28 * 28 * 64 * 2 bytes; a 16-bit dW of
    // 64 * 3 * 3 * 64 elements.
    const auto dense = wgrad::deriveGeometry({2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1});
    ASSERT_TRUE(dense);
    EXPECT_EQ(dense->ho, 28);
    EXPECT_EQ(dense->wo, 28);
    EXPECT_EQ(dense->cpg, 64);
    EXPECT_EQ(dense->kpg, 64);
    EXPECT_EQ(dense->tensorBytes, (std::array<int64_t, 3>{200704, 200704, 73728}));
    EXPECT_EQ(dense->dwElements, 36864);

    // Strided and unpadded: (28 - 3) / 2 + 1 = 13; dY carries K, X carries C.
    const auto strided = wgrad::deriveGeometry({16, 128, 256, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, 1});
    ASSERT_TRUE(strided);
    EXPECT_EQ(strided->ho, 13);
    EXPECT_EQ(strided->wo, 13);
    EXPECT_EQ(strided->tensorBytes[0], 16 * 13 * 13 * 256 * 2);
    EXPECT_EQ(strided->tensorBytes[1], 16 * 28 * 28 * 128 * 2);

    // Dilated: the effective filter is 5 wide, so Ho = 30 + 4 - 5 + 1.
    const auto dilated = wgrad::deriveGeometry({2, 64, 64, 30, 30, 3, 3, 1, 1, 2, 2, 2, 2, 1});
    ASSERT_TRUE(dilated);
    EXPECT_EQ(dilated->ho, 30);

    // Grouped: dW holds C / groups channels per filter.
    const auto grouped = wgrad::deriveGeometry({8, 256, 512, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 32});
    ASSERT_TRUE(grouped);
    EXPECT_EQ(grouped->cpg, 8);
    EXPECT_EQ(grouped->kpg, 16);
    EXPECT_EQ(grouped->dwElements, 512 * 3 * 3 * 8);

    // Rectangular filter and asymmetric extents.
    const auto rect = wgrad::deriveGeometry({1, 16, 16, 17, 40, 1, 7, 1, 2, 0, 3, 1, 1, 1});
    ASSERT_TRUE(rect);
    EXPECT_EQ(rect->ho, 17);
    EXPECT_EQ(rect->wo, 20);
}

TEST(TestGfx942ConvWgradGeometry, DeriveGeometryRefusesWhatTheGraphContractRefuses)
{
    // A filter wider than the padded input: Python's floor gives Ho <= 0.
    EXPECT_FALSE(wgrad::deriveGeometry({1, 16, 16, 1, 8, 2, 1, 2, 1, 0, 0, 1, 1, 1}));
    // groups not dividing C or K, and groups beyond grid z.
    EXPECT_FALSE(wgrad::deriveGeometry({1, 30, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 4}));
    EXPECT_FALSE(wgrad::deriveGeometry({1, 32, 30, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 4}));
    EXPECT_FALSE(wgrad::deriveGeometry({1, 65536, 65536, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 65536}));
    // Nonpositive extents, stride and dilation; negative padding.
    EXPECT_FALSE(wgrad::deriveGeometry({0, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_FALSE(wgrad::deriveGeometry({1, 16, 16, 8, 8, 3, 3, 0, 1, 1, 1, 1, 1, 1}));
    EXPECT_FALSE(wgrad::deriveGeometry({1, 16, 16, 8, 8, 3, 3, 1, 1, 1, 1, 0, 1, 1}));
    EXPECT_FALSE(wgrad::deriveGeometry({1, 16, 16, 8, 8, 3, 3, 1, 1, -1, 1, 1, 1, 1}));
    // X past 2^31 bytes as a whole (no per-image slicing for wgrad): 2^30 elements.
    EXPECT_TRUE(wgrad::deriveGeometry({1, 1024, 16, 1024, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1}));
    EXPECT_FALSE(wgrad::deriveGeometry({2, 1024, 16, 1024, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1}));
    // dW counted at fp32 width: 2^29 elements is past the ABI.
    EXPECT_FALSE(wgrad::deriveGeometry({1, 16384, 32768, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1}));
}

// ---------------------------------------------------------------------------------------
// The split-K degree
// ---------------------------------------------------------------------------------------

// SplitKMatchesResolveConvSplitK (hand-pasted ImplicitInstance rows that left splitK at its
// default of 1, so its hardcoded degrees are exactly the ones the resolveSplitK fix changed)
// is superseded by PlansMatchTheAdapterOverTheCatalog/FullPlansMatchTheAdapter, generated
// from the real shipped catalog, plus TheSplitKDegreeIsALaunchValueNotTheBakedPlaceholder
// below, which covers the splitK==1 short-circuit explicitly.

TEST(TestGfx942ConvWgradGeometry, CkFormulaSplitKFollowsSelectSplitKWgrad)
{
    // gfx942 has no fitted split-K cost model: conv_split_k.select_split_k_wgrad's CK
    // formula picks the degree. Expected values are Python's for the same arguments.
    // Ungrouped: floor(2 waves * 304 CUs / 5 tiles), clamped to wg_K.
    EXPECT_EQ(wgrad::ckFormulaSplitK(64, 576, 1568, 128, 128, 1, 256), 121);
    EXPECT_EQ(wgrad::ckFormulaSplitK(16, 16, int64_t{1} << 24, 16, 16, 1, 64), 608);
    // Grouped: 16 target waves per CU over the real no-split CTA count, then snapped to a
    // multiple of the XCD step (8 XCDs / gcd(tiles, 8)).
    EXPECT_EQ(wgrad::ckFormulaSplitK(1, 9, 100352, 16, 64, 256, 256), 8);
    EXPECT_EQ(wgrad::ckFormulaSplitK(8, 72, 6272, 64, 64, 32, 256), 16);
}

TEST(TestGfx942ConvWgradGeometry, TheSplitKDegreeIsALaunchValueNotTheBakedPlaceholder)
{
    // A one-stage binary built split (its packed-atomic epilogue) runs the resolved degree,
    // which lands on the plan, grid z and ks; its dW is cleared at any degree. The same
    // tile built unsplit (split_k == 1) has no atomic epilogue and stays at degree 1.
    const Problem dense3x3{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    ImplicitInstance atomic{128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE};
    atomic.splitK = 2;
    const auto a = wgrad::instancePlan(dense3x3, implicitInstance(atomic));
    ASSERT_TRUE(a);
    // resolve_conv_split_k on gfx942: CK formula floor(608 / 5) = 121, under the 16-bit
    // atomic's cap of 128.
    EXPECT_EQ(a->splitK, 121);
    EXPECT_EQ(a->steps[0].grid[2], 121);
    EXPECT_TRUE(a->zeroDw);
    // ks = ceil(1568 / (32 * 121)) * 32.
    EXPECT_EQ(a->steps[0].values[wgrad::IMPLICIT_KERNARG_NAMES.size() - 2], 32);
    auto unsplit = atomic;
    unsplit.splitK = 1;
    const auto u = wgrad::instancePlan(dense3x3, implicitInstance(unsplit));
    ASSERT_TRUE(u);
    EXPECT_EQ(u->splitK, 1);
    EXPECT_FALSE(u->zeroDw);
    // The default epilogue emits no packed 16-bit atomic: an atomic-built binary with it
    // runs unsplit, but still clears dW (its epilogue accumulates at degree 1 too).
    auto defaultEpilogue = atomic;
    defaultEpilogue.epilogue = Epilogue::DEFAULT;
    const auto d = wgrad::instancePlan(dense3x3, implicitInstance(defaultEpilogue));
    if(d)
    {
        EXPECT_EQ(d->splitK, 1);
        EXPECT_TRUE(d->zeroDw);
    }
    // A one-stage binary built split still needs an even store vector: vector_size_c 1
    // refuses, the unsplit-built twin serves.
    atomic.vectorC = 1;
    unsplit.vectorC = 1;
    EXPECT_FALSE(wgrad::instancePlan(dense3x3, implicitInstance(atomic)));
    EXPECT_TRUE(wgrad::instancePlan(dense3x3, implicitInstance(unsplit)));
}

// ---------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------

TEST(TestGfx942ConvWgradGeometry, TwoStagePlansZeroTheScratchAndFoldItIntoDw)
{
    const Problem dense3x3{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    ImplicitInstance twoStage{128, 128, 32, 2, 2, 8, 8, 1, Epilogue::DEFAULT, 1, 2, true};
    twoStage.wsReplicas = 8;
    // Odd dW rows (Y * X * cpg = 297) are the packed atomic's to refuse, not the
    // scratch's: the two-stage instance serves both.
    auto oddC = dense3x3;
    oddC.c = 33;
    twoStage.vectorB = 1;
    for(const auto& problem : {dense3x3, oddC})
    {
        const auto plan = wgrad::instancePlan(problem, implicitInstance(twoStage));
        ASSERT_TRUE(plan) << describe(problem);
        EXPECT_FALSE(wgrad::staticRefusal(implicitInstance(twoStage)));
        EXPECT_GE(plan->splitK, 2);
        EXPECT_FALSE(plan->zeroDw);
        EXPECT_TRUE(plan->zeroWs);
        const auto wgN = problem.y * problem.x * problem.c;
        // groups * ws_replicas * wg_M * wg_N f32.
        const auto ws = plan->workspace(wgrad::Buffer::WS);
        ASSERT_TRUE(ws);
        EXPECT_EQ(ws->offset, 0);
        EXPECT_EQ(ws->bytes, 8 * 64 * wgN * 4);
        ASSERT_EQ(plan->steps.size(), 2U);
        const auto& stage1 = plan->steps[0];
        EXPECT_EQ(stage1.abi, wgrad::Abi::IMPLICIT_TWO_STAGE);
        EXPECT_EQ(stage1.values.size(), wgrad::IMPLICIT_KERNARG_NAMES.size() + 1);
        EXPECT_EQ(stage1.values[wgrad::TWO_STAGE_WS_POSITION], ws->bytes);
        EXPECT_EQ(stage1.values.back(), plan->splitK);
        EXPECT_EQ(stage1.grid[2], plan->splitK);
        const auto& stage2 = plan->steps[1];
        EXPECT_EQ(stage2.kernel, wgrad::PlanKernel::WS_REDUCE);
        EXPECT_EQ(stage2.abi, wgrad::Abi::WS_REDUCE);
        EXPECT_EQ(stage2.buffers,
                  (std::array<wgrad::Buffer, 3>{
                      wgrad::Buffer::WS, wgrad::Buffer::NONE, wgrad::Buffer::DW}));
        EXPECT_EQ(stage2.block, 256);
        EXPECT_EQ(stage2.grid, (std::array<int64_t, 3>{(wgN + 63) / 64, 16, 1}));
        EXPECT_EQ(stage2.values,
                  (std::vector<int32_t>{64,
                                        static_cast<int32_t>(wgN),
                                        static_cast<int32_t>(ws->bytes),
                                        static_cast<int32_t>(64 * wgN * 2),
                                        1}));
    }
    // One K tile: gfx942's CK formula still splits it (the extra slices add zeros) where
    // gfx950's cost model refuses -- the plan is two-stage at the formula's degree.
    const Problem oneTile{1, 64, 64, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, FP16};
    const auto oneTilePlan = wgrad::instancePlan(oneTile, implicitInstance(twoStage));
    ASSERT_TRUE(oneTilePlan);
    EXPECT_TRUE(oneTilePlan->zeroWs);
    auto oneStage = twoStage;
    oneStage.twoStage = false;
    oneStage.splitK = 1;
    EXPECT_TRUE(wgrad::instancePlan(oneTile, implicitInstance(oneStage)));
}

TEST(TestGfx942ConvWgradGeometry, HelpersNeverServeOnTheirOwn)
{
    const Problem dense3x3{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    DirectKnobs fold{};
    fold.taps = 9;
    AotInstance helper{wgrad::KernelFamily::HELPER,
                       FP16,
                       std::nullopt,
                       wgrad::DirectInstance{DirectVariant::FOLD, {}, fold}};
    EXPECT_EQ(wgrad::staticRefusal(helper), wgrad::HELPER_REFUSAL);
    EXPECT_FALSE(wgrad::instancePlan(dense3x3, helper));
    AotInstance stage2{wgrad::KernelFamily::HELPER,
                       FP16,
                       std::nullopt,
                       wgrad::DirectInstance{DirectVariant::WS_REDUCE, {}, {}, 8}};
    EXPECT_EQ(wgrad::staticRefusal(stage2), wgrad::HELPER_REFUSAL);
    EXPECT_FALSE(wgrad::instancePlan(dense3x3, stage2));
}

TEST(TestGfx942ConvWgradGeometry, TheInstanceDtypeIsPinned)
{
    const Problem fp16{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    auto bf16 = fp16;
    bf16.dtype = BF16;
    auto instance
        = implicitInstance({128, 128, 32, 2, 2, 8, 8, 8, Epilogue::CSHUFFLE, 1, 1, false, true});
    EXPECT_TRUE(wgrad::instancePlan(fp16, instance));
    EXPECT_FALSE(wgrad::instancePlan(bf16, instance));
    instance.dtype = BF16;
    EXPECT_TRUE(wgrad::instancePlan(bf16, instance));
}

TEST(TestGfx942ConvWgradGeometry, DirectInstancesServeTheirCompiledGeometryOnly)
{
    const auto c = serializedCases()[0];
    auto problem = c.problem;
    EXPECT_TRUE(wgrad::instancePlan(problem, c.instance));
    // Another filter, padding or stride; per-axis stride or padding; dilation.
    for(const auto& mutate : std::vector<std::function<void(Problem&)>>{
            [](Problem& p) { p.y = p.x = 5; },
            [](Problem& p) { p.padH = p.padW = 0; },
            [](Problem& p) { p.strideH = p.strideW = 2; },
            [](Problem& p) { p.strideW = 2; },
            [](Problem& p) { p.padW = 0; },
            [](Problem& p) { p.dilationH = p.dilationW = 2; }})
    {
        auto changed = problem;
        mutate(changed);
        EXPECT_FALSE(wgrad::instancePlan(changed, c.instance)) << describe(changed);
    }
    // direct_wgrad's runtime channel counts: any cpg / kpg >= 16 its C tiles cover.
    problem.c = problem.k = 32;
    EXPECT_TRUE(wgrad::instancePlan(problem, c.instance));
    problem.c = 8;
    EXPECT_FALSE(wgrad::instancePlan(problem, c.instance));
}

// ---------------------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------------------

/// A kernel's metadata, as the descriptor loader delivers it.
struct Metadata
{
    std::map<std::string, std::variant<int64_t, std::string>> fields;

    std::optional<int64_t> intField(std::string_view field) const
    {
        const auto it = fields.find(std::string(field));
        const auto* value = it == fields.end() ? nullptr : std::get_if<int64_t>(&it->second);
        return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
    }

    const std::string* stringField(std::string_view field) const
    {
        const auto it = fields.find(std::string(field));
        return it == fields.end() ? nullptr : std::get_if<std::string>(&it->second);
    }
};

/// Every KMD field at the direct/helper placeholders and defaults.
Metadata baseMetadata()
{
    Metadata m;
    m.fields = {{"dtype", std::string("fp16")},
                {"layout", std::string("NHWC")},
                {"wave_size", int64_t{64}},
                {"instance_id", std::string(40, 'a')},
                {"pipeline", std::string("direct")},
                {"epilogue", std::string("direct")},
                {"lds_layout", std::string("default")},
                {"vector_size_a", int64_t{0}},
                {"vector_size_b", int64_t{0}},
                {"vector_size_c", int64_t{0}},
                {"group_merge", int64_t{1}},
                {"ws_replicas", int64_t{0}},
                {"lds_k_pad", int64_t{0}},
                {"split_k", int64_t{1}}};
    for(const auto field : wgrad::GEMM_TUNING_FIELDS)
    {
        m.fields[std::string(field)] = int64_t{0};
    }
    for(const auto field : wgrad::GEMM_FLAG_FIELDS)
    {
        m.fields[std::string(field)] = int64_t{0};
    }
    for(const auto field : wgrad::DIRECT_CAPABILITY_FIELDS)
    {
        m.fields[std::string(field)] = int64_t{0};
    }
    for(const auto& [field, member] : wgrad::DIRECT_KNOB_FIELDS)
    {
        m.fields[std::string(field)] = int64_t{0};
    }
    return m;
}

Metadata implicitMetadata()
{
    auto m = baseMetadata();
    m.fields["kernel_family"] = int64_t{0};
    m.fields["direct_variant"] = std::string("none");
    m.fields["pipeline"] = std::string("compv3");
    m.fields["epilogue"] = std::string("cshuffle");
    for(const auto& [field, value] : std::map<std::string, int64_t>{{"tile_m", 128},
                                                                    {"tile_n", 128},
                                                                    {"tile_k", 32},
                                                                    {"warp_m", 2},
                                                                    {"warp_n", 2},
                                                                    {"warp_tile_m", 16},
                                                                    {"warp_tile_n", 16},
                                                                    {"warp_tile_k", 32},
                                                                    {"vector_size_a", 8},
                                                                    {"vector_size_b", 8},
                                                                    {"vector_size_c", 8},
                                                                    {"lds_k_outer", 1},
                                                                    {"grouped", 1},
                                                                    {"prefetch_addr", 1},
                                                                    {"split_k", 1}})
    {
        m.fields[field] = value;
    }
    return m;
}

Metadata dwMetadata()
{
    auto m = baseMetadata();
    m.fields["kernel_family"] = int64_t{1};
    m.fields["direct_variant"] = std::string("direct_wgrad_dw");
    for(const auto& [field, value] : std::map<std::string, int64_t>{{"filter_h", 3},
                                                                    {"filter_w", 3},
                                                                    {"stride_h", 1},
                                                                    {"stride_w", 1},
                                                                    {"pad_h", 1},
                                                                    {"pad_w", 1},
                                                                    {"dilation_h", 1},
                                                                    {"dilation_w", 1},
                                                                    {"cpg", 1},
                                                                    {"kpg", 1},
                                                                    {"lanes_c", 8},
                                                                    {"lanes_w", 8},
                                                                    {"prefetch", 2},
                                                                    {"rows_per_seg", 32},
                                                                    {"target_blocks", 2048}})
    {
        m.fields[field] = value;
    }
    return m;
}

Metadata foldMetadata(int64_t taps)
{
    auto m = baseMetadata();
    m.fields["kernel_family"] = int64_t{2};
    m.fields["direct_variant"] = std::string("direct_wgrad_fold");
    m.fields["taps"] = taps;
    return m;
}

Metadata wsReduceMetadata(int64_t wsReplicas)
{
    auto m = baseMetadata();
    m.fields["kernel_family"] = int64_t{2};
    m.fields["direct_variant"] = std::string("direct_wgrad_ws_reduce");
    m.fields["ws_replicas"] = wsReplicas;
    return m;
}

Metadata twoStageMetadata()
{
    auto m = implicitMetadata();
    m.fields["two_stage"] = int64_t{1};
    m.fields["split_k"] = int64_t{2};
    m.fields["ws_replicas"] = int64_t{8};
    m.fields["epilogue"] = std::string("default");
    m.fields["vector_size_c"] = int64_t{1};
    return m;
}

TEST(TestGfx942ConvWgradGeometry, TiledQGroupsDoubleTheBlockAndHalveTheSegmentSplits)
{
    // DirectWgradTiledSpec(...).launch_geometry() and is_valid_wgrad_tiled_spec on a bf16
    // f3x3p1 problem: a second wave group per block doubles the threads, and the two
    // groups share one block's segments, so a segment-bound grid halves. pingpong only
    // staggers the groups and launches like q_groups 2.
    DirectKnobs knobs{};
    knobs.waveK16 = knobs.waveC16 = 2;
    knobs.wavesK = knobs.wavesC = 2;
    knobs.rowsPerSeg = 30;
    knobs.prefetch = 2;
    knobs.targetBlocks = 256;
    knobs.unfoldN = 1;
    const wgrad::DirectProblem small{2, 28, 28, 1, 64, 64, 3, 3, 1, 1, 28, 28};
    const wgrad::DirectProblem big{32, 56, 56, 1, 256, 256, 3, 3, 1, 1, 56, 56};
    const auto launch = [&](const wgrad::DirectProblem& p, int64_t qGroups, int64_t pingpong) {
        auto k = knobs;
        k.qGroups = qGroups;
        k.pingpong = pingpong;
        return wgrad::tiledLaunch(p, k);
    };
    for(const auto& [p, qGroups, pingpong, grid, block] :
        std::vector<std::tuple<wgrad::DirectProblem, int64_t, int64_t, int64_t, int64_t>>{
            {small, 0, 0, 2, 256},
            {small, 1, 0, 2, 256},
            {small, 2, 0, 1, 512},
            {small, 2, 1, 1, 512},
            {big, 1, 0, 256, 256},
            {big, 2, 1, 256, 512}})
    {
        const auto l = launch(p, qGroups, pingpong);
        ASSERT_TRUE(l) << qGroups << " " << pingpong;
        EXPECT_EQ(l->grid, (std::array<int64_t, 3>{grid, 1, 1})) << qGroups << " " << pingpong;
        EXPECT_EQ(l->block, block) << qGroups << " " << pingpong;
    }
    // q_groups is 1 or 2, and pingpong needs the second group.
    EXPECT_FALSE(launch(small, 3, 0));
    EXPECT_FALSE(launch(small, 1, 1));
    EXPECT_FALSE(launch(small, 2, 2));
}

TEST(TestGfx942ConvWgradGeometry, ParsesEachFamilyFromItsMetadata)
{
    const auto implicit = wgrad::parseAotInstance(implicitMetadata());
    ASSERT_TRUE(implicit && implicit->implicit);
    EXPECT_EQ(implicit->implicit->tileM, 128);
    EXPECT_EQ(implicit->implicit->vectorC, 8);
    EXPECT_EQ(implicit->implicit->splitK, 1);
    EXPECT_FALSE(implicit->implicit->twoStage);
    EXPECT_TRUE(implicit->implicit->grouped);
    EXPECT_EQ(implicit->implicit->epilogue, Epilogue::CSHUFFLE);

    const auto dw = wgrad::parseAotInstance(dwMetadata());
    ASSERT_TRUE(dw && dw->direct);
    EXPECT_EQ(dw->direct->variant, DirectVariant::DW);
    EXPECT_EQ(dw->direct->knobs.lanesW, 8);
    // The parsed instance plans exactly as the serialized case built by hand.
    const auto dwCase = serializedCases()[4];
    const auto plan = wgrad::instancePlan(dwCase.problem, *dw);
    ASSERT_TRUE(plan);
    EXPECT_EQ(wgrad::planDigest(*plan), dwCase.digest);

    const auto fold = wgrad::parseAotInstance(foldMetadata(9));
    ASSERT_TRUE(fold && fold->direct);
    EXPECT_EQ(fold->family, wgrad::KernelFamily::HELPER);
    EXPECT_EQ(fold->direct->knobs.taps, 9);

    const auto parsedTwoStage = wgrad::parseAotInstance(twoStageMetadata());
    ASSERT_TRUE(parsedTwoStage && parsedTwoStage->implicit);
    EXPECT_TRUE(parsedTwoStage->implicit->twoStage);
    EXPECT_EQ(parsedTwoStage->implicit->wsReplicas, 8);

    const auto stage2 = wgrad::parseAotInstance(wsReduceMetadata(8));
    ASSERT_TRUE(stage2 && stage2->direct);
    EXPECT_EQ(stage2->family, wgrad::KernelFamily::HELPER);
    EXPECT_EQ(stage2->direct->variant, DirectVariant::WS_REDUCE);
    EXPECT_EQ(stage2->direct->wsReplicas, 8);
}

TEST(TestGfx942ConvWgradGeometry, RefusesMalformedOrOutOfContractMetadata)
{
    const auto refuses = [](Metadata m, const std::string& field, auto value) {
        m.fields[field] = value;
        EXPECT_FALSE(wgrad::parseAotInstance(m)) << field;
    };
    auto missing = implicitMetadata();
    missing.fields.erase("tile_k");
    EXPECT_FALSE(wgrad::parseAotInstance(missing));
    refuses(implicitMetadata(), "dtype", std::string("fp32"));
    refuses(implicitMetadata(), "layout", std::string("NCHW"));
    refuses(implicitMetadata(), "wave_size", int64_t{32});
    refuses(implicitMetadata(), "kernel_family", int64_t{3});
    refuses(implicitMetadata(), "direct_variant", std::string("direct_wgrad"));
    refuses(implicitMetadata(), "tile_m", int64_t{0});
    refuses(implicitMetadata(), "vector_size_b", int64_t{3});
    refuses(implicitMetadata(), "grouped", int64_t{2});
    refuses(implicitMetadata(), "split_k", int64_t{0});
    refuses(implicitMetadata(), "group_merge", int64_t{0});
    refuses(implicitMetadata(), "epilogue", std::string("direct"));
    refuses(implicitMetadata(), "filter_h", int64_t{3});
    refuses(implicitMetadata(), "waves_k", int64_t{1});
    refuses(implicitMetadata(), "warp_n", int64_t{16});
    auto twoStageUnsplit = implicitMetadata();
    twoStageUnsplit.fields["two_stage"] = int64_t{1};
    EXPECT_FALSE(wgrad::parseAotInstance(twoStageUnsplit)) << "two_stage with split_k 1";

    refuses(dwMetadata(), "kernel_family", int64_t{2});
    refuses(dwMetadata(), "tile_m", int64_t{64});
    refuses(dwMetadata(), "split_k", int64_t{-1});
    refuses(dwMetadata(), "filter_w", int64_t{5});
    refuses(dwMetadata(), "dilation_h", int64_t{2});
    refuses(dwMetadata(), "stride_w", int64_t{2});
    refuses(dwMetadata(), "kpg", int64_t{0});
    refuses(dwMetadata(), "taps", int64_t{9});
    refuses(dwMetadata(), "lanes_c", int64_t{-1});
    refuses(foldMetadata(9), "kernel_family", int64_t{1});
    refuses(foldMetadata(9), "filter_h", int64_t{3});
    refuses(foldMetadata(9), "prefetch", int64_t{2});
    refuses(foldMetadata(9), "ws_replicas", int64_t{8});
    refuses(wsReduceMetadata(8), "ws_replicas", int64_t{0});
    refuses(wsReduceMetadata(8), "taps", int64_t{9});
    refuses(wsReduceMetadata(8), "kernel_family", int64_t{1});
    refuses(dwMetadata(), "ws_replicas", int64_t{8});

    // direct_wgrad takes both channel counts at run time; gather bakes cpg only.
    auto runtime = dwMetadata();
    runtime.fields["direct_variant"] = std::string("direct_wgrad");
    runtime.fields["lanes_c"] = runtime.fields["lanes_w"] = runtime.fields["prefetch"]
        = runtime.fields["rows_per_seg"] = runtime.fields["target_blocks"] = int64_t{0};
    EXPECT_FALSE(wgrad::parseAotInstance(runtime));
    runtime.fields["cpg"] = runtime.fields["kpg"] = int64_t{0};
    EXPECT_TRUE(wgrad::parseAotInstance(runtime));
    auto gather = runtime;
    gather.fields["direct_variant"] = std::string("direct_wgrad_gather");
    EXPECT_FALSE(wgrad::parseAotInstance(gather));
    gather.fields["cpg"] = int64_t{24};
    EXPECT_TRUE(wgrad::parseAotInstance(gather));
}

// ---------------------------------------------------------------------------------------
// The real shipped catalog, end to end (Gfx942ConvWgradTestVectors.hpp, generated by
// projects/hipdnn/tools/IngestorGenerator/tools/make_conv_wgrad_aot_test_vectors.py from
// configs/gfx942_conv_wgrad_aot_{fp16,bf16}.instances.json): every catalog instance,
// decoded through parseAotInstance exactly as the native pack reads a descriptor, planned
// on a seeded synthetic problem set and compared against the rocKE adapter's own plan
// digest. Supersedes PlansMatchTheAdapterOverTheGoldenTable / OnePlanPerFamilySerializes-
// AsTheAdapterDoes's hand-built instances with the catalog hipDNN actually ships.
// ---------------------------------------------------------------------------------------

TEST(TestGfx942ConvWgradGeometry, EveryCatalogInstanceParses)
{
    for(const auto* dtype : {"fp16", "bf16"})
    {
        size_t helpers = 0;
        size_t implicitCount = 0;
        std::map<std::string, size_t> directByVariant;
        for(const auto& entry : catalog::catalog(dtype))
        {
            ASSERT_TRUE(entry.instance) << entry.text("instance_id");
            EXPECT_EQ(entry.instance->implicit.has_value(), entry.integer("kernel_family") == 0);
            if(entry.isHelper())
            {
                ++helpers;
            }
            else if(entry.integer("kernel_family") == 0)
            {
                ++implicitCount;
            }
            else
            {
                ++directByVariant[entry.text("direct_variant")];
            }
        }
        // The gfx942 catalog (gfx942_wgrad.json, pruned to keep every cases_wgrad.txt case
        // within 1% of its best non-packed-atomic instance, plus the helpers its
        // pipelines launch): fp16 is 164 implicit GEMM and the two-stage Stage 2; bf16 is
        // 82 implicit GEMM, 2
        // direct_wgrad_dw_cols and their fold, cast and the Stage 2. These track the
        // catalog: a re-pruned one updates them (and test_make_conv_wgrad_aot_config.py's).
        const bool fp16 = std::string_view(dtype) == "fp16";
        EXPECT_EQ(implicitCount, fp16 ? 164U : 82U) << dtype;
        EXPECT_EQ(helpers, fp16 ? 1U : 3U) << dtype;
        size_t directTotal = 0;
        for(const auto& [variant, count] : directByVariant)
        {
            directTotal += count;
        }
        EXPECT_EQ(directTotal, fp16 ? 0U : 2U) << dtype;
        EXPECT_EQ(directByVariant["direct_wgrad_dw_cols"], directTotal) << dtype;
        EXPECT_EQ(catalog::catalog(dtype).size(), fp16 ? 165U : 87U) << dtype;
    }
}

TEST(TestGfx942ConvWgradGeometry, PlansMatchTheAdapterOverTheCatalog)
{
    // gfx942_conv_wgrad_aot_plan over every catalog instance of the problem's dtype: the
    // same instances serve, and their plans fold to the adapter's digest. On a digest
    // mismatch, make_conv_wgrad_aot_test_vectors.py's render() names each serving plan's
    // digest. The fold helper never serves a problem on its own (it is loaded by its
    // folded pipeline's handler, not matched), so it is excluded here exactly as the
    // adapter excludes it (gfx942_conv_wgrad_aot_plan's HELPER_REFUSAL).
    std::map<std::string, size_t> servedByFamily;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = catalog::toProblem(row.problem, row.dtype);
        const auto& entries = catalog::catalog(row.dtype);
        ASSERT_EQ(std::string_view(row.served).size(), entries.size());
        uint64_t folded = 0xCBF29CE484222325ULL;
        for(size_t i = 0; i < entries.size(); ++i)
        {
            const auto& entry = entries[i];
            const bool expected = row.served[i] == '1';
            const auto plan
                = entry.isHelper() ? std::nullopt : wgrad::instancePlan(problem, *entry.instance);
            EXPECT_EQ(plan.has_value(), expected)
                << entry.text("direct_variant") << " " << entry.text("instance_id") << " on "
                << describe(problem) << " " << row.dtype;
            if(!plan)
            {
                continue;
            }
            const auto digest = wgrad::planDigest(*plan);
            for(int byte = 0; byte < 8; ++byte)
            {
                folded ^= (digest >> (8 * byte)) & 0xFFU;
                folded *= 0x100000001B3ULL;
            }
            ++servedByFamily[entry.integer("kernel_family") == 0 ? std::string("implicit")
                                                                 : entry.text("direct_variant")];
        }
        EXPECT_EQ(folded, row.digest) << describe(problem) << " " << row.dtype;
    }
    // Implicit GEMM and the one direct pipeline the catalog ships are among the served rows.
    for(const auto* family : {"implicit", "direct_wgrad_dw_cols"})
    {
        EXPECT_GT(servedByFamily[family], 0U) << family;
    }
}

TEST(TestGfx942ConvWgradGeometry, FullPlansMatchTheAdapter)
{
    // One full plan per instance kind the catalog ships (one-stage, two-stage, direct).
    ASSERT_GE(vectors::fullPlans().size(), 3U);
    for(const auto& full : vectors::fullPlans())
    {
        const auto& row = vectors::servedMatrix()[full.row];
        const auto problem = catalog::toProblem(row.problem, row.dtype);
        const auto& entry = catalog::catalog(row.dtype)[full.instance];
        SCOPED_TRACE(entry.text("direct_variant") + " " + entry.text("instance_id") + " on "
                     + describe(problem));
        const auto plan = wgrad::instancePlan(problem, *entry.instance);
        ASSERT_TRUE(plan);
        EXPECT_EQ(wgrad::serializePlan(*plan), full.serialized);
    }
}

// ---------------------------------------------------------------------------------------
// The native symbols (registry, matcher, workspaceBytes / prepare lockstep)
// ---------------------------------------------------------------------------------------

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols GFX942_CONV_WGRAD_AOT{"hipkernel:Gfx942ConvWgradAot",
                                            "hipkernel.gfx942_conv_wgrad_aot.graph_match",
                                            "",
                                            "hipkernel.gfx942_conv_wgrad_aot.kernel_match",
                                            "hipkernel.gfx942_conv_wgrad_aot.score",
                                            "hipkernel.gfx942_conv_wgrad_aot.dispatch",
                                            "gfx942_conv_wgrad.x.uid",
                                            "gfx942_conv_wgrad.dy.uid",
                                            "gfx942_conv_wgrad.dw.uid"};

/// A ConvolutionWrwAttributes graph of @p p: X (uid 1), dY (uid 2), dW (uid 3), packed
/// NHWC, dW of the activation dtype or fp32.
flatbuffers::FlatBufferBuilder buildWgradGraph(const Problem& p, bool dwFloat)
{
    const auto g = wgrad::deriveGeometry(p);
    const auto dtype
        = p.dtype == BF16 ? data_objects::DataType::BFLOAT16 : data_objects::DataType::HALF;
    const auto nhwc = [](int64_t n, int64_t c, int64_t h, int64_t w) {
        return std::pair{std::vector<int64_t>{n, c, h, w},
                         std::vector<int64_t>{h * w * c, 1, w * c, c}};
    };
    const std::array<std::pair<std::vector<int64_t>, std::vector<int64_t>>, 3> shapes{
        nhwc(p.n, p.c, p.hi, p.wi), nhwc(p.n, p.k, g->ho, g->wo), nhwc(p.k, g->cpg, p.y, p.x)};
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        const auto dims = builder.CreateVector(shapes[i].first);
        const auto strides = builder.CreateVector(shapes[i].second);
        data_objects::TensorAttributesBuilder tensorBuilder(builder);
        tensorBuilder.add_uid(static_cast<int64_t>(i + 1));
        tensorBuilder.add_data_type(i == 2 && dwFloat ? data_objects::DataType::FLOAT : dtype);
        tensorBuilder.add_dims(dims);
        tensorBuilder.add_strides(strides);
        tensorBuilder.add_virtual_(false);
        tensorBuilder.add_alignment(16);
        tensors.push_back(tensorBuilder.Finish());
    }
    const std::vector<int64_t> stride{p.strideH, p.strideW};
    const std::vector<int64_t> dilation{p.dilationH, p.dilationW};
    const std::vector<int64_t> padding{p.padH, p.padW};
    const auto attributes = data_objects::CreateConvolutionWrwAttributesDirect(
                                builder,
                                1,
                                2,
                                3,
                                &padding,
                                &padding,
                                &stride,
                                &dilation,
                                data_objects::ConvMode::CROSS_CORRELATION)
                                .Union();
    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "gfx942_conv_wgrad_test",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::ConvolutionWrwAttributes,
                                       attributes)};
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_tensors(tensorVector);
    graphBuilder.add_nodes(nodeVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

GraphFixture wgradFixture(const Problem& p, bool dwFloat)
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    return GraphFixture(buildWgradGraph(p, dwFloat), properties);
}

DescriptorId idOf(uint32_t value, uint8_t tag)
{
    DescriptorId id{};
    id[0] = tag;
    id[15] = static_cast<uint8_t>(value);
    return id;
}

/// Kernel @p id of the engine (dispatch) @p engine, in a pack of its own: the shipped
/// catalog shards instances and helpers into separate packs of one dispatch.
KernelDefinition kernelOf(const Metadata& metadata, uint32_t id, uint8_t engine)
{
    KernelDefinition kernel;
    kernel.kernelId = idOf(id, engine);
    kernel.packId = idOf(id, static_cast<uint8_t>(engine + 1));
    kernel.dispatchId = idOf(0, engine);
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.name = "gfx942_conv_wgrad_aot_test_" + std::to_string(id);
    for(const auto& [name, value] : metadata.fields)
    {
        if(const auto* integer = std::get_if<int64_t>(&value))
        {
            kernel.metadata[name] = *integer;
        }
        else
        {
            kernel.metadata[name] = std::get<std::string>(value);
        }
    }
    return kernel;
}

/// Runs the pack's KernelPrepareFn over @p kernels as the engine's shared state does at
/// load, which also files each helper for its engine.
void prepareAll(std::vector<KernelDefinition>& kernels)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto prepareFn
        = KernelPrepareRegistry::tryResolve(std::string(GFX942_CONV_WGRAD_AOT.kernelMatcher));
    ASSERT_NE(prepareFn, nullptr);
    for(auto& kernel : kernels)
    {
        kernel.prepared = prepareFn(kernel);
        kernel.preparedBy = prepareFn;
    }
}

/// What prepare() did with @p kernel: refused it against the contract (the message), or
/// got past the plan to code loading, which a descriptor naming no archive cannot do.
std::optional<std::string> prepareRefusal(const IKernelDispatchHandler<Handle>& handler,
                                          const GraphFixture& fixture,
                                          const BoundTokens& bound,
                                          const KernelDefinition& kernel)
{
    try
    {
        handler.prepare(fixture.context(), bound, kernel);
        ADD_FAILURE() << "preparation loaded a kernel from an empty archive path";
        return std::nullopt;
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        if(message.find("convolution contract") != std::string::npos
           || message.find("helper") != std::string::npos)
        {
            return message;
        }
        return std::nullopt;
    }
}

TEST(TestGfx942ConvWgradNative, GraphMatcherBindsTheWrwTensorsAndRefusesOtherNodes)
{
    const Problem dense3x3{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    const auto fixture = wgradFixture(dense3x3, false);
    const auto bound = matchesGraph(GFX942_CONV_WGRAD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_WGRAD_AOT.inputAToken), std::optional<int64_t>(1));
    EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_WGRAD_AOT.inputBToken), std::optional<int64_t>(2));
    EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_WGRAD_AOT.outputToken), std::optional<int64_t>(3));
    EXPECT_TRUE(matchesGraph(GFX942_CONV_WGRAD_AOT, wgradFixture(dense3x3, true).context()));
    // Another architecture.
    DeviceProperties gfx950;
    gfx950.gcnArchName = "gfx950";
    gfx950.warpSize = 64;
    EXPECT_FALSE(matchesGraph(GFX942_CONV_WGRAD_AOT,
                              GraphFixture(buildWgradGraph(dense3x3, false), gfx950).context()));
}

TEST(TestGfx942ConvWgradNative, WorkspaceBytesAndPrepareRefuseInLockstep)
{
    auto oneStage = implicitMetadata();
    std::vector<KernelDefinition> kernels{kernelOf(oneStage, 1, 70),
                                          kernelOf(twoStageMetadata(), 2, 70),
                                          kernelOf(dwMetadata(), 3, 70),
                                          kernelOf(foldMetadata(9), 4, 70),
                                          // The same depthwise instance in an engine
                                          // with its fold but not its cast.
                                          kernelOf(dwMetadata(), 5, 72),
                                          kernelOf(wsReduceMetadata(8), 6, 70),
                                          // The same two-stage instance in an engine
                                          // without its Stage 2.
                                          kernelOf(twoStageMetadata(), 7, 72),
                                          // The one-slab Stage 2: a direct pipeline's
                                          // cast into a 16-bit dW.
                                          kernelOf(wsReduceMetadata(1), 8, 70),
                                          // A fold beside the same depthwise instance in
                                          // an engine without its cast.
                                          kernelOf(foldMetadata(9), 9, 72),
                                          // The same depthwise instance in an engine
                                          // without its fold.
                                          kernelOf(dwMetadata(), 10, 74)};
    prepareAll(kernels);
    const auto& handler = dispatchHandler(GFX942_CONV_WGRAD_AOT);

    const Problem dense3x3{2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, FP16};
    const Problem depthwise{4, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, FP16};
    struct Case
    {
        const char* what;
        Problem problem;
        bool dwFloat;
        size_t kernel;
        bool served;
        size_t workspace;
    };
    const std::vector<Case> cases{
        {"one-stage implicit", dense3x3, false, 0, true, 0},
        // groups * ws_replicas * kpg * Y * X * cpg f32.
        {"two-stage implicit", dense3x3, false, 1, true, 8 * 64 * 576 * 4},
        {"two-stage without its Stage 2", dense3x3, false, 6, false, 0},
        {"the Stage 2 itself", dense3x3, false, 5, false, 0},
        {"two-stage on an fp32 dW", dense3x3, true, 1, false, 0},
        {"implicit on an fp32 dW", dense3x3, true, 0, false, 0},
        {"depthwise with its fold", depthwise, true, 2, true, 129024},
        // The slots, then the fp32 dW (C * 9 f32) at the next 256-byte boundary.
        {"depthwise cast into a 16-bit dW", depthwise, false, 2, true, 129024 + 64 * 9 * 4},
        {"depthwise without its cast", depthwise, false, 4, false, 0},
        {"depthwise with its fold but not its cast", depthwise, true, 4, true, 129024},
        {"the cast itself", depthwise, false, 7, false, 0},
        {"depthwise without its fold", depthwise, true, 9, false, 0},
        {"the fold itself", depthwise, true, 3, false, 0},
        {"depthwise on a dense graph", dense3x3, true, 2, false, 0},
    };
    for(const auto& c : cases)
    {
        const auto fixture = wgradFixture(c.problem, c.dwFloat);
        const auto bound = matchesGraph(GFX942_CONV_WGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound) << c.what;
        const auto& kernel = kernels[c.kernel];
        // The matcher, workspaceBytes() and prepare() take one path (aotKernelPlan), so
        // each refusal is all three's and each admission all three's.
        EXPECT_EQ(matchesKernel(GFX942_CONV_WGRAD_AOT, fixture.context(), kernel, *bound), c.served)
            << c.what;
        EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, kernel), c.workspace) << c.what;
        const auto refusal = prepareRefusal(handler, fixture, *bound, kernel);
        EXPECT_EQ(!refusal.has_value(), c.served) << c.what << ": " << refusal.value_or("");
        const auto score = scoreKernel(GFX942_CONV_WGRAD_AOT, fixture.context(), kernel, *bound);
        if(c.served)
        {
            EXPECT_GT(score, 0.0) << c.what;
        }
        else
        {
            EXPECT_EQ(score, -1.0) << c.what;
        }
    }
    // A helper's refusal names the adapter's reason.
    const auto denseFixture = wgradFixture(dense3x3, false);
    const auto denseBound = matchesGraph(GFX942_CONV_WGRAD_AOT, denseFixture.context());
    ASSERT_TRUE(denseBound);
    const auto helperRefusal = prepareRefusal(handler, denseFixture, *denseBound, kernels[5]);
    ASSERT_TRUE(helperRefusal);
    EXPECT_NE(helperRefusal->find(std::string(wgrad::HELPER_REFUSAL)), std::string::npos)
        << *helperRefusal;
    // Bindings other than the graph matcher's are refused.
    auto otherBound = *denseBound;
    otherBound[std::string(GFX942_CONV_WGRAD_AOT.outputToken)] = int64_t{7};
    EXPECT_THROW(handler.prepare(denseFixture.context(), otherBound, kernels[0]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// The shipped catalog as the engine loads it: one dispatch, every kernel in the pack
/// make_conv_wgrad_aot_config.shard_name(spec, "family") puts it in -- implicit GEMM in
/// {dtype}_implicit, each direct pipeline and each helper variant in {dtype}_{variant} --
/// so every helper sits in a pack other than its instance's.
TEST(TestGfx942ConvWgradNative, TheShippedPacksServeEveryPlannedInstance)
{
    const auto idOf16 = [](size_t value, uint8_t tag) {
        DescriptorId id{};
        id[0] = tag;
        id[14] = static_cast<uint8_t>(value >> 8);
        id[15] = static_cast<uint8_t>(value);
        return id;
    };
    constexpr uint8_t ENGINE = 0x90;
    std::map<std::string, std::vector<KernelDefinition>> byDtype;
    std::map<std::string, size_t> packs;
    size_t next = 0;
    for(const auto* dtype : {"fp16", "bf16"})
    {
        for(const auto& entry : catalog::catalog(dtype))
        {
            const auto shard = std::string(dtype) + "_"
                               + (entry.integer("kernel_family") == 0
                                      ? std::string("implicit")
                                      : entry.text("direct_variant"));
            const auto pack = packs.emplace(shard, packs.size()).first->second;
            KernelDefinition kernel;
            kernel.kernelId = idOf16(next++, ENGINE + 1);
            kernel.packId = idOf16(pack, ENGINE + 2);
            kernel.dispatchId = idOf16(0, ENGINE);
            kernel.source.kind = KernelSourceKind::KPACK;
            kernel.name = "gfx942_conv_wgrad_aot_" + entry.text("instance_id");
            for(const auto& [name, value] : entry.fields)
            {
                if(const auto* integer = std::get_if<int64_t>(&value))
                {
                    kernel.metadata[name] = *integer;
                }
                else
                {
                    kernel.metadata[name] = std::get<std::string>(value);
                }
            }
            byDtype[dtype].push_back(std::move(kernel));
        }
        prepareAll(byDtype[dtype]);
    }
    // The shards the catalog ships: implicit and the Stage 2 per dtype, plus bf16's
    // direct_wgrad_dw_cols and its fold.
    EXPECT_EQ(packs.size(), 6U);

    std::map<std::string, size_t> servedBySteps;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = catalog::toProblem(row.problem, row.dtype);
        const auto fixture = wgradFixture(problem, problem.dwFloat);
        const auto bound = matchesGraph(GFX942_CONV_WGRAD_AOT, fixture.context());
        const auto& kernels = byDtype.at(row.dtype);
        if(!bound)
        {
            // A problem outside the graph contract (a nonpositive dY, say) serves nothing.
            EXPECT_EQ(std::string(row.served).find('1'), std::string::npos)
                << describe(problem);
            continue;
        }
        const auto& entries = catalog::catalog(row.dtype);
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const bool expected = row.served[i] == '1';
            EXPECT_EQ(matchesKernel(GFX942_CONV_WGRAD_AOT, fixture.context(), kernels[i], *bound),
                      expected)
                << entries[i].text("direct_variant") << " " << entries[i].text("instance_id")
                << " on " << describe(problem) << " " << row.dtype
                << (problem.dwFloat ? " fp32 dW" : "");
            if(expected)
            {
                const auto plan = wgrad::instancePlan(problem, *entries[i].instance);
                ++servedBySteps[std::to_string(plan->steps.size())];
            }
        }
    }
    // One-, two- and three-step plans (a folded pipeline cast into a 16-bit dW) all serve.
    for(const auto* steps : {"1", "2", "3"})
    {
        EXPECT_GT(servedBySteps[steps], 0U) << steps;
    }
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

} // namespace
