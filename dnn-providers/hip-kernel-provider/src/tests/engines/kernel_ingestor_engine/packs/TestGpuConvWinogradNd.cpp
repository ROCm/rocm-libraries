// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvDeterminismTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvWinogradNdProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <initializer_list>
#include <vector>
namespace
{
using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;

long long off(const long long* s, long long n, long long c, long long d, long long h, long long w)
{
    return n * s[0] + c * s[1] + d * s[2] + h * s[3] + w * s[4];
}
long long span(const long long* s, long long n, long long c, const long long* spatial)
{
    return off(s, n - 1, c - 1, spatial[0] - 1, spatial[1] - 1, spatial[2] - 1) + 1;
}
KernelDefinition ndKernel(int bm, int chunk)
{
    KernelDefinition k;
    k.name = "winograd_nd.numeric";
    k.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
    k.source.sourceFile = "kernels/ConvWinogradNd.cpp";
    k.source.entryPoint = "ConvWinogradNdGemm";
    k.metadata["direction"] = std::string("FWD");
    k.metadata["dtype"] = std::string("FLOAT");
    k.metadata["arch"] = std::string("gfx1250");
    k.metadata["atom"] = int64_t(5);
    k.metadata["precision"] = int64_t(0);
    k.metadata["gemm_bm"] = int64_t(bm);
    k.metadata["gemm_bn"] = int64_t(64);
    k.metadata["gemm_bk"] = int64_t(16);
    k.metadata["chunk_tiles"] = int64_t(chunk);
    return k;
}
// Independent direct forward-convolution reference in original graph roles.
std::vector<double>
    direct(const ConvNaiveArgs& a, const float* x, const float* w, bool magnitude = false)
{
    std::vector<double> result(static_cast<size_t>(span(a.yStrides, a.n, a.k, a.outSpatial)), 0.0);
    const long long cg = a.c / a.groups;
    const long long kg = a.k / a.groups;
    for(long long n = 0; n < a.n; ++n)
    {
        for(long long k = 0; k < a.k; ++k)
        {
            for(long long od = 0; od < a.outSpatial[0]; ++od)
            {
                for(long long oh = 0; oh < a.outSpatial[1]; ++oh)
                {
                    for(long long ow = 0; ow < a.outSpatial[2]; ++ow)
                    {
                        const auto yo = off(a.yStrides, n, k, od, oh, ow);
                        for(long long c = 0; c < cg; ++c)
                        {
                            for(long long rd = 0; rd < a.filter[0]; ++rd)
                            {
                                for(long long rh = 0; rh < a.filter[1]; ++rh)
                                {
                                    for(long long rw = 0; rw < a.filter[2]; ++rw)
                                    {
                                        const long long d
                                            = od - a.prePadding[0] + rd * a.dilation[0];
                                        const long long h
                                            = oh - a.prePadding[1] + rh * a.dilation[1];
                                        const long long v
                                            = ow - a.prePadding[2] + rw * a.dilation[2];
                                        if(d < 0 || d >= a.inSpatial[0] || h < 0
                                           || h >= a.inSpatial[1] || v < 0 || v >= a.inSpatial[2])
                                        {
                                            continue;
                                        }
                                        const auto xo
                                            = off(a.xStrides, n, (k / kg) * cg + c, d, h, v);
                                        const bool flip = a.flipFilter != 0;
                                        const auto wo = off(a.wStrides,
                                                            k,
                                                            c,
                                                            flip ? a.filter[0] - 1 - rd : rd,
                                                            flip ? a.filter[1] - 1 - rh : rh,
                                                            flip ? a.filter[2] - 1 - rw : rw);
                                        const double product = static_cast<double>(x[xo])
                                                               * static_cast<double>(w[wo]);
                                        result[static_cast<size_t>(yo)]
                                            += magnitude ? std::fabs(product) : product;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}
void runNd(bool last,
           bool reflect,
           int dilationH,
           int dilationW,
           int channelsPerGroup,
           int outputsPerGroup,
           std::initializer_list<int> chunks)
{
    ConvShape shape;
    shape.n = 1;
    shape.c = 2 * static_cast<int64_t>(channelsPerGroup);
    shape.k = 2 * static_cast<int64_t>(outputsPerGroup);
    shape.groups = 2;
    shape.channelsLast = last;
    shape.dataType = DataType::FLOAT;
    SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(shape.dataType));
    shape.input = {13, 17};
    shape.filter = {3, 3};
    shape.dilation = {dilationH, dilationW};
    shape.stride = {1, 1};
    shape.prePadding = {dilationH + 1, dilationW - 1};
    shape.postPadding = {dilationH - 1, dilationW + 1};
    shape.mode = reflect ? ConvMode::CONVOLUTION : ConvMode::CROSS_CORRELATION;
    SCOPED_TRACE(::testing::Message() << "C/g " << channelsPerGroup << " K/g " << outputsPerGroup
                                      << " channelsLast " << last << " reflected filter " << reflect
                                      << " dilation " << dilationH << "x" << dilationW);
    auto spec = specFor(shape, ConvDirection::FORWARD);
    // Deliberately mix packed output layout and filter layout with input layout.
    spec.y.strides = packedStrides(spec.y.dims, !last);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const PackSymbols pack{"",
                           "hipkernel.conv_winograd_nd_fwd.graph_match",
                           "",
                           "hipkernel.conv_winograd_nd.kernel_match",
                           "",
                           "hipkernel.conv_winograd_nd.dispatch",
                           "",
                           "",
                           ""};
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound);
    const auto problem = conv_winograd_nd::parseProblem(fixture.context(), ConvDirection::FORWARD);
    ASSERT_TRUE(problem);
    Tensor<float> x(spec.x.dims, spec.x.strides);
    Tensor<float> w(spec.w.dims, spec.w.strides);
    Tensor<float> y(spec.y.dims, spec.y.strides);
    auto& input = x;
    auto& output = y;
    input.fillTensorWithRandomValues(-1.0f, 1.0f, 7123);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, 9213);
    const auto& handler = dispatchHandler(pack);
    for(const int bm : {64, 128})
    {
        for(const int chunk : chunks)
        {
            auto kernel = ndKernel(bm, chunk);
            if(!conv_winograd_nd::configRunsOn(*conv_winograd_nd::configOf(kernel),
                                               fixture.context().deviceProperties.gcnArchName))
            {
                GTEST_SKIP() << "Dilated forward convolution requires gfx1250";
            }
            ASSERT_TRUE(matchesKernel(pack, fixture.context(), kernel, *bound));
            const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
            ASSERT_NE(prepared, nullptr);
            const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
            const hipdnn_data_sdk::utilities::Workspace<> workspace(workspaceBytes);
            const Handle handle;
            // Repeat at the same filter address with different contents, and change the input binding.
            Tensor<float> otherInput(input.dims(), input.strides());
            otherInput.fillTensorWithRandomValues(-1.0f, 1.0f, 5171);
            for(int repeat = 0; repeat < 2; ++repeat)
            {
                auto& selected = repeat != 0 ? otherInput : input;
                w.fillTensorWithRandomValues(
                    -1.0f, 1.0f, 9213u + static_cast<unsigned int>(repeat));
                const auto ref = direct(problem->base.args,
                                        static_cast<const float*>(selected.rawHostData()),
                                        static_cast<const float*>(w.rawHostData()));
                const auto magnitude = direct(problem->base.args,
                                              static_cast<const float*>(selected.rawHostData()),
                                              static_cast<const float*>(w.rawHostData()),
                                              true);
                output.fillWithSentinelValue();
                std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
                    {{X_UID, selected.rawDeviceData()},
                     {W_UID, w.rawDeviceData()},
                     {Y_UID, output.rawDeviceData()}}};
                handler.launch(handle, *prepared, buffers.data(), 3, workspace.get());
                ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
                EXPECT_TRUE(conv_determinism_test::relaunchReproducesOutput(
                    [&] { handler.launch(handle, *prepared, buffers.data(), 3, workspace.get()); },
                    handle.getStream(),
                    output,
                    workspace.get(),
                    workspaceBytes));
                output.markDeviceModified();
                const auto* actual = static_cast<const float*>(output.rawHostData());
                // Existing F2 FP32 tolerance, scaled by sum-|products| in the same residue tile.
                const auto& a = problem->args;
                for(long long n = 0; n < a.n; ++n)
                {
                    for(long long c = 0; c < a.co; ++c)
                    {
                        for(long long h = 0; h < a.output[0]; ++h)
                        {
                            for(long long v = 0; v < a.output[1]; ++v)
                            {
                                const auto offset = [&](long long yy, long long xx) {
                                    return n * a.outStride[0] + c * a.outStride[1]
                                           + yy * a.outStride[2] + xx * a.outStride[3];
                                };
                                const auto i = static_cast<size_t>(offset(h, v));
                                double scale = 0.0;
                                for(int dy = -1; dy <= 1; ++dy)
                                {
                                    for(int dx = -1; dx <= 1; ++dx)
                                    {
                                        const long long yy = h + dy * a.dilation[0];
                                        const long long xx = v + dx * a.dilation[1];
                                        if(yy >= 0 && yy < a.output[0] && xx >= 0
                                           && xx < a.output[1])
                                        {
                                            scale = std::max(
                                                scale,
                                                magnitude[static_cast<size_t>(offset(yy, xx))]);
                                        }
                                    }
                                }
                                EXPECT_NEAR(static_cast<double>(static_cast<float>(actual[i])),
                                            ref[i],
                                            1e-5 * scale + 2.0 * 1.2e-7 * std::fabs(ref[i]) + 1e-6)
                                    << "bm " << bm << " chunk " << chunk << " offset " << i;
                            }
                        }
                    }
                }
            }
        }
    }
}
TEST(TestGpuConvWinogradNd, AnisotropicAndHighDilationResidues)
{
    SKIP_IF_NO_DEVICES();
    for(auto dilation : {std::array<int, 2>{2, 4},
                         std::array<int, 2>{4, 2},
                         std::array<int, 2>{6, 12},
                         std::array<int, 2>{24, 36}})
    {
        for(const bool last : {false, true})
        {
            for(const bool reflect : {false, true})
            {
                runNd(last, reflect, dilation[0], dilation[1], 3, 5, {256, 4096, 3});
            }
        }
    }
}
TEST(TestGpuConvWinogradNd, NchwMultipleReductionAndOutputRowTails)
{
    SKIP_IF_NO_DEVICES();
    // C/g = 17 crosses BK16; K/g = 65 crosses BM64. The 96 residue tiles cross
    // BN64 in a 65-tile chunk and leave another chunk, each with a partial BN64 block.
    runNd(false, false, 2, 4, 17, 65, {65, 3});
}
TEST(TestGpuConvWinogradNd, NhwcMultipleReductionAndOutputRowTailsConvolution)
{
    SKIP_IF_NO_DEVICES();
    // Three BK16 steps and a row past both BM64 and BM128, with mixed output layout
    // and reflected filters. Both chunk sizes require multiple launches.
    runNd(true, true, 4, 2, 33, 129, {65, 3});
}
}
#endif
