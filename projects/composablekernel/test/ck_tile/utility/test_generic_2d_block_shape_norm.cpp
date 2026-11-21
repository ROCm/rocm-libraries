// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include "ck_tile/ops/rmsnorm2d.hpp"

// RMSNorm and LayerNorm with the rmsnorm2d and layernorm2d generators' shapes whose rows are
// narrower than a wave, where each wave holds several rows. Generic2dBlockShape used to give
// these shapes 2-4x the threads, and the extra waves wrote rows of the next workgroup
// (https://github.com/ROCm/rocm-libraries/issues/15811).

namespace {

template <ck_tile::index_t ThreadPerBlock_M,
          ck_tile::index_t ThreadPerBlock_N,
          ck_tile::index_t Vector_N>
using NormShape =
    ck_tile::Generic2dBlockShape<ck_tile::sequence<ThreadPerBlock_M, ThreadPerBlock_N * Vector_N>,
                                 ck_tile::sequence<ThreadPerBlock_M, ThreadPerBlock_N>,
                                 ck_tile::sequence<1, Vector_N>>;

template <typename Shape>
struct RmsnormKernel
{
    using Traits   = ck_tile::Rmsnorm2dFwdTraits<true,  // kPadN
                                                 false, // kSaveInvRms
                                                 false, // kSaveUnquant
                                                 false, // kTwoPass
                                                 ck_tile::Rmsnorm2dFusedAddEnum::NO_ADD,
                                                 ck_tile::Rmsnorm2dFusedQuantEnum::NO_SWEEP,
                                                 ck_tile::Rmsnorm2dSensitiveEnum::NO_SPECIFIC_MODEL>;
    using Problem  = ck_tile::Rmsnorm2dFwdPipelineProblem<float,
                                                          float,
                                                          float,
                                                          float,
                                                          float,
                                                          float,
                                                          float,
                                                          float,
                                                          Shape,
                                                          Traits>;
    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<float, float, false, true, false>>;
    using Kernel = ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<Problem>, Epilogue>;

    static auto
    Kargs(const void* x, const void* gamma, void* y, ck_tile::index_t m, ck_tile::index_t n)
    {
        return Kernel::MakeKargs(ck_tile::Rmsnorm2dFwdHostArgs{x,
                                                               nullptr,
                                                               nullptr,
                                                               gamma,
                                                               y,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               1e-5f,
                                                               m,
                                                               n,
                                                               n,
                                                               n,
                                                               n,
                                                               n});
    }
};

template <typename Shape>
struct LayernormKernel
{
    using Traits   = ck_tile::Layernorm2dFwdTraits<true,  // kPadN
                                                   false, // kSaveMeanInvStd
                                                   false, // kFastFDiv
                                                   true,  // kWelford
                                                   false, // kTwoPass
                                                   ck_tile::Layernorm2dXBiasEnum::NO_BIAS,
                                                   ck_tile::Layernorm2dFusedAddEnum::NO_ADD,
                                                   ck_tile::Layernorm2dFusedQuantEnum::NO_SWEEP>;
    using Problem  = ck_tile::Layernorm2dFwdPipelineProblem<float,
                                                            float,
                                                            float,
                                                            float,
                                                            float,
                                                            float,
                                                            float,
                                                            float,
                                                            float,
                                                            float,
                                                            Shape,
                                                            Traits>;
    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<float, float, false, true, false>>;
    using Kernel =
        ck_tile::Layernorm2dFwd<ck_tile::Layernorm2dFwdPipelineOnePass<Problem>, Epilogue>;

    static auto
    Kargs(const void* x, const void* gamma, void* y, ck_tile::index_t m, ck_tile::index_t n)
    {
        // beta is gamma's buffer: any [n] input will do.
        return Kernel::MakeKargs(ck_tile::Layernorm2dFwdHostArgs{x,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 gamma,
                                                                 gamma,
                                                                 y,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 1e-5f,
                                                                 m,
                                                                 n,
                                                                 n,
                                                                 n,
                                                                 n,
                                                                 n});
    }
};

template <typename Shape>
void Reference(const ck_tile::HostTensor<float>& x,
               const ck_tile::HostTensor<float>& gamma,
               ck_tile::HostTensor<float>& y,
               bool layernorm)
{
    const ck_tile::index_t m = x.get_lengths()[0];
    const ck_tile::index_t n = x.get_lengths()[1];
    if(layernorm)
    {
        ck_tile::HostTensor<float> mean({m}), inv_std({m});
        ck_tile::reference_layernorm2d_fwd<float, float, float, float, float, float, float>(
            x, gamma, gamma, y, mean, inv_std, 1e-5f);
    }
    else
    {
        ck_tile::HostTensor<float> inv_rms({m});
        ck_tile::HostTensor<ck_tile::null_type> unquant({m, n});
        ck_tile::reference_rmsnorm2d_fwd<float, float, float, float, float, ck_tile::null_type>(
            x, gamma, y, inv_rms, unquant, 1e-5f);
    }
}

// Runs `workgroups` workgroups of the kernel over an m x n tensor whose y starts as `sentinel`,
// and returns y.
template <typename Norm>
std::vector<float> Run(const ck_tile::HostTensor<float>& x,
                       const ck_tile::HostTensor<float>& gamma,
                       ck_tile::index_t workgroups,
                       float sentinel)
{
    using Kernel             = typename Norm::Kernel;
    const ck_tile::index_t m = x.get_lengths()[0];
    const ck_tile::index_t n = x.get_lengths()[1];

    std::vector<float> y(m * n, sentinel);
    ck_tile::DeviceMem x_buf(x.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y.size() * sizeof(float));
    x_buf.ToDevice(x.data());
    gamma_buf.ToDevice(gamma.data());
    y_buf.ToDevice(y.data());

    const auto kargs = Norm::Kargs(
        x_buf.GetDeviceBuffer(), gamma_buf.GetDeviceBuffer(), y_buf.GetDeviceBuffer(), m, n);
    ck_tile::launch_kernel(
        ck_tile::stream_config{},
        ck_tile::make_kernel<1>(Kernel{}, dim3(workgroups), dim3(Kernel::BlockSize()), 0, kargs));
    y_buf.FromDevice(y.data());
    return y;
}

template <typename Shape, typename Norm>
void Check(bool layernorm)
{
    using Kernel = typename Norm::Kernel;
    // The block has exactly the shape's threads.
    EXPECT_EQ(Kernel::BlockSize(), Shape::ThreadPerBlock_M * Shape::ThreadPerBlock_N);

    for(const ck_tile::index_t n : {Shape::Block_N, Shape::Block_N - Shape::Vector_N})
    {
        // Four workgroups, each normalizing its rows.
        const ck_tile::index_t m = 4 * Shape::Block_M;
        ck_tile::HostTensor<float> x({m, n}), gamma({n}), y_ref({m, n});
        ck_tile::FillUniformDistribution<float>{-1.f, 1.f, 11939}(x);
        ck_tile::FillUniformDistribution<float>{0.5f, 1.5f, 7}(gamma);
        Reference<Shape>(x, gamma, y_ref, layernorm);

        const std::vector<float> y = Run<Norm>(x, gamma, m / Shape::Block_M, 0.f);
        ck_tile::HostTensor<float> y_dev({m, n});
        std::copy(y.begin(), y.end(), y_dev.mData.begin());
        EXPECT_TRUE(ck_tile::check_err(y_dev, y_ref, "wrong norm", 1e-4, 1e-5)) << "n = " << n;

        // Workgroup 0 alone: it writes its own rows and leaves the next workgroup's untouched.
        constexpr float sentinel    = 1234.5f;
        const std::vector<float> y0 = Run<Norm>(x, gamma, 1, sentinel);
        for(ck_tile::index_t i = 0; i < 2 * Shape::Block_M; ++i)
        {
            for(ck_tile::index_t j = 0; j < n; ++j)
            {
                if(i < Shape::Block_M)
                    EXPECT_NEAR(y0[i * n + j], y_ref(i, j), 1e-4) << "row " << i << ", n = " << n;
                else
                    EXPECT_EQ(y0[i * n + j], sentinel) << "row " << i << ", n = " << n;
            }
        }
    }
}

} // namespace

// The generators' narrow shapes: {ThreadPerBlock_M, ThreadPerBlock_N, Vector_N}.

TEST(Generic2dBlockShapeNorm, Rmsnorm_4x16x4)
{
    Check<NormShape<4, 16, 4>, RmsnormKernel<NormShape<4, 16, 4>>>(false);
}
TEST(Generic2dBlockShapeNorm, Rmsnorm_4x16x8)
{
    Check<NormShape<4, 16, 8>, RmsnormKernel<NormShape<4, 16, 8>>>(false);
}
TEST(Generic2dBlockShapeNorm, Rmsnorm_8x8x8)
{
    Check<NormShape<8, 8, 8>, RmsnormKernel<NormShape<8, 8, 8>>>(false);
}
TEST(Generic2dBlockShapeNorm, Rmsnorm_8x32x8)
{
    Check<NormShape<8, 32, 8>, RmsnormKernel<NormShape<8, 32, 8>>>(false);
}

TEST(Generic2dBlockShapeNorm, Layernorm_4x16x4)
{
    Check<NormShape<4, 16, 4>, LayernormKernel<NormShape<4, 16, 4>>>(true);
}
TEST(Generic2dBlockShapeNorm, Layernorm_4x16x8)
{
    Check<NormShape<4, 16, 8>, LayernormKernel<NormShape<4, 16, 8>>>(true);
}
TEST(Generic2dBlockShapeNorm, Layernorm_8x8x8)
{
    Check<NormShape<8, 8, 8>, LayernormKernel<NormShape<8, 8, 8>>>(true);
}
TEST(Generic2dBlockShapeNorm, Layernorm_8x32x8)
{
    Check<NormShape<8, 32, 8>, LayernormKernel<NormShape<8, 32, 8>>>(true);
}
