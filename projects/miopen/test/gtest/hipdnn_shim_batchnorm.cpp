// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shim-surface tests for the forwarded batchnorm entry points. The derived
// suites reuse the bn.hpp fixtures, which reach compute only through the public
// miopenBatchNorm* API and check against the fusionHost.hpp CPU reference. The
// plain tests cover what those fixtures cannot reach: epsilon and momentum in
// the plan key, running stats updated in place, null save and running pointers,
// and the cases the hipDNN path declines while MIOpen serves them. Each of those
// records whether the call was served so the parity harness can pin the
// expected divergences.
//
// Not covered, because MIOpen itself has no defined behavior to compare with:
// null alpha/beta pointers (MIOpen dereferences them), null dscale/dbias on
// backward, and a null bias with RELU on backward activation (the MIOpen kernel
// reads the null device pointer).

#include "bn.hpp"
#include "hipdnn_shim_helpers.hpp"

#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

namespace hs = hipdnn_shim_test;

namespace {

using TrainData = BNFwdTrainTestData<float, float, float, float, float, double, BN2DTestCase>;
using InferData = BNInferTestData<float, float, float, float, float, double, BN2DTestCase>;
using BwdData   = BNBwdTestData<float, float, float, float, float, float, double, BN2DTestCase>;

constexpr double kTol = 4e-3;

// The epsilon the hipDNN provider hardcodes when backward recomputes the stats.
constexpr double kProviderEpsilon = 1e-5;

BN2DTestCase SmallCase()
{
    return {2, 4, 8, 8, miopen::batchnorm::Direction::ForwardTraining, true, true};
}

void SetActivation(miopenActivationMode_t& mode,
                   double& alpha,
                   double& beta,
                   miopenActivationMode_t new_mode)
{
    mode  = new_mode;
    alpha = 0.1;
    beta  = 0.3;
}

void SetUpTrain(TrainData& d, miopenBatchNormMode_t bn_mode = miopenBNSpatial)
{
    d.SetUpImpl(SmallCase(), bn_mode, miopenTensorNCHW);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

void SetUpInfer(InferData& d, miopenBatchNormMode_t bn_mode = miopenBNSpatial)
{
    d.SetUpImpl(SmallCase(), bn_mode, miopenTensorNCHW, false);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

void SetUpBwd(BwdData& d, miopenBatchNormMode_t bn_mode = miopenBNSpatial)
{
    d.SetUpImpl(SmallCase(), bn_mode, miopenTensorNCHW);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

miopenStatus_t TrainV1(miopen::Handle& handle, TrainData& d)
{
    return miopenBatchNormalizationForwardTraining(&handle,
                                                   d.bn_mode,
                                                   &d.alpha,
                                                   &d.beta,
                                                   &d.input.desc,
                                                   d.in_dev.get(),
                                                   &d.output.desc,
                                                   d.out_dev.get(),
                                                   &d.scale.desc,
                                                   d.scale_dev.get(),
                                                   d.shift_dev.get(),
                                                   d.averageFactor,
                                                   d.runMean_dev.get(),
                                                   d.runVariance_dev.get(),
                                                   d.epsilon,
                                                   d.saveMean_dev.get(),
                                                   d.saveVariance_dev.get());
}

// Which of the V2 save and running pointers are passed; the others are null.
struct TrainPointers
{
    bool save_mean;
    bool save_var;
    bool run_mean;
    bool run_var;
};

miopenStatus_t TrainV2(miopen::Handle& handle, TrainData& d, TrainPointers p)
{
    return miopenBatchNormalizationForwardTraining_V2(&handle,
                                                      d.bn_mode,
                                                      &d.alpha,
                                                      &d.beta,
                                                      &d.input.desc,
                                                      d.in_dev.get(),
                                                      &d.output.desc,
                                                      d.out_dev.get(),
                                                      &d.scale.desc,
                                                      &d.shift.desc,
                                                      &d.saveMean.desc,
                                                      &d.saveVariance.desc,
                                                      d.scale_dev.get(),
                                                      d.shift_dev.get(),
                                                      d.averageFactor,
                                                      p.run_mean ? d.runMean_dev.get() : nullptr,
                                                      p.run_var ? d.runVariance_dev.get() : nullptr,
                                                      d.epsilon,
                                                      p.save_mean ? d.saveMean_dev.get() : nullptr,
                                                      p.save_var ? d.saveVariance_dev.get()
                                                                 : nullptr);
}

miopenStatus_t
TrainActivation(miopen::Handle& handle, TrainData& d, miopenActivationDescriptor_t activ)
{
    return miopenBatchNormForwardTrainingActivation(&handle,
                                                    d.bn_mode,
                                                    &d.alpha,
                                                    &d.beta,
                                                    &d.input.desc,
                                                    d.in_dev.get(),
                                                    &d.output.desc,
                                                    d.out_dev.get(),
                                                    &d.scale.desc,
                                                    &d.shift.desc,
                                                    &d.saveMean.desc,
                                                    &d.saveVariance.desc,
                                                    d.scale_dev.get(),
                                                    d.shift_dev.get(),
                                                    d.averageFactor,
                                                    d.runMean_dev.get(),
                                                    d.runVariance_dev.get(),
                                                    d.epsilon,
                                                    d.saveMean_dev.get(),
                                                    d.saveVariance_dev.get(),
                                                    activ);
}

miopenStatus_t InferV1(miopen::Handle& handle, InferData& d, bool with_estimates)
{
    return miopenBatchNormalizationForwardInference(&handle,
                                                    d.bn_mode,
                                                    &d.alpha,
                                                    &d.beta,
                                                    &d.input.desc,
                                                    d.in_dev.get(),
                                                    &d.output.desc,
                                                    d.out_dev.get(),
                                                    &d.scale.desc,
                                                    d.scale_dev.get(),
                                                    d.shift_dev.get(),
                                                    with_estimates ? d.estMean_dev.get() : nullptr,
                                                    with_estimates ? d.estVariance_dev.get()
                                                                   : nullptr,
                                                    d.epsilon);
}

miopenStatus_t BwdV1(miopen::Handle& handle, BwdData& d, bool with_saved_stats)
{
    return miopenBatchNormalizationBackward(&handle,
                                            d.bn_mode,
                                            &d.alphaDataDiff,
                                            &d.betaDataDiff,
                                            &d.alphaParamDiff,
                                            &d.betaParamDiff,
                                            &d.input.desc,
                                            d.in_dev.get(),
                                            &d.dy.desc,
                                            d.dy_dev.get(),
                                            &d.output.desc,
                                            d.out_dev.get(),
                                            &d.bnScale.desc,
                                            d.bnScale_dev.get(),
                                            d.dScale_dev.get(),
                                            d.dBias_dev.get(),
                                            d.epsilon,
                                            with_saved_stats ? d.savedMean_dev.get() : nullptr,
                                            with_saved_stats ? d.savedInvVar_dev.get() : nullptr);
}

miopenStatus_t BwdActivation(miopen::Handle& handle,
                             BwdData& d,
                             bool with_saved_stats,
                             miopenActivationDescriptor_t activ)
{
    return miopenBatchNormBackwardActivation(&handle,
                                             d.bn_mode,
                                             &d.alphaDataDiff,
                                             &d.betaDataDiff,
                                             &d.alphaParamDiff,
                                             &d.betaParamDiff,
                                             &d.input.desc,
                                             d.in_dev.get(),
                                             &d.dy.desc,
                                             d.dy_dev.get(),
                                             &d.output.desc,
                                             d.out_dev.get(),
                                             &d.bnScale.desc,
                                             &d.dBias.desc,
                                             &d.savedMean.desc,
                                             &d.savedInvVar.desc,
                                             d.bnScale_dev.get(),
                                             d.bnBias_dev.get(),
                                             d.dScale_dev.get(),
                                             d.dBias_dev.get(),
                                             d.epsilon,
                                             with_saved_stats ? d.savedMean_dev.get() : nullptr,
                                             with_saved_stats ? d.savedInvVar_dev.get() : nullptr,
                                             activ);
}

// Reads the device results back and compares them with the CPU reference. Only
// the stat groups the call was given in full are compared: MIOpen skips the
// save or running update when any pointer of that group is null.
void CheckTrain(miopen::Handle& handle, TrainData& d, bool save, bool running)
{
    d.output.data = handle.Read<float>(d.out_dev, d.output.data.size());
    if(save)
    {
        d.saveMean.data     = handle.Read<float>(d.saveMean_dev, d.saveMean.data.size());
        d.saveVariance.data = handle.Read<float>(d.saveVariance_dev, d.saveVariance.data.size());
    }
    if(running)
    {
        d.runMean.data     = handle.Read<float>(d.runMean_dev, d.runMean.data.size());
        d.runVariance.data = handle.Read<float>(d.runVariance_dev, d.runVariance.data.size());
    }

    test::ComputeCPUBNFwdTrain(d);
    activationHostInfer(
        d.activ_mode, 0.0, d.activ_beta, d.activ_alpha, d.out_ref.data, d.out_ref.data);

    test::CompareTensor<float>(d.output, d.out_ref, kTol);
    if(save)
    {
        test::CompareTensor<float>(d.saveMean, d.saveMean_ref, kTol);
        test::CompareTensor<float>(d.saveVariance, d.saveVariance_ref, kTol);
    }
    if(running)
    {
        test::CompareTensor<float>(d.runMean, d.runMean_ref, kTol);
        test::CompareTensor<float>(d.runVariance, d.runVariance_ref, kTol);
    }
}

void CheckInfer(miopen::Handle& handle, InferData& d)
{
    d.output.data = handle.Read<float>(d.out_dev, d.output.data.size());
    test::ComputeCPUBNInference(d);
    test::CompareTensor<float>(d.output, d.out_ref, kTol);
}

void CheckBwd(miopen::Handle& handle, BwdData& d)
{
    d.output.data = handle.Read<float>(d.out_dev, d.output.data.size());
    d.dScale.data = handle.Read<float>(d.dScale_dev, d.dScale.data.size());
    d.dBias.data  = handle.Read<float>(d.dBias_dev, d.dBias.data.size());

    test::ComputeCPUBNBwd(d);

    test::CompareTensor<float>(d.output, d.out_ref, kTol);
    test::CompareTensor<float>(d.dScale, d.dScale_ref, kTol);
    test::CompareTensor<float>(d.dBias, d.dBias_ref, kTol);
}

// Backward without saved stats recomputes them from x. The CPU reference reads
// only savedMean and savedInvVar, so give it the true per-channel stats of the
// NCHW input (biased variance, as the kernels use).
void FillTrueStats(BwdData& d, double eps)
{
    const auto& lens    = d.input.desc.GetLengths();
    const std::size_t n = lens[0];
    const std::size_t c = lens[1];
    std::size_t spatial = 1;
    for(std::size_t i = 2; i < lens.size(); ++i)
        spatial *= lens[i];
    const double count = static_cast<double>(n * spatial);

    for(std::size_t ci = 0; ci < c; ++ci)
    {
        double sum = 0.0;
        for(std::size_t ni = 0; ni < n; ++ni)
            for(std::size_t i = 0; i < spatial; ++i)
                sum += d.input.data[(ni * c + ci) * spatial + i];
        const double mean = sum / count;

        double sq = 0.0;
        for(std::size_t ni = 0; ni < n; ++ni)
        {
            for(std::size_t i = 0; i < spatial; ++i)
            {
                const double diff = d.input.data[(ni * c + ci) * spatial + i] - mean;
                sq += diff * diff;
            }
        }

        d.savedMean.data[ci]   = static_cast<float>(mean);
        d.savedInvVar.data[ci] = static_cast<float>(1.0 / std::sqrt(sq / count + eps));
    }
}

// A served call or an expected decline. A decline must come from the hipDNN path
// with the given reason; MIOpen serves every case these tests send it.
bool RecordOutcome(const std::string& case_name, const hs::CallResult& result, const char* reason)
{
    const bool served = result.status == miopenStatusSuccess;
    EXPECT_TRUE(served || result.status == miopenStatusUnsupportedOp ||
                result.status == miopenStatusNotImplemented)
        << case_name << ": " << result.message;
    if(!served)
        hs::ExpectForwardedDecline(result, reason);
    hs::RecordServed(case_name, served);
    return served;
}

// Each derived suite records a hash of everything its fixture reads back. The CPU
// reference check runs in the base TearDown and passes within kTol; the hash goes
// further, because the parity comparison needs it to match bit for bit, which only
// holds if the forwarded call ran the same kernel with the same configuration that
// MIOpen picks. Whether to hash is decided before the base TearDown runs, because
// that is where a skipped or failed test returns before reading anything back.
template <typename Base>
struct HashTrainOutputs : Base
{
    void TearDown() override
    {
        const bool read = !this->test_skipped && !this->HasFailure();
        Base::TearDown();
        if(!read || this->HasFailure())
            return;
        const auto& d = this->bn_fwd_train_test_data;
        hs::OutputHash hash;
        hash.Add(d.output.data);
        hash.Add(d.saveMean.data);
        hash.Add(d.saveVariance.data);
        hash.Add(d.runMean.data);
        hash.Add(d.runVariance.data);
        if(this->api_type == BNApiType::testBNAPIV3)
        {
            hash.Add(d.nextRunMean.data);
            hash.Add(d.nextRunVariance.data);
        }
        hs::RecordOutputHash(hash);
    }
};

template <typename Base>
struct HashInferOutputs : Base
{
    void TearDown() override
    {
        const bool read = !this->test_skipped && !this->HasFailure();
        Base::TearDown();
        if(!read || this->HasFailure())
            return;
        hs::OutputHash hash;
        hash.Add(this->bn_infer_test_data.output.data);
        hs::RecordOutputHash(hash);
    }
};

template <typename Base>
struct HashBwdOutputs : Base
{
    void TearDown() override
    {
        const bool read = !this->test_skipped && !this->HasFailure();
        Base::TearDown();
        if(!read || this->HasFailure())
            return;
        const auto& d = this->bn_bwd_test_data;
        hs::OutputHash hash;
        hash.Add(d.output.data);
        hash.Add(d.dScale.data);
        hash.Add(d.dBias.data);
        hs::RecordOutputHash(hash);
    }
};

} // namespace

// The derived suites below use the Smoke prefix, like hipdnn_shim_conv*.cpp. The
// HipdnnShim in the suite name is what puts them on the shim surface.

// ---------------------------------------------------------------------------
// Forward training

struct GPU_HipdnnShimBNFwdTrain_FP32
    : HashTrainOutputs<BNFwdTrainTest<float, float, float, float, float, double, BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNFwdTrain_FP16 : HashTrainOutputs<BNFwdTrainTest<half_float::half,
                                                                       half_float::half,
                                                                       float,
                                                                       float,
                                                                       float,
                                                                       double,
                                                                       BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNFwdTrain_BFP16
    : HashTrainOutputs<
          BNFwdTrainTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNFwdTrain3D_FP32
    : HashTrainOutputs<BNFwdTrainTest<float, float, float, float, float, double, BN3DTestCase>>
{
};

struct GPU_HipdnnShimBNFwdTrainRelu_FP32
    : HashTrainOutputs<BNFwdTrainTest<float, float, float, float, float, double, BN2DTestCase>>
{
};

TEST_P(GPU_HipdnnShimBNFwdTrain_FP32, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNFwdTrain_FP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNFwdTrain_BFP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNFwdTrain3D_FP32, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNFwdTrainRelu_FP32, MatchesCpuReference) {}

namespace {

std::vector<BN3DTestCase> ShimNetwork3D()
{
    // clang-format off
    return {
        {2, 4,   3, 16, 16, miopen::batchnorm::Direction::ForwardTraining, true, true},
        {4, 8,   4, 8,  8,  miopen::batchnorm::Direction::ForwardTraining, true, true},
        {2, 256, 1, 1,  1,  miopen::batchnorm::Direction::ForwardTraining, true, true},
    };
    // clang-format on
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNFwdTrain_FP32,
    testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIV3}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNFwdTrain_FP16,
    testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIV3}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNFwdTrain_BFP16,
    testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIV3}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNFwdTrain3D_FP32,
    testing::Combine(testing::ValuesIn(ShimNetwork3D()),
                     testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIV3}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN3DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNFwdTrainRelu_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationRELU})),
                         TestNameGenerator<BN2DTestCase>());

// ---------------------------------------------------------------------------
// Forward inference

struct GPU_HipdnnShimBNInfer_FP32
    : HashInferOutputs<BNInferTest<float, float, float, float, float, double, BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNInfer_FP16 : HashInferOutputs<BNInferTest<half_float::half,
                                                                 half_float::half,
                                                                 float,
                                                                 float,
                                                                 float,
                                                                 double,
                                                                 BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNInfer_BFP16
    : HashInferOutputs<BNInferTest<bfloat16, bfloat16, float, float, float, double, BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNInferRelu_FP32
    : HashInferOutputs<BNInferTest<float, float, float, float, float, double, BN2DTestCase>>
{
};

TEST_P(GPU_HipdnnShimBNInfer_FP32, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNInfer_FP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNInfer_BFP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNInferRelu_FP32, MatchesCpuReference) {}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNInfer_FP32,
    testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNInfer_FP16,
    testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNInfer_BFP16,
    testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                     testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNInferRelu_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIInvVar}),
                                          testing::ValuesIn({miopenActivationRELU})),
                         TestNameGenerator<BN2DTestCase>());

// ---------------------------------------------------------------------------
// Backward

struct GPU_HipdnnShimBNBwd_FP32
    : HashBwdOutputs<BNBwdTest<float, float, float, float, float, float, double, BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNBwd_FP16 : HashBwdOutputs<BNBwdTest<half_float::half,
                                                           half_float::half,
                                                           half_float::half,
                                                           float,
                                                           float,
                                                           float,
                                                           double,
                                                           BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNBwd_BFP16
    : HashBwdOutputs<
          BNBwdTest<bfloat16, bfloat16, bfloat16, float, float, float, double, BN2DTestCase>>
{
};

struct GPU_HipdnnShimBNBwdRelu_FP32
    : HashBwdOutputs<BNBwdTest<float, float, float, float, float, float, double, BN2DTestCase>>
{
};

TEST_P(GPU_HipdnnShimBNBwd_FP32, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNBwd_FP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNBwd_BFP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNBwdRelu_FP32, MatchesCpuReference) {}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNBwd_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNBwd_FP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNBwd_BFP16,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNBwdRelu_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationRELU})),
                         TestNameGenerator<BN2DTestCase>());

// ---------------------------------------------------------------------------
// Output hash

// The published FNV-1a 64 test vectors. They pin the constants and the byte order of
// OutputHash, and the split case shows that hashing in several Add calls is the same as
// hashing the joined bytes, which the multi-output parity hashes rely on.
TEST(CPU_HipdnnShimOutputHash_NONE, MatchesPublishedFnv1aVectors)
{
    const auto hash_of = [](std::initializer_list<std::string> parts) {
        hs::OutputHash hash;
        for(const auto& part : parts)
            hash.Add(std::vector<char>(part.begin(), part.end()));
        return hash.Hex();
    };
    EXPECT_EQ(hash_of({}), "cbf29ce484222325");
    EXPECT_EQ(hash_of({"a"}), "af63dc4c8601ec8c");
    EXPECT_EQ(hash_of({"foobar"}), "85944171f73967e8");
    EXPECT_EQ(hash_of({"foo", "bar"}), "85944171f73967e8");
}

// ---------------------------------------------------------------------------
// Plan key, in-place running stats and null outputs

// Epsilon and momentum are compile-time scalars in the hipDNN plan, so they must
// be part of the plan key. The second call has the same shape as the first and
// must not reuse its plan.
TEST(GPU_HipdnnShimBNPlanKey_FP32, EpsilonAndMomentumSelectTheirOwnPlan)
{
    auto&& handle = get_handle();

    TrainData defaults;
    SetUpTrain(defaults);
    ASSERT_EQ(TrainV1(handle, defaults), miopenStatusSuccess);
    CheckTrain(handle, defaults, true, true);

    TrainData changed;
    SetUpTrain(changed);
    changed.epsilon       = 0.5;
    changed.averageFactor = 0.9;
    ASSERT_EQ(TrainV1(handle, changed), miopenStatusSuccess);
    CheckTrain(handle, changed, true, true);
}

// V1 binds the same running-stat buffers as input and output. Each call must
// apply the running update exactly once.
TEST(GPU_HipdnnShimBNInPlace_FP32, RunningStatsUpdateOncePerCall)
{
    auto&& handle = get_handle();

    TrainData d;
    SetUpTrain(d);
    ASSERT_EQ(TrainV1(handle, d), miopenStatusSuccess);
    ASSERT_EQ(TrainV1(handle, d), miopenStatusSuccess);

    // The CPU reference updates runMean_ref and runVariance_ref in place, so
    // running it twice gives the expected stats after two calls.
    test::ComputeCPUBNFwdTrain(d);
    CheckTrain(handle, d, true, true);
}

// MIOpen serves every combination and skips the save or running update whose
// pointers are not all given. The hipDNN graph always produces the saved stats
// and needs the running stats all or none.
TEST(GPU_HipdnnShimBNNullOutputs_FP32, StatusMatchesMiopen)
{
    struct NullCase
    {
        const char* name;
        TrainPointers pointers;
        const char* decline_reason;
    };

    const std::vector<NullCase> cases = {
        {"save_only", {true, true, false, false}, nullptr},
        {"running_only", {false, false, true, true}, "always produces the saved mean"},
        {"neither", {false, false, false, false}, "always produces the saved mean"},
        {"save_var_missing", {true, false, true, true}, "always produces the saved mean"},
        {"run_var_missing", {true, true, true, false}, "all be given or all be null"},
    };

    auto&& handle = get_handle();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        TrainData d;
        SetUpTrain(d);

        const auto result = hs::CaptureCall([&] { return TrainV2(handle, d, c.pointers); });
        const bool served = RecordOutcome(c.name, result, c.decline_reason);
        if(c.decline_reason == nullptr)
            EXPECT_TRUE(served);
        if(served)
        {
            const bool save    = c.pointers.save_mean && c.pointers.save_var;
            const bool running = c.pointers.run_mean && c.pointers.run_var;
            CheckTrain(handle, d, save, running);
        }
    }
}

// ---------------------------------------------------------------------------
// Gaps: MIOpen serves these, the hipDNN path declines them

TEST(GPU_HipdnnShimBNGaps_FP32, PerActivationMode)
{
    auto&& handle      = get_handle();
    const char* reason = "supports only spatial mode";

    {
        SCOPED_TRACE("train");
        TrainData d;
        SetUpTrain(d, miopenBNPerActivation);
        const auto result = hs::CaptureCall([&] { return TrainV1(handle, d); });
        if(RecordOutcome("per_activation_train", result, reason))
            CheckTrain(handle, d, true, true);
    }
    {
        SCOPED_TRACE("infer");
        InferData d;
        SetUpInfer(d, miopenBNPerActivation);
        const auto result = hs::CaptureCall([&] { return InferV1(handle, d, true); });
        if(RecordOutcome("per_activation_infer", result, reason))
            CheckInfer(handle, d);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        SetUpBwd(d, miopenBNPerActivation);
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
        if(RecordOutcome("per_activation_bwd", result, reason))
            CheckBwd(handle, d);
    }
}

TEST(GPU_HipdnnShimBNGaps_FP32, ClampActivation)
{
    auto&& handle = get_handle();

    TrainData d;
    SetUpTrain(d);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationCLAMP);

    hs::OwnedActivationDescriptor activ;
    ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
    ASSERT_EQ(
        miopenSetActivationDescriptor(activ.handle, d.activ_mode, d.activ_alpha, d.activ_beta, 0.0),
        miopenStatusSuccess);

    const auto result = hs::CaptureCall([&] { return TrainActivation(handle, d, activ.handle); });
    if(RecordOutcome("clamp_train", result, "supports only the RELU activation"))
        CheckTrain(handle, d, true, true);
}

// MIOpen silently runs training when the estimated stats are null; there is no
// inference result to compare, so only the outcome is recorded.
TEST(GPU_HipdnnShimBNGaps_FP32, InferenceWithoutEstimatedStats)
{
    auto&& handle = get_handle();

    InferData d;
    SetUpInfer(d);
    const auto result = hs::CaptureCall([&] { return InferV1(handle, d, false); });
    RecordOutcome("infer_no_estimates", result, "needs the estimated mean and variance");
}

// Epsilon is 1e-5 so the shim gets past its epsilon check and declines for the
// missing saved stats, which its inference node needs.
TEST(GPU_HipdnnShimBNGaps_FP32, BackwardActivationWithoutSavedStats)
{
    auto&& handle = get_handle();

    BwdData d;
    SetUpBwd(d);
    d.epsilon = kProviderEpsilon;
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationRELU);

    hs::OwnedActivationDescriptor activ;
    ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
    ASSERT_EQ(
        miopenSetActivationDescriptor(activ.handle, d.activ_mode, d.activ_alpha, d.activ_beta, 0.0),
        miopenStatusSuccess);

    const auto result =
        hs::CaptureCall([&] { return BwdActivation(handle, d, false, activ.handle); });
    if(RecordOutcome("bwd_act_no_saved_stats", result, "needs the saved"))
    {
        FillTrueStats(d, d.epsilon);
        CheckBwd(handle, d);
    }
}

// The provider recomputes the stats with a fixed epsilon of 1e-5, so any other
// epsilon is declined and 1e-5 is served.
TEST(GPU_HipdnnShimBNGaps_FP32, BackwardWithoutSavedStats)
{
    auto&& handle = get_handle();

    {
        SCOPED_TRACE("flt_epsilon");
        BwdData d;
        SetUpBwd(d);
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, false); });
        if(RecordOutcome("bwd_no_saved_stats_flt_epsilon", result, "always uses"))
        {
            FillTrueStats(d, d.epsilon);
            CheckBwd(handle, d);
        }
    }
    {
        SCOPED_TRACE("provider_epsilon");
        BwdData d;
        SetUpBwd(d);
        d.epsilon         = kProviderEpsilon;
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, false); });
        EXPECT_TRUE(RecordOutcome("bwd_no_saved_stats_provider_epsilon", result, nullptr));
        if(result.status == miopenStatusSuccess)
        {
            FillTrueStats(d, d.epsilon);
            CheckBwd(handle, d);
        }
    }
}

// Restores the shared handle's tuning policy even when an ASSERT returns early, so
// the tests after this one keep running under None.
struct ScopedHandleTuningPolicy
{
    ScopedHandleTuningPolicy(miopen::Handle& h, miopenTuningPolicy_t policy) : handle(h)
    {
        EXPECT_EQ(miopenGetTuningPolicy(&handle, &saved), miopenStatusSuccess);
        EXPECT_EQ(miopenSetTuningPolicy(&handle, policy), miopenStatusSuccess);
    }
    ~ScopedHandleTuningPolicy() { miopenSetTuningPolicy(&handle, saved); }

    miopen::Handle& handle;
    miopenTuningPolicy_t saved = miopenTuningPolicyNone;
};

// MIOpen tunes the spatial training and backward solvers when the handle asks for
// Search; the provider runs on a handle of its own and never sees that policy, so
// the shim declines rather than run a perf config MIOpen would not have picked.
// Inference is not tunable and stays served. The shape is used by no other test,
// because a Search caches its tuned invoker on the handle under the shape's network
// config, and in the disabled run alone that would change what later tests run.
// Training passes no running stats, since a Search re-runs the kernel and would
// apply the running-stat update more than once.
TEST(GPU_HipdnnShimBNGaps_FP32, HandleTuningPolicy)
{
    auto&& handle = get_handle();
    const BN2DTestCase tuning_case{
        3, 5, 7, 9, miopen::batchnorm::Direction::ForwardTraining, true, true};
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicySearch);
    const char* reason = "tuning policy";

    {
        SCOPED_TRACE("train");
        TrainData d;
        d.SetUpImpl(tuning_case, miopenBNSpatial, miopenTensorNCHW);
        SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
        const auto result =
            hs::CaptureCall([&] { return TrainV2(handle, d, {true, true, false, false}); });
        if(RecordOutcome("tuning_policy_train", result, reason))
            CheckTrain(handle, d, true, false);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        d.SetUpImpl(tuning_case, miopenBNSpatial, miopenTensorNCHW);
        SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
        if(RecordOutcome("tuning_policy_bwd", result, reason))
            CheckBwd(handle, d);
    }
}

// ---------------------------------------------------------------------------
// alpha/beta other than 1/0

// MIOpen rejects these with UnknownError on training and BadParm elsewhere; the
// hipDNN path must return the same status.
TEST(GPU_HipdnnShimBNAlphaBeta_FP32, StatusMatchesMiopen)
{
    auto&& handle      = get_handle();
    const char* reason = "only alpha=1 and beta=0";

    {
        SCOPED_TRACE("train");
        TrainData d;
        SetUpTrain(d);
        d.alpha           = 2.0f;
        const auto result = hs::CaptureCall([&] { return TrainV1(handle, d); });
        EXPECT_EQ(result.status, miopenStatusUnknownError) << result.message;
        hs::RecordStatus("alpha_beta_train", result.status);
        if(result.status != miopenStatusSuccess)
            hs::ExpectForwardedDecline(result, reason);
    }
    {
        SCOPED_TRACE("infer");
        InferData d;
        SetUpInfer(d);
        d.alpha           = 2.0f;
        const auto result = hs::CaptureCall([&] { return InferV1(handle, d, true); });
        EXPECT_EQ(result.status, miopenStatusBadParm) << result.message;
        hs::RecordStatus("alpha_beta_infer", result.status);
        if(result.status != miopenStatusSuccess)
            hs::ExpectForwardedDecline(result, reason);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        SetUpBwd(d);
        d.alphaDataDiff   = 2.0f;
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
        EXPECT_EQ(result.status, miopenStatusBadParm) << result.message;
        hs::RecordStatus("alpha_beta_bwd", result.status);
        if(result.status != miopenStatusSuccess)
            hs::ExpectForwardedDecline(result, reason);
    }
}
