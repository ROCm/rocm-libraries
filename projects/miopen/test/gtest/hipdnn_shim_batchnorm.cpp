// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shim-surface tests for the forwarded batchnorm entry points. The derived
// suites reuse the bn.hpp fixtures, which reach compute only through the public
// miopenBatchNorm* API and check against the fusionHost.hpp CPU reference. The
// plain tests cover what those fixtures cannot reach: epsilon and momentum in
// the plan key, running stats updated in place, null save and running pointers,
// inference without estimated stats, per-activation mode, clipped activations,
// backward without saved stats, and the calls the hipDNN path declines while
// MIOpen serves them. Each of those records whether the call was served so the
// parity harness can pin the expected divergences. The rest record the status,
// for problems both modes must fail the same way: alpha/beta other than 1/0, and
// problems no MIOpen solver takes.
//
// Not covered, because MIOpen itself has no defined behavior to compare with:
// null alpha/beta pointers (MIOpen dereferences them), null dscale/dbias on
// backward, and a null bias with RELU on backward activation (the MIOpen kernel
// reads the null device pointer).

#include "bn.hpp"
#include "hipdnn_shim_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
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

void SetUpTrain(TrainData& d,
                miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
                miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    d.SetUpImpl(SmallCase(), bn_mode, tensor_layout);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

void SetUpInfer(InferData& d,
                miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
                miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    d.SetUpImpl(SmallCase(), bn_mode, tensor_layout, false);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

void SetUpBwd(BwdData& d,
              miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
              miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    d.SetUpImpl(SmallCase(), bn_mode, tensor_layout);
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

// Reads the previous running stats from runMean/runVariance and writes the updated
// ones to nextRunMean/nextRunVariance.
miopenStatus_t
TrainActivationV2(miopen::Handle& handle, TrainData& d, miopenActivationDescriptor_t activ)
{
    return miopenBatchNormForwardTrainingActivation_V2(&handle,
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
                                                       d.nextRunMean_dev.get(),
                                                       d.nextRunVariance_dev.get(),
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
    activationHostInfer(
        d.activ_mode, 0.0, d.activ_beta, d.activ_alpha, d.out_ref.data, d.out_ref.data);
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

struct BatchStats
{
    std::vector<double> mean;
    std::vector<double> variance; // biased, as the kernels use
};

// The true per-channel stats of an NCHW input.
BatchStats ComputeBatchStats(const tensor<float>& input)
{
    const auto& lens    = input.desc.GetLengths();
    const std::size_t n = lens[0];
    const std::size_t c = lens[1];
    std::size_t spatial = 1;
    for(std::size_t i = 2; i < lens.size(); ++i)
        spatial *= lens[i];
    const double count = static_cast<double>(n * spatial);

    BatchStats stats{std::vector<double>(c), std::vector<double>(c)};
    for(std::size_t ci = 0; ci < c; ++ci)
    {
        double sum = 0.0;
        for(std::size_t ni = 0; ni < n; ++ni)
            for(std::size_t i = 0; i < spatial; ++i)
                sum += input.data[(ni * c + ci) * spatial + i];
        const double mean = sum / count;

        double sq = 0.0;
        for(std::size_t ni = 0; ni < n; ++ni)
        {
            for(std::size_t i = 0; i < spatial; ++i)
            {
                const double diff = input.data[(ni * c + ci) * spatial + i] - mean;
                sq += diff * diff;
            }
        }

        stats.mean[ci]     = mean;
        stats.variance[ci] = sq / count;
    }
    return stats;
}

// Backward without saved stats recomputes them from x. The CPU reference reads
// only savedMean and savedInvVar, so give it the true stats of the input.
void FillTrueStats(BwdData& d, double eps)
{
    const BatchStats stats = ComputeBatchStats(d.input);
    for(std::size_t ci = 0; ci < stats.mean.size(); ++ci)
    {
        d.savedMean.data[ci]   = static_cast<float>(stats.mean[ci]);
        d.savedInvVar.data[ci] = static_cast<float>(1.0 / std::sqrt(stats.variance[ci] + eps));
    }
}

// True when every element still holds the NaN the test data poisoned it with.
bool Untouched(miopen::Handle& handle, const miopen::Allocator::ManageDataPtr& dev, std::size_t n)
{
    const auto values = handle.Read<float>(dev, n);
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isnan(v); });
}

// A served call or an expected decline. A decline must come from the hipDNN path
// with the given reason; MIOpen serves every case these tests send it.
bool RecordOutcome(const std::string& case_name, const hs::CallResult& result, const char* reason)
{
    const bool served = result.status == miopenStatusSuccess;
    EXPECT_TRUE(served || result.status == miopenStatusUnsupportedOp)
        << case_name << ": " << result.message;
    if(!served)
        hs::ExpectForwardedDecline(result, reason);
    hs::RecordServed(case_name, served);
    return served;
}

// A call both paths must serve.
bool ExpectServed(const std::string& case_name, const hs::CallResult& result)
{
    const bool served = RecordOutcome(case_name, result, nullptr);
    EXPECT_TRUE(served) << case_name << ": " << result.message;
    return served;
}

// Each derived suite records a hash of everything its fixture reads back. The CPU
// reference check runs in the base TearDown and passes within kTol; the hash goes
// further, because the parity comparison needs it to match bit for bit, which only
// holds if the forwarded call ran the same kernel with the same configuration that
// MIOpen picks. Whether to hash is decided before the base TearDown runs, because
// that is where a skipped or failed test returns before reading anything back.
// SetUp resets the process-wide PRNG first, so a test's data, and its hash, do not
// depend on which tests ran before it (a shuffle or a narrower filter).
template <typename Base>
struct HashTrainOutputs : Base
{
    void SetUp() override
    {
        prng::reset_seed();
        Base::SetUp();
    }

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
    void SetUp() override
    {
        prng::reset_seed();
        Base::SetUp();
    }

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
    void SetUp() override
    {
        prng::reset_seed();
        Base::SetUp();
    }

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

struct GPU_HipdnnShimBNInfer3D_FP32
    : HashInferOutputs<BNInferTest<float, float, float, float, float, double, BN3DTestCase>>
{
};

struct GPU_HipdnnShimBNInferRelu_FP32
    : HashInferOutputs<BNInferTest<float, float, float, float, float, double, BN2DTestCase>>
{
};

TEST_P(GPU_HipdnnShimBNInfer_FP32, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNInfer_FP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNInfer_BFP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNInfer3D_FP32, MatchesCpuReference) {}
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

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_HipdnnShimBNInfer3D_FP32,
    testing::Combine(testing::ValuesIn(ShimNetwork3D()),
                     testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                     testing::ValuesIn({miopenBNSpatial}),
                     testing::ValuesIn({testBNAPIV1, testBNAPIV2, testBNAPIInvVar}),
                     testing::ValuesIn({miopenActivationPASTHRU})),
    TestNameGenerator<BN3DTestCase>());

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

struct GPU_HipdnnShimBNBwd3D_FP32
    : HashBwdOutputs<BNBwdTest<float, float, float, float, float, float, double, BN3DTestCase>>
{
};

struct GPU_HipdnnShimBNBwdRelu_FP32
    : HashBwdOutputs<BNBwdTest<float, float, float, float, float, float, double, BN2DTestCase>>
{
};

TEST_P(GPU_HipdnnShimBNBwd_FP32, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNBwd_FP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNBwd_BFP16, MatchesCpuReference) {}
TEST_P(GPU_HipdnnShimBNBwd3D_FP32, MatchesCpuReference) {}
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
                         GPU_HipdnnShimBNBwd3D_FP32,
                         testing::Combine(testing::ValuesIn(ShimNetwork3D()),
                                          testing::ValuesIn({miopenTensorNCDHW, miopenTensorNDHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1, testBNAPIV2}),
                                          testing::ValuesIn({miopenActivationPASTHRU})),
                         TestNameGenerator<BN3DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNBwdRelu_FP32,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({miopenBNSpatial}),
                                          testing::ValuesIn({testBNAPIV1}),
                                          testing::ValuesIn({miopenActivationRELU})),
                         TestNameGenerator<BN2DTestCase>());

// ---------------------------------------------------------------------------
// Plan key, in-place running stats and null outputs

namespace {

// Epsilon and momentum are compile-time scalars in the hipDNN plan, so each must
// be part of the plan key on its own. The second call has the same shape as the
// first and differs only in what `change` sets; a key that missed it would reuse
// the first plan and give wrong numbers without failing the call.
template <typename Change>
void ExpectOwnPlan(Change change)
{
    auto&& handle = get_handle();

    TrainData defaults;
    SetUpTrain(defaults);
    ASSERT_EQ(TrainV1(handle, defaults), miopenStatusSuccess);
    CheckTrain(handle, defaults, true, true);

    TrainData changed;
    SetUpTrain(changed);
    change(changed);
    ASSERT_EQ(TrainV1(handle, changed), miopenStatusSuccess);
    CheckTrain(handle, changed, true, true);
}

} // namespace

TEST(GPU_HipdnnShimBNPlanKey_FP32, EpsilonSelectsItsOwnPlan)
{
    ExpectOwnPlan([](TrainData& d) { d.epsilon = 0.5; });
}

TEST(GPU_HipdnnShimBNPlanKey_FP32, MomentumSelectsItsOwnPlan)
{
    ExpectOwnPlan([](TrainData& d) { d.averageFactor = 0.9; });
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

// The bn.hpp fixtures call the V1 activation entry point whenever an activation is
// set, whatever the API version, so this is the only test that reaches the V2 one.
// Its previous and next running stats are separate buffers: the previous ones must
// stay untouched and the next ones must hold the update.
TEST(GPU_HipdnnShimBNFwdTrainActivationV2_FP32, SeparateRunningStatsMatchCpuReference)
{
    auto&& handle = get_handle();

    TrainData d;
    SetUpTrain(d);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationRELU);

    hs::OwnedActivationDescriptor activ;
    ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
    ASSERT_EQ(
        miopenSetActivationDescriptor(activ.handle, d.activ_mode, d.activ_alpha, d.activ_beta, 0.0),
        miopenStatusSuccess);

    const auto prev_mean     = d.runMean.data;
    const auto prev_variance = d.runVariance.data;
    ASSERT_EQ(TrainActivationV2(handle, d, activ.handle), miopenStatusSuccess);

    EXPECT_EQ(handle.Read<float>(d.runMean_dev, prev_mean.size()), prev_mean);
    EXPECT_EQ(handle.Read<float>(d.runVariance_dev, prev_variance.size()), prev_variance);

    // CheckTrain reads the running stats from runMean_dev and runVariance_dev.
    std::swap(d.runMean_dev, d.nextRunMean_dev);
    std::swap(d.runVariance_dev, d.nextRunVariance_dev);
    CheckTrain(handle, d, true, true);
}

// MIOpen serves every combination and skips the save or running update whose
// pointers are not all given. The hipDNN graph always produces the saved stats,
// so the shim writes them to scratch when the caller does not take them. A buffer
// of a skipped group must be left as it was: the saved mean keeps its poison and
// the running mean its initial value.
TEST(GPU_HipdnnShimBNNullOutputs_FP32, StatusMatchesMiopen)
{
    struct NullCase
    {
        const char* name;
        TrainPointers pointers;
    };

    const std::vector<NullCase> cases = {
        {"save_only", {true, true, false, false}},
        {"running_only", {false, false, true, true}},
        {"neither", {false, false, false, false}},
        {"save_var_missing", {true, false, true, true}},
        {"run_var_missing", {true, true, true, false}},
    };

    auto&& handle = get_handle();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        TrainData d;
        SetUpTrain(d);
        const auto initial_run_mean = d.runMean.data;

        const auto result = hs::CaptureCall([&] { return TrainV2(handle, d, c.pointers); });
        EXPECT_TRUE(RecordOutcome(c.name, result, nullptr));
        if(result.status != miopenStatusSuccess)
            continue;

        const bool save    = c.pointers.save_mean && c.pointers.save_var;
        const bool running = c.pointers.run_mean && c.pointers.run_var;
        if(c.pointers.save_mean && !save)
            EXPECT_TRUE(Untouched(handle, d.saveMean_dev, d.saveMean.data.size()));
        if(c.pointers.run_mean && !running)
            EXPECT_EQ(handle.Read<float>(d.runMean_dev, initial_run_mean.size()), initial_run_mean);
        CheckTrain(handle, d, save, running);
    }
}

// ---------------------------------------------------------------------------
// Problems the shim reshapes or maps before forwarding. They are checked against
// the CPU reference and not hashed: where the graph differs from MIOpen's kernel,
// the results match MIOpen's only to rounding.

// Per-activation mode is folded into spatial mode over C*H*W channels of one
// element each, which gives every element its own stats over the batch. MIOpen's
// per-activation kernels read every tensor in memory order, whatever its layout,
// and so does the fold. With the parameters laid out like x, as here, that is the
// per-element answer for NHWC too.
TEST(GPU_HipdnnShimBNPerActivation_FP32, MatchesCpuReference)
{
    auto&& handle = get_handle();

    for(const auto tensor_layout : {miopenTensorNCHW, miopenTensorNHWC})
    {
        const std::string suffix = tensor_layout == miopenTensorNHWC ? "_nhwc" : "";
        SCOPED_TRACE(tensor_layout == miopenTensorNHWC ? "NHWC" : "NCHW");
        {
            SCOPED_TRACE("train");
            TrainData d;
            SetUpTrain(d, miopenBNPerActivation, tensor_layout);
            const auto result = hs::CaptureCall([&] { return TrainV1(handle, d); });
            if(ExpectServed("per_activation_train" + suffix, result))
                CheckTrain(handle, d, true, true);
        }
        {
            SCOPED_TRACE("infer");
            InferData d;
            SetUpInfer(d, miopenBNPerActivation, tensor_layout);
            const auto result = hs::CaptureCall([&] { return InferV1(handle, d, true); });
            if(ExpectServed("per_activation_infer" + suffix, result))
                CheckInfer(handle, d);
        }
        {
            SCOPED_TRACE("bwd");
            BwdData d;
            SetUpBwd(d, miopenBNPerActivation, tensor_layout);
            const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
            if(ExpectServed("per_activation_bwd" + suffix, result))
                CheckBwd(handle, d);
        }
    }
}

// CLIPPEDRELU and CLAMP forward as ReLU with an upper clip, and with both clips.
// The clips are set wide enough that every branch of the activation and of its
// gradient is taken. Backward is covered as well as forward, because it is the
// direction that takes the activation's gradient and so has the most to get wrong.
TEST(GPU_HipdnnShimBNClippedActivation_FP32, MatchesCpuReference)
{
    struct ClipCase
    {
        const char* name;
        miopenActivationMode_t mode;
        double alpha;
        double beta;
    };

    const std::vector<ClipCase> cases = {
        {"clipped_relu", miopenActivationCLIPPEDRELU, 1.0, 0.0},
        {"clamp", miopenActivationCLAMP, -0.5, 1.0},
    };

    auto&& handle = get_handle();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        const std::string name = c.name;
        {
            SCOPED_TRACE("train");
            TrainData d;
            SetUpTrain(d);
            d.activ_mode  = c.mode;
            d.activ_alpha = c.alpha;
            d.activ_beta  = c.beta;

            hs::OwnedActivationDescriptor activ;
            ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
            ASSERT_EQ(miopenSetActivationDescriptor(
                          activ.handle, d.activ_mode, d.activ_alpha, d.activ_beta, 0.0),
                      miopenStatusSuccess);

            const auto result =
                hs::CaptureCall([&] { return TrainActivation(handle, d, activ.handle); });
            if(ExpectServed(name + "_train", result))
                CheckTrain(handle, d, true, true);
        }
        {
            SCOPED_TRACE("bwd");
            BwdData d;
            SetUpBwd(d);
            d.activ_mode  = c.mode;
            d.activ_alpha = c.alpha;
            d.activ_beta  = c.beta;

            hs::OwnedActivationDescriptor activ;
            ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
            ASSERT_EQ(miopenSetActivationDescriptor(
                          activ.handle, d.activ_mode, d.activ_alpha, d.activ_beta, 0.0),
                      miopenStatusSuccess);

            const auto result =
                hs::CaptureCall([&] { return BwdActivation(handle, d, true, activ.handle); });
            if(ExpectServed(name + "_bwd", result))
                CheckBwd(handle, d);
        }
    }
}

// MIOpen silently runs training when the estimated stats are null, so the output
// is x normalized with its own batch stats: the inference reference given those
// stats as its estimates.
TEST(GPU_HipdnnShimBNInferWithoutEstimates_FP32, RunsTraining)
{
    auto&& handle = get_handle();

    InferData d;
    SetUpInfer(d);
    const auto result = hs::CaptureCall([&] { return InferV1(handle, d, false); });
    EXPECT_TRUE(RecordOutcome("infer_no_estimates", result, nullptr));
    if(result.status != miopenStatusSuccess)
        return;

    const BatchStats stats = ComputeBatchStats(d.input);
    for(std::size_t ci = 0; ci < stats.mean.size(); ++ci)
    {
        d.estMean.data[ci]     = static_cast<float>(stats.mean[ci]);
        d.estVariance.data[ci] = static_cast<float>(stats.variance[ci]);
    }
    CheckInfer(handle, d);
}

// The provider recomputes the stats itself only for plain backward at an epsilon
// of 1e-5. For any other epsilon, and for backward with an activation, whose
// graph needs the saved stats, the shim runs training first to compute them into
// scratch. Every combination is served; the CPU reference is given the true stats
// of the input at the call's epsilon.
TEST(GPU_HipdnnShimBNBwdWithoutSavedStats_FP32, MatchesCpuReference)
{
    struct NoStatsCase
    {
        const char* name;
        bool provider_epsilon;
        miopenActivationMode_t mode;
    };

    // BwdData's own epsilon is FLT_EPSILON.
    const std::vector<NoStatsCase> cases = {
        {"bwd_no_saved_stats_provider_epsilon", true, miopenActivationPASTHRU},
        {"bwd_no_saved_stats_flt_epsilon", false, miopenActivationPASTHRU},
        {"bwd_act_no_saved_stats", true, miopenActivationRELU},
        {"bwd_act_no_saved_stats_flt_epsilon", false, miopenActivationRELU},
    };

    auto&& handle = get_handle();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        BwdData d;
        SetUpBwd(d);
        if(c.provider_epsilon)
            d.epsilon = kProviderEpsilon;
        SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, c.mode);

        hs::OwnedActivationDescriptor activ;
        if(c.mode != miopenActivationPASTHRU)
        {
            ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
            ASSERT_EQ(miopenSetActivationDescriptor(
                          activ.handle, d.activ_mode, d.activ_alpha, d.activ_beta, 0.0),
                      miopenStatusSuccess);
        }

        const auto result = hs::CaptureCall([&] {
            return c.mode == miopenActivationPASTHRU
                       ? BwdV1(handle, d, false)
                       : BwdActivation(handle, d, false, activ.handle);
        });
        if(ExpectServed(c.name, result))
        {
            FillTrueStats(d, d.epsilon);
            CheckBwd(handle, d);
        }
    }
}

// ---------------------------------------------------------------------------
// Tuning policies

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

// Used by no test that runs under None, because a tuning policy caches its tuned
// invoker on the handle under the shape's network config, and in the disabled run
// alone that would change what later tests run.
const BN2DTestCase& TuningCase()
{
    static const BN2DTestCase tuning_case{
        3, 5, 7, 9, miopen::batchnorm::Direction::ForwardTraining, true, true};
    return tuning_case;
}

// Training passes no running stats, since a search re-runs the kernel and would
// apply the running-stat update more than once.
hs::CallResult
TrainUnderPolicy(miopen::Handle& handle, TrainData& d, miopenBatchNormMode_t mode = miopenBNSpatial)
{
    d.SetUpImpl(TuningCase(), mode, miopenTensorNCHW);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
    return hs::CaptureCall([&] { return TrainV2(handle, d, {true, true, false, false}); });
}

hs::CallResult
BwdUnderPolicy(miopen::Handle& handle, BwdData& d, miopenBatchNormMode_t mode = miopenBNSpatial)
{
    d.SetUpImpl(TuningCase(), mode, miopenTensorNCHW);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
    return hs::CaptureCall([&] { return BwdV1(handle, d, true); });
}

// MIOpen tunes the spatial training and backward solvers when the handle asks for
// Search. The provider runs on a handle of its own, so the shim carries Search
// over as the plan's benchmarking knob, under which the provider sets Search on
// its handle. Inference is not tunable and is served without the knob.
TEST(GPU_HipdnnShimBNTuning_FP32, SearchPolicyIsServed)
{
    auto&& handle = get_handle();
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicySearch);

    {
        SCOPED_TRACE("train");
        TrainData d;
        if(ExpectServed("tuning_policy_train", TrainUnderPolicy(handle, d)))
            CheckTrain(handle, d, true, false);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        if(ExpectServed("tuning_policy_bwd", BwdUnderPolicy(handle, d)))
            CheckBwd(handle, d);
    }
    {
        SCOPED_TRACE("infer");
        InferData d;
        d.SetUpImpl(TuningCase(), miopenBNSpatial, miopenTensorNCHW, false);
        const auto result = hs::CaptureCall([&] { return InferV1(handle, d, true); });
        if(ExpectServed("tuning_policy_infer", result))
            CheckInfer(handle, d);
    }
}

// MIOpen's per-activation solvers are not tunable, so MIOpen serves them under
// any policy, and so does the hipDNN path, even under the ones it declines for
// spatial calls.
TEST(GPU_HipdnnShimBNTuning_FP32, PerActivationIgnoresPolicy)
{
    auto&& handle = get_handle();
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicyDbClean);

    {
        SCOPED_TRACE("train");
        TrainData d;
        if(ExpectServed("tuning_policy_per_activation_train",
                        TrainUnderPolicy(handle, d, miopenBNPerActivation)))
            CheckTrain(handle, d, true, false);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        if(ExpectServed("tuning_policy_per_activation_bwd",
                        BwdUnderPolicy(handle, d, miopenBNPerActivation)))
            CheckBwd(handle, d);
    }
}

// ---------------------------------------------------------------------------
// Gaps: MIOpen serves these, the hipDNN path declines them

// The provider sets only None or Search on its handle, so it cannot run the
// policies that also write or clean the perf db. The shim declines them for
// training and backward; DbUpdate stands for all three.
TEST(GPU_HipdnnShimBNGaps_FP32, DbUpdatePolicyIsDeclined)
{
    auto&& handle = get_handle();
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicyDbUpdate);
    const char* reason = "tuning policy";

    {
        SCOPED_TRACE("train");
        TrainData d;
        if(RecordOutcome("tuning_policy_dbupdate_train", TrainUnderPolicy(handle, d), reason))
            CheckTrain(handle, d, true, false);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        if(RecordOutcome("tuning_policy_dbupdate_bwd", BwdUnderPolicy(handle, d), reason))
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

// ---------------------------------------------------------------------------
// Problems no MIOpen solver takes

namespace {

// A tensor with an uninitialized device buffer, for calls both modes reject
// before any kernel runs.
struct RawTensor
{
    RawTensor(miopen::Handle& handle, miopenDataType_t type, std::vector<std::size_t> lens)
        : desc(type, std::move(lens)), dev(handle.Create(desc.GetNumBytes()))
    {
    }

    miopen::TensorDescriptor desc;
    miopen::Allocator::ManageDataPtr dev;
};

std::vector<std::size_t> StatLens(const std::vector<std::size_t>& x_lens)
{
    std::vector<std::size_t> lens(x_lens.size(), 1);
    lens[1] = x_lens[1];
    return lens;
}

// y takes x's type, dx takes dy's, and scale, bias and the statistics take
// stat_type.
struct RawProblem
{
    RawProblem(miopen::Handle& handle,
               const std::vector<std::size_t>& x_lens,
               miopenDataType_t x_type,
               miopenDataType_t dy_type,
               miopenDataType_t stat_type)
        : x(handle, x_type, x_lens),
          y(handle, x_type, x_lens),
          dy(handle, dy_type, x_lens),
          dx(handle, dy_type, x_lens),
          scale(handle, stat_type, StatLens(x_lens)),
          bias(handle, stat_type, StatLens(x_lens)),
          mean(handle, stat_type, StatLens(x_lens)),
          variance(handle, stat_type, StatLens(x_lens)),
          dscale(handle, stat_type, StatLens(x_lens)),
          dbias(handle, stat_type, StatLens(x_lens))
    {
        EXPECT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
        SetActivation(miopenActivationPASTHRU);
    }

    void SetActivation(miopenActivationMode_t mode)
    {
        EXPECT_EQ(miopenSetActivationDescriptor(activ.handle, mode, 0.1, 0.3, 0.0),
                  miopenStatusSuccess);
    }

    miopenStatus_t Train(miopen::Handle& handle)
    {
        return miopenBatchNormForwardTrainingActivation(&handle,
                                                        miopenBNSpatial,
                                                        &one,
                                                        &zero,
                                                        &x.desc,
                                                        x.dev.get(),
                                                        &y.desc,
                                                        y.dev.get(),
                                                        &scale.desc,
                                                        &bias.desc,
                                                        &mean.desc,
                                                        &variance.desc,
                                                        scale.dev.get(),
                                                        bias.dev.get(),
                                                        0.1,
                                                        mean.dev.get(),
                                                        variance.dev.get(),
                                                        1e-5,
                                                        nullptr,
                                                        nullptr,
                                                        activ.handle);
    }

    miopenStatus_t Infer(miopen::Handle& handle)
    {
        return miopenBatchNormForwardInferenceActivation(&handle,
                                                         miopenBNSpatial,
                                                         &one,
                                                         &zero,
                                                         &x.desc,
                                                         x.dev.get(),
                                                         &y.desc,
                                                         y.dev.get(),
                                                         &scale.desc,
                                                         &bias.desc,
                                                         &mean.desc,
                                                         &variance.desc,
                                                         scale.dev.get(),
                                                         bias.dev.get(),
                                                         mean.dev.get(),
                                                         variance.dev.get(),
                                                         1e-5,
                                                         activ.handle);
    }

    miopenStatus_t Bwd(miopen::Handle& handle)
    {
        return miopenBatchNormalizationBackward(&handle,
                                                miopenBNSpatial,
                                                &one,
                                                &zero,
                                                &one,
                                                &zero,
                                                &x.desc,
                                                x.dev.get(),
                                                &dy.desc,
                                                dy.dev.get(),
                                                &dx.desc,
                                                dx.dev.get(),
                                                &scale.desc,
                                                scale.dev.get(),
                                                dscale.dev.get(),
                                                dbias.dev.get(),
                                                1e-5,
                                                mean.dev.get(),
                                                variance.dev.get());
    }

    RawTensor x, y, dy, dx, scale, bias, mean, variance, dscale, dbias;
    hs::OwnedActivationDescriptor activ;
    float one  = 1.0f;
    float zero = 0.0f;
};

} // namespace

// MIOpen fails these after its parameter checks, with UnknownError for a rank
// its problem description does not know and NotImplemented when no solver
// applies. The hipDNN path must return the same status rather than decline them
// as unsupported. Each problem passes the parameter checks, so y matches x on
// forward and dx matches dy on backward.
TEST(GPU_HipdnnShimBNNoSolver_FP32, StatusMatchesMiopen)
{
    auto&& handle = get_handle();
    const std::vector<std::size_t> nchw{2, 4, 8, 8};
    const char* no_solver = "no batchnorm solver";

    {
        SCOPED_TRACE("rank 3 train");
        RawProblem p(handle, {2, 4, 64}, miopenFloat, miopenFloat, miopenFloat);
        const auto result = hs::CaptureCall([&] { return p.Train(handle); });
        hs::ExpectStatusParity(
            "no_solver_rank3_train", result, miopenStatusUnknownError, "Unknown spatial dim");
    }
    {
        SCOPED_TRACE("tanh train");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.SetActivation(miopenActivationTANH);
        const auto result = hs::CaptureCall([&] { return p.Train(handle); });
        hs::ExpectStatusParity(
            "no_solver_tanh_train", result, miopenStatusNotImplemented, no_solver);
    }
    {
        SCOPED_TRACE("tanh infer");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.SetActivation(miopenActivationTANH);
        const auto result = hs::CaptureCall([&] { return p.Infer(handle); });
        hs::ExpectStatusParity(
            "no_solver_tanh_infer", result, miopenStatusNotImplemented, no_solver);
    }
    {
        SCOPED_TRACE("double train");
        RawProblem p(handle, nchw, miopenDouble, miopenDouble, miopenFloat);
        const auto result = hs::CaptureCall([&] { return p.Train(handle); });
        hs::ExpectStatusParity(
            "no_solver_double_train", result, miopenStatusNotImplemented, no_solver);
    }
    {
        SCOPED_TRACE("half scale train");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenHalf);
        const auto result = hs::CaptureCall([&] { return p.Train(handle); });
        hs::ExpectStatusParity(
            "no_solver_half_scale_train", result, miopenStatusNotImplemented, no_solver);
    }
    {
        SCOPED_TRACE("mixed bwd");
        RawProblem p(handle, nchw, miopenFloat, miopenHalf, miopenFloat);
        const auto result = hs::CaptureCall([&] { return p.Bwd(handle); });
        hs::ExpectStatusParity(
            "no_solver_mixed_bwd", result, miopenStatusNotImplemented, no_solver);
    }
}
