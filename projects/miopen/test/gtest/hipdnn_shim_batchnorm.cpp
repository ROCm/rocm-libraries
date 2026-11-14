// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shim-surface tests for the forwarded batchnorm entry points. The derived
// suites reuse the bn.hpp fixtures, which reach compute only through the public
// miopenBatchNorm* API and check against the fusionHost.hpp CPU reference. The
// plain tests cover what those fixtures cannot reach: epsilon, momentum, saved
// stats and graph kind in the plan key, running stats updated in place, null
// save and running pointers, inference without estimated stats, per-activation
// mode, clipped activations, backward without saved stats, and the calls the
// hipDNN path declines while MIOpen serves them. Most of those record whether
// the call was served, so the parity harness can pin the expected divergences.
// The plan-key, in-place and V2 activation tests require the call to be served.
// The rest record the status, for problems both modes must fail the same way:
// alpha/beta other than 1/0, tensors MIOpen rejects as BadParm, and problems no
// MIOpen solver takes. The declined-problem test runs only with forwarding on,
// and checks the reason the hipDNN path gives for problems it does not forward.
//
// Not covered, because MIOpen itself has no defined behavior to compare with:
// null alpha/beta pointers (MIOpen dereferences them), null dscale/dbias on
// backward, and a null bias with RELU on backward activation (the MIOpen kernel
// reads the null device pointer).

#include "bn.hpp"
#include "hipdnn_shim_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hs = hipdnn_shim_test;

MIOPEN_LIB_ENV_VAR(MIOPEN_WARN_SEARCH)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_DISABLE_USER_DB)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_DISABLE_SYSTEM_DB)

namespace {

using TrainData = BNFwdTrainTestData<float, float, float, float, float, double, BN2DTestCase>;
using InferData = BNInferTestData<float, float, float, float, float, double, BN2DTestCase>;
using BwdData   = BNBwdTestData<float, float, float, float, float, float, double, BN2DTestCase>;

using Train3DData = BNFwdTrainTestData<float, float, float, float, float, double, BN3DTestCase>;
using Infer3DData = BNInferTestData<float, float, float, float, float, double, BN3DTestCase>;
using Bwd3DData   = BNBwdTestData<float, float, float, float, float, float, double, BN3DTestCase>;

constexpr double kTol = 4e-3;

// The epsilon the hipDNN provider hardcodes when backward recomputes the stats.
constexpr double kProviderEpsilon = 1e-5;

template <typename Case>
Case SmallCase();

template <>
BN2DTestCase SmallCase<BN2DTestCase>()
{
    return {3, 5, 6, 10, miopen::batchnorm::Direction::ForwardTraining, true, true};
}

// D, H and W all differ, so a fold that mixed them up would not line up.
template <>
BN3DTestCase SmallCase<BN3DTestCase>()
{
    return {3, 5, 2, 3, 4, miopen::batchnorm::Direction::ForwardTraining, true, true};
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

// The setup, call and check helpers below are templates so they serve the 2D and
// 3D data alike; each picks the small case of its data's rank.
template <typename Data>
void SetUpTrain(Data& d,
                miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
                miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    d.SetUpImpl(SmallCase<decltype(d.bn_config)>(), bn_mode, tensor_layout);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

template <typename Data>
void SetUpInfer(Data& d,
                miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
                miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    d.SetUpImpl(SmallCase<decltype(d.bn_config)>(), bn_mode, tensor_layout, false);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

template <typename Data>
void SetUpBwd(Data& d,
              miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
              miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    d.SetUpImpl(SmallCase<decltype(d.bn_config)>(), bn_mode, tensor_layout);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
}

template <typename Data>
miopenStatus_t TrainV1(miopen::Handle& handle, Data& d)
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

template <typename Data>
miopenStatus_t InferV1(miopen::Handle& handle, Data& d, bool with_mean, bool with_variance)
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
                                                    with_mean ? d.estMean_dev.get() : nullptr,
                                                    with_variance ? d.estVariance_dev.get()
                                                                  : nullptr,
                                                    d.epsilon);
}

template <typename Data>
miopenStatus_t InferV1(miopen::Handle& handle, Data& d, bool with_estimates)
{
    return InferV1(handle, d, with_estimates, with_estimates);
}

// The inverse variance entry takes no epsilon. It reads the second estimate as
// an inverse variance.
miopenStatus_t InferInvVariance(miopen::Handle& handle, InferData& d, bool with_estimates)
{
    return miopenBatchNormalizationForwardInferenceInvVariance(
        &handle,
        d.bn_mode,
        &d.alpha,
        &d.beta,
        &d.input.desc,
        d.in_dev.get(),
        &d.output.desc,
        d.out_dev.get(),
        &d.scale.desc,
        &d.shift.desc,
        &d.estMean.desc,
        &d.estVariance.desc,
        d.scale_dev.get(),
        d.shift_dev.get(),
        with_estimates ? d.estMean_dev.get() : nullptr,
        with_estimates ? d.estVariance_dev.get() : nullptr);
}

miopenStatus_t
InferActivation(miopen::Handle& handle, InferData& d, miopenActivationDescriptor_t activ)
{
    return miopenBatchNormForwardInferenceActivation(&handle,
                                                     d.bn_mode,
                                                     &d.alpha,
                                                     &d.beta,
                                                     &d.input.desc,
                                                     d.in_dev.get(),
                                                     &d.output.desc,
                                                     d.out_dev.get(),
                                                     &d.scale.desc,
                                                     &d.shift.desc,
                                                     &d.estMean.desc,
                                                     &d.estVariance.desc,
                                                     d.scale_dev.get(),
                                                     d.shift_dev.get(),
                                                     d.estMean_dev.get(),
                                                     d.estVariance_dev.get(),
                                                     d.epsilon,
                                                     activ);
}

template <typename Data>
miopenStatus_t BwdV1(miopen::Handle& handle, Data& d, bool with_saved_stats)
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

template <typename Data>
miopenStatus_t BwdActivation(miopen::Handle& handle,
                             Data& d,
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
template <typename Data>
void CheckTrain(miopen::Handle& handle, Data& d, bool save, bool running)
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

template <typename Data>
void CheckInfer(miopen::Handle& handle, Data& d)
{
    d.output.data = handle.Read<float>(d.out_dev, d.output.data.size());
    test::ComputeCPUBNInference(d);
    activationHostInfer(
        d.activ_mode, 0.0, d.activ_beta, d.activ_alpha, d.out_ref.data, d.out_ref.data);
    auto tolerance = kTol;
#if WORKAROUND_SWDEV_547301
    // The bn.hpp inference fixture's tolerance, which Navi4x needs (SWDEV-547301).
    tolerance = miopen::StartsWith(handle.GetDeviceName(), "gfx120") ? 8e-3 : kTol;
#endif
    test::CompareTensor<float>(d.output, d.out_ref, tolerance);
}

template <typename Data>
void CheckBwd(miopen::Handle& handle, Data& d)
{
    using Dx      = typename decltype(d.output.data)::value_type;
    using DScale  = typename decltype(d.dScale.data)::value_type;
    d.output.data = handle.Read<Dx>(d.out_dev, d.output.data.size());
    d.dScale.data = handle.Read<DScale>(d.dScale_dev, d.dScale.data.size());
    d.dBias.data  = handle.Read<DScale>(d.dBias_dev, d.dBias.data.size());

    test::ComputeCPUBNBwd(d);

    test::CompareTensor<Dx>(d.output, d.out_ref, kTol);
    test::CompareTensor<DScale>(d.dScale, d.dScale_ref, kTol);
    test::CompareTensor<DScale>(d.dBias, d.dBias_ref, kTol);
}

// Backward without saved stats recomputes them from x. The CPU reference reads
// only savedMean and savedInvVar, so give it the true stats of the input.
template <typename Data>
void FillTrueStats(Data& d, double eps)
{
    tensor<float> y{d.input.desc};
    std::decay_t<decltype(d.savedMean)> none;
    if(d.bn_mode == miopenBNPerActivation)
        batchNormPerActHostFwdTrain(
            d.input, y, d.bnScale, d.bnBias, eps, 0.0, d.savedMean, d.savedInvVar, none, none);
    else
        batchNormSpatialHostFwdTrain(
            d.input, y, d.bnScale, d.bnBias, eps, 0.0, d.savedMean, d.savedInvVar, none, none);
}

// MIOpen silently runs training when the estimated stats are null, so the output
// is x normalized with its own batch stats: the training reference's output.
void CheckInferWithoutEstimates(miopen::Handle& handle, InferData& d)
{
    d.output.data = handle.Read<float>(d.out_dev, d.output.data.size());
    tensor<float> none;
    batchNormSpatialHostFwdTrain(
        d.input, d.out_ref, d.scale, d.shift, d.epsilon, 0.0, none, none, none, none);
    test::CompareTensor<float>(d.output, d.out_ref, kTol);
}

// The test data seeds each tensor from its lengths, so dy holds the same values as
// x. Backward with the true stats of x then cancels to a residue of order
// epsilon / variance: rounding noise, which the compare fails on in both modes.
// Draw dy from a seed of its own.
template <typename Data>
void DecoupleDy(miopen::Handle& handle, Data& d)
{
    using Dy = typename decltype(d.dy.data)::value_type;
    prng::reset_seed(1);
    for(auto& v : d.dy.data)
        v = prng::gen_descreet_uniform_sign<Dy>(2e-3, 1000);
    d.dy_dev = handle.Write(d.dy.data);
}

// True when every element still holds the NaN the test data poisoned it with.
bool Untouched(miopen::Handle& handle, const miopen::Allocator::ManageDataPtr& dev, std::size_t n)
{
    const auto values = handle.Read<float>(dev, n);
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isnan(v); });
}

// A call both paths must serve.
bool ExpectServed(const std::string& case_name, const hs::CallResult& result)
{
    const bool served = hs::RecordGapOutcome(case_name, result);
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

namespace {

// The test data seeds each tensor from its lengths, so scale, shift and estMean
// hold the same value per channel. A channel whose value is negative enough
// leaves every output at or below zero, and ReLU then zeroes all of it. With two
// or three channels all of them can, and the fixture fails its "CPU data is all
// zeros" check; with eight or more the odds are negligible.
std::vector<BN2DTestCase> ReluNetwork2D()
{
    auto cases = Network2DSmall<BN2DTestCase>();
    cases.erase(
        std::remove_if(cases.begin(), cases.end(), [](const BN2DTestCase& tc) { return tc.C < 8; }),
        cases.end());
    return cases;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimBNInferRelu_FP32,
                         testing::Combine(testing::ValuesIn(ReluNetwork2D()),
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

// Plain backward without saved stats, at the provider's epsilon, runs a graph that
// recomputes the stats from x. The same problem with saved stats must not reuse
// it: that plan would ignore the saved stats the caller passed, which the test
// data draws at random, far from the true ones.
TEST(GPU_HipdnnShimBNPlanKey_FP32, SavedStatsSelectTheirOwnPlan)
{
    auto&& handle = get_handle();

    {
        SCOPED_TRACE("without saved stats");
        BwdData d;
        SetUpBwd(d);
        DecoupleDy(handle, d);
        d.epsilon = kProviderEpsilon;
        ASSERT_EQ(BwdV1(handle, d, false), miopenStatusSuccess);
        FillTrueStats(d, d.epsilon);
        CheckBwd(handle, d);
    }
    {
        SCOPED_TRACE("with saved stats");
        BwdData d;
        SetUpBwd(d);
        d.epsilon = kProviderEpsilon;
        ASSERT_EQ(BwdV1(handle, d, true), miopenStatusSuccess);
        CheckBwd(handle, d);
    }
}

// Training without running stats and inference with estimates bind the same six
// tensors at the same epsilon, so only the graph kind tells their plans apart. An
// inference call that reused the training plan would normalize x with its own
// batch stats, not the estimates, which the test data draws at random.
TEST(GPU_HipdnnShimBNPlanKey_FP32, GraphKindSelectsItsOwnPlan)
{
    auto&& handle = get_handle();

    {
        SCOPED_TRACE("training without running stats");
        TrainData d;
        SetUpTrain(d);
        d.epsilon = kProviderEpsilon;
        ASSERT_EQ(TrainV2(handle, d, {true, true, false, false}), miopenStatusSuccess);
        CheckTrain(handle, d, true, false);
    }
    {
        SCOPED_TRACE("inference with estimates");
        InferData d;
        SetUpInfer(d);
        d.epsilon = kProviderEpsilon;
        ASSERT_EQ(InferV1(handle, d, true), miopenStatusSuccess);
        CheckInfer(handle, d);
    }
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
        ExpectServed(c.name, result);
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

namespace {

// Runs train, infer and bwd in per-activation mode at one layout and checks each
// against the CPU reference.
template <typename Train, typename Infer, typename Bwd>
void RunPerActivation(miopen::Handle& handle,
                      miopenTensorLayout_t tensor_layout,
                      const std::string& suffix)
{
    {
        SCOPED_TRACE("train");
        Train d;
        SetUpTrain(d, miopenBNPerActivation, tensor_layout);
        const auto result = hs::CaptureCall([&] { return TrainV1(handle, d); });
        if(ExpectServed("per_activation_train" + suffix, result))
            CheckTrain(handle, d, true, true);
    }
    {
        SCOPED_TRACE("infer");
        Infer d;
        SetUpInfer(d, miopenBNPerActivation, tensor_layout);
        const auto result = hs::CaptureCall([&] { return InferV1(handle, d, true); });
        if(ExpectServed("per_activation_infer" + suffix, result))
            CheckInfer(handle, d);
    }
    {
        SCOPED_TRACE("bwd");
        Bwd d;
        SetUpBwd(d, miopenBNPerActivation, tensor_layout);
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
        if(ExpectServed("per_activation_bwd" + suffix, result))
            CheckBwd(handle, d);
    }
}

} // namespace

// Per-activation mode is folded into spatial mode over C*H*W channels of one
// element each, which gives every element its own stats over the batch. MIOpen's
// per-activation kernels read every tensor in memory order, whatever its layout,
// and so does the fold. With the parameters laid out like x, as here, that is the
// per-element answer for NHWC too. A 5D tensor folds the same way, over C*D*H*W
// channels; MIOpen reshapes it to 4D before any solver sees it, so it serves 3D
// per-activation as well.
TEST(GPU_HipdnnShimBNPerActivation_FP32, MatchesCpuReference)
{
    auto&& handle = get_handle();

    for(const auto tensor_layout : {miopenTensorNCHW, miopenTensorNHWC})
    {
        const bool nhwc = tensor_layout == miopenTensorNHWC;
        SCOPED_TRACE(nhwc ? "NHWC" : "NCHW");
        RunPerActivation<TrainData, InferData, BwdData>(handle, tensor_layout, nhwc ? "_nhwc" : "");
    }
}

TEST(GPU_HipdnnShimBNPerActivation3D_FP32, MatchesCpuReference)
{
    auto&& handle = get_handle();

    for(const auto tensor_layout : {miopenTensorNCDHW, miopenTensorNDHWC})
    {
        const bool ndhwc = tensor_layout == miopenTensorNDHWC;
        SCOPED_TRACE(ndhwc ? "NDHWC" : "NCDHW");
        RunPerActivation<Train3DData, Infer3DData, Bwd3DData>(
            handle, tensor_layout, ndhwc ? "_ndhwc" : "_ncdhw");
    }
}

// MIOpen's per-activation kernels ignore the activation, so the call is served and
// its results are those without it
// (https://github.com/ROCm/rocm-libraries/issues/15445). TANH is one the provider
// has no mapping for. Training and backward serve it all the same; inference has
// no solver for it in either mode, so it fails with NotImplemented on both paths.
TEST(GPU_HipdnnShimBNPerActivation_FP32, IgnoresActivation)
{
    auto&& handle = get_handle();

    for(const auto mode : {miopenActivationRELU, miopenActivationTANH})
    {
        const std::string name = mode == miopenActivationRELU ? "relu" : "tanh";
        SCOPED_TRACE(name);

        hs::OwnedActivationDescriptor activ;
        ASSERT_EQ(miopenCreateActivationDescriptor(&activ.handle), miopenStatusSuccess);
        ASSERT_EQ(miopenSetActivationDescriptor(activ.handle, mode, 0.1, 0.3, 0.0),
                  miopenStatusSuccess);
        {
            SCOPED_TRACE("train");
            TrainData d;
            SetUpTrain(d, miopenBNPerActivation);
            const auto result =
                hs::CaptureCall([&] { return TrainActivation(handle, d, activ.handle); });
            if(ExpectServed("per_activation_" + name + "_train", result))
                CheckTrain(handle, d, true, true);
        }
        {
            SCOPED_TRACE("infer");
            InferData d;
            SetUpInfer(d, miopenBNPerActivation);
            const auto result =
                hs::CaptureCall([&] { return InferActivation(handle, d, activ.handle); });
            if(mode == miopenActivationTANH)
                hs::ExpectStatusParity("per_activation_tanh_infer",
                                       result,
                                       miopenStatusNotImplemented,
                                       "no batchnorm solver");
            else if(ExpectServed("per_activation_" + name + "_infer", result))
                CheckInfer(handle, d);
        }
        {
            SCOPED_TRACE("bwd");
            BwdData d;
            SetUpBwd(d, miopenBNPerActivation);
            const auto result =
                hs::CaptureCall([&] { return BwdActivation(handle, d, true, activ.handle); });
            if(ExpectServed("per_activation_" + name + "_bwd", result))
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

// The training it runs must keep the caller's epsilon. The default one is 1e-5,
// so a second case uses an epsilon large next to the variance. MIOpen runs
// inference only when both estimates are given, so one missing estimate is
// enough to run training.
TEST(GPU_HipdnnShimBNInferWithoutEstimates_FP32, RunsTraining)
{
    auto&& handle = get_handle();

    struct NoEstimatesCase
    {
        std::string name;
        double epsilon;
        bool with_mean;
        bool with_variance;
    };
    for(const auto& c : {NoEstimatesCase{"infer_no_estimates", 1e-5, false, false},
                         NoEstimatesCase{"infer_no_estimates_large_epsilon", 0.5, false, false},
                         NoEstimatesCase{"infer_no_estimated_mean", 0.5, false, true},
                         NoEstimatesCase{"infer_no_estimated_variance", 0.5, true, false}})
    {
        SCOPED_TRACE(c.name);
        InferData d;
        SetUpInfer(d);
        d.epsilon = c.epsilon;
        const auto result =
            hs::CaptureCall([&] { return InferV1(handle, d, c.with_mean, c.with_variance); });
        if(ExpectServed(c.name, result))
            CheckInferWithoutEstimates(handle, d);
    }
}

// The inverse variance entry has no epsilon, so the training it runs uses the
// default one, 1e-5, and the reference does too.
TEST(GPU_HipdnnShimBNInferWithoutEstimates_FP32, InvVarianceRunsTrainingAtDefaultEpsilon)
{
    auto&& handle = get_handle();
    InferData d;
    SetUpInfer(d);
    d.epsilon         = 1e-5;
    const auto result = hs::CaptureCall([&] { return InferInvVariance(handle, d, false); });
    if(ExpectServed("infer_inv_variance_no_estimates", result))
        CheckInferWithoutEstimates(handle, d);
}

// The provider recomputes the stats itself only for plain backward at an epsilon
// of 1e-5. For any other epsilon, and for backward with an activation, whose
// graph needs the saved stats, the shim runs training first to compute them into
// scratch. Every combination is served; the CPU reference is given the true stats
// of the input at the call's epsilon. The other epsilon is large next to the
// variance, so that results computed at the provider's epsilon instead fail. Both
// paths are also run over NHWC, per-activation, FP16 and BF16. Per-activation
// drops the activation, so it runs without one.
namespace {

struct NoStatsCase
{
    std::string name;
    double epsilon;
    miopenActivationMode_t mode;
    miopenBatchNormMode_t bn_mode      = miopenBNSpatial;
    miopenTensorLayout_t tensor_layout = miopenTensorNCHW;
};

constexpr double kLargeEpsilon = 0.5;

// The four combinations of epsilon and activation, with names that start with
// the given prefix.
std::vector<NoStatsCase> NoStatsCases(const std::string& prefix,
                                      miopenBatchNormMode_t bn_mode      = miopenBNSpatial,
                                      miopenTensorLayout_t tensor_layout = miopenTensorNCHW)
{
    return {
        {prefix, kProviderEpsilon, miopenActivationPASTHRU, bn_mode, tensor_layout},
        {prefix + "_large_epsilon", kLargeEpsilon, miopenActivationPASTHRU, bn_mode, tensor_layout},
        {prefix + "_act", kProviderEpsilon, miopenActivationRELU, bn_mode, tensor_layout},
        {prefix + "_act_large_epsilon",
         kLargeEpsilon,
         miopenActivationRELU,
         bn_mode,
         tensor_layout},
    };
}

template <typename Data>
void RunBwdWithoutSavedStats(const std::vector<NoStatsCase>& cases)
{
    auto&& handle = get_handle();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        Data d;
        SetUpBwd(d, c.bn_mode, c.tensor_layout);
        DecoupleDy(handle, d);
        d.epsilon = c.epsilon;
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

using BwdHalfData = BNBwdTestData<half_float::half,
                                  half_float::half,
                                  half_float::half,
                                  float,
                                  float,
                                  float,
                                  double,
                                  BN2DTestCase>;
using BwdBfloat16Data =
    BNBwdTestData<bfloat16, bfloat16, bfloat16, float, float, float, double, BN2DTestCase>;

} // namespace

TEST(GPU_HipdnnShimBNBwdWithoutSavedStats_FP32, MatchesCpuReference)
{
    auto cases = NoStatsCases("bwd_no_saved_stats");
    for(auto&& c : NoStatsCases("bwd_no_saved_stats_nhwc", miopenBNSpatial, miopenTensorNHWC))
        cases.push_back(c);
    for(auto&& c : NoStatsCases("bwd_no_saved_stats_per_activation", miopenBNPerActivation))
        if(c.mode == miopenActivationPASTHRU)
            cases.push_back(c);
    RunBwdWithoutSavedStats<BwdData>(cases);
}

TEST(GPU_HipdnnShimBNBwdWithoutSavedStats_FP16, MatchesCpuReference)
{
    RunBwdWithoutSavedStats<BwdHalfData>(NoStatsCases("bwd_no_saved_stats_fp16"));
}

TEST(GPU_HipdnnShimBNBwdWithoutSavedStats_BFP16, MatchesCpuReference)
{
    RunBwdWithoutSavedStats<BwdBfloat16Data>(NoStatsCases("bwd_no_saved_stats_bfp16"));
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

// Sets MIOPEN_FIND_ENFORCE in the process environment, which the shim and the
// provider read, and in MIOpen's own copy, which MIOpen reads once and keeps.
// Pinning MIOpen's copy first means MIOpen never reads the value set here, so it
// cannot keep it after the test.
struct ScopedFindEnforce
{
    explicit ScopedFindEnforce(const std::string& value) : miopen_copy(MIOPEN_FIND_ENFORCE, value)
    {
        if(const char* prev = std::getenv("MIOPEN_FIND_ENFORCE"))
            saved = prev;
        Set(value.c_str());
    }
    ~ScopedFindEnforce() { Set(saved ? saved->c_str() : nullptr); }

    ScopedFindEnforce(const ScopedFindEnforce&)            = delete;
    ScopedFindEnforce& operator=(const ScopedFindEnforce&) = delete;

    static void Set(const char* value)
    {
#ifdef _WIN32
        _putenv_s("MIOPEN_FIND_ENFORCE", value != nullptr ? value : "");
#else
        if(value != nullptr)
            setenv("MIOPEN_FIND_ENFORCE", value, 1);
        else
            unsetenv("MIOPEN_FIND_ENFORCE");
#endif
    }

    const ScopedEnvironment<std::string> miopen_copy;
    std::optional<std::string> saved;
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

// Used only under Search: MIOpen searches only for a shape the handle holds no
// tuned invoker for, so no other test may run this shape.
const BN2DTestCase& SearchCase()
{
    static const BN2DTestCase search_case{
        3, 5, 7, 11, miopen::batchnorm::Direction::ForwardTraining, true, true};
    return search_case;
}

// Training passes no running stats, since a search re-runs the kernel and would
// apply the running-stat update more than once.
hs::CallResult TrainUnderPolicy(miopen::Handle& handle,
                                TrainData& d,
                                miopenBatchNormMode_t mode    = miopenBNSpatial,
                                const BN2DTestCase& test_case = TuningCase())
{
    d.SetUpImpl(test_case, mode, miopenTensorNCHW);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
    return hs::CaptureCall([&] { return TrainV2(handle, d, {true, true, false, false}); });
}

hs::CallResult BwdUnderPolicy(miopen::Handle& handle,
                              BwdData& d,
                              miopenBatchNormMode_t mode    = miopenBNSpatial,
                              const BN2DTestCase& test_case = TuningCase())
{
    d.SetUpImpl(test_case, mode, miopenTensorNCHW);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
    return hs::CaptureCall([&] { return BwdV1(handle, d, true); });
}

hs::CallResult InferUnderPolicy(miopen::Handle& handle,
                                InferData& d,
                                bool with_estimates,
                                const BN2DTestCase& test_case = TuningCase())
{
    d.SetUpImpl(test_case, miopenBNSpatial, miopenTensorNCHW, false);
    SetActivation(d.activ_mode, d.activ_alpha, d.activ_beta, miopenActivationPASTHRU);
    return hs::CaptureCall([&] { return InferV1(handle, d, with_estimates); });
}

// The environment the search-observing tests run under: MIOPEN_WARN_SEARCH raises
// MIOpen's "Search Started" line to a warning, and the perf dbs are off, because a
// record found there, from an earlier run or another test, is loaded in place of a
// search. Pass user_db to keep the user db on.
struct SearchLogScope
{
    explicit SearchLogScope(bool user_db = false)
        : no_user_db(MIOPEN_DEBUG_DISABLE_USER_DB, !user_db)
    {
    }

    const ScopedEnvironment<bool> warn_search{MIOPEN_WARN_SEARCH, true};
    const ScopedEnvironment<bool> no_user_db;
    const ScopedEnvironment<bool> no_system_db{MIOPEN_DEBUG_DISABLE_SYSTEM_DB, true};
};

// Whether MIOpen searched during the call, from the "Search Started" line that
// MIOPEN_WARN_SEARCH raises to a warning.
bool Searched(const hs::CallResult& result)
{
    return result.logged.find("Search Started:") != std::string::npos;
}

// MIOpen tunes the spatial training and backward solvers when the handle asks for
// Search. The provider runs on a handle of its own, so the shim carries Search
// over as the plan's benchmarking knob, under which the provider sets Search on
// its handle. Inference is not tunable and is served without the knob.
//
// The results alone cannot tell a search from a run under None, so the test also
// looks for MIOpen's "Search Started" line, which MIOPEN_WARN_SEARCH raises to a
// warning. Both perf dbs are off, because a record found there, from an earlier
// run or another test, is loaded in place of a search.
TEST(GPU_HipdnnShimBNTuning_FP32, SearchPolicyIsServed)
{
    auto&& handle = get_handle();
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicySearch);
    const SearchLogScope search_log;

    {
        SCOPED_TRACE("train");
        TrainData d;
        const auto result = TrainUnderPolicy(handle, d, miopenBNSpatial, SearchCase());
        if(ExpectServed("tuning_policy_train", result))
        {
            EXPECT_TRUE(Searched(result)) << result.logged;
            CheckTrain(handle, d, true, false);
        }
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        const auto result = BwdUnderPolicy(handle, d, miopenBNSpatial, SearchCase());
        if(ExpectServed("tuning_policy_bwd", result))
        {
            EXPECT_TRUE(Searched(result)) << result.logged;
            CheckBwd(handle, d);
        }
    }
    {
        SCOPED_TRACE("infer");
        InferData d;
        const auto result = InferUnderPolicy(handle, d, true, SearchCase());
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

// Used only by NonePolicyDoesNotSearch.
const BN2DTestCase& NoSearchCase()
{
    static const BN2DTestCase no_search_case{
        3, 5, 7, 13, miopen::batchnorm::Direction::ForwardTraining, true, true};
    return no_search_case;
}

// Under None MIOpen loads a perf-db record or runs the default config, and never
// searches, so the hipDNN path must not search either. Both perf dbs are off and
// MIOPEN_FIND_ENFORCE is empty, which MIOpen reads as no enforcement, so the policy
// alone decides.
TEST(GPU_HipdnnShimBNTuning_FP32, NonePolicyDoesNotSearch)
{
    auto&& handle = get_handle();
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicyNone);
    const ScopedFindEnforce find_enforce("");
    const SearchLogScope search_log;

    {
        SCOPED_TRACE("train");
        TrainData d;
        const auto result = TrainUnderPolicy(handle, d, miopenBNSpatial, NoSearchCase());
        if(ExpectServed("none_policy_train", result))
        {
            EXPECT_FALSE(Searched(result)) << result.logged;
            CheckTrain(handle, d, true, false);
        }
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        const auto result = BwdUnderPolicy(handle, d, miopenBNSpatial, NoSearchCase());
        if(ExpectServed("none_policy_bwd", result))
        {
            EXPECT_FALSE(Searched(result)) << result.logged;
            CheckBwd(handle, d);
        }
    }
}

// Used only by FindEnforceUnderNonePolicyMatchesMiopen.
const BN2DTestCase& FindEnforceCase()
{
    static const BN2DTestCase find_enforce_case{
        3, 5, 7, 15, miopen::batchnorm::Direction::ForwardTraining, true, true};
    return find_enforce_case;
}

struct FindEnforceStep
{
    const char* value;
    bool searches;
    // Whether MIOpen logs that it removed a perf-db record; unset where that depends
    // on what an earlier run left behind.
    std::optional<bool> removes;
};

void ExpectFindEnforceOutcome(const std::string& case_name,
                              const hs::CallResult& result,
                              const FindEnforceStep& step)
{
    const bool searched = Searched(result);
    ::testing::Test::RecordProperty("parity_searched_" + case_name, searched ? "true" : "false");
    EXPECT_EQ(searched, step.searches) << result.logged;
    if(step.removes)
    {
        const bool removed = result.logged.find("Perf Db: record removed:") != std::string::npos;
        ::testing::Test::RecordProperty("parity_removed_" + case_name, removed ? "true" : "false");
        EXPECT_EQ(removed, *step.removes) << result.logged;
    }
}

// MIOPEN_FIND_ENFORCE takes over from a None handle, and the hipDNN path must follow
// it as MIOpen does: in whether a call searches, and in what it leaves in the user
// perf db. The steps run in order and see each other only through that db, since
// each gets a handle of its own and so cannot reuse an invoker or plan an earlier
// step cached:
//
//   DB_CLEAN          removes any record an earlier run left
//   SEARCH            finds no record, so searches and stores one
//   SEARCH            loads that record instead of searching
//   SEARCH_DB_UPDATE  searches even though there is a record
//   NONE, DB_UPDATE   load the record, no search
//   DB_CLEAN          removes the record
//   SEARCH            searches again, so the record really went
//   DB_CLEAN          removes it, so the next run starts the same way
//
// The user db is on and the system db off. MIOpen opens each perf db once per
// process, and if an earlier test opened the user db while it was off, its records
// stay in memory, which the steps cannot tell apart from the file.
TEST(GPU_HipdnnShimBNTuning_FP32, FindEnforceUnderNonePolicyMatchesMiopen)
{
    const SearchLogScope search_log(true);

    const std::vector<FindEnforceStep> steps = {
        {"DB_CLEAN", false, std::nullopt},
        {"SEARCH", true, false},
        {"SEARCH", false, false},
        {"SEARCH_DB_UPDATE", true, false},
        {"NONE", false, false},
        {"DB_UPDATE", false, false},
        {"DB_CLEAN", false, true},
        {"SEARCH", true, false},
        {"DB_CLEAN", false, true},
    };

    for(std::size_t i = 0; i < steps.size(); ++i)
    {
        const auto& step = steps[i];
        const auto name  = "find_enforce_" + std::to_string(i) + "_" + step.value;
        SCOPED_TRACE(name);
        const ScopedFindEnforce find_enforce(step.value);

        hs::Owned<miopenHandle_t, miopenDestroy> owned_handle;
        ASSERT_EQ(miopenCreateWithStream(&owned_handle.handle, get_handle().GetStream()),
                  miopenStatusSuccess);
        auto&& handle = miopen::deref(owned_handle.handle);

        {
            TrainData d;
            const auto result = TrainUnderPolicy(handle, d, miopenBNSpatial, FindEnforceCase());
            if(ExpectServed(name + "_train", result))
            {
                ExpectFindEnforceOutcome(name + "_train", result, step);
                CheckTrain(handle, d, true, false);
            }
        }
        {
            BwdData d;
            const auto result = BwdUnderPolicy(handle, d, miopenBNSpatial, FindEnforceCase());
            if(ExpectServed(name + "_bwd", result))
            {
                ExpectFindEnforceOutcome(name + "_bwd", result, step);
                CheckBwd(handle, d);
            }
        }
    }
}

// The shim declines a Search handle only under the MIOPEN_FIND_ENFORCE values that
// write or clean the perf db. Under SEARCH, here also spelled as its number 3, both
// MIOpen and the provider search, so the call is served and searches. Each value
// gets a handle of its own, so the second cannot reuse the first one's invoker or
// plan, and both perf dbs are off, so neither can load a record.
TEST(GPU_HipdnnShimBNTuning_FP32, SearchPolicyUnderFindEnforceSearchIsServed)
{
    const SearchLogScope search_log;

    for(const char* value : {"SEARCH", "3"})
    {
        SCOPED_TRACE(value);
        const ScopedFindEnforce find_enforce(value);

        hs::Owned<miopenHandle_t, miopenDestroy> owned_handle;
        ASSERT_EQ(miopenCreateWithStream(&owned_handle.handle, get_handle().GetStream()),
                  miopenStatusSuccess);
        auto&& handle = miopen::deref(owned_handle.handle);
        const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicySearch);

        TrainData d;
        const auto result = TrainUnderPolicy(handle, d, miopenBNSpatial, SearchCase());
        if(ExpectServed("search_policy_find_enforce_" + std::string(value) + "_train", result))
        {
            EXPECT_TRUE(Searched(result)) << result.logged;
            CheckTrain(handle, d, true, false);
        }
    }
}

// ---------------------------------------------------------------------------
// Gaps: MIOpen serves these, the hipDNN path declines them

// The hipDNN path declines a tuning gap with UnsupportedOp. RecordGapOutcome also
// accepts NotImplemented, which conv declines need, so this checks the status too.
bool RecordTuningGap(const std::string& case_name, const hs::CallResult& result, const char* reason)
{
    const bool served = hs::RecordGapOutcome(case_name, result, reason);
    if(hs::ForwardingEnabled() && !served)
        EXPECT_EQ(result.status, miopenStatusUnsupportedOp) << case_name << ": " << result.message;
    return served;
}

// Training, backward, and inference without the estimates (which runs training),
// each a call MIOpen serves under the current policy and the hipDNN path declines.
// The case names are prefix + "_train", "_bwd" and "_infer".
void ExpectTrainingPolicyGap(miopen::Handle& handle, const std::string& prefix, const char* reason)
{
    {
        SCOPED_TRACE("train");
        TrainData d;
        if(RecordTuningGap(prefix + "_train", TrainUnderPolicy(handle, d), reason))
            CheckTrain(handle, d, true, false);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        if(RecordTuningGap(prefix + "_bwd", BwdUnderPolicy(handle, d), reason))
            CheckBwd(handle, d);
    }
    {
        SCOPED_TRACE("infer");
        InferData d;
        const auto result = InferUnderPolicy(handle, d, false);
        if(RecordTuningGap(prefix + "_infer", result, reason))
            CheckInferWithoutEstimates(handle, d);
    }
}

// The provider sets only None or Search on its handle, so it cannot run the
// policies that also write or clean the perf db. The shim declines them for
// training and backward, and for inference without the estimated stats, which
// runs training. Each of the three policies gets its own cases.
TEST(GPU_HipdnnShimBNGaps_FP32, WriteOrCleanPoliciesAreDeclined)
{
    auto&& handle = get_handle();

    for(const auto& [tuning_policy, prefix] :
        {std::pair<miopenTuningPolicy_t, std::string>{miopenTuningPolicyDbUpdate,
                                                      "tuning_policy_dbupdate"},
         {miopenTuningPolicySearchDbUpdate, "tuning_policy_searchdbupdate"},
         {miopenTuningPolicyDbClean, "tuning_policy_dbclean"}})
    {
        SCOPED_TRACE(prefix);
        const ScopedHandleTuningPolicy policy(handle, tuning_policy);
        ExpectTrainingPolicyGap(handle, prefix, "tuning policy");
    }
}

// MIOpen lets a Search handle win over MIOPEN_FIND_ENFORCE, but the provider sets
// None on its handle whenever the variable is set, so the variable would win
// there. Under DB_CLEAN the forwarded call would drop the perf-db entry MIOpen
// tunes, so the shim declines it. DB_UPDATE and SEARCH_DB_UPDATE are declined the
// same way. DB_CLEAN runs every path; the others run training only, in spellings
// MIOpen also accepts: lower case, and the action's number (5 is DB_CLEAN).
TEST(GPU_HipdnnShimBNGaps_FP32, SearchPolicyUnderFindEnforceIsDeclined)
{
    auto&& handle = get_handle();
    const ScopedHandleTuningPolicy policy(handle, miopenTuningPolicySearch);
    const char* reason = "MIOPEN_FIND_ENFORCE";

    {
        SCOPED_TRACE("DB_CLEAN");
        const ScopedFindEnforce find_enforce("DB_CLEAN");
        ExpectTrainingPolicyGap(handle, "find_enforce_db_clean", reason);
    }

    const std::pair<const char*, const char*> others[] = {
        {"db_update", "DB_UPDATE"}, {"search_db_update", "search_db_update"}, {"5", "5"}};
    for(const auto& [name, value] : others)
    {
        SCOPED_TRACE(value);
        const ScopedFindEnforce other(value);
        TrainData d;
        if(RecordTuningGap(
               "find_enforce_" + std::string(name) + "_train", TrainUnderPolicy(handle, d), reason))
            CheckTrain(handle, d, true, false);
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
        hs::ExpectStatusParity("alpha_beta_train", result, miopenStatusUnknownError, reason);
    }
    {
        SCOPED_TRACE("infer");
        InferData d;
        SetUpInfer(d);
        d.alpha           = 2.0f;
        const auto result = hs::CaptureCall([&] { return InferV1(handle, d, true); });
        hs::ExpectStatusParity("alpha_beta_infer", result, miopenStatusBadParm, reason);
    }
    {
        SCOPED_TRACE("bwd");
        BwdData d;
        SetUpBwd(d);
        d.alphaDataDiff   = 2.0f;
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
        hs::ExpectStatusParity("alpha_beta_bwd", result, miopenStatusBadParm, reason);
    }
    {
        SCOPED_TRACE("bwd param diff");
        BwdData d;
        SetUpBwd(d);
        d.alphaParamDiff  = 2.0f;
        const auto result = hs::CaptureCall([&] { return BwdV1(handle, d, true); });
        hs::ExpectStatusParity("alpha_beta_bwd_param_diff", result, miopenStatusBadParm, reason);
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

    RawTensor(miopen::Handle& handle, miopen::TensorDescriptor desc_in)
        : desc(std::move(desc_in)), dev(handle.Create(desc.GetNumBytes()))
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

// The hipDNN path repeats MIOpen's parameter checks, so it must reject these
// with BadParm as MIOpen does: statistics of another rank than x, on forward
// and on backward, and a training x that is not packed.
TEST(GPU_HipdnnShimBNBadParm_FP32, StatusMatchesMiopen)
{
    auto&& handle = get_handle();
    const std::vector<std::size_t> nchw{2, 4, 8, 8};
    const miopen::TensorDescriptor rank3_stats(miopenFloat, std::vector<std::size_t>{1, 4, 1});
    const char* rank_reason = "as many dimensions as x";

    {
        SCOPED_TRACE("rank mismatch train");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.scale           = RawTensor(handle, rank3_stats);
        const auto result = hs::CaptureCall([&] { return p.Train(handle); });
        hs::ExpectStatusParity(
            "bad_parm_rank_mismatch_train", result, miopenStatusBadParm, rank_reason);
    }
    {
        // Backward passes the scale descriptor for the statistics too.
        SCOPED_TRACE("rank mismatch bwd");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.scale           = RawTensor(handle, rank3_stats);
        const auto result = hs::CaptureCall([&] { return p.Bwd(handle); });
        hs::ExpectStatusParity(
            "bad_parm_rank_mismatch_bwd", result, miopenStatusBadParm, rank_reason);
    }
    {
        SCOPED_TRACE("non-packed train");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.x = RawTensor(
            handle,
            miopen::TensorDescriptor(miopenFloat, nchw, std::vector<std::size_t>{512, 128, 16, 1}));
        const auto result = hs::CaptureCall([&] { return p.Train(handle); });
        hs::ExpectStatusParity(
            "bad_parm_non_packed_train", result, miopenStatusBadParm, "x must be packed");
    }
}

// Problems that pass MIOpen's checks but that the hipDNN path does not forward.
// Natively MIOpen runs or rejects them by its own rules, so only the forwarding
// mode is checked. The reason is checked, not just the status, because each
// problem is built to reach one decline first. Inference is used where it can
// be, since its parameter checks do not test packedness.
TEST(GPU_HipdnnShimBNDeclined_FP32, UnsupportedProblemsAreDeclinedWithReason)
{
    if(!hs::ForwardingEnabled())
        return;

    auto&& handle = get_handle();
    const std::vector<std::size_t> nchw{2, 4, 8, 8};
    const miopen::TensorDescriptor padded(
        miopenFloat, nchw, std::vector<std::size_t>{512, 128, 16, 1});

    auto expect_declined_because = [&](const hs::CallResult& result, const char* reason) {
        // With a check missing the call is served, so its work must finish
        // before the buffers are freed.
        handle.Finish();
        EXPECT_EQ(result.status, miopenStatusUnsupportedOp) << reason;
        hs::ExpectForwardedDecline(result, reason);
    };

    {
        SCOPED_TRACE("non-packed infer");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.x = RawTensor(handle, padded);
        p.y = RawTensor(handle, padded);
        expect_declined_because(hs::CaptureCall([&] { return p.Infer(handle); }),
                                "supports only packed tensors");
    }
    {
        SCOPED_TRACE("non-packed bwd");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.x  = RawTensor(handle, padded);
        p.dy = RawTensor(handle, padded);
        p.dx = RawTensor(handle, padded);
        expect_declined_because(hs::CaptureCall([&] { return p.Bwd(handle); }),
                                "supports only packed tensors");
    }
    {
        // Packed NHWC strides, while x is NCHW.
        SCOPED_TRACE("y layout infer");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.y = RawTensor(
            handle,
            miopen::TensorDescriptor(miopenFloat, nchw, std::vector<std::size_t>{256, 1, 32, 4}));
        expect_declined_because(hs::CaptureCall([&] { return p.Infer(handle); }),
                                "same shape and layout");
    }
    {
        SCOPED_TRACE("spatial scale infer");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        p.scale = RawTensor(handle, miopen::TensorDescriptor(miopenFloat, nchw));
        expect_declined_because(hs::CaptureCall([&] { return p.Infer(handle); }),
                                "statistics shaped {1, C, 1, ...}");
    }
    {
        SCOPED_TRACE("vectorized infer");
        RawProblem p(handle, nchw, miopenFloat, miopenFloat, miopenFloat);
        const miopen::TensorDescriptor vectorized(miopenFloat, miopenTensorNCHWc4, nchw);
        p.x = RawTensor(handle, vectorized);
        p.y = RawTensor(handle, vectorized);
        expect_declined_because(hs::CaptureCall([&] { return p.Infer(handle); }),
                                "vectorized tensor layouts");
    }
}
