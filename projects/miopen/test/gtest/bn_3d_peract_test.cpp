// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "../random.hpp"
#include "../verify.hpp"
#include "../fusionHost.hpp"
#include "test_operations.hpp"
#include "network_data.hpp"

namespace {
constexpr double MIO_BN_TEST_EPSILON      = 1e-5;
constexpr double MIO_BN_TEST_EXPAVGFACTOR = 0.1;

enum class BN3DPerActTestType
{
    ForwardTraining,
    ForwardInferenceRecalc,
    ForwardInferenceUseEstimated,
    BackwardRecalc,
    BackwardUseSaved
};

struct BN3DPerActTestCase
{
    std::size_t n, c, d, h, w;
    BN3DPerActTestType test_type;

    friend std::ostream& operator<<(std::ostream& ss, const BN3DPerActTestCase& tc)
    {
        return ss << "(n: " << tc.n << " c: " << tc.c << " d: " << tc.d << " h: " << tc.h
                  << " w: " << tc.w << " type: " << static_cast<int>(tc.test_type) << ")";
    }
};

enum class BN3DPerActTestSet
{
    Standard, // 4 types: FwdTrain, FwdInferenceRecalc, BwdRecalc, BwdUseSaved
    Full      // 5 types: includes ForwardInferenceUseEstimated
};

std::vector<BN3DPerActTestCase>
GetBN3DPerActTestCases(BN3DPerActTestSet test_set = BN3DPerActTestSet::Standard)
{
    std::vector<BN3DPerActTestCase> test_cases;
    // Match ctest behavior:
    // FP32 runs 5 types (Full), while FP16/BF16 run 4 types (Standard)
    std::vector<BN3DPerActTestType> types;
    if(test_set == BN3DPerActTestSet::Full)
    {
        types = {BN3DPerActTestType::ForwardTraining,
                 BN3DPerActTestType::ForwardInferenceRecalc,
                 BN3DPerActTestType::ForwardInferenceUseEstimated,
                 BN3DPerActTestType::BackwardRecalc,
                 BN3DPerActTestType::BackwardUseSaved};
    }
    else
    {
        types = {BN3DPerActTestType::ForwardTraining,
                 BN3DPerActTestType::ForwardInferenceRecalc,
                 BN3DPerActTestType::BackwardRecalc,
                 BN3DPerActTestType::BackwardUseSaved};
    }

    // Use batch size factor 4 to match ctest behavior (like other BN 3D tests)
    for(const auto& shape : get_3d_bn_peract_inputs(4))
    {
        const auto n = shape[0];
        // Match ctest logic: skip ALL test cases when n == 1 (not just training/backward)
        // From bn_peract_test.cpp: if(n == 1) { return; }
        if(n == 1)
        {
            continue; // Skip all test cases for batch size 1
        }
        for(const auto& type : types)
        {
            test_cases.push_back({shape[0], shape[1], shape[2], shape[3], shape[4], type});
        }
    }
    return test_cases;
}

template <typename T>
struct GPU_Bn3dPerAct : public ::testing::TestWithParam<BN3DPerActTestCase>
{
    using AccDataType   = std::conditional_t<std::is_same_v<T, double>, double, float>;
    using InputDataType = T;

    // CPU Verification Tensors (Zero-Copy)
    tensor<T> dy;
    tensor<AccDataType> bnScale;
    tensor<AccDataType> bnBias;
    tensor<AccDataType> dScale_ref;
    tensor<AccDataType> dBias_ref;
    tensor<AccDataType> savedMean;
    tensor<AccDataType> savedInvVar;
    tensor<AccDataType> saveMean_ref;
    tensor<AccDataType> saveVariance_ref;
    tensor<AccDataType> runMean_ref;
    tensor<AccDataType> runVariance_ref;
    tensor<AccDataType> estMean;
    tensor<AccDataType> estVariance;

    miopenBatchNormMode_t bn_mode = miopenBNPerActivation;
    bool useInverseVariance       = false;
    double averageFactor          = MIO_BN_TEST_EXPAVGFACTOR;

    miopenActivationMode_t activ_mode = miopenActivationPASTHRU;
    double activ_alpha                = 1.0;
    double activ_beta                 = 0.0;
    double activ_gamma                = 1.0;

    std::size_t n, c, d, h, w;
    tensor<T> input;
    tensor<T> output;
    tensor<AccDataType> out_ref;
    tensor<AccDataType> scale;
    tensor<AccDataType> shift;
    tensor<AccDataType> runMean;
    tensor<AccDataType> runVar;
    miopen::TensorDescriptor derivedBnDesc;
    miopenTensorLayout_t bn_layout;
    miopenTensorLayout_t derived_layout; // Layout for 4D derived tensors (scale, shift, etc.)
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev;
    miopen::Allocator::ManageDataPtr runMean_dev;
    miopen::Allocator::ManageDataPtr runVar_dev;
    miopen::Allocator::ManageDataPtr out_dev;

    float alpha         = 1.0f;
    float beta          = 0.0f;
    double epsilon      = MIO_BN_TEST_EPSILON;
    double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
    double tolerance    = 5e-3;

    void SetUp() override
    {
        prng::reset_seed();
        const auto& tc = this->GetParam();
        n              = tc.n;
        c              = tc.c;
        d              = tc.d;
        h              = tc.h;
        w              = tc.w;

        if(n == 1)
        {
            GTEST_SKIP() << "Invalid batch size for batch norm tests";
        }

        auto&& handle = get_handle();

        input  = tensor<T>(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
        output = tensor<T>(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});

        auto input_layout_opt = input.desc.GetLayoutEnum();
        bn_layout = (input_layout_opt && input_layout_opt.value() != 0) ? input_layout_opt.value()
                                                                        : miopenTensorNCDHW;
        if(bn_layout == 0)
            bn_layout = miopenTensorNCDHW;
        out_ref = tensor<AccDataType>(bn_layout, std::vector<std::size_t>{n, c, d, h, w});

        input.generate(uniform_signed_initializer<T>(2e-3, 1000));

        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);
        auto derived_num_dims = derivedBnDesc.GetLengths().size();
        derived_layout        = (derived_num_dims == 5) ? miopenTensorNCDHW : miopenTensorNCHW;

        if(!derivedBnDesc.GetLayoutEnum() ||
           derivedBnDesc.GetLayoutEnum().value() != derived_layout)
        {
            derivedBnDesc = miopen::TensorDescriptor(
                derivedBnDesc.GetType(), derived_layout, derivedBnDesc.GetLengths());
        }

        scale   = tensor<AccDataType>(derived_layout, derivedBnDesc.GetLengths());
        shift   = tensor<AccDataType>(derived_layout, derivedBnDesc.GetLengths());
        runMean = tensor<AccDataType>(derived_layout, derivedBnDesc.GetLengths());
        runVar  = tensor<AccDataType>(derived_layout, derivedBnDesc.GetLengths());

        scale.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        shift.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        runMean.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        runVar.generate(uniform_unsigned_initializer<AccDataType>(2e-3, 1000));

        in_dev      = handle.Write(input.data);
        scale_dev   = handle.Write(scale.data);
        shift_dev   = handle.Write(shift.data);
        runMean_dev = handle.Write(runMean.data);
        runVar_dev  = handle.Write(runVar.data);
        out_dev     = handle.Write(output.data);

        // Initialize CPU verification tensors
        auto derived_lengths = derivedBnDesc.GetLengths();
        dy               = tensor<T>(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
        bnScale          = tensor<AccDataType>(derived_layout, derived_lengths);
        bnBias           = tensor<AccDataType>(derived_layout, derived_lengths);
        dScale_ref       = tensor<AccDataType>(derived_layout, derived_lengths);
        dBias_ref        = tensor<AccDataType>(derived_layout, derived_lengths);
        savedMean        = tensor<AccDataType>(derived_layout, derived_lengths);
        savedInvVar      = tensor<AccDataType>(derived_layout, derived_lengths);
        saveMean_ref     = tensor<AccDataType>(derived_layout, derived_lengths);
        saveVariance_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        runMean_ref      = tensor<AccDataType>(derived_layout, derived_lengths);
        runVariance_ref  = tensor<AccDataType>(derived_layout, derived_lengths);
        estMean          = tensor<AccDataType>(derived_layout, derived_lengths);
        estVariance      = tensor<AccDataType>(derived_layout, derived_lengths);

        if(std::is_same_v<T, bfloat16>)
            tolerance = 0.6;
        else
            tolerance = 0.5;
    }

    template <typename TensorType>
    void EnsureValidLayout(TensorType& t, miopenTensorLayout_t default_layout)
    {
        auto layout_t = t.desc.GetLayout_t();
        auto num_dims = t.desc.GetLengths().size();
        miopenTensorLayout_t valid_layout;
        if(num_dims == 5)
            valid_layout = miopenTensorNCDHW;
        else if(num_dims == 4)
            valid_layout = miopenTensorNCHW;
        else
            valid_layout = default_layout;

        if(layout_t == 0 ||
           (num_dims == 5 && layout_t != miopenTensorNCDHW && layout_t != miopenTensorNDHWC) ||
           (num_dims == 4 && layout_t != miopenTensorNCHW && layout_t != miopenTensorNHWC &&
            layout_t != miopenTensorCHWN && layout_t != miopenTensorNCHWc4 &&
            layout_t != miopenTensorNCHWc8 && layout_t != miopenTensorCHWNc4 &&
            layout_t != miopenTensorCHWNc8))
        {
            t.desc = miopen::TensorDescriptor(t.desc.GetType(), valid_layout, t.desc.GetLengths());
        }
        if(t.data.size() < t.desc.GetElementSpace())
            t.data.resize(t.desc.GetElementSpace());
    }

    void InitializeCPUTensors()
    {
        auto derived_lengths = derivedBnDesc.GetLengths();
        if(dy.data.empty())
            dy = tensor<T>(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
        if(bnScale.data.empty())
            bnScale = tensor<AccDataType>(derived_layout, derived_lengths);
        if(bnBias.data.empty())
            bnBias = tensor<AccDataType>(derived_layout, derived_lengths);
        if(dScale_ref.data.empty())
            dScale_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        if(dBias_ref.data.empty())
            dBias_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        if(savedMean.data.empty())
            savedMean = tensor<AccDataType>(derived_layout, derived_lengths);
        if(savedInvVar.data.empty())
            savedInvVar = tensor<AccDataType>(derived_layout, derived_lengths);
        if(saveMean_ref.data.empty())
            saveMean_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        if(saveVariance_ref.data.empty())
            saveVariance_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        if(runMean_ref.data.empty())
            runMean_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        if(runVariance_ref.data.empty())
            runVariance_ref = tensor<AccDataType>(derived_layout, derived_lengths);
        if(estMean.data.empty())
            estMean = tensor<AccDataType>(derived_layout, derived_lengths);
        if(estVariance.data.empty())
            estVariance = tensor<AccDataType>(derived_layout, derived_lengths);
    }

    void RunTest()
    {
        this->InitializeCPUTensors();
        const auto& test_case = this->GetParam();
        auto&& handle         = get_handle();

        switch(test_case.test_type)
        {
        case BN3DPerActTestType::ForwardTraining: {
            tensor<AccDataType> saveMean(this->derived_layout, this->derivedBnDesc.GetLengths());
            tensor<AccDataType> saveInvVar(this->derived_layout, this->derivedBnDesc.GetLengths());
            auto saveMean_dev   = handle.Write(saveMean.data);
            auto saveInvVar_dev = handle.Write(saveInvVar.data);

            miopenStatus_t status = miopenBatchNormalizationForwardTraining(&handle,
                                                                            miopenBNPerActivation,
                                                                            &alpha,
                                                                            &beta,
                                                                            &input.desc,
                                                                            in_dev.get(),
                                                                            &output.desc,
                                                                            out_dev.get(),
                                                                            &derivedBnDesc,
                                                                            scale_dev.get(),
                                                                            shift_dev.get(),
                                                                            expAvgFactor,
                                                                            runMean_dev.get(),
                                                                            runVar_dev.get(),
                                                                            epsilon,
                                                                            saveMean_dev.get(),
                                                                            saveInvVar_dev.get());

            ASSERT_EQ(status, miopenStatusSuccess);

            this->output.data  = handle.Read<T>(out_dev, this->output.data.size());
            saveMean.data      = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());
            saveInvVar.data    = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());
            this->runMean.data = handle.Read<AccDataType>(runMean_dev, this->runMean.data.size());
            this->runVar.data  = handle.Read<AccDataType>(runVar_dev, this->runVar.data.size());

            this->saveMean_ref     = saveMean;
            this->saveVariance_ref = saveInvVar;
            this->runMean_ref      = this->runMean;
            this->runVariance_ref  = this->runVar;

            this->EnsureValidLayout(this->input, miopenTensorNCDHW);
            this->EnsureValidLayout(this->output, miopenTensorNCDHW);
            this->EnsureValidLayout(this->out_ref, this->bn_layout);
            this->EnsureValidLayout(this->scale, this->derived_layout);
            this->EnsureValidLayout(this->shift, this->derived_layout);
            this->EnsureValidLayout(this->saveMean_ref, this->derived_layout);
            this->EnsureValidLayout(this->saveVariance_ref, this->derived_layout);
            this->EnsureValidLayout(this->runMean_ref, this->derived_layout);
            this->EnsureValidLayout(this->runVariance_ref, this->derived_layout);

            test::ComputeCPUBNFwdTrain<T, AccDataType>(*this);

            test::CompareTensor(this->output, this->out_ref, tolerance);
            test::CompareTensor(saveMean, this->saveMean_ref, tolerance);
            test::CompareTensor(saveInvVar, this->saveVariance_ref, tolerance);
            test::CompareTensor(this->runMean, this->runMean_ref, tolerance);
            test::CompareTensor(this->runVar, this->runVariance_ref, tolerance);
            break;
        }
        case BN3DPerActTestType::ForwardInferenceRecalc:
        case BN3DPerActTestType::ForwardInferenceUseEstimated: {
            void* p_est_mean =
                (test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated)
                    ? runMean_dev.get()
                    : nullptr;
            void* p_est_var =
                (test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated)
                    ? runVar_dev.get()
                    : nullptr;
            miopenStatus_t status = miopenBatchNormalizationForwardInference(&handle,
                                                                             miopenBNPerActivation,
                                                                             &alpha,
                                                                             &beta,
                                                                             &input.desc,
                                                                             in_dev.get(),
                                                                             &output.desc,
                                                                             out_dev.get(),
                                                                             &derivedBnDesc,
                                                                             scale_dev.get(),
                                                                             shift_dev.get(),
                                                                             p_est_mean,
                                                                             p_est_var,
                                                                             epsilon);

            ASSERT_EQ(status, miopenStatusSuccess);

            this->output.data = handle.Read<T>(out_dev, this->output.data.size());

            this->estMean            = this->runMean;
            this->estVariance        = this->runVar;
            this->useInverseVariance = false;

            this->EnsureValidLayout(this->input, miopenTensorNCDHW);
            this->EnsureValidLayout(this->output, miopenTensorNCDHW);
            this->EnsureValidLayout(this->out_ref, this->bn_layout);
            this->EnsureValidLayout(this->scale, this->derived_layout);
            this->EnsureValidLayout(this->shift, this->derived_layout);
            this->EnsureValidLayout(this->estMean, this->derived_layout);
            this->EnsureValidLayout(this->estVariance, this->derived_layout);

            if(test_case.test_type == BN3DPerActTestType::ForwardInferenceRecalc)
            {
                this->saveMean_ref     = this->runMean;
                this->saveVariance_ref = this->runVar;
                this->runMean_ref      = this->runMean;
                this->runVariance_ref  = this->runVar;
                this->EnsureValidLayout(this->input, miopenTensorNCDHW);
                this->EnsureValidLayout(this->out_ref, this->bn_layout);
                test::ComputeCPUBNFwdTrain<T, AccDataType>(*this);
                this->estMean            = this->saveMean_ref;
                this->estVariance        = this->saveVariance_ref;
                this->useInverseVariance = true;
            }
            test::ComputeCPUBNInference<T, AccDataType>(*this);
            test::CompareTensor(this->output, this->out_ref, tolerance);
            break;
        }
        case BN3DPerActTestType::BackwardRecalc: {
            tensor<T> dy_input(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
            dy_input.generate(uniform_signed_initializer<T>(2e-3, 1000));
            auto dy_dev = handle.Write(dy_input.data);

            tensor<T> dx_output(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
            tensor<AccDataType> dscale(this->derived_layout, this->derivedBnDesc.GetLengths());
            tensor<AccDataType> dshift(this->derived_layout, this->derivedBnDesc.GetLengths());
            auto dx_dev     = handle.Write(dx_output.data);
            auto dscale_dev = handle.Write(dscale.data);
            auto dshift_dev = handle.Write(dshift.data);

            miopenStatus_t status = miopenBatchNormalizationBackward(&handle,
                                                                     miopenBNPerActivation,
                                                                     &alpha,
                                                                     &beta,
                                                                     &alpha,
                                                                     &beta,
                                                                     &input.desc,
                                                                     in_dev.get(),
                                                                     &dy_input.desc,
                                                                     dy_dev.get(),
                                                                     &dx_output.desc,
                                                                     dx_dev.get(),
                                                                     &derivedBnDesc,
                                                                     scale_dev.get(),
                                                                     dscale_dev.get(),
                                                                     dshift_dev.get(),
                                                                     epsilon,
                                                                     nullptr,
                                                                     nullptr);

            ASSERT_EQ(status, miopenStatusSuccess);

            dx_output.data = handle.Read<T>(dx_dev, dx_output.data.size());
            dscale.data    = handle.Read<AccDataType>(dscale_dev, dscale.data.size());
            dshift.data    = handle.Read<AccDataType>(dshift_dev, dshift.data.size());

            this->dy          = dy_input;
            this->bnScale     = this->scale;
            this->bnBias      = this->shift;
            this->dScale_ref  = dscale;
            this->dBias_ref   = dshift;
            this->savedMean   = this->runMean;
            this->savedInvVar = this->runVar;

            this->EnsureValidLayout(this->input, miopenTensorNCDHW);
            this->EnsureValidLayout(this->dy, miopenTensorNCDHW);
            this->EnsureValidLayout(this->out_ref, miopenTensorNCDHW);
            this->EnsureValidLayout(this->bnScale, this->derived_layout);
            this->EnsureValidLayout(this->bnBias, this->derived_layout);
            this->EnsureValidLayout(this->dScale_ref, this->derived_layout);
            this->EnsureValidLayout(this->dBias_ref, this->derived_layout);

            this->saveMean_ref     = this->runMean;
            this->saveVariance_ref = this->runVar;
            this->runMean_ref      = this->runMean;
            this->runVariance_ref  = this->runVar;
            this->EnsureValidLayout(this->saveMean_ref, this->derived_layout);
            this->EnsureValidLayout(this->saveVariance_ref, this->derived_layout);

            test::ComputeCPUBNFwdTrain<T, AccDataType>(*this);
            this->savedMean   = this->saveMean_ref;
            this->savedInvVar = this->saveVariance_ref;
            this->EnsureValidLayout(this->savedMean, this->derived_layout);
            this->EnsureValidLayout(this->savedInvVar, this->derived_layout);

            test::ComputeCPUBNBwd<T, AccDataType>(*this);
            test::CompareTensor(dx_output, this->out_ref, tolerance);
            test::CompareTensor(dscale, this->dScale_ref, tolerance);
            test::CompareTensor(dshift, this->dBias_ref, tolerance);
            break;
        }
        case BN3DPerActTestType::BackwardUseSaved: {
            tensor<AccDataType> saveMean(this->derived_layout, this->derivedBnDesc.GetLengths());
            tensor<AccDataType> saveInvVar(this->derived_layout, this->derivedBnDesc.GetLengths());
            auto saveMean_dev   = handle.Write(saveMean.data);
            auto saveInvVar_dev = handle.Write(saveInvVar.data);

            miopenStatus_t status = miopenBatchNormalizationForwardTraining(&handle,
                                                                            miopenBNPerActivation,
                                                                            &alpha,
                                                                            &beta,
                                                                            &input.desc,
                                                                            in_dev.get(),
                                                                            &output.desc,
                                                                            out_dev.get(),
                                                                            &derivedBnDesc,
                                                                            scale_dev.get(),
                                                                            shift_dev.get(),
                                                                            expAvgFactor,
                                                                            runMean_dev.get(),
                                                                            runVar_dev.get(),
                                                                            epsilon,
                                                                            saveMean_dev.get(),
                                                                            saveInvVar_dev.get());

            ASSERT_EQ(status, miopenStatusSuccess);

            saveMean.data   = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());
            saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());

            tensor<T> dy_input(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
            dy_input.generate(uniform_signed_initializer<T>(2e-3, 1000));
            auto dy_dev = handle.Write(dy_input.data);

            tensor<T> dx_output(miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w});
            tensor<AccDataType> dscale(this->derived_layout, this->derivedBnDesc.GetLengths());
            tensor<AccDataType> dshift(this->derived_layout, this->derivedBnDesc.GetLengths());
            auto dx_dev     = handle.Write(dx_output.data);
            auto dscale_dev = handle.Write(dscale.data);
            auto dshift_dev = handle.Write(dshift.data);

            status = miopenBatchNormalizationBackward(&handle,
                                                      miopenBNPerActivation,
                                                      &alpha,
                                                      &beta,
                                                      &alpha,
                                                      &beta,
                                                      &input.desc,
                                                      in_dev.get(),
                                                      &dy_input.desc,
                                                      dy_dev.get(),
                                                      &dx_output.desc,
                                                      dx_dev.get(),
                                                      &derivedBnDesc,
                                                      scale_dev.get(),
                                                      dscale_dev.get(),
                                                      dshift_dev.get(),
                                                      epsilon,
                                                      saveMean_dev.get(),
                                                      saveInvVar_dev.get());

            ASSERT_EQ(status, miopenStatusSuccess);

            dx_output.data = handle.Read<T>(dx_dev, dx_output.data.size());
            dscale.data    = handle.Read<AccDataType>(dscale_dev, dscale.data.size());
            dshift.data    = handle.Read<AccDataType>(dshift_dev, dshift.data.size());

            this->dy          = dy_input;
            this->bnScale     = this->scale;
            this->bnBias      = this->shift;
            this->dScale_ref  = dscale;
            this->dBias_ref   = dshift;
            this->savedMean   = saveMean;
            this->savedInvVar = saveInvVar;

            this->EnsureValidLayout(this->input, miopenTensorNCDHW);
            this->EnsureValidLayout(this->dy, miopenTensorNCDHW);
            this->EnsureValidLayout(this->out_ref, miopenTensorNCDHW);
            this->EnsureValidLayout(this->bnScale, this->derived_layout);
            this->EnsureValidLayout(this->bnBias, this->derived_layout);
            this->EnsureValidLayout(this->dScale_ref, this->derived_layout);
            this->EnsureValidLayout(this->dBias_ref, this->derived_layout);
            this->EnsureValidLayout(this->savedMean, this->derived_layout);
            this->EnsureValidLayout(this->savedInvVar, this->derived_layout);

            test::ComputeCPUBNBwd<T, AccDataType>(*this);
            test::CompareTensor(dx_output, this->out_ref, tolerance);
            test::CompareTensor(dscale, this->dScale_ref, tolerance);
            test::CompareTensor(dshift, this->dBias_ref, tolerance);
            break;
        }
        }
    }
};

using GPU_Bn3dPerAct_FP32  = GPU_Bn3dPerAct<float>;
using GPU_Bn3dPerAct_FP16  = GPU_Bn3dPerAct<half_float::half>;
using GPU_Bn3dPerAct_BFP16 = GPU_Bn3dPerAct<bfloat16>;
using GPU_Bn3dPerAct_FP64  = GPU_Bn3dPerAct<double>;

} // namespace

TEST_P(GPU_Bn3dPerAct_FP32, Test) { this->RunTest(); }
TEST_P(GPU_Bn3dPerAct_FP16, Test) { this->RunTest(); }
TEST_P(GPU_Bn3dPerAct_BFP16, Test) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3dPerAct_FP32,
                         testing::ValuesIn(GetBN3DPerActTestCases(BN3DPerActTestSet::Full)));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3dPerAct_FP16,
                         testing::ValuesIn(GetBN3DPerActTestCases(BN3DPerActTestSet::Standard)));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3dPerAct_BFP16,
                         testing::ValuesIn(GetBN3DPerActTestCases(BN3DPerActTestSet::Standard)));
