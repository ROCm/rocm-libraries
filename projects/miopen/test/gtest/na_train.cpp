// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "../fusionHost.hpp"
#include "compare_helper.hpp"
#include <miopen/stringutils.hpp>
#include <sstream>

#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

namespace {
constexpr float MIO_BN_TEST_EXPAVGFACTOR = 0.99f;
constexpr float MIO_BN_TEST_EPSILON      = 1e-5; // FLT_EPSILON
constexpr int batch_factor               = 4;

using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);
ptr_FusionPlanDesc GetManagedFusionPlanDesc(miopenTensorDescriptor_t inputDesc)
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
    return ptr_FusionPlanDesc{fusePlanDesc};
}

ptr_FusionPlanArgs GetManageFusionPlanArgs()
{
    miopenOperatorArgs_t fusionArgs;
    miopenCreateOperatorArgs(&fusionArgs);
    return ptr_FusionPlanArgs{fusionArgs};
}

ptr_ActivationDesc GetManagedActivDesc()
{
    miopenActivationDescriptor_t activdesc;
    miopenCreateActivationDescriptor(&activdesc);
    return ptr_ActivationDesc{activdesc};
}

template <class T, class U>
struct verify_fwd_batchnorm_spatial_activ
{
    tensor<T> x;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenFusionPlanDescriptor_t fusionplan;
    miopenFusionOpDescriptor_t bNormFwdOp;
    miopenFusionOpDescriptor_t activFwdOp;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    double alpha;
    double beta;

    verify_fwd_batchnorm_spatial_activ(miopenFusionPlanDescriptor_t pfwdfusionplan,
                                       const tensor<T>& pinput,
                                       miopenActivationDescriptor_t pactivDesc,
                                       const tensor<U>& pbnscale,
                                       const tensor<U>& pbnbias,
                                       miopenFusionOpDescriptor_t pbNormFwdOp,
                                       miopenFusionOpDescriptor_t pactivFwdOp)
    {
        x            = pinput;
        activDesc    = pactivDesc;
        bnscale      = pbnscale;
        bnbias       = pbnbias;
        fusionplan   = pfwdfusionplan;
        bNormFwdOp   = pbNormFwdOp;
        activFwdOp   = pactivFwdOp;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNSpatial);
        ssn = ssc = ssh = ssw        = 0;
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
        alpha                        = 1.;
        beta                         = 0.;
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {
        auto bout = x;
        std::fill(bout.begin(), bout.end(), 0.);
        auto aout = x;
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        batchNormSpatialHostFwdTrain(x,
                                     bout,
                                     bnscale,
                                     bnbias,
                                     epsilon,
                                     expAvgFactor,
                                     savedMean,
                                     savedInvVar,
                                     runMean,
                                     runVar);
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        return std::make_tuple(aout, runMean, runVar, savedMean, savedInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();

        auto baout = x;
        std::fill(baout.begin(), baout.end(), 0.);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        auto in_dev              = handle.Write(x.data);
        auto out_dev             = handle.Write(baout.data);
        auto bnscale_dev         = handle.Write(bnscale.data);
        auto bnbias_dev          = handle.Write(bnbias.data);
        auto savedMean_dev       = handle.Write(savedMean.data);
        auto savedInvVar_dev     = handle.Write(savedInvVar.data);
        auto runningMean_dev     = handle.Write(runMean.data);
        auto runningVariance_dev = handle.Write(runVar.data);
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormForward(ptr_fusionargs.get(),
                                        bNormFwdOp,
                                        &alpha,
                                        &beta,
                                        bnscale_dev.get(),
                                        bnbias_dev.get(),
                                        savedMean_dev.get(),
                                        savedInvVar_dev.get(),
                                        runningMean_dev.get(),
                                        runningVariance_dev.get(),
                                        expAvgFactor,
                                        epsilon);
        miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activFwdOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

        auto lclxdesc = x.desc;
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                &lclxdesc,
                                in_dev.get(),
                                &lclxdesc,
                                out_dev.get(),
                                ptr_fusionargs.get());

        baout.data       = handle.Read<T>(out_dev, baout.data.size());
        runMean.data     = handle.Read<U>(runningMean_dev, runMean.data.size());
        runVar.data      = handle.Read<U>(runningVariance_dev, runVar.data.size());
        savedMean.data   = handle.Read<U>(savedMean_dev, savedMean.data.size());
        savedInvVar.data = handle.Read<U>(savedInvVar_dev, savedInvVar.data.size());

        return std::make_tuple(baout, runMean, runVar, savedMean, savedInvVar);
    }

    void fail() const
    {
        GTEST_FAIL() << "Forward Train Spatial Batch Normalization + Activation: " << std::endl
                     << "Input tensor: " << x.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_bwd_batchnorm_spatial_activ
{
    tensor<T> x;
    tensor<T> y;
    tensor<T> dy;
    tensor<U> savedMean;
    tensor<U> savedInvVar;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenFusionPlanDescriptor_t fusionplan;
    miopenFusionOpDescriptor_t bNormBwdOp;
    miopenFusionOpDescriptor_t activBwdOp;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    std::size_t input_n, input_c, input_h, input_w;
    double alpha;
    double beta;

    verify_bwd_batchnorm_spatial_activ(miopenFusionPlanDescriptor_t pbwdfusionplan,
                                       const tensor<T>& pdyin,
                                       const tensor<T>& pxin,
                                       const tensor<T>& pyin,
                                       miopenActivationDescriptor_t pactivDesc,
                                       const tensor<U>& pbnscale,
                                       const tensor<U>& pbnbias,
                                       const tensor<U>& psavedMean,
                                       const tensor<U>& psavedInvVar,
                                       miopenFusionOpDescriptor_t pbNormBwdOp,
                                       miopenFusionOpDescriptor_t pactivBwdOp)
    {
        x            = pxin;
        y            = pyin;
        dy           = pdyin;
        savedMean    = psavedMean;
        savedInvVar  = psavedInvVar;
        activDesc    = pactivDesc;
        bnscale      = pbnscale;
        bnbias       = pbnbias;
        fusionplan   = pbwdfusionplan;
        bNormBwdOp   = pbNormBwdOp;
        activBwdOp   = pactivBwdOp;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNSpatial);
        ssn = ssc = ssh = ssw        = 0;
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
        input_n = input_c = input_h = input_w        = 0;
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(x.desc.GetLengths());
        alpha                                        = 1.;
        beta                                         = 0.;
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {
        auto dx = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        auto bout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(bout.begin(), bout.end(), 0.);

        auto aout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        batchNormActivSpatialHostBwdTrain(activ_mode,
                                          activ_gamma,
                                          activ_beta,
                                          activ_alpha,
                                          x,
                                          dy,
                                          y,
                                          dx,
                                          bnscale,
                                          bnbias,
                                          dgamma,
                                          dbeta,
                                          savedMean,
                                          savedInvVar);

        return std::make_tuple(dx, dgamma, dbeta);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();
        auto dx       = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        auto xin_dev         = handle.Write(x.data);
        auto dxout_dev       = handle.Write(dx.data);
        auto yin_dev         = handle.Write(y.data);
        auto dyin_dev        = handle.Write(dy.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);
        auto dgamma_dev      = handle.Write(dgamma.data);
        auto dbeta_dev       = handle.Write(dbeta.data);
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormBackward(ptr_fusionargs.get(),
                                         bNormBwdOp,
                                         &alpha,
                                         &beta,
                                         xin_dev.get(),
                                         bnscale_dev.get(),
                                         bnbias_dev.get(),
                                         dgamma_dev.get(),
                                         dbeta_dev.get(),
                                         savedMean_dev.get(),
                                         savedInvVar_dev.get());
        miopenSetOpArgsActivBackward(ptr_fusionargs.get(),
                                     activBwdOp,
                                     &alpha,
                                     &beta,
                                     yin_dev.get(),
                                     nullptr,
                                     activ_alpha,
                                     activ_beta,
                                     activ_gamma);

        auto lcldydesc = dy.desc;
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                &lcldydesc,
                                dyin_dev.get(),
                                &lcldydesc,
                                dxout_dev.get(),
                                ptr_fusionargs.get());

        dx.data     = handle.Read<T>(dxout_dev, dx.data.size());
        dgamma.data = handle.Read<U>(dgamma_dev, dgamma.data.size());
        dbeta.data  = handle.Read<U>(dbeta_dev, dbeta.data.size());
        return std::make_tuple(dx, dgamma, dbeta);
    }

    void fail() const { GTEST_FAIL(); }
};

template <class T, class U>
struct verify_fwd_batchnorm_peract_activ
{
    tensor<T> x;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenFusionPlanDescriptor_t fusionplan;
    miopenFusionOpDescriptor_t bNormFwdOp;
    miopenFusionOpDescriptor_t activFwdOp;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    std::size_t input_n, input_c, input_h, input_w;
    double alpha;
    double beta;

    verify_fwd_batchnorm_peract_activ(miopenFusionPlanDescriptor_t pfwdfusionplan,
                                      const tensor<T>& pinput,
                                      miopenActivationDescriptor_t pactivDesc,
                                      const tensor<U>& pbnscale,
                                      const tensor<U>& pbnbias,
                                      miopenFusionOpDescriptor_t pbNormFwdOp,
                                      miopenFusionOpDescriptor_t pactivFwdOp)
    {
        x            = pinput;
        activDesc    = pactivDesc;
        bnscale      = pbnscale;
        bnbias       = pbnbias;
        fusionplan   = pfwdfusionplan;
        bNormFwdOp   = pbNormFwdOp;
        activFwdOp   = pactivFwdOp;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNPerActivation);
        ssn = ssc = ssh = ssw = 0;
        input_n = input_c = input_h = input_w        = 0;
        std::tie(ssn, ssc, ssh, ssw)                 = miopen::tien<4>(derivedBnDesc.GetLengths());
        alpha                                        = 1.;
        beta                                         = 0.;
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(x.desc.GetLengths());
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {
        auto bout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(bout.begin(), bout.end(), 0.);

        auto aout = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        batchNormPerActHostFwdTrain(x,
                                    bout,
                                    bnscale,
                                    bnbias,
                                    epsilon,
                                    expAvgFactor,
                                    savedMean,
                                    savedInvVar,
                                    runMean,
                                    runVar);
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        return std::make_tuple(aout, runMean, runVar, savedMean, savedInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();
        auto baout    = x;
        std::fill(baout.begin(), baout.end(), 0.);

        auto savedMean   = tensor<U>{ssn, ssc, ssh, ssw};
        auto savedInvVar = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(savedMean.begin(), savedMean.end(), 0.);
        std::fill(savedInvVar.begin(), savedInvVar.end(), 0.);

        auto runMean = tensor<U>{ssn, ssc, ssh, ssw};
        auto runVar  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(runMean.begin(), runMean.end(), 0.);
        std::fill(runVar.begin(), runVar.end(), 0.);

        auto in_dev              = handle.Write(x.data);
        auto out_dev             = handle.Write(baout.data);
        auto bnscale_dev         = handle.Write(bnscale.data);
        auto bnbias_dev          = handle.Write(bnbias.data);
        auto savedMean_dev       = handle.Write(savedMean.data);
        auto savedInvVar_dev     = handle.Write(savedInvVar.data);
        auto runningMean_dev     = handle.Write(runMean.data);
        auto runningVariance_dev = handle.Write(runVar.data);
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormForward(ptr_fusionargs.get(),
                                        bNormFwdOp,
                                        &alpha,
                                        &beta,
                                        bnscale_dev.get(),
                                        bnbias_dev.get(),
                                        savedMean_dev.get(),
                                        savedInvVar_dev.get(),
                                        runningMean_dev.get(),
                                        runningVariance_dev.get(),
                                        expAvgFactor,
                                        epsilon);
        miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activFwdOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

        auto lclxdesc = x.desc;
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                &lclxdesc,
                                in_dev.get(),
                                &lclxdesc,
                                out_dev.get(),
                                ptr_fusionargs.get());

        baout.data       = handle.Read<T>(out_dev, baout.data.size());
        runMean.data     = handle.Read<U>(runningMean_dev, runMean.data.size());
        runVar.data      = handle.Read<U>(runningVariance_dev, runVar.data.size());
        savedMean.data   = handle.Read<U>(savedMean_dev, savedMean.data.size());
        savedInvVar.data = handle.Read<U>(savedInvVar_dev, savedInvVar.data.size());

        return std::make_tuple(baout, runMean, runVar, savedMean, savedInvVar);
    }

    void fail() const
    {
        GTEST_FAIL() << "Forward Train Per Activation Batch Normalization + Activation: "
                     << std::endl
                     << "Input tensor: " << x.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_bwd_batchnorm_peract_activ
{
    tensor<T> x;
    tensor<T> y;
    tensor<T> dy;
    tensor<U> savedMean;
    tensor<U> savedInvVar;
    miopenActivationDescriptor_t activDesc{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    miopenFusionPlanDescriptor_t fusionplan;
    miopenFusionOpDescriptor_t bNormBwdOp;
    miopenFusionOpDescriptor_t activBwdOp;
    miopen::TensorDescriptor derivedBnDesc{};
    double epsilon;
    double expAvgFactor;
    std::size_t ssn, ssc, ssh, ssw;
    double alpha;
    double beta;
    std::size_t input_n, input_c, input_h, input_w;

    verify_bwd_batchnorm_peract_activ(miopenFusionPlanDescriptor_t pbwdfusionplan,
                                      const tensor<T>& pdyin,
                                      const tensor<T>& pxin,
                                      const tensor<T>& pyin,
                                      miopenActivationDescriptor_t pactivDesc,
                                      const tensor<U>& pbnscale,
                                      const tensor<U>& pbnbias,
                                      const tensor<U>& psavedMean,
                                      const tensor<U>& psavedInvVar,
                                      miopenFusionOpDescriptor_t pbNormBwdOp,
                                      miopenFusionOpDescriptor_t pactivBwdOp)
    {
        x            = pxin;
        y            = pyin;
        dy           = pdyin;
        savedMean    = psavedMean;
        savedInvVar  = psavedInvVar;
        activDesc    = pactivDesc;
        bnscale      = pbnscale;
        bnbias       = pbnbias;
        fusionplan   = pbwdfusionplan;
        bNormBwdOp   = pbNormBwdOp;
        activBwdOp   = pactivBwdOp;
        epsilon      = MIO_BN_TEST_EPSILON;
        expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x.desc, miopenBNPerActivation);
        ssn = ssc = ssh = ssw = 0;
        input_n = input_c = input_h = input_w        = 0;
        std::tie(ssn, ssc, ssh, ssw)                 = miopen::tien<4>(derivedBnDesc.GetLengths());
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(x.desc.GetLengths());
        alpha                                        = 1.;
        beta                                         = 0.;
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {
        auto dx = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        batchNormActivPerActHostBwdTrain(activ_mode,
                                         activ_gamma,
                                         activ_beta,
                                         activ_alpha,
                                         x,
                                         dy,
                                         y,
                                         dx,
                                         bnscale,
                                         bnbias,
                                         dgamma,
                                         dbeta,
                                         savedMean,
                                         savedInvVar);

        return std::make_tuple(dx, dgamma, dbeta);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();
        auto dx       = tensor<T>{input_n, input_c, input_h, input_w};
        std::fill(dx.begin(), dx.end(), 0.);

        auto dgamma = tensor<U>{ssn, ssc, ssh, ssw};
        auto dbeta  = tensor<U>{ssn, ssc, ssh, ssw};
        std::fill(dgamma.begin(), dgamma.end(), 0.);
        std::fill(dbeta.begin(), dbeta.end(), 0.);

        auto xin_dev         = handle.Write(x.data);
        auto dxout_dev       = handle.Write(dx.data);
        auto yin_dev         = handle.Write(y.data);
        auto dyin_dev        = handle.Write(dy.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);
        auto dgamma_dev      = handle.Write(dgamma.data);
        auto dbeta_dev       = handle.Write(dbeta.data);
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormBackward(ptr_fusionargs.get(),
                                         bNormBwdOp,
                                         &alpha,
                                         &beta,
                                         xin_dev.get(),
                                         bnscale_dev.get(),
                                         bnbias_dev.get(),
                                         dgamma_dev.get(),
                                         dbeta_dev.get(),
                                         savedMean_dev.get(),
                                         savedInvVar_dev.get());
        miopenSetOpArgsActivBackward(ptr_fusionargs.get(),
                                     activBwdOp,
                                     &alpha,
                                     &beta,
                                     yin_dev.get(),
                                     nullptr,
                                     activ_alpha,
                                     activ_beta,
                                     activ_gamma);

        auto lcldydesc = dy.desc;
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                &lcldydesc,
                                dyin_dev.get(),
                                &lcldydesc,
                                dxout_dev.get(),
                                ptr_fusionargs.get());

        dx.data     = handle.Read<T>(dxout_dev, dx.data.size());
        dgamma.data = handle.Read<U>(dgamma_dev, dgamma.data.size());
        dbeta.data  = handle.Read<U>(dbeta_dev, dbeta.data.size());
        return std::make_tuple(dx, dgamma, dbeta);
    }

    void fail() const { GTEST_FAIL(); }
};

static std::string transform_mode(std::string s)
{
    return miopen::RemovePrefix(miopen::ToUpper(s), "MIOPENACTIVATION");
}

using TestCase = std::tuple<int, std::vector<int>, double, double, double, std::string>;

auto GenCases(int batchNormMode, bool full = false)
{
    if(full)
    {
        return ::testing::Combine(::testing::ValuesIn({batchNormMode}),
                                  ::testing::ValuesIn((batchNormMode == 1)
                                                          ? get_bn_spatial_inputs(batch_factor)
                                                          : get_bn_peract_inputs(batch_factor)),
                                  ::testing::ValuesIn({double{0.5}}),
                                  ::testing::ValuesIn({double{0.5}}),
                                  ::testing::ValuesIn({double{0.5}}),
                                  ::testing::ValuesIn({std::string{"MIOPENACTIVATIONRELU"},
                                                       std::string{"MIOPENACTIVATIONLOGISTIC"},
                                                       std::string{"MIOPENACTIVATIONABS"}}));
    }
    return ::testing::Combine(::testing::ValuesIn({0}),
                              ::testing::ValuesIn(std::set<std::vector<int>>{{16, 32, 8, 8}}),
                              ::testing::ValuesIn({double{0.5}}),
                              ::testing::ValuesIn({double{0.5}}),
                              ::testing::ValuesIn({double{0.5}}),
                              ::testing::ValuesIn({std::string{"MIOPENACTIVATIONRELU"}}));
}

auto GetSmokePeractCases()
{
    static auto cases = GenCases(0);
    return cases;
}

auto GetSmokeSpatialCases()
{
    static auto cases = GenCases(1);
    return cases;
}

auto GetFullPeractCases()
{
    static auto cases = GenCases(0, true);
    return cases;
}

auto GetFullSpatialCases()
{
    static auto cases = GenCases(1, true);
    return cases;
}

template <class T>
struct na_fusion_test : public testing::TestWithParam<TestCase>
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    ptr_ActivationDesc ptr_activdesc = nullptr;

    miopenActivationMode_t activ_mode = miopenActivationRELU;
    std::string amode;
    miopenBatchNormMode_t bnmode{};
    int batchnormMode = 1;

    const uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;
    double alpha = 0., beta = 0., gamma = 0.;
    double tolerance = 80.f;

    void SetUp() override
    {
        std::vector<int> nchw{};
        std::tie(batchnormMode, nchw, alpha, beta, gamma, amode) = GetParam();
        input = tensor<T>{nchw[0], nchw[1], nchw[2], nchw[3]};
        input.generate(tensor_elem_gen_integer{max_value});
        // NOLINTBEGIN(*-braces-around-statements)
        if(amode == "PASSTHRU")
            activ_mode = miopenActivationPASTHRU;
        else if(amode == "LOGISTIC")
            activ_mode = miopenActivationLOGISTIC;
        else if(amode == "TANH")
            activ_mode = miopenActivationTANH;
        else if(amode == "RELU")
            activ_mode = miopenActivationRELU;
        else if(amode == "SOFTRELU")
            activ_mode = miopenActivationSOFTRELU;
        else if(amode == "ABS")
            activ_mode = miopenActivationABS;
        else if(amode == "POWER")
            activ_mode = miopenActivationPOWER;
        else if(amode == "CLIPPEDRELU")
            activ_mode = miopenActivationCLIPPEDRELU;
        else if(amode == "LEAKYRELU")
            activ_mode = miopenActivationLEAKYRELU;
        else if(amode == "ELU")
            activ_mode = miopenActivationELU;
        amode = transform_mode(amode);
        // NOLINTEND(*-braces-around-statements)
    }

    void Run()
    {

        std::size_t input_n, input_c, input_h, input_w;
        std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());
        tolerance     = std::min(80 * double(input.desc.GetElementSize()),
                             1280 * sqrt(double(input.desc.GetElementSize())));
        ptr_activdesc = GetManagedActivDesc();
        miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, alpha, beta, gamma);
        auto&& handle = get_handle();

        miopenFusionOpDescriptor_t bNormFwdOp = nullptr;
        miopenFusionOpDescriptor_t activFwdOp = nullptr;
        auto ptr_fwdfusionplan                = GetManagedFusionPlanDesc(&input.desc);

        miopenFusionOpDescriptor_t bNormBwdOp = nullptr;
        miopenFusionOpDescriptor_t activBwdOp = nullptr;
        auto ptr_bwdfusionplan                = GetManagedFusionPlanDesc(&input.desc);

        std::size_t ssn, ssc, ssh, ssw;
        if(batchnormMode == 1)
        {
            bnmode = miopenBNSpatial;

            miopen::TensorDescriptor derivedBnDesc{};
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bnmode);
            std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
            scale =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            shift =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            miopenCreateOpBatchNormForward(ptr_fwdfusionplan.get(), &bNormFwdOp, bnmode, true);
            miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);
            miopenStatus_t miopenFwdError =
                miopenCompileFusionPlan(&handle, ptr_fwdfusionplan.get());
            if(miopenFwdError != miopenStatusSuccess)
            {
                std::cerr << "BatchNorm+Activation Spatial Forward Training plan not supported."
                          << std::endl;
                return;
            }

            auto fwdTrain = test_helpers::CompareResults(
                verify_fwd_batchnorm_spatial_activ<T, PREC_TYPE>{ptr_fwdfusionplan.get(),
                                                                 input,
                                                                 ptr_activdesc.get(),
                                                                 scale,
                                                                 shift,
                                                                 bNormFwdOp,
                                                                 activFwdOp},
                tolerance);
            //        Tuple returns: (aout, runMean, runVar, savedMean, savedInvVar);
            auto y_in        = std::get<0>(fwdTrain.second);
            auto savedMean   = std::get<3>(fwdTrain.second);
            auto savedInvVar = std::get<4>(fwdTrain.second);
            auto dyin        = tensor<T>{input_n, input_c, input_h, input_w}.generate(
                tensor_elem_gen_integer{max_value});

            miopenCreateOpBatchNormBackward(ptr_bwdfusionplan.get(), &bNormBwdOp, bnmode);
            miopenCreateOpActivationBackward(ptr_bwdfusionplan.get(), &activBwdOp, activ_mode);
            miopenStatus_t miopenBwdError =
                miopenCompileFusionPlan(&handle, ptr_bwdfusionplan.get());
            if(miopenBwdError != miopenStatusSuccess)
            {
                std::cerr << "BatchNorm+Activation Spatial Backward Training plan not supported."
                          << std::endl;
                return;
            }
            test_helpers::CompareResults(
                verify_bwd_batchnorm_spatial_activ<T, PREC_TYPE>{ptr_bwdfusionplan.get(),
                                                                 dyin,
                                                                 input,
                                                                 y_in,
                                                                 ptr_activdesc.get(),
                                                                 scale,
                                                                 shift,
                                                                 savedMean,
                                                                 savedInvVar,
                                                                 bNormBwdOp,
                                                                 activBwdOp},
                tolerance);
        }
        else if(batchnormMode == 0)
        {
            bnmode = miopenBNPerActivation;
            miopen::TensorDescriptor derivedBnDesc{};
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bnmode);
            std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());
            scale =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            shift =
                tensor<PREC_TYPE>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{max_value});
            miopenCreateOpBatchNormForward(ptr_fwdfusionplan.get(), &bNormFwdOp, bnmode, true);
            miopenCreateOpActivationForward(ptr_fwdfusionplan.get(), &activFwdOp, activ_mode);
            miopenStatus_t miopenFwdError =
                miopenCompileFusionPlan(&handle, ptr_fwdfusionplan.get());
            if(miopenFwdError != miopenStatusSuccess)
            {
                std::cerr
                    << "BatchNorm+Activation Per Activation Forward Training plan not supported."
                    << std::endl;
                return;
            }

            auto fwdTrain = test_helpers::CompareResults(
                verify_fwd_batchnorm_peract_activ<T, PREC_TYPE>{ptr_fwdfusionplan.get(),
                                                                input,
                                                                ptr_activdesc.get(),
                                                                scale,
                                                                shift,
                                                                bNormFwdOp,
                                                                activFwdOp},
                tolerance);
            auto y_in        = std::get<0>(fwdTrain.second);
            auto savedMean   = std::get<3>(fwdTrain.second);
            auto savedInvVar = std::get<4>(fwdTrain.second);
            auto dyin        = tensor<T>{input_n, input_c, input_h, input_w}.generate(
                tensor_elem_gen_integer{max_value});

            miopenCreateOpBatchNormBackward(ptr_bwdfusionplan.get(), &bNormBwdOp, bnmode);
            miopenCreateOpActivationBackward(ptr_bwdfusionplan.get(), &activBwdOp, activ_mode);
            miopenStatus_t miopenBwdError =
                miopenCompileFusionPlan(&handle, ptr_bwdfusionplan.get());
            if(miopenBwdError != miopenStatusSuccess)
            {
                std::cerr
                    << "BatchNorm+Activation Per Activation Backward Training plan not supported."
                    << std::endl;
                return;
            }
            test_helpers::CompareResults(
                verify_bwd_batchnorm_peract_activ<T, PREC_TYPE>{ptr_bwdfusionplan.get(),
                                                                dyin,
                                                                input,
                                                                y_in,
                                                                ptr_activdesc.get(),
                                                                scale,
                                                                shift,
                                                                savedMean,
                                                                savedInvVar,
                                                                bNormBwdOp,
                                                                activBwdOp},
                tolerance);
        }
    }
};

struct TestNameGenerator
{
    std::string operator()(const ::testing::TestParamInfo<TestCase>& param_info)
    {
        std::stringstream ss{};
        auto replace_dot = [](double value) // assuming there's only one
        {
            std::string str{std::to_string(value)};
            auto i = str.find('.');
            if(i != std::string::npos)
                str[i] = '_';
            return str;
        };

        auto print_nchw = [](std::vector<int> const& vec) {
            std::stringstream vec_ss{};
            for(auto el : vec)
            {
                vec_ss << std::to_string(el) << "_";
            }
            return vec_ss.str();
        };

        ss << "nchw_" << print_nchw(std::get<1>(param_info.param)) << "_alpha_"
           << replace_dot(std::get<2>(param_info.param)) << "_beta_"
           << replace_dot(std::get<3>(param_info.param)) << "_gamma_"
           << replace_dot(std::get<4>(param_info.param)) << "_amode_"
           << std::get<5>(param_info.param);
        return ss.str();
    }
};
template <typename T>
class na_fusion_test_peract : public na_fusion_test<T>
{
};

template <typename T>
class na_fusion_test_spatial : public na_fusion_test<T>
{
};
} // namespace

using GPU_na_fusion_test_bn_peract_FP16 = na_fusion_test_peract<half_float::half>;
using GPU_na_fusion_test_bn_peract_FP32 = na_fusion_test_peract<float>;

TEST_P(GPU_na_fusion_test_bn_peract_FP32, TestFloat32) { Run(); }
TEST_P(GPU_na_fusion_test_bn_peract_FP16, TestFloat16) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_na_fusion_test_bn_peract_FP32,
                         GetSmokePeractCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_na_fusion_test_bn_peract_FP16,
                         GetSmokePeractCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_na_fusion_test_bn_peract_FP32,
                         GetFullPeractCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_na_fusion_test_bn_peract_FP16,
                         GetFullPeractCases(),
                         TestNameGenerator{});

using GPU_na_fusion_test_bn_spatial_FP16 = na_fusion_test_spatial<half_float::half>;
using GPU_na_fusion_test_bn_spatial_FP32 = na_fusion_test_spatial<float>;

TEST_P(GPU_na_fusion_test_bn_spatial_FP32, TestFloat32) { Run(); }
TEST_P(GPU_na_fusion_test_bn_spatial_FP16, TestFloat16) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_na_fusion_test_bn_spatial_FP32,
                         GetSmokeSpatialCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_na_fusion_test_bn_spatial_FP16,
                         GetSmokeSpatialCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_na_fusion_test_bn_spatial_FP32,
                         GetFullSpatialCases(),
                         TestNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_na_fusion_test_bn_spatial_FP16,
                         GetFullSpatialCases(),
                         TestNameGenerator{});
