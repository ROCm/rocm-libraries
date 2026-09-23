// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MLO_SOFTMAXHOST_H_
#define MLO_SOFTMAXHOST_H_

#include <miopen/tensor.hpp>
#include <miopen/tensor_extra.hpp>

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#define NEGATIVE_INF_FP32 (-1e20)
#define NEGATIVE_INF_FP16 (-1e5)

template <typename T>
T logaddexp(T x, T y, T neg_inf)
{
    T a = std::max(x, y);
    T b = std::min(x, y);
    T c = b - a;

    return c <= neg_inf ? std::max(a, neg_inf) : std::max(T(a + log(T(1) + exp(b - a))), neg_inf);
}

template <typename Tgpu, typename Tcheck /* the data type used in CPU checkings (usually double) */>
int mloSoftmaxForwardRunHost(miopenTensorDescriptor_t inputTensor,
                             miopenTensorDescriptor_t outputTensor,
                             Tgpu* in,
                             Tcheck* outhost,
                             float alpha,
                             float beta,
                             miopenSoftmaxAlgorithm_t algo,
                             miopenSoftmaxMode_t mode)
{
    int n, c, h, w, in_nstr, in_cstr, in_hstr, in_wstr;
    int out_nstr, out_cstr, out_hstr, out_wstr;
    miopenGet4dTensorDescriptorLengths(inputTensor, &n, &c, &h, &w);
    miopenGet4dTensorDescriptorStrides(inputTensor, &in_nstr, &in_cstr, &in_hstr, &in_wstr);
    miopenGet4dTensorDescriptorStrides(outputTensor, &out_nstr, &out_cstr, &out_hstr, &out_wstr);

    Tcheck max_val = (sizeof(Tgpu) == 4) ? 3.402823466e+38f : 65504.;
    std::vector<Tcheck> channel_max((mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? n : n * h * w),
                                    static_cast<Tcheck>(-max_val));
    std::vector<Tcheck> results(n * c * h * w, static_cast<Tcheck>(0.0));

    int ret = 0;

    if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
    {
        for(int i = 0; i < n; i++)
        {
            if(algo == MIOPEN_SOFTMAX_FAST)
            {
                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] = static_cast<Tcheck>(
                                in[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr]);
                        }
            }
            else
            {
                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            channel_max[i] = std::max(
                                static_cast<Tcheck>(
                                    in[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr]),
                                channel_max[i]);
                        }

                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                static_cast<Tcheck>(
                                    in[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr]) -
                                channel_max[i];
                        }
            }

            if(algo == MIOPEN_SOFTMAX_LOG)
            {
                Tcheck neg_inf = static_cast<Tcheck>(
                    miopen::deref(inputTensor).GetType() == miopenHalf ? NEGATIVE_INF_FP16
                                                                       : NEGATIVE_INF_FP32);
                channel_max[i] = neg_inf;
                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            channel_max[i] = logaddexp(results[(i * c + j) * h * w + s0 * w + s1],
                                                       channel_max[i],
                                                       neg_inf);
                        }

                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            outhost[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr] =
                                alpha *
                                    (results[(i * c + j) * h * w + s0 * w + s1] - channel_max[i]) +
                                beta * outhost[i * out_nstr + j * out_cstr + s0 * out_hstr +
                                               s1 * out_wstr];
                        }
            }
            else
            {
                channel_max[i] = 0.0;
                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                exp(results[(i * c + j) * h * w + s0 * w + s1]);
                            channel_max[i] += results[(i * c + j) * h * w + s0 * w + s1];
                        }

                for(int j = 0; j < c; j++)
                    for(int s0 = 0; s0 < h; s0++)
                        for(int s1 = 0; s1 < w; s1++)
                        {
                            outhost[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr] =
                                alpha *
                                    (results[(i * c + j) * h * w + s0 * w + s1] / channel_max[i]) +
                                beta * outhost[i * out_nstr + j * out_cstr + s0 * out_hstr +
                                               s1 * out_wstr];
                        }
            }
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            for(int s0 = 0; s0 < h; s0++)
                for(int s1 = 0; s1 < w; s1++)
                {
                    if(algo == MIOPEN_SOFTMAX_FAST)
                    {
                        for(int j = 0; j < c; j++)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] = static_cast<Tcheck>(
                                in[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr]);
                        }
                    }
                    else
                    {
                        for(int j = 0; j < c; j++)
                        {
                            channel_max[i * h * w + s0 * w + s1] = std::max(
                                static_cast<Tcheck>(
                                    in[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr]),
                                channel_max[i * h * w + s0 * w + s1]);
                        }

                        for(int j = 0; j < c; j++)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                static_cast<Tcheck>(
                                    in[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr]) -
                                channel_max[i * h * w + s0 * w + s1];
                        }
                    }

                    if(algo == MIOPEN_SOFTMAX_LOG)
                    {
                        Tcheck neg_inf = static_cast<Tcheck>(
                            miopen::deref(inputTensor).GetType() == miopenHalf ? NEGATIVE_INF_FP16
                                                                               : NEGATIVE_INF_FP32);
                        channel_max[i * h * w + s0 * w + s1] = results[i * c * h * w + s0 * w + s1];
                        for(int j = 1; j < c; j++)
                        {
                            channel_max[i * h * w + s0 * w + s1] =
                                logaddexp(results[(i * c + j) * h * w + s0 * w + s1],
                                          channel_max[i * h * w + s0 * w + s1],
                                          neg_inf);
                        }

                        for(int j = 0; j < c; j++)
                        {
                            outhost[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr] =
                                alpha * (results[(i * c + j) * h * w + s0 * w + s1] -
                                         channel_max[i * h * w + s0 * w + s1]) +
                                beta * outhost[i * out_nstr + j * out_cstr + s0 * out_hstr +
                                               s1 * out_wstr];
                        }
                    }
                    else
                    {
                        channel_max[i * h * w + s0 * w + s1] = 0.0;
                        for(int j = 0; j < c; j++)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                exp(results[(i * c + j) * h * w + s0 * w + s1]);
                            channel_max[i * h * w + s0 * w + s1] +=
                                results[(i * c + j) * h * w + s0 * w + s1];
                        }

                        for(int j = 0; j < c; j++)
                        {
                            outhost[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr] =
                                alpha * (results[(i * c + j) * h * w + s0 * w + s1] /
                                         channel_max[i * h * w + s0 * w + s1]) +
                                beta * outhost[i * out_nstr + j * out_cstr + s0 * out_hstr +
                                               s1 * out_wstr];
                        }
                    }
                }
        }
    }

    return ret;
}

template <typename Tgpu /* the data type used in GPU computations (usually half) */,
          typename Tcheck /* the data type used in CPU checkings (usually double) */>
int mloSoftmaxBackwardRunHost(miopenTensorDescriptor_t dInputTensor,
                              miopenTensorDescriptor_t dOutputTensor,
                              Tgpu* out,
                              Tgpu* dout,
                              Tcheck* dinhost,
                              float alpha,
                              float beta,
                              miopenSoftmaxAlgorithm_t algo,
                              miopenSoftmaxMode_t mode)
{
    int n, c, h, w, in_nstr, in_cstr, in_hstr, in_wstr;
    int out_nstr, out_cstr, out_hstr, out_wstr;
    miopenGet4dTensorDescriptorLengths(dOutputTensor, &n, &c, &h, &w);
    miopenGet4dTensorDescriptorStrides(dInputTensor, &in_nstr, &in_cstr, &in_hstr, &in_wstr);
    miopenGet4dTensorDescriptorStrides(dOutputTensor, &out_nstr, &out_cstr, &out_hstr, &out_wstr);

    std::vector<Tcheck> channel_dot((mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? n : n * h * w),
                                    static_cast<Tcheck>(0.0));
    std::vector<Tcheck> results(n * c * h * w, static_cast<Tcheck>(0.0));

    int ret = 0;

    for(int i = 0; i < n; i++)
    {
        if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        {
            for(int j = 0; j < c; j++)
                for(int s0 = 0; s0 < h; s0++)
                    for(int s1 = 0; s1 < w; s1++)
                    {
                        if(algo == MIOPEN_SOFTMAX_LOG)
                        {
                            channel_dot[i] += static_cast<Tcheck>(
                                dout[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr]);
                        }
                        else
                        {
                            channel_dot[i] +=
                                static_cast<Tcheck>(out[i * out_nstr + j * out_cstr +
                                                        s0 * out_hstr + s1 * out_wstr]) *
                                static_cast<Tcheck>(dout[i * out_nstr + j * out_cstr +
                                                         s0 * out_hstr + s1 * out_wstr]);
                        }
                    }

            for(int j = 0; j < c; j++)
                for(int s0 = 0; s0 < h; s0++)
                    for(int s1 = 0; s1 < w; s1++)
                    {
                        if(algo == MIOPEN_SOFTMAX_LOG)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                static_cast<Tcheck>(dout[i * out_nstr + j * out_cstr +
                                                         s0 * out_hstr + s1 * out_wstr]) -
                                channel_dot[i] * std::exp(out[i * out_nstr + j * out_cstr +
                                                              s0 * out_hstr + s1 * out_wstr]);
                        }
                        else
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                static_cast<Tcheck>(dout[i * out_nstr + j * out_cstr +
                                                         s0 * out_hstr + s1 * out_wstr]) -
                                channel_dot[i];

                            results[(i * c + j) * h * w + s0 * w + s1] *= static_cast<Tcheck>(
                                out[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr]);
                        }
                        dinhost[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr] =
                            alpha * results[(i * c + j) * h * w + s0 * w + s1] +
                            beta * dinhost[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr];
                    }
        }
        else
        {
            for(int s0 = 0; s0 < h; s0++)
                for(int s1 = 0; s1 < w; s1++)
                {
                    for(int j = 0; j < c; j++)
                    {
                        if(algo == MIOPEN_SOFTMAX_LOG)
                        {
                            channel_dot[i * h * w + s0 * w + s1] += static_cast<Tcheck>(
                                dout[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr]);
                        }
                        else
                        {
                            channel_dot[i * h * w + s0 * w + s1] +=
                                static_cast<Tcheck>(out[i * out_nstr + j * out_cstr +
                                                        s0 * out_hstr + s1 * out_wstr]) *
                                static_cast<Tcheck>(dout[i * out_nstr + j * out_cstr +
                                                         s0 * out_hstr + s1 * out_wstr]);
                        }
                    }

                    for(int j = 0; j < c; j++)
                    {
                        if(algo == MIOPEN_SOFTMAX_LOG)
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                static_cast<Tcheck>(dout[i * out_nstr + j * out_cstr +
                                                         s0 * out_hstr + s1 * out_wstr]) -
                                channel_dot[i * h * w + s0 * w + s1] *
                                    std::exp(out[i * out_nstr + j * out_cstr + s0 * out_hstr +
                                                 s1 * out_wstr]);
                        }
                        else
                        {
                            results[(i * c + j) * h * w + s0 * w + s1] =
                                static_cast<Tcheck>(dout[i * out_nstr + j * out_cstr +
                                                         s0 * out_hstr + s1 * out_wstr]) -
                                channel_dot[i * h * w + s0 * w + s1];

                            results[(i * c + j) * h * w + s0 * w + s1] *= static_cast<Tcheck>(
                                out[i * out_nstr + j * out_cstr + s0 * out_hstr + s1 * out_wstr]);
                        }
                        dinhost[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr] =
                            alpha * results[(i * c + j) * h * w + s0 * w + s1] +
                            beta * dinhost[i * in_nstr + j * in_cstr + s0 * in_hstr + s1 * in_wstr];
                    }
                }
        }
    }

    return ret;
}

#endif
