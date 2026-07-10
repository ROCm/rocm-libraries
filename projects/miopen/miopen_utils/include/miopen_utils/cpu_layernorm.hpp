// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef GUARD_CPU_CONV_HPP
#define GUARD_CPU_CONV_HPP

#include <miopen_utils/tensor_holder.hpp>

template <class T>
void cpu_layernorm_forward(tensor<T> input,
                           tensor<T> weight,
                           tensor<T> bias,
                           tensor<T>& ref_output,
                           tensor<T>& ref_mean,
                           tensor<T>& ref_rstd,
                           float eps,
                           int32_t dim,
                           miopenNormMode_t mode,
                           bool use_multithread = false)
{
    auto layout   = input.desc.GetLayoutEnum();
    size_t stride = 1;
    if(dim > 1 && layout.has_value() &&
       (layout.value() == miopenTensorNHWC || layout.value() == miopenTensorNDHWC))
    {
        stride = input.desc.GetLengths()[1]; // stride = C
    }

    auto dims         = input.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i < dim)
        {
            if(!(stride > 1 && i == 1))
            {
                outer_size *= dims[i];
            }
        }
        else
        {
            inner_size *= dims[i];
        }
    }

    size_t min_grain = use_multithread ? 8 : outer_size;
    miopen::par_for(outer_size, min_grain, [&](int32_t o) {
        miopen::ford(stride)([&](int32_t s) {
            double mean_v = 0.0;
            double var_v  = 0.0;

            miopen::ford(inner_size)([&](int32_t i) {
                double tmp = static_cast<double>(input[o * inner_size * stride + i * stride + s]);
                mean_v += tmp;
                var_v += tmp * tmp;
            });

            mean_v        = mean_v / inner_size;
            var_v         = var_v / inner_size - mean_v * mean_v;
            double rstd_v = 1.0 / sqrt(var_v + eps);

            ref_mean[o * stride + s] = static_cast<T>(mean_v);
            ref_rstd[o * stride + s] = static_cast<T>(rstd_v);

            miopen::ford(inner_size)([&](int32_t i) {
                double weight_v =
                    (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1.0 : static_cast<float>(weight[i]);
                double bias_v =
                    (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 0.0 : static_cast<float>(bias[i]);

                ref_output[o * inner_size * stride + i * stride + s] = static_cast<T>(
                    (static_cast<double>(input[o * inner_size * stride + i * stride + s]) -
                     mean_v) *
                        rstd_v * weight_v +
                    bias_v);
            });
        });
    });
}

template <class T>
void cpu_layernorm_backward(tensor<T> dy,
                            tensor<T> x,
                            tensor<T> weight,
                            tensor<T> mean,
                            tensor<T> rstd,
                            tensor<T>& ref_dx,
                            int32_t dim,
                            miopenNormMode_t mode,
                            bool use_multithread = false)
{
    auto layout   = dy.desc.GetLayoutEnum();
    size_t stride = 1;
    if(dim > 1 && (layout == miopenTensorNHWC || layout == miopenTensorNDHWC))
    {
        stride = dy.desc.GetLengths()[1]; // stride = C
    }

    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i < dim)
        {
            if(!(stride > 1 && i == 1))
            {
                outer_size *= dims[i];
            }
        }
        else
        {
            inner_size *= dims[i];
        }
    }

    size_t min_grain = use_multithread ? 8 : outer_size;
    miopen::par_for(outer_size, min_grain, [&](int32_t o) {
        miopen::ford(stride)([&](int32_t s) {
            double sum_dy_weight   = 0.0;
            double sum_dy_weight_x = 0.0;

            miopen::ford(inner_size)([&](int32_t i) {
                double pweight =
                    (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1.0 : static_cast<double>(weight[i]);
                double pdy = (dy.GetSize() != 0)
                                 ? static_cast<double>(dy[o * inner_size * stride + i * stride + s])
                                 : 0.0;
                double px  = static_cast<double>(x[o * inner_size * stride + i * stride + s]);

                sum_dy_weight += pdy * pweight;
                sum_dy_weight_x += pdy * px * pweight;
            });

            double scale = 1.0 / static_cast<double>(inner_size);
            double prstd = static_cast<double>(rstd[o * stride + s]);
            double pmean = static_cast<double>(mean[o * stride + s]);
            double a = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
            double b = prstd * sum_dy_weight * scale - a * pmean;

            miopen::ford(inner_size)([&](int32_t i) {
                double pweight =
                    (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1.0 : static_cast<double>(weight[i]);
                double pdy = (dy.GetSize() != 0)
                                 ? static_cast<double>(dy[o * inner_size * stride + i * stride + s])
                                 : 0.0;
                double val = prstd * pdy * pweight -
                             a * static_cast<double>(x[o * inner_size * stride + i * stride + s]) -
                             b;

                ref_dx[o * inner_size * stride + i * stride + s] = static_cast<T>(val);
            });
        });
    });
}

template <class T>
void cpu_layernorm_backward_weight_bias(tensor<T> dy,
                                        tensor<T> x,
                                        tensor<T> mean,
                                        tensor<T> rstd,
                                        tensor<T>& ref_dw,
                                        tensor<T>& ref_db,
                                        int32_t dim,
                                        bool use_multithread = false)
{
    auto layout   = dy.desc.GetLayoutEnum();
    size_t stride = 1;
    if(dim > 1 && (layout == miopenTensorNHWC || layout == miopenTensorNDHWC))
    {
        stride = dy.desc.GetLengths()[1]; // stride = C
    }

    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i < dim)
        {
            if(!(stride > 1 && i == 1))
            {
                outer_size *= dims[i];
            }
        }
        else
        {
            inner_size *= dims[i];
        }
    }

    size_t min_grain = use_multithread ? 8 : inner_size;
    miopen::par_for(inner_size, min_grain, [&](int32_t i) {
        double sum_dw = 0.0;
        double sum_db = 0.0;

        miopen::ford(stride)([&](int32_t s) {
            miopen::ford(outer_size)([&](int32_t o) {
                double prstd = static_cast<double>(rstd[o * stride + s]);
                double pmean = static_cast<double>(mean[o * stride + s]);
                double pdy   = (dy.GetSize() != 0)
                                   ? static_cast<double>(dy[o * inner_size * stride + i * stride + s])
                                   : 0;
                double px    = static_cast<double>(x[o * inner_size * stride + i * stride + s]);

                sum_dw += pdy * (px - pmean) * prstd;
                sum_db += pdy;
            });
        });

        ref_dw[i] = sum_dw;
        ref_db[i] = sum_db;
    });
}

#endif
