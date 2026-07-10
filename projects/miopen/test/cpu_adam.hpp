/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_CPU_ADAM_HPP
#define GUARD_CPU_ADAM_HPP

#include <miopen_utils/tensor_holder.hpp>

#include <common_utils/float_equal.hpp>

template <typename T1, typename T2>
void cpu_adam(tensor<T1>& params,
              tensor<T2>& grads,
              tensor<T1>& exp_avgs,
              tensor<T1>& exp_avg_sqs,
              tensor<T1>& max_exp_avg_sqs,
              float lr,
              float beta1,
              float beta2,
              float weight_decay,
              float eps,
              bool amsgrad,
              bool maximize,
              bool adamw,
              bool is_amp,
              int32_t grad_scale,
              bool found_inf,
              int32_t step_count,
              bool multi_threaded)
{
    if(is_amp && found_inf)
        return;

    const float one_minus_lr_by_weight_decay = adamw ? 1.0f - lr * weight_decay : 0.0f;
    const float one_minus_beta1              = 1.0 - beta1;
    const float one_minus_beta2              = 1.0 - beta2;
    const float inv_grad_scale               = 1.0f / static_cast<float>(grad_scale);
    const size_t n                           = params.GetSize();
    const size_t min_grain                   = multi_threaded ? 8 : n;

    miopen::par_for(n, min_grain, [&](int32_t i) {
        T1 param          = params[i];
        T1 exp_avg        = exp_avgs[i];
        T1 exp_avg_sq     = exp_avg_sqs[i];
        T1 max_exp_avg_sq = amsgrad ? max_exp_avg_sqs[i] : static_cast<T1>(0);

        float sqrt_max_exp_avg_sq = amsgrad ? sqrt(max_exp_avg_sq) : 0.0f;

        T1 grad = grads[i];
        if(maximize)
            grad = -grad;

        if(is_amp)
            grad *= inv_grad_scale;

        float grad_by_one_minus_beta1        = grad * one_minus_beta1;
        float square_grad_by_one_minus_beta2 = grad * grad * one_minus_beta2;

        for(int32_t step = 1; step <= step_count; ++step)
        {
            const float bias_correction1 = 1.0 - pow(beta1, step);
            const float bias_correction2 = 1.0 - pow(beta2, step);

            if(!miopen::float_equal_sentinel(weight_decay, 0.f))
            {
                if(adamw)
                    param *= one_minus_lr_by_weight_decay;
                else
                {
                    auto updated_grad = grad;
                    updated_grad += param * weight_decay;
                    grad_by_one_minus_beta1        = updated_grad * one_minus_beta1;
                    square_grad_by_one_minus_beta2 = updated_grad * updated_grad * one_minus_beta2;
                }
            }

            exp_avg    = exp_avg * beta1 + grad_by_one_minus_beta1;
            exp_avg_sq = exp_avg_sq * beta2 + square_grad_by_one_minus_beta2;

            float denom = 0.0f;
            if(amsgrad)
            {
                if(exp_avg_sq > max_exp_avg_sq)
                {
                    max_exp_avg_sq      = exp_avg_sq;
                    sqrt_max_exp_avg_sq = sqrt(max_exp_avg_sq);
                }

                denom = sqrt_max_exp_avg_sq / sqrt(bias_correction2) + eps;
            }
            else
            {
                denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
            }

            param -= (lr * exp_avg) / (bias_correction1 * denom);
        }

        params[i] = param;
    });
}

#endif
