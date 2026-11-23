// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused batchnorm inference and activation over a packed NCHW tensor:
//
//   out = act(scale[c] * (x - mean[c]) * inv_variance[c] + bias[c])
//
// HIP_PLUGIN_BNPW_TYPE is the io element type; the per-channel tensors are float.
// HIP_PLUGIN_BNPW_ACTIVATION selects act: 0 identity, 1 relu, 2 clipped relu (upper bound
// beta), 3 clamp (bounds alpha and beta).

#define HIP_PLUGIN_BNPW_IDENTITY 0
#define HIP_PLUGIN_BNPW_RELU 1
#define HIP_PLUGIN_BNPW_CLIPPED_RELU 2
#define HIP_PLUGIN_BNPW_CLAMP 3

extern "C" __global__ void BatchnormInferencePointwise(const HIP_PLUGIN_BNPW_TYPE* x,
                                                       const float* mean,
                                                       const float* invVariance,
                                                       const float* scale,
                                                       const float* bias,
                                                       HIP_PLUGIN_BNPW_TYPE* out,
                                                       float alpha,
                                                       float beta,
                                                       int elementCount,
                                                       int channels,
                                                       int spatialSize)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(index >= elementCount)
    {
        return;
    }

    const int channel = (index / spatialSize) % channels;
    float value
        = scale[channel] * (static_cast<float>(x[index]) - mean[channel]) * invVariance[channel]
          + bias[channel];

#if HIP_PLUGIN_BNPW_ACTIVATION == HIP_PLUGIN_BNPW_RELU
    value = fmaxf(value, 0.0f);
#elif HIP_PLUGIN_BNPW_ACTIVATION == HIP_PLUGIN_BNPW_CLIPPED_RELU
    value = fminf(fmaxf(value, 0.0f), beta);
#elif HIP_PLUGIN_BNPW_ACTIVATION == HIP_PLUGIN_BNPW_CLAMP
    value = fminf(fmaxf(value, alpha), beta);
#endif

    out[index] = static_cast<HIP_PLUGIN_BNPW_TYPE>(value);
}
