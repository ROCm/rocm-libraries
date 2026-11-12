// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The small-input-channel forward convolution kernels (CONV_SMALLC_FWD; CONV_SMALLC_BWD runs
// stride-1 backward data with few output-gradient channels as the forward conv over dy with the
// filter transposed and reflected), compiled by HIPRTC per catalog config. The pack defines:
//   CONV_SMALLC_T        operand and output type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_SMALLC_FMA      1 for the FMA kernel (256 threads), 0 for the MMA kernel
//   CONV_SMALLC_ATOM     16x16 MMA atom (ConvIgemmCore.h), MMA kernel only
//   CONV_SMALLC_CMAX     largest input channel count the config serves
//   CONV_SMALLC_KD, _R, _S       filter extent
//   CONV_SMALLC_SD, _SH, _SW     stride
//   CONV_SMALLC_DD, _DH, _DW     dilation
//   CONV_SMALLC_FM, _FN, _WM, _WQ, _WN   tile (ConvSmallcCore.h)
//   CONV_SMALLC_EP       epilogue (MMA kernel): 0 through LDS slots, 1 from registers (default 0)

#include "ConvIgemmCore.h"
#include "ConvSmallcCore.h"

#ifndef CONV_SMALLC_FMA
#define CONV_SMALLC_FMA 0
#endif
#ifndef CONV_SMALLC_EP
#define CONV_SMALLC_EP 0
#endif
#if CONV_SMALLC_FMA
#define CONV_SMALLC_BLOCK 256
#else
#define CONV_SMALLC_BLOCK \
    (CONV_SMALLC_WM * CONV_SMALLC_WQ * CONV_SMALLC_WN * CONV_SMALLC_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_SMALLC_BLOCK) ConvSmallc(SmallcArgs a)
{
#if CONV_SMALLC_FMA
    smallcConvFma<CONV_SMALLC_T,
                  CONV_SMALLC_T,
                  CONV_SMALLC_CMAX,
                  CONV_SMALLC_KD,
                  CONV_SMALLC_R,
                  CONV_SMALLC_S,
                  CONV_SMALLC_SD,
                  CONV_SMALLC_SH,
                  CONV_SMALLC_SW,
                  CONV_SMALLC_DD,
                  CONV_SMALLC_DH,
                  CONV_SMALLC_DW,
                  CONV_SMALLC_FM,
                  CONV_SMALLC_FN,
                  CONV_SMALLC_WM,
                  CONV_SMALLC_WQ,
                  CONV_SMALLC_WN>(a);
#else
    smallcConv<CONV_SMALLC_T,
               CONV_SMALLC_ATOM,
               CONV_SMALLC_T,
               CONV_SMALLC_CMAX,
               CONV_SMALLC_KD,
               CONV_SMALLC_R,
               CONV_SMALLC_S,
               CONV_SMALLC_SD,
               CONV_SMALLC_SH,
               CONV_SMALLC_SW,
               CONV_SMALLC_DD,
               CONV_SMALLC_DH,
               CONV_SMALLC_DW,
               CONV_SMALLC_FM,
               CONV_SMALLC_FN,
               CONV_SMALLC_WM,
               CONV_SMALLC_WQ,
               CONV_SMALLC_WN,
               CONV_SMALLC_EP>(a);
#endif
}

/// Formats the graph's filter into the rows this config reads.
extern "C" __global__ void __launch_bounds__(256) ConvSmallcFormat(SmallcFormatArgs a)
{
    smallcFormatWeights<CONV_SMALLC_T>(a);
}
