// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The FFT convolution kernels (CONV_FFT_FWD, CONV_FFT_BWD, CONV_FFT_WRW), compiled by HIPRTC per
// catalog config: one source, nine entry points, all taking ConvFftArgs by value.
//   ConvFftData      data windows -> spectra (x forward / backward-weights, dy backward data and
//                    the zero-padded dy windows of backward weights)
//   ConvFftFilter    filter -> conjugated spectra (forward, backward data)
//   ConvFftProduct   per-bin complex product over channels (forward, backward data)
//   ConvFftOutput    product spectra -> y or dx tiles
//   ConvFftWgrad     per-bin product summed over tiles (backward weights), chunk by chunk
//   ConvFftWgradOut  weight-gradient spectra -> dw
//   ConvFftDepthwise       depthwise forward / backward data: transform, filter multiply, inverse
//   ConvFftDepthwiseWgrad  depthwise backward weights: per-partition sums of conj(DY) X per bin
//   ConvFftDepthwiseFold   depthwise backward weights: partitions summed in order
// The pack defines:
//   CONV_FFT_T    element type of the tensors: float, _Float16, or unsigned short (bf16 bits)
//   CONV_FFT_TH   transform rows (1 for a 1-D transform), CONV_FFT_TW transform columns
//   CONV_FFT_PB   tiles (or filters) a transform block
//   CONV_FFT_TB   threads of a transform block (at least one a row pair: PB TH / 2, or PB / 2)
//   CONV_FFT_GN   output channels of a product block tile (16, 32 or 64; 256 threads)
//   CONV_FFT_TM   tiles a product thread (4, or 8 with 64 output channels; default 4)

#include "ConvFftCore.h"

#ifndef CONV_FFT_T
#error "CONV_FFT_T must name the element type"
#endif
#ifndef CONV_FFT_TM
#define CONV_FFT_TM 4
#endif

extern "C" __global__ void __launch_bounds__(CONV_FFT_TB) ConvFftData(ConvFftArgs a)
{
    fftDataTransform<CONV_FFT_T, CONV_FFT_TH, CONV_FFT_TW, CONV_FFT_PB>(a);
}

extern "C" __global__ void __launch_bounds__(CONV_FFT_TB) ConvFftFilter(ConvFftArgs a)
{
    fftFilterTransform<CONV_FFT_T, CONV_FFT_TH, CONV_FFT_TW, CONV_FFT_PB>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvFftProduct(ConvFftArgs a)
{
    fftProduct<CONV_FFT_GN, CONV_FFT_TM>(a);
}

extern "C" __global__ void __launch_bounds__(CONV_FFT_TB) ConvFftOutput(ConvFftArgs a)
{
    fftOutputInverse<CONV_FFT_T, CONV_FFT_TH, CONV_FFT_TW, CONV_FFT_PB>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvFftWgrad(ConvFftArgs a)
{
    fftWgradProduct<CONV_FFT_GN>(a);
}

extern "C" __global__ void __launch_bounds__(CONV_FFT_TB) ConvFftWgradOut(ConvFftArgs a)
{
    fftWgradInverse<CONV_FFT_T, CONV_FFT_TH, CONV_FFT_TW, CONV_FFT_PB>(a);
}

extern "C" __global__ void __launch_bounds__(CONV_FFT_TB) ConvFftDepthwise(ConvFftArgs a)
{
    fftDepthwise<CONV_FFT_T, CONV_FFT_TH, CONV_FFT_TW, CONV_FFT_PB>(a);
}

// The depthwise backward-weights kernel holds two spectra a tile; a config whose two spectra
// exceed the 64 KiB every target has compiles it as a stub (the pack never launches it: its plan
// declines that config), so the module's other kernels still build.
#define CONV_FFT_SPECTRA_BYTES (CONV_FFT_PB * CONV_FFT_TH * (CONV_FFT_TW / 2 + 1) * 8)

extern "C" __global__ void __launch_bounds__(CONV_FFT_TB) ConvFftDepthwiseWgrad(ConvFftArgs a)
{
#if 2 * CONV_FFT_SPECTRA_BYTES <= 65536
    fftDepthwiseWgrad<CONV_FFT_T, CONV_FFT_TH, CONV_FFT_TW, CONV_FFT_PB, CONV_FFT_TB>(a);
#else
    (void)a;
    __builtin_trap();
#endif
}

extern "C" __global__ void __launch_bounds__(256) ConvFftDepthwiseFold(ConvFftArgs a)
{
    fftDepthwiseFold<CONV_FFT_TH, CONV_FFT_TW>(a);
}
