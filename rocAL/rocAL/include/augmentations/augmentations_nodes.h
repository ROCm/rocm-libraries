/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include "augmentations/geometry_augmentations/node_warp_affine.h"
#include "augmentations/color_augmentations/node_exposure.h"
#include "augmentations/color_augmentations/node_vignette.h"
#include "augmentations/effects_augmentations/node_jitter.h"
#include "augmentations/effects_augmentations/node_snp_noise.h"
#include "augmentations/effects_augmentations/node_snow.h"
#include "augmentations/effects_augmentations/node_rain.h"
#include "augmentations/color_augmentations/node_color_temperature.h"
#include "augmentations/effects_augmentations/node_fog.h"
#include "augmentations/effects_augmentations/node_pixelate.h"
#include "augmentations/geometry_augmentations/node_lens_correction.h"
#include "augmentations/color_augmentations/node_gamma.h"
#include "augmentations/geometry_augmentations/node_flip.h"
#include "augmentations/geometry_augmentations/node_crop_resize.h"
#include "augmentations/color_augmentations/node_brightness.h"
#include "augmentations/color_augmentations/node_contrast.h"
#include "augmentations/color_augmentations/node_blur.h"
#include "augmentations/geometry_augmentations/node_fisheye.h"
#include "augmentations/color_augmentations/node_blend.h"
#include "augmentations/geometry_augmentations/node_resize.h"
#include "augmentations/geometry_augmentations/node_rotate.h"
#include "augmentations/color_augmentations/node_color_twist.h"
#include "augmentations/color_augmentations/node_hue.h"
#include "augmentations/color_augmentations/node_saturation.h"
#include "augmentations/geometry_augmentations/node_crop_mirror_normalize.h"
#include "augmentations/geometry_augmentations/node_resize_mirror_normalize.h"
#include "augmentations/geometry_augmentations/node_resize_crop_mirror.h"
#include "augmentations/node_ssd_random_crop.h"
#include "augmentations/geometry_augmentations/node_crop.h"
#include "augmentations/geometry_augmentations/node_random_crop.h"
#include "augmentations/node_copy.h"
#include "augmentations/node_nop.h"
#include "augmentations/node_sequence_rearrange.h"
#include "augmentations/audio_augmentations/node_preemphasis_filter.h"
#include "augmentations/audio_augmentations/node_spectrogram.h"
#include "augmentations/audio_augmentations/node_to_decibels.h"
#include "augmentations/audio_augmentations/node_resample.h"
#include "augmentations/node_uniform_distribution.h"
#include "augmentations/node_normal_distribution.h"
#include "augmentations/arithmetic_augmentations/node_tensor_mul_scalar.h"
#include "augmentations/arithmetic_augmentations/node_tensor_add_tensor.h"
#include "augmentations/audio_augmentations/node_non_silent_region_detection.h"
#include "augmentations/geometry_augmentations/node_slice.h"
#include "augmentations/effects_augmentations/node_normalize.h"
#include "augmentations/audio_augmentations/node_mel_filter_bank.h"
#include "augmentations/geometry_augmentations/node_transpose.h"
