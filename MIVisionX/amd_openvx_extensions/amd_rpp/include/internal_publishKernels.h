/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _PUBLISH_KERNELS_H_
#define _PUBLISH_KERNELS_H_

#include "internal_rpp.h"

#ifndef SHARED_PUBLIC
#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif
#endif

#define RPP_MAX_TENSOR_DIMS 5

extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);
vx_status ADD_KERNEL(std::function<vx_status(vx_context)>);
vx_status get_kernels_to_publish();

vx_status AbsoluteDifferencebatchPD_Register(vx_context);
vx_status AccumulatebatchPD_Register(vx_context);
vx_status AccumulateSquaredbatchPD_Register(vx_context);
vx_status AccumulateWeightedbatchPD_Register(vx_context);
vx_status AddbatchPD_Register(vx_context);
vx_status BitwiseANDbatchPD_Register(vx_context);
vx_status BitwiseNOTbatchPD_Register(vx_context);
vx_status BlendbatchPD_Register(vx_context);
vx_status BlurbatchPD_Register(vx_context);
vx_status BoxFilterbatchPD_Register(vx_context);
vx_status BrightnessbatchPD_Register(vx_context);
vx_status CannyEdgeDetector_Register(vx_context);
vx_status ChannelCombinebatchPD_Register(vx_context);
vx_status ChannelExtractbatchPD_Register(vx_context);
vx_status ColorTemperaturebatchPD_Register(vx_context);
vx_status ColorTwistbatchPD_Register(vx_context);
vx_status ContrastbatchPD_Register(vx_context);
vx_status CopybatchPD_Register(vx_context);
vx_status CropMirrorNormalizePD_Register(vx_context);
vx_status CropPD_Register(vx_context);
vx_status CustomConvolutionbatchPD_Register(vx_context);
vx_status DataObjectCopybatchPD_Register(vx_context);
vx_status DilatebatchPD_Register(vx_context);
vx_status ErodebatchPD_Register(vx_context);
vx_status ExclusiveORbatchPD_Register(vx_context);
vx_status ExposurebatchPD_Register(vx_context);
vx_status FastCornerDetector_Register(vx_context);
vx_status FisheyebatchPD_Register(vx_context);
vx_status FlipbatchPD_Register(vx_context);
vx_status FogbatchPD_Register(vx_context);
vx_status GammaCorrectionbatchPD_Register(vx_context);
vx_status GaussianFilterbatchPD_Register(vx_context);
vx_status GaussianImagePyramidbatchPD_Register(vx_context);
vx_status HarrisCornerDetector_Register(vx_context);
vx_status Histogram_Register(vx_context);
vx_status HistogramBalancebatchPD_Register(vx_context);
vx_status HistogramEqualizebatchPD_Register(vx_context);
vx_status HuebatchPD_Register(vx_context);
vx_status InclusiveORbatchPD_Register(vx_context);
vx_status JitterbatchPD_Register(vx_context);
vx_status LaplacianImagePyramid_Register(vx_context);
vx_status LensCorrectionbatchPD_Register(vx_context);
vx_status LocalBinaryPatternbatchPD_Register(vx_context);
vx_status LookUpTablebatchPD_Register(vx_context);
vx_status MagnitudebatchPD_Register(vx_context);
vx_status MaxbatchPD_Register(vx_context);
vx_status MeanStddev_Register(vx_context);
vx_status MedianFilterbatchPD_Register(vx_context);
vx_status MinbatchPD_Register(vx_context);
vx_status MinMaxLoc_Register(vx_context);
vx_status MultiplybatchPD_Register(vx_context);
vx_status NoisebatchPD_Register(vx_context);
vx_status NonLinearFilterbatchPD_Register(vx_context);
vx_status NonMaxSupressionbatchPD_Register(vx_context);
vx_status NopbatchPD_Register(vx_context);
vx_status PhasebatchPD_Register(vx_context);
vx_status PixelatebatchPD_Register(vx_context);
vx_status RainbatchPD_Register(vx_context);
vx_status RandomCropLetterBoxbatchPD_Register(vx_context);
vx_status RandomShadowbatchPD_Register(vx_context);
vx_status remap_Register(vx_context);
vx_status ResizebatchPD_Register(vx_context);
vx_status ResizeCropbatchPD_Register(vx_context);
vx_status ResizeCropMirrorPD_Register(vx_context);
vx_status ResizeMirrorNormalizeTensor_Register(vx_context);
vx_status Resizetensor_Register(vx_context);
vx_status RotatebatchPD_Register(vx_context);
vx_status SaturationbatchPD_Register(vx_context);
vx_status ScalebatchPD_Register(vx_context);
vx_status SequenceRearrangebatchPD_Register(vx_context);
vx_status SnowbatchPD_Register(vx_context);
vx_status SobelbatchPD_Register(vx_context);
vx_status SubtractbatchPD_Register(vx_context);
vx_status TensorAdd_Register(vx_context);
vx_status TensorLookup_Register(vx_context);
vx_status TensorMatrixMultiply_Register(vx_context);
vx_status TensorMultiply_Register(vx_context);
vx_status TensorSubtract_Register(vx_context);
vx_status ThresholdingbatchPD_Register(vx_context);
vx_status VignettebatchPD_Register(vx_context);
vx_status WarpAffinebatchPD_Register(vx_context);
vx_status WarpPerspectivebatchPD_Register(vx_context);

vx_status Blend_Register(vx_context);
vx_status Blur_Register(vx_context);
vx_status Brightness_Register(vx_context);
vx_status ColorTemperature_Register(vx_context);
vx_status ColorTwist_Register(vx_context);
vx_status Contrast_Register(vx_context);
vx_status Copy_Register(vx_context);
vx_status Crop_Register(vx_context);
vx_status CropMirrorNormalize_Register(vx_context);
vx_status Exposure_Register(vx_context);
vx_status FishEye_Register(vx_context);
vx_status Flip_Register(vx_context);
vx_status Fog_Register(vx_context);
vx_status GammaCorrection_Register(vx_context);
vx_status Glitch_Register(vx_context);
vx_status Hue_Register(vx_context);
vx_status Jitter_Register(vx_context);
vx_status LensCorrection_Register(vx_context);
vx_status Noise_Register(vx_context);
vx_status Nop_Register(vx_context);
vx_status Pixelate_Register(vx_context);
vx_status PreemphasisFilter_Register(vx_context);
vx_status Rain_Register(vx_context);
vx_status Resize_Register(vx_context);
vx_status ResizeCrop_Register(vx_context);
vx_status ResizeCropMirror_Register(vx_context);
vx_status ResizeMirrorNormalize_Register(vx_context);
vx_status Rotate_Register(vx_context);
vx_status Saturation_Register(vx_context);
vx_status SequenceRearrange_Register(vx_context);
vx_status Snow_Register(vx_context);
vx_status Vignette_Register(vx_context);
vx_status WarpAffine_Register(vx_context);
vx_status SequenceRearrange_Register(vx_context);
vx_status Spectrogram_Register(vx_context);
vx_status Downmix_Register(vx_context);
vx_status ToDecibels_Register(vx_context);
vx_status Resample_Register(vx_context);
vx_status TensorMulScalar_Register(vx_context);
vx_status TensorAddTensor_Register(vx_context);
vx_status NonSilentRegionDetection_Register(vx_context);
vx_status Slice_Register(vx_context);
vx_status Normalize_Register(vx_context);
vx_status MelFilterBank_Register(vx_context);
vx_status Transpose_Register(vx_context);

// kernel names
#define VX_KERNEL_RPP_NOPBATCHPD_NAME                           "org.rpp.NopbatchPD"
#define VX_KERNEL_RPP_COPYBATCHPD_NAME                          "org.rpp.CopybatchPD"
#define VX_KERNEL_RPP_BRIGHTNESSBATCHPD_NAME                    "org.rpp.BrightnessbatchPD"
#define VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD_NAME               "org.rpp.GammaCorrectionbatchPD"
#define VX_KERNEL_RPP_BLENDBATCHPD_NAME                         "org.rpp.BlendbatchPD"
#define VX_KERNEL_RPP_BLURBATCHPD_NAME                          "org.rpp.BlurbatchPD"
#define VX_KERNEL_RPP_CONTRASTBATCHPD_NAME                      "org.rpp.ContrastbatchPD"
#define VX_KERNEL_RPP_PIXELATEBATCHPD_NAME                      "org.rpp.PixelatebatchPD"
#define VX_KERNEL_RPP_JITTERBATCHPD_NAME                        "org.rpp.JitterbatchPD"
#define VX_KERNEL_RPP_SNOWBATCHPD_NAME                          "org.rpp.SnowbatchPD"
#define VX_KERNEL_RPP_NOISEBATCHPD_NAME                         "org.rpp.NoisebatchPD"
#define VX_KERNEL_RPP_RANDOMSHADOWBATCHPD_NAME                  "org.rpp.RandomShadowbatchPD"
#define VX_KERNEL_RPP_FOGBATCHPD_NAME                           "org.rpp.FogbatchPD"
#define VX_KERNEL_RPP_RAINBATCHPD_NAME                          "org.rpp.RainbatchPD"
#define VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD_NAME           "org.rpp.RandomCropLetterBoxbatchPD"
#define VX_KERNEL_RPP_EXPOSUREBATCHPD_NAME                      "org.rpp.ExposurebatchPD"
#define VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD_NAME              "org.rpp.HistogramBalancebatchPD"
#define VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD_NAME            "org.rpp.AbsoluteDifferencebatchPD"
#define VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD_NAME            "org.rpp.AccumulateWeightedbatchPD"
#define VX_KERNEL_RPP_ACCUMULATEBATCHPD_NAME                    "org.rpp.AccumulatebatchPD"
#define VX_KERNEL_RPP_ADDBATCHPD_NAME                           "org.rpp.AddbatchPD"
#define VX_KERNEL_RPP_SUBTRACTBATCHPD_NAME                      "org.rpp.SubtractbatchPD"
#define VX_KERNEL_RPP_MAGNITUDEBATCHPD_NAME                     "org.rpp.MagnitudebatchPD"
#define VX_KERNEL_RPP_MULTIPLYBATCHPD_NAME                      "org.rpp.MultiplybatchPD"
#define VX_KERNEL_RPP_PHASEBATCHPD_NAME                         "org.rpp.PhasebatchPD"
#define VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD_NAME             "org.rpp.AccumulateSquaredbatchPD"
#define VX_KERNEL_RPP_BITWISEANDBATCHPD_NAME                    "org.rpp.BitwiseANDbatchPD"
#define VX_KERNEL_RPP_BITWISENOTBATCHPD_NAME                    "org.rpp.BitwiseNOTbatchPD"
#define VX_KERNEL_RPP_EXCLUSIVEORBATCHPD_NAME                   "org.rpp.ExclusiveORbatchPD"
#define VX_KERNEL_RPP_INCLUSIVEORBATCHPD_NAME                   "org.rpp.InclusiveORbatchPD"
#define VX_KERNEL_RPP_HISTOGRAM_NAME                            "org.rpp.Histogram"
#define VX_KERNEL_RPP_THRESHOLDINGBATCHPD_NAME                  "org.rpp.ThresholdingbatchPD"
#define VX_KERNEL_RPP_MAXBATCHPD_NAME                           "org.rpp.MaxbatchPD"
#define VX_KERNEL_RPP_MINBATCHPD_NAME                           "org.rpp.MinbatchPD"
#define VX_KERNEL_RPP_MINMAXLOC_NAME                            "org.rpp.MinMaxLoc"
#define VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD_NAME             "org.rpp.HistogramEqualizebatchPD"
#define VX_KERNEL_RPP_MEANSTDDEV_NAME                           "org.rpp.MeanStddev"
#define VX_KERNEL_RPP_FLIPBATCHPD_NAME                          "org.rpp.FlipbatchPD"
#define VX_KERNEL_RPP_RESIZEBATCHPD_NAME                        "org.rpp.ResizebatchPD"
#define VX_KERNEL_RPP_RESIZECROPBATCHPD_NAME                    "org.rpp.ResizeCropbatchPD"
#define VX_KERNEL_RPP_ROTATEBATCHPD_NAME                        "org.rpp.RotatebatchPD"
#define VX_KERNEL_RPP_WARPAFFINEBATCHPD_NAME                    "org.rpp.WarpAffinebatchPD"
#define VX_KERNEL_RPP_FISHEYEBATCHPD_NAME                       "org.rpp.FisheyebatchPD"
#define VX_KERNEL_RPP_LENSCORRECTIONBATCHPD_NAME                "org.rpp.LensCorrectionbatchPD"
#define VX_KERNEL_RPP_SCALEBATCHPD_NAME                         "org.rpp.ScalebatchPD"
#define VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD_NAME               "org.rpp.WarpPerspectivebatchPD"
#define VX_KERNEL_RPP_DILATEBATCHPD_NAME                        "org.rpp.DilatebatchPD"
#define VX_KERNEL_RPP_ERODEBATCHPD_NAME                         "org.rpp.ErodebatchPD"
#define VX_KERNEL_RPP_HUEBATCHPD_NAME                           "org.rpp.HuebatchPD"
#define VX_KERNEL_RPP_SATURATIONBATCHPD_NAME                    "org.rpp.SaturationbatchPD"
#define VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD_NAME              "org.rpp.ColorTemperaturebatchPD"
#define VX_KERNEL_RPP_VIGNETTEBATCHPD_NAME                      "org.rpp.VignettebatchPD"
#define VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD_NAME                "org.rpp.ChannelExtractbatchPD"
#define VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD_NAME                "org.rpp.ChannelCombinebatchPD"
#define VX_KERNEL_RPP_LOOKUPTABLEBATCHPD_NAME                   "org.rpp.LookUpTablebatchPD"
#define VX_KERNEL_RPP_BOXFILTERBATCHPD_NAME                     "org.rpp.BoxFilterbatchPD"
#define VX_KERNEL_RPP_SOBELBATCHPD_NAME                         "org.rpp.SobelbatchPD"
#define VX_KERNEL_RPP_MEDIANFILTERBATCHPD_NAME                  "org.rpp.MedianFilterbatchPD"
#define VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD_NAME             "org.rpp.CustomConvolutionbatchPD"
#define VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD_NAME              "org.rpp.NonMaxSupressionbatchPD"
#define VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD_NAME                "org.rpp.GaussianFilterbatchPD"
#define VX_KERNEL_RPP_NONLINEARFILTERBATCHPD_NAME               "org.rpp.NonLinearFilterbatchPD"
#define VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD_NAME            "org.rpp.LocalBinaryPatternbatchPD"
#define VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD_NAME                "org.rpp.DataObjectCopybatchPD"
#define VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD_NAME          "org.rpp.GaussianImagePyramidbatchPD"
#define VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID_NAME                "org.rpp.LaplacianImagePyramid"
#define VX_KERNEL_RPP_CANNYEDGEDETECTOR_NAME                    "org.rpp.CannyEdgeDetector"
#define VX_KERNEL_RPP_HARRISCORNERDETECTOR_NAME                 "org.rpp.HarrisCornerDetector"
#define VX_KERNEL_RPP_FASTCORNERDETECTOR_NAME                   "org.rpp.FastCornerDetector"
#define VX_KERNEL_RPP_REMAP_NAME                                "org.rpp.remap"
#define VX_KERNEL_RPP_TENSORADD_NAME                            "org.rpp.TensorAdd"
#define VX_KERNEL_RPP_TENSORSUBTRACT_NAME                       "org.rpp.TensorSubtract"
#define VX_KERNEL_RPP_TENSORMULTIPLY_NAME                       "org.rpp.TensorMultiply"
#define VX_KERNEL_RPP_TENSORMATRIXMULTIPLY_NAME                 "org.rpp.TensorMatrixMultiply"
#define VX_KERNEL_RPP_TENSORLOOKUP_NAME                         "org.rpp.TensorLookup"
#define VX_KERNEL_RPP_COLORTWISTBATCHPD_NAME                    "org.rpp.ColorTwistbatchPD"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD_NAME           "org.rpp.CropMirrorNormalizebatchPD"
#define VX_KERNEL_RPP_CROPPD_NAME                               "org.rpp.CropPD"
#define VX_KERNEL_RPP_RESIZECROPMIRRORPD_NAME                   "org.rpp.ResizeCropMirrorPD"
#define VX_KERNEL_RPP_RESIZEMIRRORNORMALIZETENSOR_NAME          "org.rpp.ResizeMirrorNormalizeTensor"
#define VX_KERNEL_RPP_SEQUENCEREARRANGEBATCHPD_NAME             "org.rpp.SequenceRearrangebatchPD"
#define VX_KERNEL_RPP_RESIZETENSOR_NAME                         "org.rpp.Resizetensor"

//tensor
#define VX_KERNEL_RPP_BLEND_NAME                                "org.rpp.Blend"
#define VX_KERNEL_RPP_BLUR_NAME                                 "org.rpp.Blur"
#define VX_KERNEL_RPP_BRIGHTNESS_NAME                           "org.rpp.Brightness"
#define VX_KERNEL_RPP_COLORTEMPERATURE_NAME                     "org.rpp.ColorTemperature"
#define VX_KERNEL_RPP_COLORTWIST_NAME                           "org.rpp.ColorTwist"
#define VX_KERNEL_RPP_CONTRAST_NAME                             "org.rpp.Contrast"
#define VX_KERNEL_RPP_COPY_NAME                                 "org.rpp.Copy"
#define VX_KERNEL_RPP_CROP_NAME                                 "org.rpp.Crop"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZE_NAME                  "org.rpp.CropMirrorNormalize"
#define VX_KERNEL_RPP_EXPOSURE_NAME                             "org.rpp.Exposure"
#define VX_KERNEL_RPP_FISHEYE_NAME                              "org.rpp.FishEye"
#define VX_KERNEL_RPP_FLIP_NAME                                 "org.rpp.Flip"
#define VX_KERNEL_RPP_FOG_NAME                                  "org.rpp.Fog"
#define VX_KERNEL_RPP_GAMMACORRECTION_NAME                      "org.rpp.GammaCorrection"
#define VX_KERNEL_RPP_GLITCH_NAME                               "org.rpp.Glitch"
#define VX_KERNEL_RPP_HUE_NAME                                  "org.rpp.Hue"
#define VX_KERNEL_RPP_JITTER_NAME                               "org.rpp.Jitter"
#define VX_KERNEL_RPP_LENSCORRECTION_NAME                       "org.rpp.LensCorrection"
#define VX_KERNEL_RPP_NOISE_NAME                                "org.rpp.Noise"
#define VX_KERNEL_RPP_NOP_NAME                                  "org.rpp.Nop"
#define VX_KERNEL_RPP_RAIN_NAME                                 "org.rpp.Rain"
#define VX_KERNEL_RPP_RESIZE_NAME                               "org.rpp.Resize"
#define VX_KERNEL_RPP_RESIZECROP_NAME                           "org.rpp.ResizeCrop"
#define VX_KERNEL_RPP_RESIZECROPMIRROR_NAME                     "org.rpp.ResizeCropMirror"
#define VX_KERNEL_RPP_RESIZEMIRRORNORMALIZE_NAME                "org.rpp.ResizeMirrorNormalize"
#define VX_KERNEL_RPP_ROTATE_NAME                               "org.rpp.Rotate"
#define VX_KERNEL_RPP_SATURATION_NAME                           "org.rpp.Saturation"
#define VX_KERNEL_RPP_SEQUENCEREARRANGE_NAME                    "org.rpp.SequenceRearrange"
#define VX_KERNEL_RPP_SNOW_NAME                                 "org.rpp.Snow"
#define VX_KERNEL_RPP_PIXELATE_NAME                             "org.rpp.Pixelate"
#define VX_KERNEL_RPP_VIGNETTE_NAME                             "org.rpp.Vignette"
#define VX_KERNEL_RPP_WARPAFFINE_NAME                           "org.rpp.WarpAffine"
#define VX_KERNEL_RPP_BRIGHTNESS_NAME                           "org.rpp.Brightness"
#define VX_KERNEL_RPP_COPY_NAME                                 "org.rpp.Copy"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZE_NAME                  "org.rpp.CropMirrorNormalize"
#define VX_KERNEL_RPP_NOP_NAME                                  "org.rpp.Nop"
#define VX_KERNEL_RPP_RESIZE_NAME                               "org.rpp.Resize"
#define VX_KERNEL_RPP_SEQUENCEREARRANGE_NAME                    "org.rpp.SequenceRearrange"
#define VX_KERNEL_RPP_PREEMPHASISFILTER_NAME                    "org.rpp.PreemphasisFilter"
#define VX_KERNEL_RPP_SPECTROGRAM_NAME                          "org.rpp.Spectrogram"
#define VX_KERNEL_RPP_DOWNMIX_NAME                              "org.rpp.Downmix"
#define VX_KERNEL_RPP_TODECIBELS_NAME                           "org.rpp.ToDecibels"
#define VX_KERNEL_RPP_RESAMPLE_NAME                             "org.rpp.Resample"
#define VX_KERNEL_RPP_TENSORMULSCALAR_NAME                      "org.rpp.TensorMulScalar"
#define VX_KERNEL_RPP_TENSORADDTENSOR_NAME                      "org.rpp.TensorAddTensor"
#define VX_KERNEL_RPP_NONSILENTREGIONDETECTION_NAME             "org.rpp.NonSilentRegionDetection"
#define VX_KERNEL_RPP_SLICE_NAME                                "org.rpp.Slice"
#define VX_KERNEL_RPP_NORMALIZE_NAME                            "org.rpp.Normalize"
#define VX_KERNEL_RPP_MELFILTERBANK_NAME                        "org.rpp.MelFilterBank"
#define VX_KERNEL_RPP_TRANSPOSE_NAME                            "org.rpp.Transpose"

#endif //_AMDVX_EXT__PUBLISH_KERNELS_H_
