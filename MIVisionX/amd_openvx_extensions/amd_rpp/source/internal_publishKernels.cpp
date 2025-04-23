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

#include "internal_publishKernels.h"
#include "vx_ext_rpp.h"

/**********************************************************************
  PUBLIC FUNCTION for OpenVX user defined functions
**********************************************************************/
extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    vx_status status = VX_SUCCESS;

    STATUS_ERROR_CHECK(get_kernels_to_publish());
    STATUS_ERROR_CHECK(Kernel_List->PUBLISH(context));

    return status;
}

/************************************************************************************************************
Add All Kernels to the Kernel List
*************************************************************************************************************/
vx_status get_kernels_to_publish()
{
    vx_status status = VX_SUCCESS;

    Kernel_List = new Kernellist(MAX_KERNELS);
    STATUS_ERROR_CHECK(ADD_KERNEL(BrightnessbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GammaCorrectionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(BlendbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(BlurbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ContrastbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(PixelatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(JitterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SnowbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NoisebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(RandomShadowbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(FogbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(RainbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(RandomCropLetterBoxbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ExposurebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(HistogramBalancebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(AbsoluteDifferencebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(AccumulateWeightedbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(AccumulatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(AddbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SubtractbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MagnitudebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MultiplybatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(PhasebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(AccumulateSquaredbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(BitwiseANDbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(BitwiseNOTbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ExclusiveORbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(InclusiveORbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Histogram_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ThresholdingbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MaxbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MinbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MinMaxLoc_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(HistogramEqualizebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MeanStddev_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(FlipbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeCropbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(RotatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(WarpAffinebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(FisheyebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(LensCorrectionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ScalebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(WarpPerspectivebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(DilatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ErodebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(HuebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SaturationbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorTemperaturebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(VignettebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ChannelExtractbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ChannelCombinebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(LookUpTablebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(BoxFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SobelbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MedianFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CustomConvolutionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NonMaxSupressionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GaussianFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NonLinearFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(LocalBinaryPatternbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(DataObjectCopybatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GaussianImagePyramidbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(LaplacianImagePyramid_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CannyEdgeDetector_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(HarrisCornerDetector_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(FastCornerDetector_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(remap_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorAdd_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorSubtract_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorMultiply_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorMatrixMultiply_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorLookup_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorTwistbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CropMirrorNormalizePD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CropPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeCropMirrorPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CopybatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NopbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeMirrorNormalizeTensor_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SequenceRearrangebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Resizetensor_Register));

    //tensor 
    STATUS_ERROR_CHECK(ADD_KERNEL(Blend_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Blur_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Brightness_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorTemperature_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorTwist_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Contrast_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Copy_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Crop_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CropMirrorNormalize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Exposure_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(FishEye_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Flip_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Fog_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GammaCorrection_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Glitch_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Hue_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Jitter_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(LensCorrection_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Noise_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Nop_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Pixelate_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(PreemphasisFilter_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Rain_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Resize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeCrop_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeCropMirror_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeMirrorNormalize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Rotate_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Saturation_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SequenceRearrange_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Snow_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Vignette_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(WarpAffine_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Spectrogram_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Downmix_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ToDecibels_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Resample_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorMulScalar_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorAddTensor_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NonSilentRegionDetection_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Slice_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Normalize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MelFilterBank_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Transpose_Register));
    return status;
}

/************************************************************************************************************
Add Kernels to the Kernel List
*************************************************************************************************************/
vx_status ADD_KERNEL(std::function<vx_status(vx_context)> func)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(Kernel_List->ADD(func));
    return status;
}
