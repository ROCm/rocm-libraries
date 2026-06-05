/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "benchmarks_common.h"

int main()
{
    // Initialize RPP HOST backend handle
    rppHandle_t handle;
#if ENABLE_PARALLEL_THREADS
    rppCreate(&handle, 1, NUM_THREADS, nullptr, RPP_HOST_BACKEND);
#else
    rppCreate(&handle, 1, 1, nullptr, RPP_HOST_BACKEND);
#endif

    int batchSizeGray = 0, maxWidthGray = 0, maxHeightGray = 0;
    int batchSizeRGB  = 0, maxWidthRGB  = 0, maxHeightRGB  = 0;

    vector<Mat> imgsGray = loadBatchImages(GRAY_IMAGE_PATH, batchSizeGray, maxWidthGray, maxHeightGray, false);
    vector<Mat> imgsRGB  = loadBatchImages(RGB_IMAGE_PATH,  batchSizeRGB,  maxWidthRGB,  maxHeightRGB,  true);

    if (imgsGray.empty() && imgsRGB.empty())
    {
        cerr << "No images found in the dataset directories!" << endl;
        rppDestroy(handle, RPP_HOST_BACKEND);
        return -1;
    }

    // Initialize global image metadata for Excel output
    if (!imgsGray.empty())
    {
        ostringstream oss;
        oss << maxWidthGray << "x" << maxHeightGray;
        grayImageSize = oss.str();
        grayImageDtype = getDtypeString(imgsGray[0].type());
        grayBatchSize = batchSizeGray;
    }
    if (!imgsRGB.empty())
    {
        ostringstream oss;
        oss << maxWidthRGB << "x" << maxHeightRGB;
        rgbImageSize = oss.str();
        rgbImageDtype = getDtypeString(imgsRGB[0].type());
        rgbBatchSize = batchSizeRGB;
    }

    // Parameters
    const float  alpha          = 1.2f;
    const float  beta           = 20.0f;
    const float  blendAlpha     = 0.5f;
    const float  gamma          = 1.5f;
    const float  hueDelta       = 10.f;
    const float  satFactor      = 1.2f;
    const float  contrastFactor = 1.3f;
    const float  contrastCenter = 128.f;
    const float  exposureStop   = 0.5f;
    const float  exposureFactor = 1.4f;
    const float  addVal         = 10.f;
    const float  subVal         = 10.f;
    const float  mulVal         = 1.2f;
    const float  noiseProb      = 0.05f;
    const float  noiseMean      = 0.f;
    const float  noiseStd       = 15.f;
    const int    resizeW        = 960;
    const int    resizeH        = 540;
    const int    cropW          = 800;
    const int    cropH          = 600;
    const int    filterKernel   = 3;
    const int    medianKernel   = 3;
    const double gaussSigma     = 5.0;
    const float  angleDeg       = 45.f;
    const double threshVal      = 128.0;
    const int    morphKernel    = 3;

    cout << "\n========================================" << endl;
    cout << "OPENCV vs RPP HOST BENCHMARKING" << endl;
    cout << "Competitive Analysis: All Kernels" << endl;
    cout << "========================================" << endl;

    // Print version and system information
    cout << "\n--- System Information ---" << endl;
    cout << "OpenCV Version: " << CV_VERSION << endl;
    cout << "RPP Version: " << getRPPVersion() << endl;
    cout << "\n--- Host Information ---" << endl;
    cout << "OS: " << getOSInfo() << endl;
    cout << "CPU: " << getCPUInfo() << endl;
    cout << "Memory: " << getMemoryInfo() << endl;
    cout << "Number of Threads: " << NUM_THREADS << endl;
    cout << "Number of Runs: " << NUM_RUNS << endl;
    cout << "========================================" << endl;

    // ==================== GRAYSCALE ====================
    if (!imgsGray.empty())
    {
        cout << "\n========== GRAYSCALE IMAGES ==========" << endl;

        cout << "\n--- Color Augmentations ---" << endl;
        benchmark_RPP_Brightness(imgsGray, false, alpha, beta, handle);
        benchmark_OpenCV_Brightness(imgsGray, false, alpha, beta);

        benchmark_RPP_GammaCorrection(imgsGray, false, gamma, handle);
        benchmark_OpenCV_GammaCorrection(imgsGray, false, gamma);

        benchmark_RPP_Contrast(imgsGray, false, contrastFactor, contrastCenter, handle);
        benchmark_OpenCV_Contrast(imgsGray, false, contrastFactor, contrastCenter);

        benchmark_RPP_Exposure(imgsGray, false, exposureFactor, handle);
        benchmark_OpenCV_Exposure(imgsGray, false, exposureStop);

        cout << "\n--- Filter Augmentations ---" << endl;
        benchmark_RPP_BoxFilter(imgsGray, false, filterKernel, handle);
        benchmark_OpenCV_BoxFilter(imgsGray, false, filterKernel);

        benchmark_RPP_MedianFilter(imgsGray, false, medianKernel, handle);
        benchmark_OpenCV_MedianFilter(imgsGray, false, medianKernel);

        benchmark_RPP_GaussianFilter(imgsGray, false, filterKernel, gaussSigma, handle);
        benchmark_OpenCV_GaussianFilter(imgsGray, false, filterKernel, gaussSigma);

        benchmark_RPP_SobelFilter(imgsGray, false, 0, handle);
        benchmark_OpenCV_SobelFilter(imgsGray, false);

        benchmark_RPP_Emboss(imgsGray, false, 3, 1.0f, handle);
        benchmark_OpenCV_Emboss(imgsGray, false, 3, 1.0f);

        cout << "\n--- Geometric Augmentations ---" << endl;
        benchmark_RPP_Crop(imgsGray, false, cropW, cropH, handle);
        benchmark_OpenCV_Crop(imgsGray, false, cropW, cropH);

        benchmark_RPP_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handle);
        benchmark_OpenCV_Resize(imgsGray, false, resizeW, resizeH, INTER_NEAREST, "Nearest");

        benchmark_RPP_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BILINEAR, "Bilinear", handle);
        benchmark_OpenCV_Resize(imgsGray, false, resizeW, resizeH, INTER_LINEAR, "Bilinear");

        benchmark_RPP_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BICUBIC, "Bicubic", handle);
        benchmark_OpenCV_Resize(imgsGray, false, resizeW, resizeH, INTER_CUBIC, "Bicubic");

        benchmark_RPP_Flip(imgsGray, false, 1, handle);
        benchmark_OpenCV_Flip(imgsGray, false, 1);

        benchmark_RPP_Flip(imgsGray, false, 0, handle);
        benchmark_OpenCV_Flip(imgsGray, false, 0);

        benchmark_RPP_Flip(imgsGray, false, -1, handle);
        benchmark_OpenCV_Flip(imgsGray, false, -1);

        benchmark_RPP_Rotate(imgsGray, false, angleDeg, handle);
        benchmark_OpenCV_Rotate(imgsGray, false, angleDeg);

        benchmark_RPP_WarpAffine(imgsGray, false, handle);
        benchmark_OpenCV_WarpAffine(imgsGray, false);

        benchmark_RPP_WarpPerspective(imgsGray, false, handle);
        benchmark_OpenCV_WarpPerspective(imgsGray, false);

        benchmark_RPP_Fisheye(imgsGray, false, handle);
        benchmark_OpenCV_Fisheye(imgsGray, false);

        benchmark_RPP_LensCorrection(imgsGray, false, handle);
        benchmark_OpenCV_LensCorrection(imgsGray, false);

        cout << "\n--- Morphological Operations ---" << endl;
        benchmark_RPP_Erode(imgsGray, false, morphKernel, handle);
        benchmark_OpenCV_Erode(imgsGray, false, morphKernel);

        benchmark_RPP_Dilate(imgsGray, false, morphKernel, handle);
        benchmark_OpenCV_Dilate(imgsGray, false, morphKernel);

        cout << "\n--- Arithmetic Operations ---" << endl;
        benchmark_RPP_AddScalar(imgsGray, false, addVal, handle);
        benchmark_OpenCV_AddScalar(imgsGray, false, addVal);

        benchmark_RPP_SubtractScalar(imgsGray, false, subVal, handle);
        benchmark_OpenCV_SubtractScalar(imgsGray, false, subVal);

        benchmark_RPP_MultiplyScalar(imgsGray, false, mulVal, handle);
        benchmark_OpenCV_MultiplyScalar(imgsGray, false, mulVal);

        benchmark_RPP_Blend(imgsGray, false, blendAlpha, handle);
        benchmark_OpenCV_Blend(imgsGray, false, blendAlpha);

        cout << "\n--- Bitwise Operations ---" << endl;
        benchmark_RPP_BitwiseAnd(imgsGray, false, handle);
        benchmark_OpenCV_BitwiseAnd(imgsGray, false);

        benchmark_RPP_BitwiseOr(imgsGray, false, handle);
        benchmark_OpenCV_BitwiseOr(imgsGray, false);

        benchmark_RPP_BitwiseNot(imgsGray, false, handle);
        benchmark_OpenCV_BitwiseNot(imgsGray, false);

        benchmark_RPP_BitwiseXor(imgsGray, false, handle);
        benchmark_OpenCV_BitwiseXor(imgsGray, false);

        cout << "\n--- Statistical Operations ---" << endl;
        benchmark_RPP_TensorMin(imgsGray, false, handle);
        benchmark_OpenCV_TensorMin(imgsGray, false);

        benchmark_RPP_TensorMax(imgsGray, false, handle);
        benchmark_OpenCV_TensorMax(imgsGray, false);

        benchmark_RPP_TensorSum(imgsGray, false, handle);
        benchmark_OpenCV_TensorSum(imgsGray, false);

        benchmark_RPP_TensorMean(imgsGray, false, handle);
        benchmark_OpenCV_TensorMean(imgsGray, false);

        benchmark_RPP_TensorStddev(imgsGray, false, handle);
        benchmark_OpenCV_TensorStddev(imgsGray, false);

        benchmark_RPP_Threshold(imgsGray, false, threshVal, handle);
        benchmark_OpenCV_Threshold(imgsGray, false, threshVal);

        cout << "\n--- Effects Augmentations ---" << endl;
        benchmark_RPP_GaussianNoise(imgsGray, false, noiseMean, noiseStd, handle);
        benchmark_OpenCV_GaussianNoise(imgsGray, false, noiseMean, noiseStd);

        benchmark_RPP_SaltAndPepperNoise(imgsGray, false, noiseProb, handle);
        benchmark_OpenCV_SaltAndPepperNoise(imgsGray, false, noiseProb);

        benchmark_RPP_JpegCompressionDistortion(imgsGray, false, 50, handle);
        benchmark_OpenCV_JpegCompressionDistortion(imgsGray, false, 50);

        cout << "\n--- Data Operations ---" << endl;
        benchmark_RPP_Copy(imgsGray, false, handle);
        benchmark_OpenCV_Copy(imgsGray, false);

        benchmark_RPP_Slice(imgsGray, false, handle);
        benchmark_OpenCV_Slice(imgsGray, false);

        benchmark_RPP_Transpose(imgsGray, false, handle);
        benchmark_OpenCV_Transpose(imgsGray, false);

        cout << "\n--- Advanced Operations ---" << endl;
        benchmark_RPP_HistogramEqualize(imgsGray, false, handle);
        benchmark_OpenCV_HistogramEqualize(imgsGray, false);

        benchmark_RPP_LUT(imgsGray, false, handle);
        benchmark_OpenCV_LUT(imgsGray, false);

        benchmark_RPP_Magnitude(imgsGray, false, handle);
        benchmark_OpenCV_Magnitude(imgsGray, false);

        benchmark_RPP_Phase(imgsGray, false, handle);
        benchmark_OpenCV_Phase(imgsGray, false);

        benchmark_RPP_Normalize(imgsGray, false, handle);
        benchmark_OpenCV_Normalize(imgsGray, false);

        benchmark_RPP_FusedMultiplyAddScalar(imgsGray, false, 1.2f, 10.0f, handle);
        benchmark_OpenCV_FusedMultiplyAddScalar(imgsGray, false, 1.2f, 10.0f);

        benchmark_RPP_Remap(imgsGray, false, handle);
        benchmark_OpenCV_Remap(imgsGray, false);
    }

    // ==================== RGB ====================
    if (!imgsRGB.empty())
    {
        cout << "\n========== RGB IMAGES ==========" << endl;

        cout << "\n--- Color Augmentations ---" << endl;
        benchmark_RPP_Brightness(imgsRGB, true, alpha, beta, handle);
        benchmark_OpenCV_Brightness(imgsRGB, true, alpha, beta);

        benchmark_RPP_GammaCorrection(imgsRGB, true, gamma, handle);
        benchmark_OpenCV_GammaCorrection(imgsRGB, true, gamma);

        benchmark_RPP_Blend(imgsRGB, true, blendAlpha, handle);
        benchmark_OpenCV_Blend(imgsRGB, true, blendAlpha);

        benchmark_RPP_Contrast(imgsRGB, true, contrastFactor, contrastCenter, handle);
        benchmark_OpenCV_Contrast(imgsRGB, true, contrastFactor, contrastCenter);

        benchmark_RPP_Exposure(imgsRGB, true, exposureFactor, handle);
        benchmark_OpenCV_Exposure(imgsRGB, true, exposureStop);

        benchmark_RPP_Hue(imgsRGB, hueDelta, handle);
        benchmark_OpenCV_Hue(imgsRGB, hueDelta);

        benchmark_RPP_Saturation(imgsRGB, satFactor, handle);
        benchmark_OpenCV_Saturation(imgsRGB, satFactor);

        benchmark_RPP_ColorToGreyscale(imgsRGB, handle);
        benchmark_OpenCV_ColorToGreyscale(imgsRGB);

        benchmark_RPP_ColorJitter(imgsRGB, 1.2f, 1.3f, 1.2f, 10.f, handle);
        benchmark_OpenCV_ColorJitter(imgsRGB, 1.2f, 1.3f, 1.2f, 10.f);

        cout << "\n--- Filter Augmentations ---" << endl;
        benchmark_RPP_BoxFilter(imgsRGB, true, filterKernel, handle);
        benchmark_OpenCV_BoxFilter(imgsRGB, true, filterKernel);

        benchmark_RPP_MedianFilter(imgsRGB, true, medianKernel, handle);
        benchmark_OpenCV_MedianFilter(imgsRGB, true, medianKernel);

        benchmark_RPP_GaussianFilter(imgsRGB, true, filterKernel, gaussSigma, handle);
        benchmark_OpenCV_GaussianFilter(imgsRGB, true, filterKernel, gaussSigma);

        benchmark_RPP_SobelFilter(imgsRGB, true, 0, handle);
        benchmark_OpenCV_SobelFilter(imgsRGB, true);

        benchmark_RPP_Emboss(imgsRGB, true, 3, 1.0f, handle);
        benchmark_OpenCV_Emboss(imgsRGB, true, 3, 1.0f);

        cout << "\n--- Geometric Augmentations ---" << endl;
        benchmark_RPP_Crop(imgsRGB, true, cropW, cropH, handle);
        benchmark_OpenCV_Crop(imgsRGB, true, cropW, cropH);

        benchmark_RPP_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handle);
        benchmark_OpenCV_Resize(imgsRGB, true, resizeW, resizeH, INTER_NEAREST, "Nearest");

        benchmark_RPP_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BILINEAR, "Bilinear", handle);
        benchmark_OpenCV_Resize(imgsRGB, true, resizeW, resizeH, INTER_LINEAR, "Bilinear");

        benchmark_RPP_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BICUBIC, "Bicubic", handle);
        benchmark_OpenCV_Resize(imgsRGB, true, resizeW, resizeH, INTER_CUBIC, "Bicubic");

        benchmark_RPP_Flip(imgsRGB, true, 1, handle);
        benchmark_OpenCV_Flip(imgsRGB, true, 1);

        benchmark_RPP_Flip(imgsRGB, true, 0, handle);
        benchmark_OpenCV_Flip(imgsRGB, true, 0);

        benchmark_RPP_Flip(imgsRGB, true, -1, handle);
        benchmark_OpenCV_Flip(imgsRGB, true, -1);

        benchmark_RPP_Rotate(imgsRGB, true, angleDeg, handle);
        benchmark_OpenCV_Rotate(imgsRGB, true, angleDeg);

        benchmark_RPP_WarpAffine(imgsRGB, true, handle);
        benchmark_OpenCV_WarpAffine(imgsRGB, true);

        benchmark_RPP_WarpPerspective(imgsRGB, true, handle);
        benchmark_OpenCV_WarpPerspective(imgsRGB, true);

        benchmark_RPP_Fisheye(imgsRGB, true, handle);
        benchmark_OpenCV_Fisheye(imgsRGB, true);

        benchmark_RPP_LensCorrection(imgsRGB, true, handle);
        benchmark_OpenCV_LensCorrection(imgsRGB, true);

        cout << "\n--- Morphological Operations ---" << endl;
        benchmark_RPP_Erode(imgsRGB, true, morphKernel, handle);
        benchmark_OpenCV_Erode(imgsRGB, true, morphKernel);

        benchmark_RPP_Dilate(imgsRGB, true, morphKernel, handle);
        benchmark_OpenCV_Dilate(imgsRGB, true, morphKernel);

        cout << "\n--- Arithmetic Operations ---" << endl;
        benchmark_RPP_AddScalar(imgsRGB, true, addVal, handle);
        benchmark_OpenCV_AddScalar(imgsRGB, true, addVal);

        benchmark_RPP_SubtractScalar(imgsRGB, true, subVal, handle);
        benchmark_OpenCV_SubtractScalar(imgsRGB, true, subVal);

        benchmark_RPP_MultiplyScalar(imgsRGB, true, mulVal, handle);
        benchmark_OpenCV_MultiplyScalar(imgsRGB, true, mulVal);

        benchmark_RPP_Blend(imgsRGB, true, blendAlpha, handle);
        benchmark_OpenCV_Blend(imgsRGB, true, blendAlpha);

        cout << "\n--- Bitwise Operations ---" << endl;
        benchmark_RPP_BitwiseAnd(imgsRGB, true, handle);
        benchmark_OpenCV_BitwiseAnd(imgsRGB, true);

        benchmark_RPP_BitwiseOr(imgsRGB, true, handle);
        benchmark_OpenCV_BitwiseOr(imgsRGB, true);

        benchmark_RPP_BitwiseNot(imgsRGB, true, handle);
        benchmark_OpenCV_BitwiseNot(imgsRGB, true);

        benchmark_RPP_BitwiseXor(imgsRGB, true, handle);
        benchmark_OpenCV_BitwiseXor(imgsRGB, true);

        cout << "\n--- Statistical Operations ---" << endl;
        benchmark_RPP_TensorMin(imgsRGB, true, handle);
        benchmark_OpenCV_TensorMin(imgsRGB, true);

        benchmark_RPP_TensorMax(imgsRGB, true, handle);
        benchmark_OpenCV_TensorMax(imgsRGB, true);

        benchmark_RPP_TensorSum(imgsRGB, true, handle);
        benchmark_OpenCV_TensorSum(imgsRGB, true);

        benchmark_RPP_TensorMean(imgsRGB, true, handle);
        benchmark_OpenCV_TensorMean(imgsRGB, true);

        benchmark_RPP_TensorStddev(imgsRGB, true, handle);
        benchmark_OpenCV_TensorStddev(imgsRGB, true);

        benchmark_RPP_Threshold(imgsRGB, true, threshVal, handle);
        benchmark_OpenCV_Threshold(imgsRGB, true, threshVal);

        cout << "\n--- Effects Augmentations ---" << endl;
        benchmark_RPP_GaussianNoise(imgsRGB, true, noiseMean, noiseStd, handle);
        benchmark_OpenCV_GaussianNoise(imgsRGB, true, noiseMean, noiseStd);

        benchmark_RPP_SaltAndPepperNoise(imgsRGB, true, noiseProb, handle);
        benchmark_OpenCV_SaltAndPepperNoise(imgsRGB, true, noiseProb);

        benchmark_RPP_NoiseShot(imgsRGB, true, 0.2f, handle);
        benchmark_OpenCV_NoiseShot(imgsRGB, true, 0.2f);

        benchmark_RPP_JpegCompressionDistortion(imgsRGB, true, 50, handle);
        benchmark_OpenCV_JpegCompressionDistortion(imgsRGB, true, 50);

        benchmark_RPP_Posterize(imgsRGB, true, 4, handle);
        benchmark_OpenCV_Posterize(imgsRGB, true, 4);

        benchmark_RPP_Solarize(imgsRGB, true, 128, handle);
        benchmark_OpenCV_Solarize(imgsRGB, true, 128);

        benchmark_RPP_ColorCast(imgsRGB, true, 20.0f, 10.0f, -15.0f, handle);
        benchmark_OpenCV_ColorCast(imgsRGB, true, 20.0f, 10.0f, -15.0f);

        benchmark_RPP_ColorTemperature(imgsRGB, true, 40, handle);
        benchmark_OpenCV_ColorTemperature(imgsRGB, true, 40);

        benchmark_RPP_ColorTwist(imgsRGB, true, handle);
        benchmark_OpenCV_ColorTwist(imgsRGB, true);

        benchmark_RPP_Vignette(imgsRGB, true, 0.5f, handle);
        benchmark_OpenCV_Vignette(imgsRGB, true, 0.5f);

        benchmark_RPP_NonLinearBlend(imgsRGB, true, 50.0f, handle);
        benchmark_OpenCV_NonLinearBlend(imgsRGB, true, 50.0f);

        cout << "\n--- Dropout Augmentations ---" << endl;
        benchmark_RPP_Erase(imgsRGB, true, 3, handle);
        benchmark_OpenCV_Erase(imgsRGB, true, 3);

        benchmark_RPP_RandomErase(imgsRGB, true, handle);
        benchmark_OpenCV_RandomErase(imgsRGB, true);

        benchmark_RPP_CoarseDropout(imgsRGB, true, 8, handle);
        benchmark_OpenCV_CoarseDropout(imgsRGB, true, 8);

        benchmark_RPP_GridDropout(imgsRGB, true, 10, 10, handle);
        benchmark_OpenCV_GridDropout(imgsRGB, true, 10, 10);

        benchmark_RPP_Gridmask(imgsRGB, true, 96, 0.6f, handle);
        benchmark_OpenCV_Gridmask(imgsRGB, true, 96, 0.6f);

        benchmark_RPP_ChannelDropout(imgsRGB, true, 0.4f, handle);
        benchmark_OpenCV_ChannelDropout(imgsRGB, true, 0.4f);

        benchmark_RPP_CutoutDropout(imgsRGB, true, 1, handle);
        benchmark_OpenCV_CutoutDropout(imgsRGB, true, 1);

        cout << "\n--- Data Operations ---" << endl;
        benchmark_RPP_Copy(imgsRGB, true, handle);
        benchmark_OpenCV_Copy(imgsRGB, true);

        benchmark_RPP_Slice(imgsRGB, true, handle);
        benchmark_OpenCV_Slice(imgsRGB, true);

        benchmark_RPP_ChannelPermute(imgsRGB, true, handle);
        benchmark_OpenCV_ChannelPermute(imgsRGB, true);

        benchmark_RPP_Transpose(imgsRGB, true, handle);
        benchmark_OpenCV_Transpose(imgsRGB, true);

        cout << "\n--- Advanced Operations ---" << endl;
        benchmark_RPP_LUT(imgsRGB, true, handle);
        benchmark_OpenCV_LUT(imgsRGB, true);

        benchmark_RPP_Magnitude(imgsRGB, true, handle);
        benchmark_OpenCV_Magnitude(imgsRGB, true);

        benchmark_RPP_Phase(imgsRGB, true, handle);
        benchmark_OpenCV_Phase(imgsRGB, true);

        benchmark_RPP_Normalize(imgsRGB, true, handle);
        benchmark_OpenCV_Normalize(imgsRGB, true);

        benchmark_RPP_FusedMultiplyAddScalar(imgsRGB, true, 1.2f, 10.0f, handle);
        benchmark_OpenCV_FusedMultiplyAddScalar(imgsRGB, true, 1.2f, 10.0f);

        benchmark_RPP_Remap(imgsRGB, true, handle);
        benchmark_OpenCV_Remap(imgsRGB, true);

        cout << "\n--- Composite Operations ---" << endl;
        benchmark_RPP_CropAndPatch(imgsRGB, true, handle);
        benchmark_OpenCV_CropAndPatch(imgsRGB, true);

        benchmark_RPP_CropMirrorNormalize(imgsRGB, true, handle);
        benchmark_OpenCV_CropMirrorNormalize(imgsRGB, true);

        benchmark_RPP_ResizeMirrorNormalize(imgsRGB, true, handle);
        benchmark_OpenCV_ResizeMirrorNormalize(imgsRGB, true);

        benchmark_RPP_ResizeCropMirror(imgsRGB, true, handle);
        benchmark_OpenCV_ResizeCropMirror(imgsRGB, true);

        benchmark_RPP_RICAP(imgsRGB, true, handle);
        benchmark_OpenCV_RICAP(imgsRGB, true);
    }

    cout << "\n========================================" << endl;
    cout << "BENCHMARKING COMPLETE" << endl;
    cout << "OpenCV vs RPP HOST Backend Comparison" << endl;
    cout << "========================================\n" << endl;

    // Export results to Excel
#if ENABLE_PARALLEL_THREADS
    string excelFilename = "opencv_vs_rpp_benchmark_results_parallel_threads_ON.xlsx";
#else
    string excelFilename = "opencv_vs_rpp_benchmark_results_parallel_threads_OFF.xlsx";
#endif
    writeResultsToExcel(excelFilename, grayscaleResults, rgbResults);

    // Cleanup RPP handle
    rppDestroy(handle, RPP_HOST_BACKEND);

    return 0;
}
