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

#ifndef BENCHMARKS_COMMON_H
#define BENCHMARKS_COMMON_H

#include <dirent.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "rpp.h"
#include "rpp_version.h"
#include "xlsxwriter.h"

using namespace std;
using namespace cv;
using namespace chrono;

// Test image paths - can be overridden via command line
#define DEFAULT_GRAY_IMAGE_PATH "1080p_128images_dataset/"
#define DEFAULT_RGB_IMAGE_PATH "1080p_128images_dataset/"

// Global configuration variables (set at runtime)
extern int NUM_RUNS;
extern int NUM_THREADS;
extern string GRAY_IMAGE_PATH;
extern string RGB_IMAGE_PATH;

// Structure to store benchmark results
struct BenchmarkResult {
    string operationName;
    string parameters;
    string imageSize;
    string dtype;
    int batchSize;
    int numRuns;
    double opencvTime;
    double rppTime;
    double speedup;

    BenchmarkResult(const string& name, const string& params, double cvTime, double rTime,
                    const string& imgSize = "", const string& dataType = "", int batch = 0,
                    int runs = 0)
        : operationName(name),
          parameters(params),
          imageSize(imgSize),
          dtype(dataType),
          batchSize(batch),
          numRuns(runs),
          opencvTime(cvTime),
          rppTime(rTime) {
        speedup = (rTime > 0) ? (cvTime / rTime) : 0.0;
    }
};

// Global vectors to store results
extern vector<BenchmarkResult> grayscaleResults;
extern vector<BenchmarkResult> rgbResults;

// Global variables for image metadata
extern string grayImageSize;
extern string grayImageDtype;
extern string rgbImageSize;
extern string rgbImageDtype;
extern int grayBatchSize;
extern int rgbBatchSize;

// Utility functions
vector<Mat> loadBatchImages(const string& directory, int& batchSize, int& maxWidth, int& maxHeight,
                            bool isColor);
void printResult(const string& name, int batchSize, bool isColor, double totalMs,
                 const string& params = "");
string getCPUInfo();
string getMemoryInfo();
string getOSInfo();
string getRPPVersion();
string getCurrentDateTime();
string getDtypeString(int cvType);
bool writeResultsToExcel(const string& filename, const vector<BenchmarkResult>& grayResults,
                         const vector<BenchmarkResult>& colorResults);

// RPP utility functions
RpptDesc createRppDescriptor(const Mat& img, RpptLayout layout = RpptLayout::NHWC);
RpptROI createFullImageROI(const Mat& img);
RpptGenericDesc toGenericDesc(const RpptDesc& desc);
RpptROI3D createFullImageROI3D(const Mat& img);

// Helper initialization functions for dropout operators
void init_grid_dropout_boxes(int batchCount, RpptRoiLtrb* anchorBoxInfoTensor,
                             RpptROI* roiTensorPtrSrc, Rpp32u gridH, Rpp32u gridW, Rpp32u& maxHoleW,
                             Rpp32u& maxHoleH, Rpp32f holeRatio, int seed);
void init_ricap_boxes(int maxWidth, int maxHeight, int batchSize, Rpp32u* permutationTensor,
                      RpptROI* roiPtrInputCropRegion);

// RPP Benchmark function declarations
void benchmark_RPP_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                              rppHandle_t handle);
void benchmark_RPP_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                   rppHandle_t handle);
void benchmark_RPP_Blend(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle);
void benchmark_RPP_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                            float contrastCenter, rppHandle_t handle);
void benchmark_RPP_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                            rppHandle_t handle);
void benchmark_RPP_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle);
void benchmark_RPP_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle);
void benchmark_RPP_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle);
void benchmark_RPP_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                               float saturation, float hue, rppHandle_t handle);
void benchmark_RPP_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                             rppHandle_t handle);
void benchmark_RPP_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                rppHandle_t handle);
void benchmark_RPP_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                  double sigma, rppHandle_t handle);
void benchmark_RPP_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                               rppHandle_t handle);
void benchmark_RPP_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                          rppHandle_t handle);
void benchmark_RPP_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                        rppHandle_t handle);
void benchmark_RPP_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                          RpptInterpolationType interpType, const string& interpName,
                          rppHandle_t handle);
void benchmark_RPP_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle);
void benchmark_RPP_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                          rppHandle_t handle);
void benchmark_RPP_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle);
void benchmark_RPP_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                          rppHandle_t handle);
void benchmark_RPP_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                             rppHandle_t handle);
void benchmark_RPP_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                  rppHandle_t handle);
void benchmark_RPP_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                  rppHandle_t handle);
void benchmark_RPP_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                             rppHandle_t handle);
void benchmark_RPP_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Normalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Normalize_SingleImage(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, Rpp32f mul,
                                          Rpp32f add, rppHandle_t handle);
void benchmark_RPP_Emboss(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stddev,
                                 rppHandle_t handle);
void benchmark_RPP_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb,
                                      rppHandle_t handle);
void benchmark_RPP_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits,
                             rppHandle_t handle);
void benchmark_RPP_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold,
                            rppHandle_t handle);
void benchmark_RPP_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor,
                             rppHandle_t handle);
void benchmark_RPP_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth,
                            Rpp32f gridRatio, rppHandle_t handle);
void benchmark_RPP_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift,
                             Rpp32f bShift, rppHandle_t handle);
void benchmark_RPP_ColorTemperature(const vector<Mat>& imgs, bool isColor, Rpp32s adjustmentValue,
                                    rppHandle_t handle);
void benchmark_RPP_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity,
                            rppHandle_t handle);
void benchmark_RPP_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev,
                                  rppHandle_t handle);
void benchmark_RPP_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                         rppHandle_t handle);
void benchmark_RPP_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numDropouts,
                                 rppHandle_t handle);
void benchmark_RPP_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow,
                               Rpp32u numGridsPerColumn, rppHandle_t handle);
void benchmark_RPP_RandomErase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_RICAP(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                  rppHandle_t handle);
void benchmark_RPP_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                 rppHandle_t handle);
void benchmark_RPP_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                             rppHandle_t handle);
void benchmark_RPP_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);

// OpenCV Benchmark function declarations
void benchmark_OpenCV_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta);
void benchmark_OpenCV_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma);
void benchmark_OpenCV_Blend(const vector<Mat>& imgs, bool isColor, float alpha);
void benchmark_OpenCV_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                               float contrastCenter);
void benchmark_OpenCV_Exposure(const vector<Mat>& imgs, bool isColor, float stop);
void benchmark_OpenCV_Hue(const vector<Mat>& imgs, float hueDelta);
void benchmark_OpenCV_Saturation(const vector<Mat>& imgs, float satFactor);
void benchmark_OpenCV_ColorToGreyscale(const vector<Mat>& imgs);
void benchmark_OpenCV_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                                  float saturation, float hue);
void benchmark_OpenCV_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                     double sigma);
void benchmark_OpenCV_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType);
void benchmark_OpenCV_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength);
void benchmark_OpenCV_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight);
void benchmark_OpenCV_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                             int interpType, const string& interpName);
void benchmark_OpenCV_Flip(const vector<Mat>& imgs, bool isColor, int flipCode);
void benchmark_OpenCV_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg);
void benchmark_OpenCV_WarpAffine(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Fisheye(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_LensCorrection(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal);
void benchmark_OpenCV_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal);
void benchmark_OpenCV_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal);
void benchmark_OpenCV_BitwiseAnd(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_BitwiseOr(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_BitwiseNot(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Threshold(const vector<Mat>& imgs, bool isColor, double thresh);
void benchmark_OpenCV_Normalize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, float mul,
                                             float add);
void benchmark_OpenCV_Transpose(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_BitwiseXor(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_HistogramEqualize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_LUT(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Magnitude(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Phase(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_WarpPerspective(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Remap(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Emboss(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorMin(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorMax(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorSum(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorMean(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorStddev(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean,
                                    float stddev);
void benchmark_OpenCV_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb);
void benchmark_OpenCV_Copy(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits);
void benchmark_OpenCV_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold);
void benchmark_OpenCV_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor);
void benchmark_OpenCV_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth,
                               Rpp32f gridRatio);
void benchmark_OpenCV_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift,
                                Rpp32f bShift);
void benchmark_OpenCV_ColorTemperature(const vector<Mat>& imgs, bool isColor,
                                       Rpp32s adjustmentValue);
void benchmark_OpenCV_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity);
void benchmark_OpenCV_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev);
void benchmark_OpenCV_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes);
void benchmark_OpenCV_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numDropouts);
void benchmark_OpenCV_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow,
                                  Rpp32u numGridsPerColumn);
void benchmark_OpenCV_RandomErase(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ColorTwist(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_CropAndPatch(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ResizeCropMirror(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_RICAP(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb);
void benchmark_OpenCV_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes);
void benchmark_OpenCV_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor,
                                                Rpp32s quality);
void benchmark_OpenCV_ChannelPermute(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Slice(const vector<Mat>& imgs, bool isColor);

#endif  // BENCHMARKS_COMMON_H
