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

// Macro for checking RPP API status
#define CHECK_RPP_STATUS(call, func_name)                                                       \
    do {                                                                                        \
        RppStatus status = (call);                                                              \
        if (status != RPP_SUCCESS) {                                                            \
            std::cerr << "RPP " << func_name << " failed with status: " << status << std::endl; \
        }                                                                                       \
    } while (0)

// ==================== RPP BENCHMARK FUNCTIONS ====================

void benchmark_RPP_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                              rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_brightness(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &alpha,
                                &beta, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "brightness");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("RPP HOST Brightness", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                   rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_gamma_correction(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &gamma,
                                      &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "gamma_correction");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST GammaCorrection", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "gamma=" + to_string(gamma));
}

void benchmark_RPP_Blend(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(imgs.size());
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        imgs[i].convertTo(imgs2[i], -1, 0.8, 30);
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_blend(imgs[i].data, imgs2[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                           &alpha, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "blend");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Blend", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "alpha=" + to_string(alpha));
}

void benchmark_RPP_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                            float contrastCenter, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_contrast(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           &contrastFactor, &contrastCenter, &rois[i],
                                           RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "contrast");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Contrast", imgs.size(), isColor,
                duration<double, milli>(end - start).count(),
                "factor=" + to_string(contrastFactor) + ", center=" + to_string(contrastCenter));
}

void benchmark_RPP_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                            rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_exposure(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           &exposureFactor, &rois[i], RpptRoiType::XYWH, handle,
                                           RPP_HOST_BACKEND),
                             "exposure");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Exposure", imgs.size(), isColor,
                duration<double, milli>(end - start).count(),
                "factor=" + to_string(exposureFactor));
}

void benchmark_RPP_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NHWC);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_hue(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &hueDelta, &rois[i],
                         RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "hue");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Hue", imgs.size(), true, duration<double, milli>(end - start).count(),
                "hueDelta=" + to_string(hueDelta));
}

void benchmark_RPP_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NHWC);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_saturation(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &satFactor,
                                &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "saturation");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Saturation", imgs.size(), true,
                duration<double, milli>(end - start).count(), "factor=" + to_string(satFactor));
}

void benchmark_RPP_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].rows, imgs[i].cols, CV_8UC1);
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NCHW);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_color_to_greyscale(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        RpptSubpixelLayout::BGRtype, handle, RPP_HOST_BACKEND),
                "color_to_greyscale");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST ColorToGreyscale", imgs.size(), true,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                               float saturation, float hue, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NHWC);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_color_jitter(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                  &brightness, &contrast, &saturation, &hue, &rois[i],
                                  RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "color_jitter");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "brightness=" << brightness << ", contrast=" << contrast
           << ", saturation=" << saturation << ", hue=" << hue;
    printResult("RPP HOST ColorJitter", imgs.size(), true,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_box_filter(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], kernelSize,
                                borderType, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "box_filter");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST BoxFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_median_filter(imgs[i].data, &srcDescs[i], out[i].data,
                                                &dstDescs[i], kernelSize, borderType, &rois[i],
                                                RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "median_filter");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST MedianFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                  double sigma, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    float stdDev = static_cast<float>(sigma);
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_gaussian_filter(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &stdDev,
                                     kernelSize, borderType, &rois[i], RpptRoiType::XYWH, handle,
                                     RPP_HOST_BACKEND),
                "gaussian_filter");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST GaussianFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                               rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), CV_8UC1);
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    Rpp32u kernelSize = 3;  // Sobel uses 3x3 kernel
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_sobel_filter(imgs[i].data, &srcDescs[i], out[i].data,
                                               &dstDescs[i], sobelType, kernelSize, &rois[i],
                                               RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "sobel_filter");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST SobelFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "type=" + to_string(sobelType));
}

void benchmark_RPP_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    vector<Rpp32f> strengthTensor(num_images, strength);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_emboss(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                         strengthTensor.data(), kernelSize, borderType, &rois[i],
                                         RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "emboss");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("RPP HOST Emboss", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                        rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(cropHeight, cropWidth, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i].xywhROI.xy.x = (imgs[i].cols - cropWidth) / 2;
        rois[i].xywhROI.xy.y = (imgs[i].rows - cropHeight) / 2;
        rois[i].xywhROI.roiWidth = cropWidth;
        rois[i].xywhROI.roiHeight = cropHeight;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                       &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "crop");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "size=" << cropWidth << "x" << cropHeight;
    printResult("RPP HOST Crop", imgs.size(), isColor, duration<double, milli>(end - start).count(),
                params.str());
}

void benchmark_RPP_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                          RpptInterpolationType interpType, const string& interpName,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    RpptImagePatch dstImgSize;
    dstImgSize.width = dstW;
    dstImgSize.height = dstH;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(dstH, dstW, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_resize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &dstImgSize,
                            interpType, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "resize");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("RPP HOST Resize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u horizontalFlag = (flipCode == 1 || flipCode == -1) ? 1 : 0;
    Rpp32u verticalFlag = (flipCode == 0 || flipCode == -1) ? 1 : 0;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_flip(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &horizontalFlag,
                          &verticalFlag, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "flip");
        }
    }
    auto end = high_resolution_clock::now();
    string name = (flipCode == 1) ? "Horizontal" : (flipCode == 0) ? "Vertical" : "Both";
    ostringstream params;
    params << "type=" << name;
    printResult("RPP HOST Flip", imgs.size(), isColor, duration<double, milli>(end - start).count(),
                params.str());
}

void benchmark_RPP_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_rotate(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                         &angleDeg, RpptInterpolationType::BILINEAR, &rois[i],
                                         RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "rotate");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "angle=" << angleDeg << "deg";
    printResult("RPP HOST Rotate", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    float affine[6] = {1.0f, 0.1f, 10.0f, 0.1f, 1.0f, 10.0f};
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_warp_affine(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                              affine, RpptInterpolationType::BILINEAR, &rois[i],
                                              RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "warp_affine");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST WarpAffine", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_fisheye(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                          &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "fisheye");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Fisheye", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Camera matrix and distortion coefficients (from reference implementation)
    struct CameraMatrix {
        Rpp32f data[9];
    };
    struct DistortionCoeffs {
        Rpp32f data[8];
    };

    vector<CameraMatrix> cameraMatrices(num_images);
    vector<DistortionCoeffs> distortionCoeffs(num_images);

    // Sample camera calibration parameters (from reference)
    CameraMatrix sampleCameraMatrix = {
        {534.07088364f, 0.0f, 341.53407554f, 0.0f, 534.11914595f, 232.94565259f, 0.0f, 0.0f, 1.0f}};
    DistortionCoeffs sampleDistortion = {
        {-0.29297164f, 0.10770696f, 0.00131038f, -0.0000311f, 0.0434798f, 0.0f, 0.0f, 0.0f}};

    for (int i = 0; i < num_images; ++i) {
        cameraMatrices[i] = sampleCameraMatrix;
        distortionCoeffs[i] = sampleDistortion;
    }

    // Table descriptor for remap tables
    RpptDesc tableDesc;
    if (num_images > 0) {
        tableDesc = createRppDescriptor(imgs[0], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        tableDesc.c = 1;
        tableDesc.strides.nStride = imgs[0].rows * imgs[0].cols;
        tableDesc.strides.hStride = imgs[0].cols;
        tableDesc.strides.wStride = tableDesc.strides.cStride = 1;
    }

    // Allocate remap tables
    size_t tableSize = num_images * imgs[0].rows * imgs[0].cols;
    vector<Rpp32f> rowRemapTable(tableSize);
    vector<Rpp32f> colRemapTable(tableSize);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_lens_correction(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                     rowRemapTable.data(), colRemapTable.data(), &tableDesc,
                                     cameraMatrices[i].data, distortionCoeffs[i].data, &rois[i],
                                     RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "lens_correction");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST LensCorrection", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize,
                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_erode_host(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                             kernelSize, &rois[i], RpptRoiType::XYWH, handle),
                             "erode_host");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST Erode", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_dilate_host(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                              kernelSize, &rois[i], RpptRoiType::XYWH, handle),
                             "dilate_host");
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST Dilate", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        // NCDHW layout for grayscale
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Create ROI3D array for the batch
    vector<RpptROI3D> roi3ds(num_images);
    for (int i = 0; i < num_images; ++i) {
        roi3ds[i].xyzwhdROI.xyz.x = 0;
        roi3ds[i].xyzwhdROI.xyz.y = 0;
        roi3ds[i].xyzwhdROI.xyz.z = 0;
        roi3ds[i].xyzwhdROI.roiWidth = width;
        roi3ds[i].xyzwhdROI.roiHeight = height;
        roi3ds[i].xyzwhdROI.roiDepth = 1;
    }

    // Create addTensor array (one value per image)
    vector<Rpp32f> addTensor(num_images, addVal);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(rppt_add_scalar(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                                         &dstGenericDesc, addTensor.data(), roi3ds.data(),
                                         RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                         "add_scalar");
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST AddScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "value=" + to_string(addVal));
}

void benchmark_RPP_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                  rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        // NCDHW layout for grayscale
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Create ROI3D array for the batch
    vector<RpptROI3D> roi3ds(num_images);
    for (int i = 0; i < num_images; ++i) {
        roi3ds[i].xyzwhdROI.xyz.x = 0;
        roi3ds[i].xyzwhdROI.xyz.y = 0;
        roi3ds[i].xyzwhdROI.xyz.z = 0;
        roi3ds[i].xyzwhdROI.roiWidth = width;
        roi3ds[i].xyzwhdROI.roiHeight = height;
        roi3ds[i].xyzwhdROI.roiDepth = 1;
    }

    // Create subtractTensor array (one value per image)
    vector<Rpp32f> subtractTensor(num_images, subVal);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(
            rppt_subtract_scalar(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                                 &dstGenericDesc, subtractTensor.data(), roi3ds.data(),
                                 RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
            "subtract_scalar");
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST SubtractScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "value=" + to_string(subVal));
}

void benchmark_RPP_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                  rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        // NCDHW layout for grayscale
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Create ROI3D array for the batch
    vector<RpptROI3D> roi3ds(num_images);
    for (int i = 0; i < num_images; ++i) {
        roi3ds[i].xyzwhdROI.xyz.x = 0;
        roi3ds[i].xyzwhdROI.xyz.y = 0;
        roi3ds[i].xyzwhdROI.xyz.z = 0;
        roi3ds[i].xyzwhdROI.roiWidth = width;
        roi3ds[i].xyzwhdROI.roiHeight = height;
        roi3ds[i].xyzwhdROI.roiDepth = 1;
    }

    // Create multiplyTensor array (one value per image)
    vector<Rpp32f> multiplyTensor(num_images, mulVal);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(
            rppt_multiply_scalar(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                                 &dstGenericDesc, multiplyTensor.data(), roi3ds.data(),
                                 RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
            "multiply_scalar");
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST MultiplyScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "value=" + to_string(mulVal));
}

void benchmark_RPP_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(imgs.size());
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_and(imgs[i].data, imgs2[i].data, &srcDescs[i],
                                              out[i].data, &dstDescs[i], &rois[i],
                                              RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "bitwise_and");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST BitwiseAnd", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(imgs.size());
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_or(imgs[i].data, imgs2[i].data, &srcDescs[i], out[i].data,
                                             &dstDescs[i], &rois[i], RpptRoiType::XYWH, handle,
                                             RPP_HOST_BACKEND),
                             "bitwise_or");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST BitwiseOr", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_bitwise_not(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &rois[i],
                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "bitwise_not");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST BitwiseNot", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    // Use Rpp8u for U8 input images (tensor_min returns U8 for U8 input)
    vector<Rpp8u> minOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_min(imgs[i].data, &srcDescs[i],
                                             &minOutputs[i * imgOutputLength], imgOutputLength,
                                             &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "tensor_min");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST TensorMin", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    // Use Rpp8u for U8 input images (tensor_max returns U8 for U8 input)
    vector<Rpp8u> maxOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_max(imgs[i].data, &srcDescs[i],
                                             &maxOutputs[i * imgOutputLength], imgOutputLength,
                                             &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "tensor_max");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST TensorMax", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    // Use Rpp64u for U8 input images (tensor_sum returns U64 for U8 input)
    vector<Rpp64u> sumOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_sum(imgs[i].data, &srcDescs[i],
                                             &sumOutputs[i * imgOutputLength], imgOutputLength,
                                             &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "tensor_sum");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST TensorSum", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    vector<Rpp32f> meanOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(
                rppt_tensor_mean(imgs[i].data, &srcDescs[i], &meanOutputs[i * imgOutputLength],
                                 imgOutputLength, &rois[i], RpptRoiType::XYWH, handle,
                                 RPP_HOST_BACKEND),
                "tensor_mean");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST TensorMean", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    vector<Rpp32f> stddevOutputs(outputLength);
    vector<Rpp32f> meanOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    // First compute mean for stddev calculation
    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);

        // Compute mean first (stddev requires mean)
        Rpp32u imgOutputLength = isColor ? 4 : 1;
        CHECK_RPP_STATUS(rppt_tensor_mean(imgs[i].data, &srcDescs[i],
                                          &meanOutputs[i * imgOutputLength], imgOutputLength,
                                          &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                         "tensor_mean");
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(
                rppt_tensor_stddev(imgs[i].data, &srcDescs[i], &stddevOutputs[i * imgOutputLength],
                                   imgOutputLength, &meanOutputs[i * imgOutputLength], &rois[i],
                                   RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "tensor_stddev");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST TensorStddev", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<Mat> grayImgs(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32f minVal = static_cast<Rpp32f>(thresh);
    Rpp32f maxVal = 255.0f;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), CV_8UC1);
        if (isColor)
            cvtColor(imgs[i], grayImgs[i], COLOR_BGR2GRAY);
        else
            grayImgs[i] = imgs[i];

        srcDescs[i] = createRppDescriptor(grayImgs[i], RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NCHW);
        rois[i] = createFullImageROI(grayImgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_threshold(grayImgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &minVal,
                               &maxVal, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "threshold");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Threshold", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "threshold=" + to_string(thresh));
}

void benchmark_RPP_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stddev,
                                 rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    unsigned long long seed = 12345;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gaussian_noise(imgs[i].data, &srcDescs[i], out[i].data,
                                                 &dstDescs[i], &mean, &stddev, seed, &rois[i],
                                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "gaussian_noise");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST GaussianNoise", imgs.size(), isColor,
                duration<double, milli>(end - start).count(),
                "mean=" + to_string(mean) + ", stddev=" + to_string(stddev));
}

void benchmark_RPP_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb,
                                      rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u seed = 12345;
    Rpp32f saltProb = 0.5f;     // 50% of noise is salt (white)
    Rpp32f saltValue = 1.0f;    // Normalized value for salt (0.0 to 1.0 range)
    Rpp32f pepperValue = 0.0f;  // Normalized value for pepper (0.0 to 1.0 range)

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_salt_and_pepper_noise(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           &noiseProb, &saltProb, &saltValue, &pepperValue, seed,
                                           &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "salt_and_pepper_noise");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST SaltAndPepperNoise", imgs.size(), isColor,
                duration<double, milli>(end - start).count(),
                "probability=" + to_string(noiseProb));
}

void benchmark_RPP_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_copy(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                       handle, RPP_HOST_BACKEND),
                             "copy");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Copy", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            int idx1 = i;
            int idx2 = (i + 1) % num_images;
            CHECK_RPP_STATUS(rppt_bitwise_xor(imgs[idx1].data, imgs[idx2].data, &srcDescs[idx1],
                                              out[i].data, &dstDescs[i], &rois[i],
                                              RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "bitwise_xor");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST BitwiseXor", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_histogram_equalize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "histogram_equalize");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST HistogramEqualize", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // For image transpose, we swap H and W dimensions
    // Input layout: NHWC (batch, height, width, channels)
    // Output layout: NWHC (batch, width, height, channels)
    // Permutation: [0, 2, 1, 3] - keep N and C, swap H and W

    // Create generic descriptors for 4D tensor (NHWC)
    RpptGenericDesc srcGenericDesc, dstGenericDesc;
    srcGenericDesc.numDims = 4;
    srcGenericDesc.offsetInBytes = 0;
    srcGenericDesc.dataType = RpptDataType::U8;
    srcGenericDesc.layout = RpptLayout::NHWC;
    srcGenericDesc.dims[0] = num_images;  // N
    srcGenericDesc.dims[1] = height;      // H
    srcGenericDesc.dims[2] = width;       // W
    srcGenericDesc.dims[3] = channels;    // C

    // Compute strides for input (NHWC)
    srcGenericDesc.strides[3] = 1;                          // C stride
    srcGenericDesc.strides[2] = channels;                   // W stride
    srcGenericDesc.strides[1] = width * channels;           // H stride
    srcGenericDesc.strides[0] = height * width * channels;  // N stride

    // Output has swapped H and W
    dstGenericDesc.numDims = 4;
    dstGenericDesc.offsetInBytes = 0;
    dstGenericDesc.dataType = RpptDataType::U8;
    dstGenericDesc.layout = RpptLayout::NHWC;
    dstGenericDesc.dims[0] = num_images;  // N
    dstGenericDesc.dims[1] = width;       // W (swapped)
    dstGenericDesc.dims[2] = height;      // H (swapped)
    dstGenericDesc.dims[3] = channels;    // C

    // Compute strides for output (NWHC)
    dstGenericDesc.strides[3] = 1;                          // C stride
    dstGenericDesc.strides[2] = channels;                   // H stride (now at position 2)
    dstGenericDesc.strides[1] = height * channels;          // W stride (now at position 1)
    dstGenericDesc.strides[0] = width * height * channels;  // N stride

    // Permutation tensor: [0, 2, 1, 3] means swap dimensions 1 and 2 (H and W)
    Rpp32u permTensor[4] = {0, 2, 1, 3};

    // ROI tensor: for each image, specify the full ROI as [start_coords, size]
    // For 4D: [n_start, h_start, w_start, c_start, n_size, h_size, w_size, c_size]
    vector<Rpp32u> roiTensor(num_images * 8);
    for (int i = 0; i < num_images; ++i) {
        int idx = i * 8;
        roiTensor[idx + 0] = 0;         // n_start
        roiTensor[idx + 1] = 0;         // h_start
        roiTensor[idx + 2] = 0;         // w_start
        roiTensor[idx + 3] = 0;         // c_start
        roiTensor[idx + 4] = 1;         // n_size (process 1 image at a time)
        roiTensor[idx + 5] = height;    // h_size
        roiTensor[idx + 6] = width;     // w_size
        roiTensor[idx + 7] = channels;  // c_size
    }

    // Concatenate all images into a batch buffer
    int imageSize = height * width * channels;
    vector<Rpp8u> inputBuffer(num_images * imageSize);
    vector<Rpp8u> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i)
        memcpy(inputBuffer.data() + i * imageSize, imgs[i].data, imageSize);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(
            rppt_transpose(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                           &dstGenericDesc, permTensor, roiTensor.data(), handle, RPP_HOST_BACKEND),
            "transpose");
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Transpose", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Create LUT - simple inversion table
    Rpp8u lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = 255 - i;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_lut(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], lut,
                                      &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "lut");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST LUT", imgs.size(), isColor, duration<double, milli>(end - start).count());
}

void benchmark_RPP_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<Mat> grad_x(num_images);
    vector<Mat> grad_y(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Compute gradients first (Sobel)
    for (int i = 0; i < num_images; ++i) {
        Mat gray = isColor ? Mat() : imgs[i];
        if (isColor) cvtColor(imgs[i], gray, COLOR_BGR2GRAY);

        Sobel(gray, grad_x[i], CV_32F, 1, 0, 3);
        Sobel(gray, grad_y[i], CV_32F, 0, 1, 3);

        out[i] = Mat::zeros(imgs[i].size(), CV_32F);
        srcDescs[i] = createRppDescriptor(grad_x[i], RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_magnitude(grad_x[i].data, grad_y[i].data, &srcDescs[i], out[i].data,
                               &dstDescs[i], &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "magnitude");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Magnitude", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<Mat> grad_x(num_images);
    vector<Mat> grad_y(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Compute gradients first (Sobel)
    for (int i = 0; i < num_images; ++i) {
        Mat gray = isColor ? Mat() : imgs[i];
        if (isColor) cvtColor(imgs[i], gray, COLOR_BGR2GRAY);

        Sobel(gray, grad_x[i], CV_32F, 1, 0, 3);
        Sobel(gray, grad_y[i], CV_32F, 0, 1, 3);

        out[i] = Mat::zeros(imgs[i].size(), CV_32F);
        srcDescs[i] = createRppDescriptor(grad_x[i], RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_phase(grad_x[i].data, grad_y[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                           &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "phase");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Phase", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Normalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // For normalize, nDim represents the dimensionality of each image (excluding batch)
    // For 2D images: nDim = 3 (H, W, C)
    // Descriptor will be 4D: numDims = nDim + 1 = 4 (N, H, W, C)
    int nDim = 3;

    // Create generic descriptor for 4D tensor (NHWC)
    RpptGenericDesc genericDesc;
    genericDesc.numDims = nDim + 1;  // 4D: N, H, W, C
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::U8;
    genericDesc.layout = RpptLayout::NHWC;
    genericDesc.dims[0] = num_images;  // N (batch)
    genericDesc.dims[1] = height;      // H
    genericDesc.dims[2] = width;       // W
    genericDesc.dims[3] = channels;    // C

    // Compute strides for NHWC layout
    genericDesc.strides[3] = 1;                          // C stride
    genericDesc.strides[2] = channels;                   // W stride
    genericDesc.strides[1] = width * channels;           // H stride
    genericDesc.strides[0] = height * width * channels;  // N stride

    // axisMask: which axes to normalize over (excluding batch dimension N)
    // Bit 0 = H, Bit 1 = W, Bit 2 = C
    // axisMask = 0b011 = 3 (normalize over H and W, keep C separate for per-channel normalization)
    Rpp32u axisMask = 3;

    // Allocate mean and stddev tensors
    // When normalizing over H and W (axisMask = 3), we keep C dimension
    // Size = num_images * channels (one mean/stddev per channel per image)
    Rpp32u meanStddevSize = num_images * channels;
    vector<Rpp32f> meanTensor(meanStddevSize);
    vector<Rpp32f> stdDevTensor(meanStddevSize);

    // computeMeanStddev: bit 0 = compute mean, bit 1 = compute stddev
    // 3 = 0b11 = compute both internally
    Rpp8u computeMeanStddev = 3;

    // Scale and shift for normalization: output = (input - mean) / stddev * scale + shift
    Rpp32f scale = 1.0f;
    Rpp32f shift = 0.0f;

    // ROI tensor: nDim * 2 values per batch item (excluding N dimension)
    // For nDim=3: [h_start, w_start, c_start, h_size, w_size, c_size]
    vector<Rpp32u> roiTensor(num_images * nDim * 2);
    for (int i = 0; i < num_images; ++i) {
        int idx = i * (nDim * 2);       // 6 values per image
        roiTensor[idx + 0] = 0;         // h_start
        roiTensor[idx + 1] = 0;         // w_start
        roiTensor[idx + 2] = 0;         // c_start
        roiTensor[idx + 3] = height;    // h_size
        roiTensor[idx + 4] = width;     // w_size
        roiTensor[idx + 5] = channels;  // c_size
    }

    // Concatenate all images into a batch buffer
    int imageSize = height * width * channels;
    vector<Rpp8u> inputBuffer(num_images * imageSize);
    vector<Rpp8u> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i)
        memcpy(inputBuffer.data() + i * imageSize, imgs[i].data, imageSize);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(
            rppt_normalize(inputBuffer.data(), &genericDesc, outputBuffer.data(), &genericDesc,
                           axisMask, meanTensor.data(), stdDevTensor.data(), computeMeanStddev,
                           scale, shift, roiTensor.data(), handle, RPP_HOST_BACKEND),
            "normalize");
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Normalize", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Normalize_SingleImage(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    int nDim = 3;  // 3D per-image: H, W, C

    // Create output buffers for each image
    vector<Mat> outputImages(num_images);
    for (int i = 0; i < num_images; ++i)
        outputImages[i] = Mat::zeros(imgs[i].size(), imgs[i].type());

    // Create generic descriptor for SINGLE IMAGE (batchSize=1)
    RpptGenericDesc srcGenericDesc, dstGenericDesc;
    srcGenericDesc.numDims = nDim + 1;  // 4D: N, H, W, C
    srcGenericDesc.offsetInBytes = 0;
    srcGenericDesc.dataType = RpptDataType::U8;
    srcGenericDesc.layout = RpptLayout::NHWC;
    srcGenericDesc.dims[0] = 1;  // *** BATCH SIZE = 1 (single image) ***
    srcGenericDesc.dims[1] = height;
    srcGenericDesc.dims[2] = width;
    srcGenericDesc.dims[3] = channels;
    srcGenericDesc.strides[3] = 1;
    srcGenericDesc.strides[2] = channels;
    srcGenericDesc.strides[1] = width * channels;
    srcGenericDesc.strides[0] = height * width * channels;

    dstGenericDesc = srcGenericDesc;

    // axisMask: Bit 0=H, Bit 1=W, Bit 2=C
    // axisMask=3 (0b011) -> normalize over H and W, keep C separate
    Rpp32u axisMask = 3;

    // ROI for SINGLE IMAGE
    vector<Rpp32u> roiTensor(nDim * 2);  // Only 6 values for one image
    roiTensor[0] = 0;                    // h_start
    roiTensor[1] = 0;                    // w_start
    roiTensor[2] = 0;                    // c_start
    roiTensor[3] = height;               // h_size
    roiTensor[4] = width;                // w_size
    roiTensor[5] = channels;             // c_size

    // Mean/stddev for SINGLE IMAGE (one value per channel)
    vector<Rpp32f> meanTensor(channels, 0.0f);
    vector<Rpp32f> stdDevTensor(channels, 0.0f);

    Rpp8u computeMeanStddev = 3;  // Compute both internally
    Rpp32f scale = 1.0f;
    Rpp32f shift = 0.0f;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_normalize(imgs[i].data, &srcGenericDesc, outputImages[i].data, &dstGenericDesc,
                               axisMask, meanTensor.data(), stdDevTensor.data(), computeMeanStddev,
                               scale, shift, roiTensor.data(), handle, RPP_HOST_BACKEND),
                "normalize");
        }
    }
    auto end = high_resolution_clock::now();

    printResult("RPP HOST Normalize", num_images, isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Create perspective transform matrix (simple example)
    float perspectiveMatrix[9] = {1.0f, 0.1f, 0.0f, 0.1f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_warp_perspective(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                      perspectiveMatrix, RpptInterpolationType::BILINEAR, &rois[i],
                                      RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "warp_perspective");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST WarpPerspective", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Create simple identity remap (with slight distortion)
    int h = imgs[0].rows;
    int w = imgs[0].cols;
    vector<Rpp32f> mapX(num_images * h * w);
    vector<Rpp32f> mapY(num_images * h * w);

    // Create table descriptor
    RpptDesc tableDesc;
    tableDesc.n = num_images;
    tableDesc.h = h;
    tableDesc.w = w;
    tableDesc.c = 1;
    tableDesc.strides.nStride = h * w;
    tableDesc.strides.hStride = w;
    tableDesc.strides.wStride = 1;
    tableDesc.strides.cStride = 1;

    for (int i = 0; i < num_images; ++i) {
        Rpp32f* mapXImg = mapX.data() + i * h * w;
        Rpp32f* mapYImg = mapY.data() + i * h * w;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                mapYImg[idx] = y + sin(x * 0.01f) * 5.0f;
                mapXImg[idx] = x + cos(y * 0.01f) * 5.0f;
            }
        }

        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_remap(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        mapY.data() + i * h * w, mapX.data() + i * h * w,
                                        &tableDesc, RpptInterpolationType::BILINEAR, &rois[i],
                                        RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "remap");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST Remap", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, Rpp32f mul,
                                          Rpp32f add, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        // NCDHW layout for grayscale
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    // Create mul and add tensors (one value per image in batch)
    vector<Rpp32f> mulTensor(num_images, mul);
    vector<Rpp32f> addTensor(num_images, add);

    // Create ROI tensor using XYZWHD format (full image ROI for each image)
    vector<RpptROI3D> roiTensor(num_images);
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(rppt_fused_multiply_add_scalar(
                             inputBuffer.data(), &genericDesc, outputBuffer.data(), &genericDesc,
                             mulTensor.data(), addTensor.data(), roiTensor.data(),
                             RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                         "fused_multiply_add_scalar");
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST FusedMultiplyAddScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp8u> bitsTensor(num_images, (Rpp8u)bits);  // FIXED: was Rpp32u

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_posterize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                            bitsTensor.data(), &rois[i], RpptRoiType::XYWH, handle,
                                            RPP_HOST_BACKEND),
                             "posterize");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "bits=" << bits;
    printResult("RPP HOST Posterize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold,
                            rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> thresholdTensor(num_images, threshold / 255.0f);  // FIXED: was Rpp8u

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_solarize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           thresholdTensor.data(), &rois[i], RpptRoiType::XYWH,
                                           handle, RPP_HOST_BACKEND),
                             "solarize");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "threshold=" << (int)threshold;
    printResult("RPP HOST Solarize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_NoiseShot(const vector<Mat>& imgs, bool isColor, Rpp32f shot_noise_factor,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> shotNoiseFactor(num_images, shot_noise_factor);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_shot_noise(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                shotNoiseFactor.data(), 12345,  // FIXED: added seed parameter
                                &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "shot_noise");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "factor=" << shot_noise_factor;
    printResult("RPP HOST NoiseShot", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth,
                            Rpp32f gridRatio, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    RpptUintVector2D translateVector = {0, 0};  // No translation

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_gridmask(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], tileWidth,
                              gridRatio, 0.0f, translateVector,  // Uses scalars!
                              &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "gridmask");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "tileWidth=" << tileWidth << ", ratio=" << gridRatio;
    printResult("RPP HOST Gridmask", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift,
                             Rpp32f bShift, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<RpptRGB> rgbTensor(num_images);
    vector<Rpp32f> alphaTensor(num_images, 1.0f);

    for (int i = 0; i < num_images; ++i) {
        rgbTensor[i].R = (Rpp8u)max(0.0f, min(255.0f, rShift));
        rgbTensor[i].G = (Rpp8u)max(0.0f, min(255.0f, gShift));
        rgbTensor[i].B = (Rpp8u)max(0.0f, min(255.0f, bShift));

        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_cast(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                             rgbTensor.data(), alphaTensor.data(), &rois[i],
                                             RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "color_cast");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "R=" << rShift << ", G=" << gShift << ", B=" << bShift;
    printResult("RPP HOST ColorCast", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_ColorTemperature(const vector<Mat>& imgs, bool isColor, Rpp32s adjustmentValue,
                                    rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32s> adjustmentTensor(num_images, adjustmentValue);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_temperature(imgs[i].data, &srcDescs[i], out[i].data,
                                                    &dstDescs[i], adjustmentTensor.data(), &rois[i],
                                                    RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "color_temperature");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "adjustment=" << adjustmentValue;
    printResult("RPP HOST ColorTemperature", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity,
                            rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> intensityTensor(num_images, vignetteIntensity);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_vignette(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           intensityTensor.data(), &rois[i], RpptRoiType::XYWH,
                                           handle, RPP_HOST_BACKEND),
                             "vignette");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "intensity=" << vignetteIntensity;
    printResult("RPP HOST Vignette", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev,
                                  rppHandle_t handle) {
    int num_images = (int)imgs.size();
    if (num_images < 2) {
        cout << "NonLinearBlend requires at least 2 images. Skipping." << endl;
        return;
    }

    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> stdDevTensor(num_images, stdDev);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images - 1; ++i) {
            CHECK_RPP_STATUS(
                rppt_non_linear_blend(imgs[i].data, imgs[i + 1].data, &srcDescs[i], out[i].data,
                                      &dstDescs[i], stdDevTensor.data(), &rois[i],
                                      RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "non_linear_blend");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "stdDev=" << stdDev;
    printResult("RPP HOST NonLinearBlend", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u boxesPerImage,
                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    int channels = isColor ? 3 : 1;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            // Create fresh arrays for each image
            vector<RpptRoiLtrb> anchorBoxInfoTensor(boxesPerImage);
            vector<Rpp32f> colorBuffer(boxesPerImage * channels);
            Rpp32u numBoxes = boxesPerImage;

            // Create random boxes
            for (Rpp32u b = 0; b < boxesPerImage; ++b) {
                int box_w = 50 + (rand() % 100);
                int box_h = 50 + (rand() % 100);
                int x = rand() % max(1, w - box_w);
                int y = rand() % max(1, h - box_h);

                anchorBoxInfoTensor[b].lt.x = x;
                anchorBoxInfoTensor[b].lt.y = y;
                anchorBoxInfoTensor[b].rb.x = x + box_w;
                anchorBoxInfoTensor[b].rb.y = y + box_h;

                // Fill color (gray)
                for (int c = 0; c < channels; ++c) colorBuffer[b * channels + c] = 128.0f;
            }

            CHECK_RPP_STATUS(rppt_erase(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        anchorBoxInfoTensor.data(), colorBuffer.data(), &numBoxes,
                                        &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "erase");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "boxes=" << boxesPerImage;
    printResult("RPP HOST Erase", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage,
                                 rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            // Random number of boxes (2 to maxBoxesPerImage)
            Rpp32u numBoxes = 2 + (rand() % (maxBoxesPerImage - 1));

            // Create fresh box array for each image
            vector<RpptRoiLtrb> anchorBoxInfoTensor(maxBoxesPerImage);

            for (Rpp32u b = 0; b < numBoxes; ++b) {
                int box_w = 30 + (rand() % 80);
                int box_h = 30 + (rand() % 80);
                int x = rand() % max(1, w - box_w);
                int y = rand() % max(1, h - box_h);

                anchorBoxInfoTensor[b].lt.x = x;
                anchorBoxInfoTensor[b].lt.y = y;
                anchorBoxInfoTensor[b].rb.x = x + box_w;
                anchorBoxInfoTensor[b].rb.y = y + box_h;
            }

            // Fill remaining boxes with zeros
            for (Rpp32u b = numBoxes; b < maxBoxesPerImage; ++b) {
                anchorBoxInfoTensor[b].lt.x = 0;
                anchorBoxInfoTensor[b].lt.y = 0;
                anchorBoxInfoTensor[b].rb.x = 0;
                anchorBoxInfoTensor[b].rb.y = 0;
            }

            CHECK_RPP_STATUS(
                rppt_coarse_dropout(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                    anchorBoxInfoTensor.data(), &numBoxes, maxBoxesPerImage,
                                    &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "coarse_dropout");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "maxBoxes=" << maxBoxesPerImage;
    printResult("RPP HOST CoarseDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow,
                               Rpp32u numGridsPerColumn, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u boxesInEachImage = numGridsPerRow * numGridsPerColumn;

    // Pre-create output buffers and descriptors
    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    Rpp32u totalBoxes = num_images * boxesInEachImage;
    Rpp32f holeRatio = 0.4f;
    int seed = 12345;  // Fixed seed for reproducibility

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        // Initialize anchor boxes for entire batch using proper helper function
        vector<RpptRoiLtrb> anchorBoxInfoTensor(totalBoxes);
        Rpp32u maxHoleW = 0, maxHoleH = 0;

        init_grid_dropout_boxes(num_images, anchorBoxInfoTensor.data(), rois.data(),
                                numGridsPerColumn, numGridsPerRow, maxHoleW, maxHoleH, holeRatio,
                                seed);

        // Process each image individually
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_grid_dropout(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                  anchorBoxInfoTensor.data() + i * boxesInEachImage,
                                  boxesInEachImage, maxHoleW, maxHoleH, &rois[i], RpptRoiType::XYWH,
                                  handle, RPP_HOST_BACKEND),
                "grid_dropout");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "grid=" << numGridsPerRow << "x" << numGridsPerColumn << ", holeRatio=" << holeRatio;
    printResult("RPP HOST GridDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_RandomErase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    int channels = isColor ? 3 : 1;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            // Random box for this image
            int box_w = 50 + (rand() % 150);
            int box_h = 50 + (rand() % 150);
            int x = rand() % max(1, w - box_w);
            int y = rand() % max(1, h - box_h);

            // Create fresh box for each image
            RpptRoiLtrb anchorBox;
            anchorBox.lt.x = x;
            anchorBox.lt.y = y;
            anchorBox.rb.x = x + box_w;
            anchorBox.rb.y = y + box_h;

            // Create noise buffer
            int bufferSize = box_w * box_h * channels;
            vector<Rpp32f> colorBuffer(bufferSize);
            for (int j = 0; j < bufferSize; ++j) colorBuffer[j] = (rand() % 256) / 255.0f;

            CHECK_RPP_STATUS(
                rppt_random_erase(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &anchorBox,
                                  colorBuffer.data(), &rois[i], RpptRoiType::XYWH, handle,
                                  RPP_HOST_BACKEND),
                "random_erase");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST RandomErase", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "");
}

void benchmark_RPP_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    if (!isColor) {
        cout << "ColorTwist requires RGB images. Skipping." << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Color twist parameters: hue shift
    vector<Rpp32f> alpha(num_images, 1.0f);
    vector<Rpp32f> beta(num_images, 0.0f);
    vector<Rpp32f> hueShift(num_images, 60.0f);  // 60 degrees
    vector<Rpp32f> saturationFactor(num_images, 1.3f);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_twist(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                              alpha.data(), beta.data(), hueShift.data(),
                                              saturationFactor.data(), &rois[i], RpptRoiType::XYWH,
                                              handle, RPP_HOST_BACKEND),
                             "color_twist");
        }
    }
    auto end = high_resolution_clock::now();
    stringstream ss;
    ss << "hue=" << hueShift[0] << ", sat=" << saturationFactor[0];
    printResult("RPP HOST ColorTwist", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), ss.str());
}

void benchmark_RPP_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    if (num_images < 2) {
        cout << "CropAndPatch requires at least 2 images. Skipping." << endl;
        return;
    }

    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    vector<RpptROI> cropRois(num_images);
    vector<RpptROI> patchRois(num_images);

    // Setup: crop from one image, patch into another
    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstRois[i] = createFullImageROI(imgs[i]);

        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Crop region (center quarter)
        cropRois[i].xywhROI.xy.x = w / 4;
        cropRois[i].xywhROI.xy.y = h / 4;
        cropRois[i].xywhROI.roiWidth = w / 2;
        cropRois[i].xywhROI.roiHeight = h / 2;

        // Patch region (same as crop)
        patchRois[i] = cropRois[i];
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images - 1; ++i) {
            // Crop from img[i], patch into img[i+1]
            CHECK_RPP_STATUS(
                rppt_crop_and_patch(imgs[i].data, imgs[i + 1].data, &srcDescs[i], out[i].data,
                                    &dstDescs[i], &dstRois[i], &cropRois[i], &patchRois[i],
                                    RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "crop_and_patch");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST CropAndPatch", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "center_quarter");
}

void benchmark_RPP_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    int channels = isColor ? 3 : 1;

    // Normalization parameters
    vector<Rpp32f> offset(num_images * channels);
    vector<Rpp32f> multiplier(num_images * channels);
    vector<Rpp32u> mirror(num_images, 1);  // 1 = horizontal flip

    if (isColor) {
        Rpp32f mean[3] = {60.0f, 80.0f, 100.0f};
        Rpp32f stdDev[3] = {0.9f, 0.9f, 0.9f};
        for (int i = 0, j = 0; i < num_images; i++, j += 3) {
            for (int c = 0; c < 3; c++) {
                offset[j + c] = -mean[c] / stdDev[c];
                multiplier[j + c] = 1.0f / stdDev[c];
            }
        }
    } else {
        Rpp32f mean = 100.0f;
        Rpp32f stdDev = 0.9f;
        for (int i = 0; i < num_images; i++) {
            offset[i] = -mean / stdDev;
            multiplier[i] = 1.0f / stdDev;
        }
    }

    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Output size (half of input as crop)
        out[i] = Mat::zeros(h / 2, w / 2, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);

        // Destination ROI (where to crop from source)
        dstRois[i].xywhROI.xy.x = w / 4;
        dstRois[i].xywhROI.xy.y = h / 4;
        dstRois[i].xywhROI.roiWidth = w / 2;
        dstRois[i].xywhROI.roiHeight = h / 2;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop_mirror_normalize(
                                 imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                 offset.data(), multiplier.data(), mirror.data(), &dstRois[i],
                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "crop_mirror_normalize");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST CropMirrorNormalize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "crop_half+flip+normalize");
}

void benchmark_RPP_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    vector<RpptImagePatch> dstImgSizes(num_images);
    int channels = isColor ? 3 : 1;

    // Normalization parameters
    vector<Rpp32f> mean(num_images * channels);
    vector<Rpp32f> stdDev(num_images * channels);
    vector<Rpp32u> mirror(num_images, 1);  // Horizontal flip

    for (int i = 0, j = 0; i < num_images; i++, j += channels) {
        if (isColor) {
            mean[j] = 60.0f;
            stdDev[j] = 1.0f;
            mean[j + 1] = 80.0f;
            stdDev[j + 1] = 1.0f;
            mean[j + 2] = 100.0f;
            stdDev[j + 2] = 1.0f;
        } else {
            mean[i] = 100.0f;
            stdDev[i] = 1.0f;
        }
    }

    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Resize to half
        dstImgSizes[i].width = w / 2;
        dstImgSizes[i].height = h / 2;

        out[i] = Mat::zeros(h / 2, w / 2, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);

        dstRois[i].xywhROI.xy.x = 0;
        dstRois[i].xywhROI.xy.y = 0;
        dstRois[i].xywhROI.roiWidth = w / 2;
        dstRois[i].xywhROI.roiHeight = h / 2;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_resize_mirror_normalize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                             dstImgSizes.data(), RpptInterpolationType::BILINEAR,
                                             mean.data(), stdDev.data(), mirror.data(), &dstRois[i],
                                             RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "resize_mirror_normalize");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST ResizeMirrorNormalize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "resize_half+flip+normalize");
}

void benchmark_RPP_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> srcRois(num_images);
    vector<RpptROI> dstRois(num_images);
    vector<RpptImagePatch> dstImgSizes(num_images);
    vector<Rpp32u> mirror(num_images, 1);  // Horizontal flip

    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Source ROI (crop from center before resize)
        srcRois[i].xywhROI.xy.x = 10;
        srcRois[i].xywhROI.xy.y = 10;
        srcRois[i].xywhROI.roiWidth = w - 20;
        srcRois[i].xywhROI.roiHeight = h - 20;

        // Resize to half of source ROI
        dstImgSizes[i].width = (w - 20) / 2;
        dstImgSizes[i].height = (h - 20) / 2;

        // Final crop size
        dstRois[i].xywhROI.roiWidth = 50;
        dstRois[i].xywhROI.roiHeight = 50;

        out[i] = Mat::zeros(50, 50, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_resize_crop_mirror(
                                 imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                 dstImgSizes.data(), RpptInterpolationType::BILINEAR, mirror.data(),
                                 &dstRois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "resize_crop_mirror");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST ResizeCropMirror", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "resize+crop+flip");
}

void benchmark_RPP_RICAP(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    if (num_images < 4) {
        cout << "RICAP requires at least 4 images. Skipping." << endl;
        return;
    }

    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    int maxWidth = imgs[0].cols;
    int maxHeight = imgs[0].rows;

    vector<Rpp32u> permutationTensor(num_images * 4);
    RpptROI roiPtrInputCropRegion[4];

    init_ricap_boxes(maxWidth, maxHeight, num_images, permutationTensor.data(),
                     roiPtrInputCropRegion);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_ricap(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        permutationTensor.data(), roiPtrInputCropRegion,
                                        RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "ricap");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST RICAP", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "4way_cutmix");
}

// Forward declaration of helper functions from dropout_helpers.cpp
void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize,
                                   int channels, int seed);
void init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                         RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, int channels,
                         int BitDepthTestMode, int seed, int dropoutType, void* colorBuffer);

void benchmark_RPP_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                  rppHandle_t handle) {
    if (!isColor) {
        cout << "RPP HOST ChannelDropout - skipped (requires RGB)" << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Generate channel dropout mask
    int channels = 3;
    vector<Rpp32f> dropoutProbability(num_images, dropoutProb);
    vector<Rpp8u> dropoutTensor(num_images * channels);
    Rpp32u seed = 12345;

    generate_channel_dropout_mask(dropoutTensor.data(), dropoutProbability.data(), num_images,
                                  channels, seed);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_channel_dropout(imgs[i].data, &srcDescs[i], out[i].data,
                                                  &dstDescs[i], dropoutTensor.data(), &rois[i],
                                                  RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "channel_dropout");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST ChannelDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "prob=" + to_string(dropoutProb));
}

void benchmark_RPP_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                 rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u boxesInEachImage = numBoxes;
    Rpp32u seed = 12345;
    int channels = isColor ? 3 : 1;

    vector<RpptRoiLtrb> anchorBoxInfoTensor(num_images * boxesInEachImage);
    vector<Rpp32u> numBoxesTensor(num_images * boxesInEachImage, numBoxes);
    vector<Rpp8u> colorBuffer(num_images * boxesInEachImage * channels);

    for (int i = 0; i < num_images; ++i) {
        rois[i] = createFullImageROI(imgs[i]);
    }

    init_cutout_dropout(num_images, boxesInEachImage, numBoxesTensor.data(),
                        anchorBoxInfoTensor.data(), rois.data(), channels, 0 /* U8_TO_U8 */, seed,
                        1 /* cutout type */, colorBuffer.data());

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_cutout_dropout(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                    anchorBoxInfoTensor.data(), colorBuffer.data(),
                                    numBoxesTensor.data(), &rois[i], RpptRoiType::XYWH, handle,
                                    RPP_HOST_BACKEND),
                "cutout_dropout");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST CutoutDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "boxes=" + to_string(numBoxes));
}

void benchmark_RPP_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    vector<Rpp32s> qualityTensor(num_images, quality);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_jpeg_compression_distortion(imgs[i].data, &srcDescs[i], out[i].data,
                                                 &dstDescs[i], qualityTensor.data(), &rois[i],
                                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "jpeg_compression_distortion");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST JpegCompressionDistortion", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "quality=" + to_string(quality));
}

void benchmark_RPP_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    if (!isColor) {
        cout << "RPP HOST ChannelPermute - skipped (requires RGB)" << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    // Permutation tensor: BGR to RGB (swap channels 0 and 2)
    // For each image, specify permutation [2, 1, 0] to swap B and R channels
    vector<Rpp32u> permutationTensor(num_images * 3);
    for (int i = 0; i < num_images; i++) {
        permutationTensor[i * 3 + 0] = 2;  // R <- B
        permutationTensor[i * 3 + 1] = 1;  // G <- G
        permutationTensor[i * 3 + 2] = 0;  // B <- R
    }

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_channel_permute(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                     permutationTensor.data(), handle, RPP_HOST_BACKEND),
                "channel_permute");
        }
    }
    auto end = high_resolution_clock::now();
    printResult("RPP HOST ChannelPermute", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "BGR->RGB");
}

void benchmark_RPP_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    int channels = isColor ? 3 : 1;
    int numDims = 3;  // H, W, C (excluding batch) for NHWC layout

    // Slice parameters: extract center region (half the size)
    vector<Rpp32s> anchorTensor(num_images * numDims);
    vector<Rpp32s> shapeTensor(num_images * numDims);
    vector<Rpp32u> roiTensor(num_images * numDims * 2);
    vector<RpptROI> rois(num_images);

    // Create output buffer (same size as input, but we'll only use the sliced portion)
    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        rois[i] = createFullImageROI(imgs[i]);
    }

    // Create generic descriptor for the batch
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 4;  // N, H, W, C for NHWC
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::U8;
    genericDesc.layout = RpptLayout::NHWC;

    genericDesc.dims[0] = num_images;    // Batch size
    genericDesc.dims[1] = imgs[0].rows;  // Height
    genericDesc.dims[2] = imgs[0].cols;  // Width
    genericDesc.dims[3] = channels;      // Channels

    // Set strides for NHWC layout
    genericDesc.strides[0] = imgs[0].rows * imgs[0].cols * channels;  // Batch stride
    genericDesc.strides[1] = imgs[0].cols * channels;                 // Height stride
    genericDesc.strides[2] = channels;                                // Width stride
    genericDesc.strides[3] = 1;                                       // Channel stride

    // Initialize anchor, shape, and ROI tensors using the same logic as test suite
    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        int idx1 = i * 3;  // 3 dims per image (H, W, C)
        int idx2 = i * 6;  // 6 values per image (3 pairs for ROI)

        // For NHWC: order is [H, W, C]
        // Anchor: starting position (extract from center, starting at 1/4 of dimensions)
        roiTensor[idx2 + 0] = anchorTensor[idx1 + 0] = h / 4;  // Y anchor
        roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = w / 4;  // X anchor
        roiTensor[idx2 + 2] = anchorTensor[idx1 + 2] = 0;      // C anchor (all channels)

        // ROI bounds
        roiTensor[idx2 + 3] = h;         // H max
        roiTensor[idx2 + 4] = w;         // W max
        roiTensor[idx2 + 5] = channels;  // C max

        // Shape: size to extract (half the image size)
        shapeTensor[idx1 + 0] = h / 2;     // H shape
        shapeTensor[idx1 + 1] = w / 2;     // W shape
        shapeTensor[idx1 + 2] = channels;  // C shape (all channels)
    }

    Rpp8u fillValue = 0;
    bool enablePadding = false;

    // Concatenate all images into a single buffer
    size_t imageSize = imgs[0].rows * imgs[0].cols * channels;
    vector<Rpp8u> inputBuffer(num_images * imageSize);
    vector<Rpp8u> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        memcpy(inputBuffer.data() + i * imageSize, imgs[i].data, imageSize);
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(
            rppt_slice(inputBuffer.data(), &genericDesc, outputBuffer.data(), &genericDesc,
                       anchorTensor.data(), shapeTensor.data(), &fillValue, enablePadding,
                       roiTensor.data(), handle, RPP_HOST_BACKEND),
            "slice");
    }
    auto end = high_resolution_clock::now();

    // Copy results back to output mats (only the sliced portion)
    for (int i = 0; i < num_images; ++i) {
        int sliceH = imgs[i].rows / 2;
        int sliceW = imgs[i].cols / 2;
        int startY = imgs[i].rows / 4;
        int startX = imgs[i].cols / 4;

        // Extract the sliced region from output buffer
        for (int y = 0; y < sliceH; ++y) {
            for (int x = 0; x < sliceW; ++x) {
                for (int c = 0; c < channels; ++c) {
                    size_t outIdx = i * imageSize + (y * imgs[i].cols + x) * channels + c;
                    size_t matIdx = ((startY + y) * imgs[i].cols + (startX + x)) * channels + c;
                    out[i].data[matIdx] = outputBuffer[outIdx];
                }
            }
        }
    }

    printResult("RPP HOST Slice", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "center_half");
}
