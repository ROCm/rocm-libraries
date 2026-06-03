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

// ==================== OPENCV BENCHMARK FUNCTIONS ====================

void benchmark_OpenCV_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            imgs[i].convertTo(out[i], -1, alpha, beta);
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("OpenCV Brightness", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
        lut.at<uchar>(i) = saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            LUT(imgs[i], lut, out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV GammaCorrection", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "gamma=" + to_string(gamma));
}

void benchmark_OpenCV_Blend(const vector<Mat>& imgs, bool isColor, float alpha)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(num_images);
    for (int i = 0; i < num_images; ++i)
        imgs[i].convertTo(imgs2[i], -1, 0.8, 30);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            addWeighted(imgs[i], alpha, imgs2[i], 1.0 - alpha, 0, out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Blend", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "alpha=" + to_string(alpha));
}

void benchmark_OpenCV_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor, float contrastCenter)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    float beta = contrastCenter * (1.f - contrastFactor);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            imgs[i].convertTo(out[i], -1, contrastFactor, beta);
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "factor=" << contrastFactor << ", center=" << contrastCenter;
    printResult("OpenCV Contrast", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_Exposure(const vector<Mat>& imgs, bool isColor, float stop)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    float scale = pow(2.0f, stop);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            imgs[i].convertTo(out[i], -1, scale, 0);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Exposure", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "stop=" + to_string(stop));
}

void benchmark_OpenCV_Hue(const vector<Mat>& imgs, float hueDelta)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat hsv;
            cvtColor(imgs[i], hsv, COLOR_BGR2HSV);
            vector<Mat> channels;
            split(hsv, channels);
            channels[0] += hueDelta;
            merge(channels, hsv);
            cvtColor(hsv, out[i], COLOR_HSV2BGR);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Hue", imgs.size(), true,
                duration_cast<milliseconds>(end - start).count(), "delta=" + to_string(hueDelta));
}

void benchmark_OpenCV_Saturation(const vector<Mat>& imgs, float satFactor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat hsv;
            cvtColor(imgs[i], hsv, COLOR_BGR2HSV);
            vector<Mat> channels;
            split(hsv, channels);
            channels[1] *= satFactor;
            merge(channels, hsv);
            cvtColor(hsv, out[i], COLOR_HSV2BGR);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Saturation", imgs.size(), true,
                duration_cast<milliseconds>(end - start).count(), "factor=" + to_string(satFactor));
}

void benchmark_OpenCV_ColorToGreyscale(const vector<Mat>& imgs)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            cvtColor(imgs[i], out[i], COLOR_BGR2GRAY);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ColorToGreyscale", imgs.size(), true,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast, float saturation, float hue)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat temp;
            imgs[i].convertTo(temp, -1, brightness, 0);
            temp.convertTo(temp, -1, contrast, 0);

            Mat hsv;
            cvtColor(temp, hsv, COLOR_BGR2HSV);
            vector<Mat> channels;
            split(hsv, channels);
            channels[0] += hue;
            channels[1] *= saturation;
            merge(channels, hsv);
            cvtColor(hsv, out[i], COLOR_HSV2BGR);
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "bright=" << brightness << ", cont=" << contrast << ", sat=" << saturation << ", hue=" << hue;
    printResult("OpenCV ColorJitter", imgs.size(), true,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            boxFilter(imgs[i], out[i], -1, Size(kernelSize, kernelSize),
                     Point(-1, -1), true, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV BoxFilter", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_OpenCV_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            medianBlur(imgs[i], out[i], kernelSize);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV MedianFilter", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_OpenCV_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize, double sigma)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            GaussianBlur(imgs[i], out[i], Size(kernelSize, kernelSize), sigma, sigma, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize << ", sigma=" << sigma;
    printResult("OpenCV GaussianFilter", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_SobelFilter(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
            Sobel(imgs[i], grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_REPLICATE);
            Sobel(imgs[i], grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_REPLICATE);
            convertScaleAbs(grad_x, abs_grad_x);
            convertScaleAbs(grad_y, abs_grad_y);
            addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out[i]);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV SobelFilter", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            int x = (imgs[i].cols - cropWidth) / 2;
            int y = (imgs[i].rows - cropHeight) / 2;
            Rect roi(x, y, cropWidth, cropHeight);
            out[i] = imgs[i](roi).clone();
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "w=" << cropWidth << ", h=" << cropHeight;
    printResult("OpenCV Crop", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH, int interpType, const string& interpName)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            resize(imgs[i], out[i], Size(dstW, dstH), 0, 0, interpType);
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("OpenCV Resize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_Flip(const vector<Mat>& imgs, bool isColor, int flipCode)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            flip(imgs[i], out[i], flipCode);
    }
    auto end = high_resolution_clock::now();
    string flipType = (flipCode == 0) ? "Vertical" : (flipCode == 1) ? "Horizontal" : "Both";
    printResult("OpenCV Flip", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "type=" + flipType);
}

void benchmark_OpenCV_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Point2f center(imgs[i].cols / 2.f, imgs[i].rows / 2.f);
            Mat rotMat = getRotationMatrix2D(center, angleDeg, 1.0);
            warpAffine(imgs[i], out[i], rotMat, imgs[i].size(), INTER_LINEAR, BORDER_REPLICATE);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Rotate", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "angle=" + to_string(angleDeg));
}

void benchmark_OpenCV_WarpAffine(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    Point2f srcTri[3] = {Point2f(0, 0), Point2f(100, 0), Point2f(0, 100)};
    Point2f dstTri[3] = {Point2f(10, 10), Point2f(90, 20), Point2f(20, 90)};
    Mat affineMat = getAffineTransform(srcTri, dstTri);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            warpAffine(imgs[i], out[i], affineMat, imgs[i].size(), INTER_LINEAR, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV WarpAffine", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            erode(imgs[i], out[i], kernel, Point(-1, -1), 1, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Erode", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_OpenCV_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            dilate(imgs[i], out[i], kernel, Point(-1, -1), 1, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Dilate", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_OpenCV_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Scalar s(addVal, addVal, addVal);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            add(imgs[i], s, out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV AddScalar", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "value=" + to_string(addVal));
}

void benchmark_OpenCV_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Scalar s(subVal, subVal, subVal);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            subtract(imgs[i], s, out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV SubtractScalar", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "value=" + to_string(subVal));
}

void benchmark_OpenCV_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            multiply(imgs[i], Scalar(mulVal, mulVal, mulVal), out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV MultiplyScalar", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "value=" + to_string(mulVal));
}

void benchmark_OpenCV_BitwiseAnd(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(num_images);
    for (int i = 0; i < num_images; ++i)
        imgs[i].convertTo(imgs2[i], -1, 0.9, 20);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            bitwise_and(imgs[i], imgs2[i], out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV BitwiseAnd", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_BitwiseOr(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(num_images);
    for (int i = 0; i < num_images; ++i)
        imgs[i].convertTo(imgs2[i], -1, 0.9, 20);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            bitwise_or(imgs[i], imgs2[i], out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV BitwiseOr", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_BitwiseNot(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            bitwise_not(imgs[i], out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV BitwiseNot", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_BitwiseXor(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(num_images);
    for (int i = 0; i < num_images; ++i)
        imgs[i].convertTo(imgs2[i], -1, 0.9, 20);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            bitwise_xor(imgs[i], imgs2[i], out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV BitwiseXor", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Threshold(const vector<Mat>& imgs, bool isColor, double thresh)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            threshold(imgs[i], out[i], thresh, 255, THRESH_BINARY);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Threshold", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "thresh=" + to_string(thresh));
}

void benchmark_OpenCV_HistogramEqualize(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            if (isColor)
            {
                Mat ycrcb;
                cvtColor(imgs[i], ycrcb, COLOR_BGR2YCrCb);
                vector<Mat> channels;
                split(ycrcb, channels);
                equalizeHist(channels[0], channels[0]);
                merge(channels, ycrcb);
                cvtColor(ycrcb, out[i], COLOR_YCrCb2BGR);
            }
            else
            {
                equalizeHist(imgs[i], out[i]);
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV HistogramEqualize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_LUT(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
        lut.at<uchar>(i) = 255 - i;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            LUT(imgs[i], lut, out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV LUT", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Magnitude(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat grad_x, grad_y;
            Sobel(imgs[i], grad_x, CV_32F, 1, 0);
            Sobel(imgs[i], grad_y, CV_32F, 0, 1);
            magnitude(grad_x, grad_y, out[i]);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Magnitude", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Phase(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat grad_x, grad_y;
            Sobel(imgs[i], grad_x, CV_32F, 1, 0);
            Sobel(imgs[i], grad_y, CV_32F, 0, 1);
            phase(grad_x, grad_y, out[i]);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Phase", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Normalize(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            normalize(imgs[i], out[i], 0, 255, NORM_MINMAX);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Normalize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_WarpPerspective(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    Point2f srcQuad[4] = {Point2f(0, 0), Point2f(100, 0), Point2f(100, 100), Point2f(0, 100)};
    Point2f dstQuad[4] = {Point2f(10, 10), Point2f(90, 20), Point2f(90, 90), Point2f(20, 80)};
    Mat perspMat = getPerspectiveTransform(srcQuad, dstQuad);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            warpPerspective(imgs[i], out[i], perspMat, imgs[i].size(), INTER_LINEAR, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV WarpPerspective", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Remap(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    if (imgs.empty()) return;

    Mat map_x(imgs[0].size(), CV_32FC1);
    Mat map_y(imgs[0].size(), CV_32FC1);
    for (int j = 0; j < map_x.rows; ++j)
    {
        for (int i = 0; i < map_x.cols; ++i)
        {
            map_x.at<float>(j, i) = i;
            map_y.at<float>(j, i) = map_x.rows - j;
        }
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            remap(imgs[i], out[i], map_x, map_y, INTER_LINEAR, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Remap", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, float mul, float add)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            imgs[i].convertTo(out[i], -1, mul, add);
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "mul=" << mul << ", add=" << add;
    printResult("OpenCV FusedMultiplyAddScalar", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_Transpose(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            transpose(imgs[i], out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Transpose", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Emboss(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    float kernel_data[] = {-1, -1, 0, -1, 0, 1, 0, 1, 1};
    Mat kernel(3, 3, CV_32F, kernel_data);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            filter2D(imgs[i], out[i], -1, kernel, Point(-1, -1), 128, BORDER_REPLICATE);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Emboss", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_TensorMin(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<double> minVals(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            double minVal, maxVal;
            minMaxLoc(imgs[i], &minVal, &maxVal);
            minVals[i] = minVal;
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV TensorMin", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_OpenCV_TensorMax(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<double> maxVals(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            double minVal, maxVal;
            minMaxLoc(imgs[i], &minVal, &maxVal);
            maxVals[i] = maxVal;
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV TensorMax", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_OpenCV_TensorSum(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Scalar> sumVals(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            sumVals[i] = cv::sum(imgs[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV TensorSum", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_OpenCV_TensorMean(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Scalar> meanVals(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            meanVals[i] = cv::mean(imgs[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV TensorMean", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_OpenCV_TensorStddev(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Scalar> meanVals(num_images);
    vector<Scalar> stddevVals(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            cv::meanStdDev(imgs[i], meanVals[i], stddevVals[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV TensorStddev", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_OpenCV_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stddev)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat noise(imgs[i].size(), CV_32FC(imgs[i].channels()));
            randn(noise, mean, stddev);
            Mat temp;
            imgs[i].convertTo(temp, CV_32FC(imgs[i].channels()));
            temp += noise;
            temp.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "mean=" << mean << ", stddev=" << stddev;
    printResult("OpenCV GaussianNoise", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            Mat noise(imgs[i].size(), CV_32F);
            randu(noise, 0, 1);

            for (int y = 0; y < out[i].rows; ++y)
            {
                for (int x = 0; x < out[i].cols; ++x)
                {
                    float val = noise.at<float>(y, x);
                    if (val < noiseProb / 2)
                        out[i].at<uchar>(y, x) = 0;
                    else if (val < noiseProb)
                        out[i].at<uchar>(y, x) = 255;
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV SaltAndPepperNoise", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "prob=" + to_string(noiseProb));
}

void benchmark_OpenCV_Copy(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            imgs[i].copyTo(out[i]);
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Copy", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    uchar mask = 0xFF << (8 - bits);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
            out[i] = imgs[i] & mask;
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Posterize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "bits=" + to_string(bits));
}

void benchmark_OpenCV_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            for (int y = 0; y < out[i].rows; ++y)
            {
                uchar* row = out[i].ptr<uchar>(y);
                for (int x = 0; x < out[i].cols * out[i].channels(); ++x)
                {
                    if (row[x] > threshold)
                        row[x] = 255 - row[x];
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Solarize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "threshold=" + to_string((int)threshold));
}

void benchmark_OpenCV_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat temp;
            imgs[i].convertTo(temp, CV_32F);
            Mat noise(imgs[i].size(), CV_32FC(imgs[i].channels()));
            randn(noise, 0, shotNoiseFactor * 255);

            for (int y = 0; y < temp.rows; ++y)
            {
                for (int x = 0; x < temp.cols; ++x)
                {
                    for (int c = 0; c < temp.channels(); ++c)
                    {
                        float val = temp.at<Vec3f>(y, x)[c];
                        float noiseVal = noise.at<Vec3f>(y, x)[c] * sqrt(val / 255.0f);
                        temp.at<Vec3f>(y, x)[c] = val + noiseVal;
                    }
                }
            }
            temp.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV NoiseShot", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "factor=" + to_string(shotNoiseFactor));
}

void benchmark_OpenCV_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth, Rpp32f gridRatio)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            Rpp32u gridSize = tileWidth * gridRatio;

            for (int y = 0; y < out[i].rows; y += tileWidth)
            {
                for (int x = 0; x < out[i].cols; x += tileWidth)
                {
                    int endY = min((int)(y + gridSize), out[i].rows);
                    int endX = min((int)(x + gridSize), out[i].cols);
                    out[i](Rect(x, y, endX - x, endY - y)).setTo(0);
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "tile=" << tileWidth << ", ratio=" << gridRatio;
    printResult("OpenCV Gridmask", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift, Rpp32f bShift)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            vector<Mat> channels;
            split(imgs[i], channels);
            channels[0] += bShift;
            channels[1] += gShift;
            channels[2] += rShift;
            merge(channels, out[i]);
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "r=" << rShift << ", g=" << gShift << ", b=" << bShift;
    printResult("OpenCV ColorCast", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_ColorTemperature(const vector<Mat>& imgs, bool isColor, Rpp32s adjustmentValue)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    float factor = adjustmentValue / 100.0f;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            vector<Mat> channels;
            split(imgs[i], channels);
            channels[2] = channels[2] * (1.0f + factor);   // Red
            channels[0] = channels[0] * (1.0f - factor);   // Blue
            merge(channels, out[i]);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ColorTemperature", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "adjustment=" + to_string(adjustmentValue));
}

void benchmark_OpenCV_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat mask(imgs[i].size(), CV_32F);
            Point2f center(imgs[i].cols / 2.f, imgs[i].rows / 2.f);
            float maxDist = sqrt(center.x * center.x + center.y * center.y);

            for (int y = 0; y < mask.rows; ++y)
            {
                for (int x = 0; x < mask.cols; ++x)
                {
                    float dx = x - center.x;
                    float dy = y - center.y;
                    float dist = sqrt(dx * dx + dy * dy);
                    mask.at<float>(y, x) = 1.0f - (dist / maxDist) * vignetteIntensity;
                }
            }

            Mat temp;
            imgs[i].convertTo(temp, CV_32F);
            vector<Mat> channels;
            split(temp, channels);
            for (auto& ch : channels)
                ch = ch.mul(mask);
            merge(channels, temp);
            temp.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Vignette", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "intensity=" + to_string(vignetteIntensity));
}

void benchmark_OpenCV_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(num_images);

    for (int i = 0; i < num_images; ++i)
        imgs[i].convertTo(imgs2[i], -1, 0.8, 30);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat temp1, temp2;
            imgs[i].convertTo(temp1, CV_32F, 1.0 / 255.0);
            imgs2[i].convertTo(temp2, CV_32F, 1.0 / 255.0);
            Mat blended = temp1.mul(temp2) * 255.0;
            blended.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV NonLinearBlend", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "stddev=" + to_string(stdDev));
}

void benchmark_OpenCV_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            for (Rpp32u b = 0; b < numBoxes; ++b)
            {
                int boxW = imgs[i].cols / 4;
                int boxH = imgs[i].rows / 4;
                int x = rand() % (imgs[i].cols - boxW);
                int y = rand() % (imgs[i].rows - boxH);
                out[i](Rect(x, y, boxW, boxH)).setTo(0);
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Erase", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "boxes=" + to_string(numBoxes));
}

void benchmark_OpenCV_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numDropouts)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            for (Rpp32u d = 0; d < numDropouts; ++d)
            {
                int w = 50 + rand() % 100;
                int h = 50 + rand() % 100;
                int x = rand() % max(1, imgs[i].cols - w);
                int y = rand() % max(1, imgs[i].rows - h);
                out[i](Rect(x, y, w, h)).setTo(0);
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV CoarseDropout", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "dropouts=" + to_string(numDropouts));
}

void benchmark_OpenCV_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow, Rpp32u numGridsPerColumn)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            int cellW = imgs[i].cols / numGridsPerRow;
            int cellH = imgs[i].rows / numGridsPerColumn;

            for (Rpp32u r = 0; r < numGridsPerRow; ++r)
            {
                for (Rpp32u c = 0; c < numGridsPerColumn; ++c)
                {
                    if (rand() % 2 == 0)
                    {
                        int x = r * cellW;
                        int y = c * cellH;
                        int w = min(cellW, imgs[i].cols - x);
                        int h = min(cellH, imgs[i].rows - y);
                        out[i](Rect(x, y, w, h)).setTo(0);
                    }
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "rows=" << numGridsPerRow << ", cols=" << numGridsPerColumn;
    printResult("OpenCV GridDropout", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_RandomErase(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            int w = 100 + rand() % 200;
            int h = 100 + rand() % 200;
            int x = rand() % max(1, imgs[i].cols - w);
            int y = rand() % max(1, imgs[i].rows - h);
            out[i](Rect(x, y, w, h)).setTo(Scalar(rand() % 256, rand() % 256, rand() % 256));
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV RandomErase", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_ColorTwist(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    float twist[] = {0.8f, 0.1f, 0.1f, 0.1f, 0.9f, 0.1f, 0.1f, 0.1f, 0.7f};
    Mat twistMat(3, 3, CV_32F, twist);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat temp;
            imgs[i].convertTo(temp, CV_32F);
            transform(temp, temp, twistMat);
            temp.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ColorTwist", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_CropAndPatch(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();
            int patchW = imgs[i].cols / 4;
            int patchH = imgs[i].rows / 4;
            int srcX = rand() % (imgs[i].cols - patchW);
            int srcY = rand() % (imgs[i].rows - patchH);
            int dstX = rand() % (imgs[i].cols - patchW);
            int dstY = rand() % (imgs[i].rows - patchH);

            Mat patch = imgs[i](Rect(srcX, srcY, patchW, patchH));
            patch.copyTo(out[i](Rect(dstX, dstY, patchW, patchH)));
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV CropAndPatch", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Scalar mean(127, 127, 127);
    Scalar stddev(50, 50, 50);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            int cropW = imgs[i].cols * 0.8;
            int cropH = imgs[i].rows * 0.8;
            int x = (imgs[i].cols - cropW) / 2;
            int y = (imgs[i].rows - cropH) / 2;

            Mat cropped = imgs[i](Rect(x, y, cropW, cropH));
            Mat flipped;
            flip(cropped, flipped, 1);

            Mat temp;
            flipped.convertTo(temp, CV_32F);
            temp = (temp - mean) / stddev;
            temp.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV CropMirrorNormalize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    Scalar mean(127, 127, 127);
    Scalar stddev(50, 50, 50);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat resized, flipped;
            resize(imgs[i], resized, Size(imgs[i].cols / 2, imgs[i].rows / 2));
            flip(resized, flipped, 1);

            Mat temp;
            flipped.convertTo(temp, CV_32F);
            temp = (temp - mean) / stddev;
            temp.convertTo(out[i], imgs[i].type());
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ResizeMirrorNormalize", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_ResizeCropMirror(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            Mat resized;
            resize(imgs[i], resized, Size(imgs[i].cols / 2, imgs[i].rows / 2));

            int cropW = resized.cols * 0.8;
            int cropH = resized.rows * 0.8;
            int x = (resized.cols - cropW) / 2;
            int y = (resized.rows - cropH) / 2;
            Mat cropped = resized(Rect(x, y, cropW, cropH));

            flip(cropped, out[i], 1);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ResizeCropMirror", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_RICAP(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    if (num_images < 4) return;

    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
            int halfW = imgs[i].cols / 2;
            int halfH = imgs[i].rows / 2;

            imgs[i % num_images](Rect(0, 0, halfW, halfH)).copyTo(out[i](Rect(0, 0, halfW, halfH)));
            imgs[(i+1) % num_images](Rect(0, 0, halfW, halfH)).copyTo(out[i](Rect(halfW, 0, halfW, halfH)));
            imgs[(i+2) % num_images](Rect(0, 0, halfW, halfH)).copyTo(out[i](Rect(0, halfH, halfW, halfH)));
            imgs[(i+3) % num_images](Rect(0, 0, halfW, halfH)).copyTo(out[i](Rect(halfW, halfH, halfW, halfH)));
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV RICAP", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb)
{
    if (!isColor)
    {
        cout << "OpenCV ChannelDropout - skipped (requires RGB)" << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    mt19937 rng(12345);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();

            // Randomly determine which channels to drop
            bernoulli_distribution keepDist(1.0f - dropoutProb);
            bool keepChannel[3];
            bool atLeastOne = false;

            for (int c = 0; c < 3; ++c)
            {
                keepChannel[c] = keepDist(rng);
                atLeastOne |= keepChannel[c];
            }

            // Ensure at least one channel is kept
            if (!atLeastOne)
                keepChannel[rng() % 3] = true;

            // Zero out dropped channels
            for (int c = 0; c < 3; ++c)
            {
                if (!keepChannel[c])
                {
                    vector<Mat> channels;
                    split(out[i], channels);
                    channels[c].setTo(0);
                    merge(channels, out[i]);
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ChannelDropout", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "prob=" + to_string(dropoutProb));
}

void benchmark_OpenCV_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            out[i] = imgs[i].clone();

            // Create cutout boxes with specific color fill
            for (Rpp32u b = 0; b < numBoxes; ++b)
            {
                // Random box size (40-60% of image dimensions)
                int boxW = static_cast<int>((0.4f + (rand() % 200) / 1000.0f) * imgs[i].cols);
                int boxH = static_cast<int>((0.4f + (rand() % 200) / 1000.0f) * imgs[i].rows);

                // Ensure box fits in image
                boxW = min(boxW, imgs[i].cols);
                boxH = min(boxH, imgs[i].rows);

                // Random position
                int x = rand() % max(1, imgs[i].cols - boxW + 1);
                int y = rand() % max(1, imgs[i].rows - boxH + 1);

                // Fill with random color (cutout style)
                Scalar color;
                if (isColor)
                    color = Scalar(rand() % 256, rand() % 256, rand() % 256);
                else
                    color = Scalar(rand() % 256);

                out[i](Rect(x, y, boxW, boxH)).setTo(color);
            }
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV CutoutDropout", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "boxes=" + to_string(numBoxes));
}

void benchmark_OpenCV_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    // JPEG compression parameters
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(quality);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            // Encode to JPEG buffer
            vector<uchar> buf;
            imencode(".jpg", imgs[i], buf, compression_params);

            // Decode from JPEG buffer
            out[i] = imdecode(buf, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV JpegCompressionDistortion", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "quality=" + to_string(quality));
}

void benchmark_OpenCV_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    // Create emboss kernel (3x3 or 5x5)
    Mat kernel;
    if (kernelSize == 3)
    {
        // 3x3 emboss kernel
        kernel = (Mat_<float>(3, 3) <<
            -1, -1,  0,
            -1,  0,  1,
             0,  1,  1);
    }
    else if (kernelSize == 5)
    {
        // 5x5 emboss kernel
        kernel = (Mat_<float>(5, 5) <<
            -1, -1, -1, -1,  0,
            -1, -1, -1,  0,  1,
            -1, -1,  0,  1,  1,
            -1,  0,  1,  1,  1,
             0,  1,  1,  1,  1);
    }
    else
    {
        // Default to 3x3
        kernel = (Mat_<float>(3, 3) <<
            -1, -1,  0,
            -1,  0,  1,
             0,  1,  1);
    }

    // Scale kernel by strength
    kernel *= strength;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            // Apply emboss filter using filter2D
            filter2D(imgs[i], out[i], -1, kernel);

            // Add 128 to center the result (convert from signed to unsigned range)
            out[i] += Scalar(128, 128, 128);
        }
    }
    auto end = high_resolution_clock::now();
    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("OpenCV Emboss", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), params.str());
}

void benchmark_OpenCV_ChannelPermute(const vector<Mat>& imgs, bool isColor)
{
    if (!isColor)
    {
        cout << "OpenCV ChannelPermute - skipped (requires RGB)" << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            // Split channels
            vector<Mat> channels;
            split(imgs[i], channels);

            // Permute: BGR to RGB (swap channels 0 and 2)
            vector<Mat> permuted = {channels[2], channels[1], channels[0]};

            // Merge back
            merge(permuted, out[i]);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV ChannelPermute", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "BGR->RGB");
}

void benchmark_OpenCV_Slice(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            // Slice from center, extract half-sized region
            int sliceH = h / 2;
            int sliceW = w / 2;
            int startY = h / 4;
            int startX = w / 4;

            // Use OpenCV Rect to extract the slice
            Rect roi(startX, startY, sliceW, sliceH);
            out[i] = imgs[i](roi).clone();
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Slice", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count(), "center_half");
}

void benchmark_OpenCV_Fisheye(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            int h = imgs[i].rows;
            int w = imgs[i].cols;
            out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());

            // Create fisheye effect using custom remap
            Mat mapX(h, w, CV_32F);
            Mat mapY(h, w, CV_32F);

            float centerX = w / 2.0f;
            float centerY = h / 2.0f;
            float radius = min(centerX, centerY);

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float dx = (x - centerX) / radius;
                    float dy = (y - centerY) / radius;
                    float r = sqrt(dx * dx + dy * dy);

                    if (r <= 1.0f)
                    {
                        // Fisheye distortion formula
                        float theta = atan(r);
                        float rNew = theta / (r + 0.0001f);

                        mapX.at<float>(y, x) = centerX + dx * rNew * radius;
                        mapY.at<float>(y, x) = centerY + dy * rNew * radius;
                    }
                    else
                    {
                        mapX.at<float>(y, x) = x;
                        mapY.at<float>(y, x) = y;
                    }
                }
            }

            remap(imgs[i], out[i], mapX, mapY, INTER_LINEAR, BORDER_REPLICATE);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV Fisheye", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}

void benchmark_OpenCV_LensCorrection(const vector<Mat>& imgs, bool isColor)
{
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    // Camera matrix (3x3)
    Mat cameraMatrix = (Mat_<double>(3, 3) <<
        534.07088364, 0.0, 341.53407554,
        0.0, 534.11914595, 232.94565259,
        0.0, 0.0, 1.0);

    // Distortion coefficients (k1, k2, p1, p2, k3)
    Mat distCoeffs = (Mat_<double>(5, 1) <<
        -0.29297164, 0.10770696, 0.00131038, -0.0000311, 0.0434798);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k)
    {
#if ENABLE_PARALLEL_THREADS
        #pragma omp parallel for num_threads(NUM_THREADS)
#endif
        for (int i = 0; i < num_images; ++i)
        {
            // Use OpenCV's undistort function for lens correction
            undistort(imgs[i], out[i], cameraMatrix, distCoeffs);
        }
    }
    auto end = high_resolution_clock::now();
    printResult("OpenCV LensCorrection", imgs.size(), isColor,
                duration_cast<milliseconds>(end - start).count());
}
