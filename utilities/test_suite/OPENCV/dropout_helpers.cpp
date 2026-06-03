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

// Helper functions from rpp_test_suite_image.h for dropout operations
#include <random>
#include <omp.h>
#include "rpp.h"

void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize, int channels, int seed)
{
    int numThreads = omp_get_max_threads();
    omp_set_dynamic(0);

#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        std::mt19937 rng(seed + batchCount);
        std::bernoulli_distribution keepDist(1.0f - dropoutProbability[batchCount]);
        Rpp8u *maskPtrTemp = dropoutTensor + (batchCount * channels);
        bool atLeastOne = false;

        for (int channel = 0; channel < channels; channel++)
        {
            maskPtrTemp[channel] = keepDist(rng);
            atLeastOne |= maskPtrTemp[channel];
        }

        if (!atLeastOne)
            maskPtrTemp[rng() % channels] = 1;
    }
}

void init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes, RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, int channels, int BitDepthTestMode, int seed, int dropoutType, void *colorBuffer)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_ratio(0.1f, 0.9f);
    std::uniform_real_distribution<float> wh_ratio_cutout(0.4f, 0.6f);

    Rpp8u *colors8u = reinterpret_cast<Rpp8u *>(colorBuffer);
    Rpp16f *colors16f = reinterpret_cast<Rpp16f *>(colorBuffer);
    Rpp32f *colors32f = reinterpret_cast<Rpp32f *>(colorBuffer);
    Rpp8s *colors8s = reinterpret_cast<Rpp8s *>(colorBuffer);

    for (int i = 0; i < batchSize; i++)
    {
        numOfBoxes[i] = maxBoxesPerImage;
        for (int j = 0; j < maxBoxesPerImage; j++)
        {
            int idx = i * maxBoxesPerImage + j;

            // Get ROI dimensions
            Rpp32f roiWidth = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.roiWidth);
            Rpp32f roiHeight = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.roiHeight);
            Rpp32f roiX = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.xy.x);
            Rpp32f roiY = static_cast<Rpp32f>(roiTensorPtrSrc[i].xywhROI.xy.y);

            // Random box dimensions (40-60% of ROI)
            Rpp32f boxWidth = roiWidth * wh_ratio_cutout(rng);
            Rpp32f boxHeight = roiHeight * wh_ratio_cutout(rng);

            // Random position within ROI
            Rpp32f maxX = roiX + roiWidth - boxWidth;
            Rpp32f maxY = roiY + roiHeight - boxHeight;
            Rpp32f boxX = roiX + (maxX - roiX) * pos_ratio(rng);
            Rpp32f boxY = roiY + (maxY - roiY) * pos_ratio(rng);

            // Set anchor box in LTRB format
            anchorBoxInfoTensor[idx].lt.x = static_cast<Rpp32u>(boxX);
            anchorBoxInfoTensor[idx].lt.y = static_cast<Rpp32u>(boxY);
            anchorBoxInfoTensor[idx].rb.x = static_cast<Rpp32u>(boxX + boxWidth);
            anchorBoxInfoTensor[idx].rb.y = static_cast<Rpp32u>(boxY + boxHeight);

            // Set random color for the box
            for (int c = 0; c < channels; c++)
            {
                int colorIdx = idx * channels + c;
                if (BitDepthTestMode == 0) // U8
                    colors8u[colorIdx] = static_cast<Rpp8u>(rng() % 256);
                else if (BitDepthTestMode == 2) // F32
                    colors32f[colorIdx] = static_cast<Rpp32f>(rng() % 256) / 255.0f;
                else if (BitDepthTestMode == 1) // F16
                    colors16f[colorIdx] = static_cast<Rpp16f>(rng() % 256) / 255.0f;
                else if (BitDepthTestMode == 6) // I8
                    colors8s[colorIdx] = static_cast<Rpp8s>((rng() % 256) - 128);
            }
        }
    }
}
