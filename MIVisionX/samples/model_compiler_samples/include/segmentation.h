#pragma once

#include <string>
#include <thread>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <inttypes.h>
#include <chrono>

#include "vx_ext_opencv.h"

// Name Output Display Windows
#define MIVisionX_LEGEND_S      "MIVisionX Image Segmentation"
#define MIVisionX_LEGEND_S_T    "MIVisionX Image Segmentation - Threshold"
#define MIVisionX_DISPLAY_S_I   "MIVisionX Image Segmentation - Input"
#define MIVisionX_DISPLAY_S_M   "MIVisionX Image Segmentation - MASK"
#define MIVisionX_DISPLAY_S_O   "MIVisionX Image Segmentation - Merged"

class Segment
{
public:
    bool initialized;
    int threshold_slider_max;
    int threshold_slider;
    double thresholdValue;
    int alpha_slider_max;
    int alpha_slider;
    double alphaValue;

    Segment();

    void initialize();

    static void threshold_on_trackbar( int threshold_slider_max, void* threshold_slider);

    static void alpha_on_trackbar( int threshold_slider_max, void* threshold_slider);

    void createLegendImage(std::string labeltext[]);

    void findClassProb(size_t start , size_t end, int width, int height, int numClasses, float* output_layer, float threshold, float* prob, unsigned char* classImg);

    void createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, cv::Mat& maskImage);

    void getMaskImage(cv::Mat& inputImage, int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, cv::Size input_geometry, cv::Mat& maskImage, std::string labelText[]);

};

