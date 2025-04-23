/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef YOLOREGION_H
#define YOLOREGION_H

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

struct box
{
    float x, y, w, h;
};

struct rect
{
    float left, top, right, bottom;
};

typedef struct _ObjectBB
{
    float x, y, w, h;
    float confidence;
    int   label;
} ObjectBB;


class CYoloRegion
{
public:
    CYoloRegion();
    ~CYoloRegion();

    void Initialize(int c, int h, int w, int size);
    int GetObjectDetections(float* in_data, const float *biases, int c, int h, int w,
                               int classes, int imgw, int imgh,
                               float thresh, float nms_thresh,
                               int blockwd,
                               std::vector<ObjectBB> &objects);
private:
    int Nb;              // number of bounding boxes
    bool initialized;
    int  frameNum;
    unsigned int outputSize;
    int totalObjectsPerClass;
    float *output;
    std::vector<box> boxes;

    // private member functions
    void Reshape(float *input, float *output, int n, int size);
    float Sigmoid(float x);
    void SoftmaxRegion(float *input, int classes, float *output);
    int argmax(float *a, int n);
    float box_iou(box a, box b);              // intersection over union
};

#endif // YOLOREGION_H

