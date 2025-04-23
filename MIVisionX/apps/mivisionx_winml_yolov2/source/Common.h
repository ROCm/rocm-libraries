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

#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdio.h>


struct DetectedObject
{
    int left, top, right, bottom;
    float x, y, w, h;       // bbox
    float confidence;
    int objType;
    std::string name;
};

struct ibox
{
    float x, y, w, h;
};

struct indexsort
{
    int iclass;
    int index;
    int channel;
    float* prob;
};

int indexsort_comparator(const void *pa, const void *pb);

float logistic_activate(float x);
void transpose(float *src, float* tar, int k, int n);
void softmax(float *input, int n, float temp, float *output);
float overlap(float x1, float w1, float x2, float w2);
float box_intersection(ibox a, ibox b);
float box_union(ibox a, ibox b);
float box_iou(ibox a, ibox b);
int max_index(float *a, int n);

