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

