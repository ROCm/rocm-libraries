#include "common.h"

int indexsort_comparator(const void *pa, const void *pb)
{
    float proba = ((indexsort *)pa)->prob[((indexsort *)pa)->index * ((indexsort *)pa)->channel + ((indexsort *)pa)->iclass];
    float probb = ((indexsort *)pb)->prob[((indexsort *)pb)->index * ((indexsort *)pb)->channel + ((indexsort *)pb)->iclass];

    float diff = proba - probb;
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float logistic_activate(float x)
{
    return 1./(1. + exp(-x));
}

void transpose(float *src, float* tar, int k, int n)
{
    int i, j, p;
    float *tmp = tar;
    for(i = 0; i < n; ++i)
    {
        for(j = 0, p = i; j < k; ++j, p += n)
        {
            *(tmp++) = src[p];
        }
    }
}
void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(ibox a, ibox b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(ibox a, ibox b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(ibox a, ibox b)
{
    return box_intersection(a, b)/box_union(a, b);
}
int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}