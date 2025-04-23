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

#include "region.h"
#include "common.h"

// biases for Nb=5
const std::string classNames20[]    = { "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};

// helper functions
// sort indexes based on comparing values in v
template <typename T>
void sort_indexes(const std::vector<T> &v, std::vector<size_t> &idx) {
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
}

// todo:: convert using SSE4 intrinsics
// reshape transpose
void CYoloRegion::Reshape(float *input, float *output, int numChannels, int n)
{
    int i, j, p;
    float *tmp = output;
    for(i = 0; i < n; ++i)
    {
        for(j = 0, p = i; j < numChannels; ++j, p += n)
        {
            *tmp++ = input[p];
        }
    }
}

// todo:: optimize
float CYoloRegion::Sigmoid(float x)
{
    return 1./(1. + exp(-x));
}

void CYoloRegion::SoftmaxRegion(float *input, int classes, float *output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < classes; i++){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < classes; i++){
        float e = exp(input[i] - largest);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < classes; i++){
        output[i] /= sum;
    }
}


inline float rect_overlap(rect &a, rect &b)
{
    float x_overlap = std::max(0.f, (std::min(a.right, b.right) - std::max(a.left, b.left)));
    float y_overlap = std::max(0.f, (std::min(a.bottom, b.bottom) - std::max(a.top, b.top)));
    return (x_overlap * y_overlap);
}

float CYoloRegion::box_iou(box a, box b)
{
    float box_intersection, box_union;
    rect ra, rb;
    ra = {a.x-a.w/2, a.y-a.h/2, a.x+a.w/2, a.y+a.h/2};
    rb = {b.x-b.w/2, b.y-b.h/2, b.x+b.w/2, b.y+a.h/2};
    box_intersection = rect_overlap(ra, rb);
    box_union = a.w*a.h + b.w*b.h - box_intersection;

    return box_intersection/box_union;
}


int CYoloRegion::argmax(float *a, int n)
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



CYoloRegion::CYoloRegion()
{
    initialized = false;
    outputSize = 0;
    frameNum = 0;
}

CYoloRegion::~CYoloRegion()
{
    initialized = false;
    if (output)
        delete [] output;
    outputSize = 0;
}


void CYoloRegion::Initialize(int c, int h, int w, int classes)
{
    int size = 4 + classes + 1;     // x,y,w,h,pc, c1...c20

    outputSize = c * h * w;
    totalObjectsPerClass = Nb * h * w;
    output = new float[outputSize];
    boxes.resize(totalObjectsPerClass);
    initialized = true;
}

// Same as doing inference for this layer
int CYoloRegion::GetObjectDetections(float* in_data, const float *biases, int c, int h, int w,
                           int classes, int imgw, int imgh,
                           float thresh, float nms_thresh,
                           int blockwd,
                           std::vector<ObjectBB> &objects)
{
    objects.clear();

    int size = 4 + classes + 1;
    Nb = 5;//biases.size();
    if(!initialized)
    {
        Initialize(c, h, w, classes);
    }

    if(!initialized)
    {
        fatal("GetObjectDetections: initialization failed");
        return -1;
    }

    int i,j,k;

    Reshape(in_data, output, size*Nb, w*h);        // reshape output

    // Initialize box, scale and probability
    for(i = 0; i < totalObjectsPerClass; ++i)
    {
        int index = i * size;
        //Box
        int n = i % Nb;
        int row = (i/Nb) / w;
        int col = (i/Nb) % w;

        boxes[i].x = (col + Sigmoid(output[index + 0])) / blockwd;      // box x location
        boxes[i].y = (row + Sigmoid(output[index + 1])) / blockwd;      //  box y location
        boxes[i].w = exp(output[index + 2]) * biases[n*2]/ blockwd; //w;
        boxes[i].h = exp(output[index + 3]) * biases[n*2+1] / blockwd; //h;

        //Scale
        output[index + 4] = Sigmoid(output[index + 4]);

        //Class Probability
        SoftmaxRegion(&output[index + 5], classes, &output[index + 5]);

        // remove the ones which has low confidance
        for(j = 0; j < classes; ++j)
        {
            output[index+5+j] *= output[index+4];
            if(output[index+5+j] < thresh) output[index+5+j] = 0;
        }
    }

    //non_max_suppression using box_iou (intersection of union)
    for(k = 0; k < classes; ++k)
    {
        std::vector<float> class_prob_vec(totalObjectsPerClass);
        std::vector<size_t> s_idx(totalObjectsPerClass);
        for(i = 0; i < totalObjectsPerClass; ++i)
        {
            class_prob_vec[i] = output[i*size + k + 5];
            s_idx[i] = i;
        }
        //std::iota(idx.begin(), idx.end(), 0);         // todo::analyse for performance
        sort_indexes(class_prob_vec, s_idx);            // sort indeces based on prob
        for(i = 0; i < totalObjectsPerClass; ++i){
            if(output[s_idx[i] * size + k + 5] == 0) continue;
            box a = boxes[s_idx[i]];
            for(j = i+1; j < totalObjectsPerClass; ++j){
                box b = boxes[s_idx[j]];
                if (box_iou(a, b) > nms_thresh){
                    output[s_idx[j] * size + 5 + k] = 0;
                }
            }
        }
    }

    // generate objects
    for(i = 0, j = 5; i < totalObjectsPerClass; ++i, j += size)
    {
        int iclass = argmax(&output[j], classes);

        float prob = output[j+iclass];

        if(prob > thresh)
        {
            box b = boxes[i];

#if 0
            // boundingbox to actual coordinates
            printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
            int left  = (b.x-b.w/2.)*imgw;
            int right = (b.x+b.w/2.)*imgw;
            int top   = (b.y-b.h/2.)*imgh;
            int bot   = (b.y+b.h/2.)*imgh;
            if(left < 0) left = 0;
            if(right > imgw-1) right = imgw-1;
            if(top < 0) top = 0;
            if(bot > imgh-1) bot = imgh-1;
#endif
            ObjectBB obj;
            obj.x = b.x;
            obj.y = b.y;
            obj.w = b.w;
            obj.h = b.h;
            obj.confidence = prob;
            obj.label = iclass;
            //std::cout << "BoundingBox(xywh): "<< i << "for frame: "<< frameNum << " (" << b.x << " " << b.y << " "<< b.w << " "<< b.h << ") " << "confidence: " << prob << " lablel: " << iclass << std::endl;
            objects.push_back(obj);
        }
    }
    frameNum++;

    return 0;
}

