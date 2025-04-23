/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MV_EXTRAS_POSTPROC_H
#define MV_EXTRAS_POSTPROC_H
#include <vector>

class CRegion;
typedef struct _BBDetectAttributes
{
    int classes;
    int imgw, imgh;
    int blockwd;        // stride of bb
    float conf_thresh;
    float nms_thresh;
} BBDetectAttributes, *pBBDetectAttributes;

typedef struct _PostprocData
{
	CRegion *region;
	const float *bb_biases;
	BBDetectAttributes BBData;
}PostprocData, *PostprocDataPtr;

struct rect
{
    float left, top, right, bottom;
};

typedef struct _ClassLabel
{
	int index;
	float probability;
}ClassLabel;

typedef struct _BBox
{
    float x, y, w, h;
    float confidence;
    int   label;
    int   imgnum;       // image number in batch where it belongs to.
} BBox;


MIVID_API_ENTRY mv_status MIVID_API_CALL mv_postproc_init(mivid_session session, int classes, int blockwd, const float *BB_biases, int bias_size, float conf_thresh, float nms_thresh, int imgw, int imgh);
MIVID_API_ENTRY void MIVID_API_CALL mv_postproc_shutdown(mivid_session handle);
MIVID_API_ENTRY mv_status MIVID_API_CALL mv_postproc_argmax(void *data, void *output, int topK, int n, int c, int h, int w);
MIVID_API_ENTRY mv_status MIVID_API_CALL mv_postproc_getBB_detections(mivid_session session, void *data, int n, int c, int h, int w, std::vector<BBox> &outputBB);

// class for detecting bounding boxes for networks like Yolo
class CRegion
{
public:
    CRegion(pBBDetectAttributes pBBAttr);
    ~CRegion();

    void Initialize(int c, int h, int w, int size);
    int GetObjectDetections(int n, int c, int h, int w, float* in_data, const float *biases, std::vector<BBox> &objects);

private:
    int Nb;              // number of bounding boxes
    bool initialized;
    int  frameNum;
    unsigned int outputSize;
    int totalObjectsPerClass;
    int classes;
    int imgh, imgw, blockwd;
    float conf_thresh, nms_thresh;
    float *output;
    std::vector<BBox> boxes;

    // private member functions
    void Reshape(float *input, float *output, int n, int size);
    float Sigmoid(float x);
    void SoftmaxRegion(float *input, int classes, float *output);
    int argmax(float *a, int n);
    float box_iou(BBox a, BBox b);              // intersection over union
};

#endif