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

#include "Region.h"
#include "Common.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>


const int N = 5;
const double biases[N * 2] = { 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 };
//const float biases[N*2] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
const std::string objectnames[] = { "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor" };

Region::Region()
{
	initialized = false;
	totalLength = 0;
}


void Region::Initialize(int c, int h, int w, int size)
{
	totalLength = c * h * w;
	totalObjects = N * h * w;
	output.resize(totalLength);
	boxes.resize(totalObjects);
	s.resize(totalObjects);

	for (int i = 0; i < totalObjects; ++i)
	{
		s[i].index = i;
		s[i].channel = size;
		s[i].prob = &output[5];
	}


	initialized = true;
}


void Region::GetDetections(float* data, int c, int h, int w,
	int classes, int imgw, int imgh,
	float thresh, float nms,
	int blockwd,
	std::vector<DetectedObject> &objects)
{
	objects.clear();

	int size = 4 + classes + 1;
	if (!initialized)
	{
		Initialize(c, h, w, size);
	}

	if (!initialized)
	{
		printf("Fail to initialize internal buffer!\n");
		return;
	}

	int i, j, k;

	transpose(data, &output[0], size*N, w*h);

	// Initialize box, scale and probability
	for (i = 0; i < h*w*N; ++i)
	{
		int index = i * size;
		//Box
		int n = i % N;
		int row = (i / N) / w;
		int col = (i / N) % w;

		boxes[i].x = (col + logistic_activate(output[index + 0])) / blockwd; //w;
		boxes[i].y = (row + logistic_activate(output[index + 1])) / blockwd; //h;
		boxes[i].w = exp(output[index + 2]) * static_cast<float>(biases[2 * n]) / blockwd; //w;
		boxes[i].h = exp(output[index + 3]) * static_cast<float>(biases[2 * n + 1]) / blockwd; //h;

		//Scale
		output[index + 4] = logistic_activate(output[index + 4]);

		//Class Probability
		softmax(&output[index + 5], classes, 1, &output[index + 5]);
		for (j = 0; j < classes; ++j)
		{
			output[index + 5 + j] *= output[index + 4];
			if (output[index + 5 + j] < thresh) output[index + 5 + j] = 0;
		}
	}

	//nms
	for (k = 0; k < classes; ++k)
	{
		for (i = 0; i < totalObjects; ++i)
		{
			s[i].iclass = k;
		}
		qsort(&s[0], totalObjects, sizeof(indexsort), indexsort_comparator);
		for (i = 0; i < totalObjects; ++i) {
			if (output[s[i].index * size + k + 5] == 0) continue;
			ibox a = boxes[s[i].index];
			for (j = i + 1; j < totalObjects; ++j) {
				ibox b = boxes[s[j].index];
				if (box_iou(a, b) > nms) {
					output[s[j].index * size + 5 + k] = 0;
				}
			}
		}
	}

	// generate objects
	for (i = 0, j = 5; i < totalObjects; ++i, j += size)
	{
		int iclass = max_index(&output[j], classes);

		float prob = output[j + iclass];

		if (prob > thresh)
		{
			ibox b = boxes[i];

			//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			int left = static_cast<int>((b.x - b.w / 2.)*imgw);
			int right = static_cast<int>((b.x + b.w / 2.)*imgw);
			int top = static_cast<int>((b.y - b.h / 2.)*imgh);
			int bot = static_cast<int>((b.y + b.h / 2.)*imgh);

			if (left < 0) left = 0;
			if (right > imgw - 1) right = imgw - 1;
			if (top < 0) top = 0;
			if (bot > imgh - 1) bot = imgh - 1;


			DetectedObject obj;
			obj.left = left;
			obj.top = top;
			obj.right = right;
			obj.bottom = bot;
			obj.x = b.x;
			obj.y = b.y;
			obj.w = b.w;
			obj.h = b.h;
			obj.confidence = prob;
			obj.objType = iclass;
			obj.name = objectnames[iclass];
			//std::cout << "BoundingBox(ltrb): "<< i << "( " << left << " " << top << " "<< right << " "<< bot << ") " << "confidence: " << prob << " lablel: " << iclass << std::endl;			
			objects.push_back(obj);
		}
	}

	return;
}
