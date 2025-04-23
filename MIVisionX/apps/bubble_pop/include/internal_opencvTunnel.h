/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _CV_TUNNEL_
#define _CV_TUNNEL_

#if _WIN32
#include <windows.h>
#endif

#include "vx_pop.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <functional>

using namespace cv;
using namespace std;

#if USE_OPENCV_4
#define CV_BGR2RGB COLOR_BGR2RGB
#define CV_AA LINE_AA
#define cvScalar Scalar
#define cvPoint Point
#endif

#define STATUS_ERROR_CHECK(call)  \
	{                             \
		vx_status status = call;  \
		if (status != VX_SUCCESS) \
			return status;        \
	}
#define PARAM_ERROR_CHECK(call)   \
	{                             \
		vx_status status = call;  \
		if (status != VX_SUCCESS) \
			goto exit;            \
	}

int VX_to_CV_Image(Mat **, vx_image);
int VX_to_CV_MATRIX(Mat **, vx_matrix);

int CV_to_VX_Pyramid(vx_pyramid, vector<Mat>);
int CV_to_VX_Image(vx_image, Mat *);

int CV_to_VX_keypoints(vector<KeyPoint>, vx_array);
int CVPoints2f_to_VX_keypoints(vector<Point2f>, vx_array);
int CV_DESP_to_VX_DESP(Mat, vx_array, int);

int match_vx_image_parameters(vx_image, vx_image);
void overlay_bubble(const Mat &, const Mat &, Mat &, Point2i);

class Kernellist
{
public:
	struct node
	{
	public:
		std::function<vx_status(vx_context)> func;
		node *next;
	};
	int count;
	Kernellist(int max)
	{
		top = nullptr;
		maxnum = max;
		count = 0;
	}

	vx_status ADD(std::function<vx_status(vx_context)> element)
	{
		vx_status status = VX_SUCCESS;
		if (count == maxnum)
			return VX_ERROR_NO_RESOURCES;
		else
		{
			node *newTop = new node;
			if (top == nullptr)
			{
				newTop->func = element;
				newTop->next = nullptr;
				top = newTop;
				count++;
			}
			else
			{
				newTop->func = element;
				newTop->next = top;
				top = newTop;
				count++;
			}
		}
		return status;
	}

	vx_status REMOVE()
	{
		vx_status status = VX_SUCCESS;
		if (top == nullptr)
			return VX_ERROR_NO_RESOURCES;
		else
		{
			node *old = top;
			top = top->next;
			count--;
			delete (old);
		}
		return status;
	}

	vx_status PUBLISH(vx_context context)
	{
		vx_status status = VX_SUCCESS;

		if (top == nullptr)
		{
			vxAddLogEntry((vx_reference)context, VX_ERROR_NO_RESOURCES, "PUBLISH Fail, Kernel list is empty");
			return VX_ERROR_NO_RESOURCES;
		}

		else
		{
			node *Kernel = top;
			for (int i = 0; i < count; i++)
			{
				STATUS_ERROR_CHECK(Kernel->func(context));
				Kernel = Kernel->next;
			}
		}
		return status;
	}

private:
	node *top;
	int maxnum;
};

static Kernellist *Kernel_List;

#endif
