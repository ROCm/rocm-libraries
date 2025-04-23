/*
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


#ifndef _WINML_TUNNEL_
#define _WINML_TUNNEL_

#include "vx_ext_winml.h"
#include "vx_winml.h"

#include "winrt/Windows.Foundation.h"
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include<algorithm>
#include<functional>
#include <fstream>

#include <Windows.h>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Foundation::Collections;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Media;
using namespace Windows::Storage;

using namespace std;

#define STATUS_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) return status;}
#define PARAM_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) goto exit;}
#define MAX_KERNELS 100

TensorFloat VX_to_ML_tensor(vx_tensor inputTensor);
vx_status ML_to_VX_tensor(IVectorView<float> inputTensor, vx_tensor outputTensor);

class Kernellist
{
public:
        struct node{ public: std::function<vx_status(vx_context)> func; node* next; };
        int count;
        Kernellist(int max){ top = nullptr; maxnum = max; count = 0;}

        vx_status ADD(std::function<vx_status(vx_context)> element)
        {
                vx_status status = VX_SUCCESS;
                if (count == maxnum) return VX_ERROR_NO_RESOURCES;
                else
                {
                        node *newTop = new node;
                        if (top == nullptr){ newTop->func = element;    newTop->next = nullptr;  top = newTop;  count++; }
                        else{ newTop->func = element;   newTop->next = top; top = newTop; count++; }
                }
                return status;
        }

        vx_status REMOVE()
        {
                vx_status status = VX_SUCCESS;
                if (top == nullptr) return VX_ERROR_NO_RESOURCES;
                else{ node * old = top; top = top->next; count--; delete(old); }
                return status;
        }

        vx_status PUBLISH(vx_context context)
        {
                vx_status status = VX_SUCCESS;

                if (top == nullptr) { vxAddLogEntry((vx_reference)context, VX_ERROR_NO_RESOURCES, "PUBLISH Fail, Kernel list is empty");  return VX_ERROR_NO_RESOURCES; }

                else
                {
                        node * Kernel = top;
                        for (int i = 0; i < count; i++){ STATUS_ERROR_CHECK(Kernel->func(context)); Kernel = Kernel->next;}
                }
                return status;
        }

private:
        node *top; int maxnum;
};

static Kernellist *Kernel_List;

#endif
