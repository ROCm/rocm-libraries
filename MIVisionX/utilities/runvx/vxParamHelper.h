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


#ifndef __VX_PARAMHELPER_H__
#define __VX_PARAMHELPER_H__

#include "vxUtils.h"

// global OpenCV image count
extern int g_numCvUse;

// process OpenCV window key refresh
int ProcessCvWindowKeyRefresh(int waitKeyDelayInMilliSeconds);

// input track bars connected to scalar objects
int GuiTrackBarInitializeScalar(vx_reference obj, int id, float valueMin, float valueMax, float valueInc);

int GuiTrackBarInitializeMatrix(vx_reference obj, int id, float valueR, float valueInc);

int GuiTrackBarShutdown(vx_reference obj);

int GuiTrackBarProcessKey(int key);


#endif /* __VX_PARAMHELPER_H__ */