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

#include "Region.h"
#include "Visualize.h"
#include "vx_ext_winml.h"
#include "vx_ext_opencv.h"
#include "vx_ext_amd.h"

class AnnieYoloDetect {
public:
	//constructor
	AnnieYoloDetect(std::string mInput, std::string modelLoc, int confidence, int mode);
	
	//destructor
	~AnnieYoloDetect();

	//run inference on given input and show the detected bounding boxes
	void detect();

private:
	const int mMode;
	const int mConfidence;
	const int mWidth = 416;
	const int mHeight = 416;
	const std::string mInput;
	const std::string mModelLoc;
	std::unique_ptr<Region> mRegion;
	std::unique_ptr<Visualize> mVisualize;
};