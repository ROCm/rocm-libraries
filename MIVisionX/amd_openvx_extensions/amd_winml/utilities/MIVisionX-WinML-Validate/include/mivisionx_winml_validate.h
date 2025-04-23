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

#pragma once

#include "winrt/Windows.Foundation.h"
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>

#include <string>
#include <fstream>
#include <sstream>

#include <Windows.h>

#define MIVISIONX_WINML_UTILITY_VERSION "1.0.0"

#ifdef _MSC_VER 
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Foundation::Collections;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Media;
using namespace Windows::Storage;

using namespace std;

inline int64_t clockCounter()
{
	return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
	return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

// Global variables
LearningModel model = nullptr;
LearningModelSession session = nullptr;
LearningModelBinding binding = nullptr;
VideoFrame imageFrame = nullptr;
vector<string> labels;
string labelsFilePath;
int deviceIndex;
const LearningModelDeviceKind deviceKindArray[5] = { LearningModelDeviceKind::Default,
												LearningModelDeviceKind::Cpu,
												LearningModelDeviceKind::DirectX,
												LearningModelDeviceKind::DirectXHighPerformance,
												LearningModelDeviceKind::DirectXMinPower
											 };
const string deviceNameArray[5] = { "Default",
								"Cpu",
								"DirectX",
								"DirectXHighPerformance",
								"DirectXMinPower"
							};

// Forward declarations
void LoadModelFromPath(hstring modelLocation);
VideoFrame LoadImageFile(hstring filePath);
void BindModel(hstring inputTensorName, hstring outputTensorName, int64_t *outputDim);
void EvaluateModel(hstring modelOutputTensorName);
void PrintResults(IVectorView<float> results);
void LoadLabels();
void EvaluateModelPlain();