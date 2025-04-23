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

#include "AnnieYoloDetect.h"
#include "Visualize.h"

using namespace std;
using namespace cv;

void printUsage() {
	cout << "Usage: " << endl;
	cout << "  * Image" << endl;
	cout << "       .\MIVisionX_winml_YoloV2.exe --image [image - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]" << endl;
	cout << "  * Camera Capture" << endl;
	cout << "       .\MIVisionX_winml_YoloV2.exe --capture [0 - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]" << endl;
	cout << "  * Video" << endl;
	cout << "       .\MIVisionX_winml_YoloV2.exe --video [video - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]" << endl;
}

void convertBackSlash(string &modelLoc) {
	for (int i = 0; i < modelLoc.size();i++)
	{
		if (modelLoc[i] == '\\')
		{
			modelLoc.insert(i, "\\");
			i++;
		}
	}
}

int main(int argc, char ** argv) {
	
	if (argc < 5 || argc > 7) {
		printUsage();
		return -1;
	}
	
	string option = argv[1];
	string input = argv[2];
	if (strcmp(argv[3], "--modelLoc") != 0) {
		printUsage();
		return -1;
	}

	string modelLoc = argv[4];
	convertBackSlash(modelLoc);
	int confidence;
	if (argc > 6) {
		if (strcmp(argv[5], "--confidence") == 0) {
			confidence = atoi(argv[6]);
		}
		else {
			printUsage();
			return -1;
		}
	}
	else
		confidence = 20;

	//option mode
	int mode;

	unique_ptr<AnnieYoloDetect> Annie;

	if (option == "--image") {
		mode = 0;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, confidence, mode);
		Annie->detect();
	}
	else if (option == "--capture") {
		mode = 1;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, confidence, mode);
		Annie->detect();
	}
	else if (option == "--video") {
		mode = 2;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, confidence, mode);
		Annie->detect();
	}
	else {
		printUsage();
		return -1;
	}
	return 0;
}
