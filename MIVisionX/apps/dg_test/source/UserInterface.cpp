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

#include <string>
#include "UserInterface.h"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace cv;
using namespace std;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define cvDestroyWindow 
#define CV_EVENT_LBUTTONDOWN EVENT_LBUTTONDOWN
#define CV_EVENT_LBUTTONUP EVENT_LBUTTONUP
#define CV_EVENT_MOUSEMOVE EVENT_MOUSEMOVE
#define CV_AA 16
#endif

UserInterface::UserInterface(const char* weights) {
    // constructs a DGtest detector 
    mDetector = make_unique<DGtest>(weights);
}

UserInterface::~UserInterface() {}

void UserInterface::startUI() {
    CallbackData callbackData;
    callbackData.windowName = mWindow;
    cvui::init(mProgressWindow);
    namedWindow(mWindow, CV_WINDOW_AUTOSIZE);
    moveWindow(mWindow, 700, 500);
    moveWindow(mProgressWindow, 1040, 500);

    Mat img(300, 300, CV_8UC3, Scalar(0, 0, 0));
    copyMakeBorder(img, img, 20, 20, 20, 20, BORDER_CONSTANT, Scalar(69,51,0));
		
    Mat progressImage(300, 250, CV_8UC3, Scalar(223, 223, 223));

    callbackData.image = img.clone();
    Mat cloneImg = progressImage.clone();
    setMouseCallback(mWindow, UserInterface::onMouse, &callbackData);
    
    imshow(callbackData.windowName, callbackData.image);
    
    int key;

    cout << endl << "Press ESC to exit" << endl;

    do {
        key = waitKey(20);
        cvui::text(cloneImg, 75, 30, "Result", 1, 0x000000);
        if (cvui::button(cloneImg, 30, 250, 70, 25, "Clear")) {
            cloneImg = progressImage.clone();
            callbackData.image = img.clone();
            imshow(mWindow, callbackData.image);
        }
        
        if (cvui::button(cloneImg, 140, 250, 70, 25, "Run")) {
            Mat crop = callbackData.image(Rect(20, 20, 300, 300));
            mDetector->runInference(crop);
            cloneImg = progressImage.clone();
            cvui::text(cloneImg, 80, 100, to_string(mDetector->getResult()), 5, 0x0000ff);
        }

        // Update cvui internal stuff
        cvui::update();
        // Show window content
        cvui::imshow(mProgressWindow, cloneImg);
        
    } while (key != 27);

    destroyAllWindows();
}

void UserInterface::onMouse(int event, int x, int y, int, void *data) {
    CallbackData *callbackData = (CallbackData *) data;
    
    switch(event){

    case CV_EVENT_LBUTTONDOWN:
        callbackData->isDrawing = true;
        callbackData->p1.x = x;
        callbackData->p1.y = y;
        callbackData->p2.x = x;
        callbackData->p2.y = y;
        break;

    case CV_EVENT_LBUTTONUP:
        callbackData->p2.x = x;
        callbackData->p2.y = y;
        callbackData->isDrawing = false;
        break;

    case CV_EVENT_MOUSEMOVE:
        if(callbackData->isDrawing) {
            circle(callbackData->image, Point(x, y), 5, Scalar(255, 255, 255), 10, CV_AA);
            imshow(callbackData->windowName, callbackData->image);
        }
        break;

    default:
        break;
    }
}
