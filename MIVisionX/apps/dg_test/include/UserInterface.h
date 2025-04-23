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

#include <opencv2/opencv.hpp>
#include "DGtest.h"

struct CallbackData {
    cv::String windowName;
    cv::Mat image;
    bool isDrawing = false;
    cv::Point p1, p2;
};

class UserInterface
{
public:
    /**
     * Constructor
     */
    UserInterface(const char* weights);
    
    /**
     * Destructor
     */
    ~UserInterface();

    /**
     * Starts the UI
     */
    void startUI();

    /**
     * Mouse function for drawing
     */
    static void onMouse(int event, int x, int y, int, void*);
 
private:
    /**
     *  Main window name for user interface
     */
    cv::String mWindow = "Palette";

    /**
     *  Progress window name for user interface
     */
    cv::String mProgressWindow = "MIVisionX DGtest";

    /**
     *  Digit detector
     */
    std::unique_ptr<DGtest> mDetector;
};
