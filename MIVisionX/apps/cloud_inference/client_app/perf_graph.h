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

#ifndef PERF_GRAPH_H
#define PERF_GRAPH_H

#include <QDialog>

namespace Ui {
class perf_graph;
}

class perf_graph : public QDialog
{
    Q_OBJECT

public:
    explicit perf_graph(QWidget *parent = 0);
    ~perf_graph();

private:
    Ui::perf_graph *ui;
    float maxFPS;

public slots:
     void closePerformanceView();
     void resetPerformanceView();
     void setModelName(QString ModelName);
     void setNumGPU(int numGPU);
     void setStartTime(QString startTime);
     void updateElapsedTime(QString elapsedTime);
     void updateFPSValue(float fps);
     void updateTotalImagesValue(int images);
};

#endif // PERF_GRAPH_H
