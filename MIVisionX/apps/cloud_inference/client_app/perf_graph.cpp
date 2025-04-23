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

#include "perf_graph.h"
#include "ui_perf_graph.h"

perf_graph::perf_graph(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::perf_graph)
{
    ui->setupUi(this);
    maxFPS = 0;
    connect(ui->close_pushButton, &QAbstractButton::clicked, this, &perf_graph::closePerformanceView);
    connect(ui->reset_pushButton, &QAbstractButton::clicked, this, &perf_graph::resetPerformanceView);
}

perf_graph::~perf_graph()
{
    delete ui;
}
void perf_graph::closePerformanceView()
{
   this->close();
}
void perf_graph::setModelName(QString ModelName)
{
   ui->modelName_label->setText(ModelName);
}
void perf_graph::setStartTime(QString startTime)
{
   ui->StartTime_label->setText(startTime);
}
void perf_graph::updateElapsedTime(QString elapsedTime)
{
   ui->elapsedTime_label->setText(elapsedTime);
}
void perf_graph::setNumGPU(int numGPU)
{
   ui->GPU_lcdNumber->display(numGPU);
}
void perf_graph::resetPerformanceView()
{
    ui->fps_lcdNumber->display(0);
    ui->images_lcdNumber->display(0);
}
void perf_graph::updateFPSValue(float fps)
{
    fps = int(fps);
    ui->fps_lcdNumber->display(fps);

    if(maxFPS < fps){
        maxFPS = fps;
        ui->maxFPS_lcdNumber->display(maxFPS);
    }
}
void perf_graph::updateTotalImagesValue(int images)
{
    ui->images_lcdNumber->display(images);
}
