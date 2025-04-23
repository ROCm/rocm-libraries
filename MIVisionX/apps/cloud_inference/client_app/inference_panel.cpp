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

#include "inference_panel.h"
#include "ui_inference_panel.h"

inference_panel::inference_panel(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::inference_panel)
{
    ui->setupUi(this);
    display_panel.setWindowIcon(QIcon(":/images/vega_icon_150.png"));
    connect(ui->viewGraph_pushButton, &QAbstractButton::clicked, this, &inference_panel::viewPerformanceGraph);
}

inference_panel::~inference_panel()
{
    delete ui;
}

void inference_panel::viewPerformanceGraph()
{
    display_panel.show();
}


