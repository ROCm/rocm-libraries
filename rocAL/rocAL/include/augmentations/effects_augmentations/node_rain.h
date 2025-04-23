/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class RainNode : public Node {
   public:
    RainNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    RainNode() = delete;
    void init(float rain_value, int rain_width, int rain_height, float rain_transparency);
    void init(FloatParam *rain_value, IntParam *rain_width, IntParam *rain_height, FloatParam *rain_transparency);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<float> _rain_value;
    ParameterVX<int> _rain_width;
    ParameterVX<int> _rain_height;
    ParameterVX<float> _rain_transparency;
    constexpr static float RAIN_VALUE_RANGE[2] = {0.15, 0.95};
    constexpr static int RAIN_WIDTH_RANGE[2] = {1, 2};
    constexpr static int RAIN_HEIGHT_RANGE[2] = {15, 17};
    constexpr static float RAIN_TRANSPARENCY_RANGE[2] = {0.2, 0.3};
};