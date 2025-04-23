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

#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class SnPNoiseNode : public Node {
   public:
    SnPNoiseNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SnPNoiseNode() = delete;
    void init(float noise_prob, float salt_prob, float salt_value, float pepper_value, int seed);
    void init(FloatParam *noise_prob_param, FloatParam *salt_prob_param, FloatParam *salt_value_param, FloatParam *pepper_value_param, int seed);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<float> _noise_prob, _salt_prob, _salt_value, _pepper_value;
    constexpr static float NOISE_PROB_RANGE[2] = {0.1, 1};
    constexpr static float SALT_PROB_RANGE[2] = {0.1, 1};
    constexpr static float SALT_RANGE[2] = {0.1, 1};
    constexpr static float PEPPER_RANGE[2] = {0, 0.5};
    int _seed;
};
