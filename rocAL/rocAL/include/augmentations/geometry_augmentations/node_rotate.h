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
#include "rocal_api_types.h"

class RotateNode : public Node {
   public:
    RotateNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    RotateNode() = delete;
    void init(float angle, RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION);
    void init(FloatParam *angle_param, RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION);
    unsigned int get_dst_width() { return _outputs[0]->info().max_shape()[0]; }
    unsigned int get_dst_height() { return _outputs[0]->info().max_shape()[1]; }
    vx_array get_angle() { return _angle.default_array(); }

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<float> _angle;
    int _interpolation_type;
    constexpr static float ROTATE_ANGLE_RANGE[2] = {0, 180};
};
