/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "parameters/parameter_vx.h"

class NormalizeNode : public Node {
   public:
     NormalizeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
     NormalizeNode() = delete;
     void init(std::vector<unsigned> &axes, std::vector<float> &mean, std::vector<float> &std_dev, float scale, float shift);

   protected:
     void create_node() override;
     void update_node() override {};

   private:
     int _axis_mask = 0;
     vx_array _mean_vx_array, _stddev_vx_array;
     std::vector<unsigned> _axes;
     std::vector<float> _mean, _std_dev;
     float _scale, _shift;
     std::vector<std::vector<uint32_t>> _normalize_roi;
     enum NormalizeModes {
         DO_NOT_COMPUTE = 0,      // Mean and Stddev values are passed from user
         COMPUTE_MEAN = 1,        // Compute mean from specified axes of input
         COMPUTE_STDDEV = 2,      // Compute stddev from specified axes of input
         COMPUTE_MEAN_STDDEV = 3  // Compute both mean and stddev from specified axes of input
     };
};
