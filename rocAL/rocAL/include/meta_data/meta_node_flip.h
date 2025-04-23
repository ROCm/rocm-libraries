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
#include <algorithm>
#include <memory>
#include <set>

#include "meta_data/bounding_box_graph.h"
#include "meta_data/meta_data.h"
#include "pipeline/node.h"
#include "augmentations/geometry_augmentations/node_flip.h"
#include "parameters/parameter_vx.h"

class FlipMetaNode : public MetaNode {
   public:
    FlipMetaNode(){};
    void update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data) override;
    std::shared_ptr<FlipNode> _node = nullptr;

   private:
    void initialize();
    std::vector<int> _h_flip_val, _v_flip_val;
};
