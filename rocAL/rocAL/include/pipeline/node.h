/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <memory>
#include <set>

#include "pipeline/graph.h"
#include "meta_data/meta_data_graph.h"
#include "pipeline/tensor.h"
class Node {
   public:
    Node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : _inputs(inputs),
                                                                                      _outputs(outputs),
                                                                                      _batch_size(outputs[0]->info().batch_size()) {}
    virtual ~Node();
    void create(std::shared_ptr<Graph> graph);
    void update_parameters();
    std::vector<Tensor *> input() { return _inputs; };
    std::vector<Tensor *> output() { return _outputs; };
    void add_next(const std::shared_ptr<Node> &node) {}      // To be implemented
    void add_previous(const std::shared_ptr<Node> &node) {}  // To be implemented
    std::shared_ptr<Graph> graph() { return _graph; }
    void set_meta_data(pMetaDataBatch meta_data_info) { _meta_data_info = meta_data_info; }
    bool _is_ssd = false;
    const Roi2DCords *get_src_roi() { return _inputs[0]->info().roi().get_2D_roi(); }
    const Roi2DCords *get_dst_roi() { return _outputs[0]->info().roi().get_2D_roi(); }

   protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    const std::vector<Tensor *> _inputs;
    const std::vector<Tensor *> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
    pMetaDataBatch _meta_data_info;
};
