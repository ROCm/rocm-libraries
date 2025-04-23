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

#include <memory>

#include "meta_data/bounding_box_graph.h"
#include "pipeline/exception.h"
#include "meta_data/meta_data_graph.h"
#include "meta_data/meta_data_reader.h"

std::shared_ptr<MetaDataGraph> create_meta_data_graph(const MetaDataConfig& config) {
    switch (config.type()) {
        case MetaDataType::Label: {
            return nullptr;
        }
        case MetaDataType::BoundingBox: {
            return std::make_shared<BoundingBoxGraph>();
        }
        case MetaDataType::PolygonMask: {
            return std::make_shared<BoundingBoxGraph>();
        }
        case MetaDataType::KeyPoints: {
            return std::make_shared<BoundingBoxGraph>();
        }

        default:
            THROW("MetaDataReader type is unsupported");
    }
}
