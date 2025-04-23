/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "loaders/numpy_source_evaluator.h"

#include "readers/image/reader_factory.h"

void NumpySourceEvaluator::create(ReaderConfig reader_cfg) {
    _reader = create_reader(std::move(reader_cfg));
    find_max_numpy_dimensions();
}

void NumpySourceEvaluator::find_max_numpy_dimensions() {
    _reader->reset();

    while (_reader->count_items()) {
        size_t fsize = _reader->open();
        if (fsize == 0) {
            _reader->close();
            continue;
        }
        const NumpyHeaderData numpy_header = _reader->get_numpy_header_data();
        _reader->close();

        if (_max_numpy_dims.size() == 0) {
            _max_numpy_dims.resize(numpy_header.array_shape.size());
            _numpy_dtype = numpy_header.type_info;
        }

        if (_max_numpy_dims.size() != numpy_header.array_shape.size()) {
            THROW("All numpy arrays must have the same number of dimensions")
        }

        if (_numpy_dtype != numpy_header.type_info) {
            THROW("All numpy arrays must have the same data type")
        }

        for (uint i = 0; i < _max_numpy_dims.size(); i++) {
            if (numpy_header.array_shape[i] > _max_numpy_dims[i]) {
                _max_numpy_dims[i] = numpy_header.array_shape[i];
            }
        }
    }
    // return the reader read pointer to the begining of the resource
    _reader->reset();
}
