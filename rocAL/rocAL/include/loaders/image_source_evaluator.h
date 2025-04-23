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
#include <map>
#include <memory>

#include "loader_module.h"
#include "readers/image/reader_factory.h"
#include "pipeline/timing_debug.h"
#include "decoders/image/turbo_jpeg_decoder.h"
enum class ImageSourceEvaluatorStatus {
    OK = 0,
    UNSUPPORTED_DECODER_TYPE,
    UNSUPPORTED_STORAGE_TYPE,
};
enum class MaxSizeEvaluationPolicy {
    MAXIMUM_FOUND_SIZE,
    MOST_FREQUENT_SIZE
};

class ImageSourceEvaluator {
   public:
    ImageSourceEvaluatorStatus create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg);
    void find_max_dimension();
    void set_size_evaluation_policy(MaxSizeEvaluationPolicy arg);
    size_t max_width();
    size_t max_height();

   private:
    class FindMaxSize {
       public:
        void set_policy(MaxSizeEvaluationPolicy arg) { _policy = arg; }
        void process_sample(unsigned val);
        unsigned get_max() { return _max; };

       private:
        MaxSizeEvaluationPolicy _policy = MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE;
        std::map<unsigned, unsigned> _hist;
        unsigned _max = 0;
        unsigned _max_count = 0;
    };
    FindMaxSize _width_max;
    FindMaxSize _height_max;
    DecoderConfig _decoder_cfg_cv;
    std::shared_ptr<Decoder> _decoder;
    std::shared_ptr<Reader> _reader;
    std::shared_ptr<MetaDataReader> _meta_data_reader;
    std::vector<unsigned char> _header_buff;
    static const size_t COMPRESSED_SIZE = 1024 * 1024;  // 1 MB
};
