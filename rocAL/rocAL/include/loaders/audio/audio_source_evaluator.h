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
#include <map>
#include <memory>

#include "loaders/loader_module.h"
#include "decoders/audio/generic_audio_decoder.h"

#ifdef ROCAL_AUDIO

enum class AudioSourceEvaluatorStatus {
    OK = 0,
    UNSUPPORTED_DECODER_TYPE,
    UNSUPPORTED_STORAGE_TYPE
};

class AudioSourceEvaluator {
   public:
    AudioSourceEvaluatorStatus Create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg);
    void FindMaxDimension();
    size_t GetMaxSamples();
    size_t GetMaxChannels();

   private:
    int _samples_max = 0, _channels_max = 0;
    std::shared_ptr<AudioDecoder> _decoder;
    std::shared_ptr<Reader> _reader;
};
#endif
