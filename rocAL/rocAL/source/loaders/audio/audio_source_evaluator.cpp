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

#include "loaders/audio/audio_source_evaluator.h"

#include "decoders/audio/audio_decoder_factory.hpp"
#include "readers/image/reader_factory.h"

#ifdef ROCAL_AUDIO

size_t AudioSourceEvaluator::GetMaxSamples() {
    return _samples_max;
}

size_t AudioSourceEvaluator::GetMaxChannels() {
    return _channels_max;
}

AudioSourceEvaluatorStatus
AudioSourceEvaluator::Create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg) {
    AudioSourceEvaluatorStatus status = AudioSourceEvaluatorStatus::OK;
    // Can initialize it to any decoder types if needed
    _decoder = create_audio_decoder(std::move(decoder_cfg));
    _reader = create_reader(std::move(reader_cfg));
    FindMaxDimension();
    return status;
}

void AudioSourceEvaluator::FindMaxDimension() {
    _reader->reset();
    auto root_folder_path = _reader->get_root_folder_path();
    auto relative_file_paths = _reader->get_file_paths_from_meta_data_reader();
    if ((relative_file_paths.size() > 0)) {
        for (auto rel_file_path : relative_file_paths) {
            std::string file_name = root_folder_path + "/" + rel_file_path;
            if (_decoder->Initialize(file_name.c_str()) != AudioDecoder::Status::OK) {
                WRN("Could not initialize audio decoder for file : " + _reader->id())
                continue;
            }
            int samples, channels;
            float sample_rate;
            if (_decoder->DecodeInfo(&samples, &channels, &sample_rate) != AudioDecoder::Status::OK) {
                WRN("Could not decode the header of the: " + _reader->id())
                continue;
            }
            if (samples <= 0 || channels <= 0)
                continue;
            _samples_max = std::max(samples, _samples_max);
            _channels_max = std::max(channels, _channels_max);
            _decoder->Release();
        }
    } else {
        while (_reader->count_items()) {
            size_t fsize = _reader->open();
            if (!fsize) continue;
            auto file_name = _reader->file_path();
            if (_decoder->Initialize(file_name.c_str()) != AudioDecoder::Status::OK) {
                WRN("Could not initialize audio decoder for file : " + _reader->id())
                continue;
            }
            int samples, channels;
            float sample_rate;
            if (_decoder->DecodeInfo(&samples, &channels, &sample_rate) != AudioDecoder::Status::OK) {
                WRN("Could not decode the header of the: " + _reader->id())
                continue;
            }
            if (samples <= 0 || channels <= 0)
                continue;
            _samples_max = std::max(samples, _samples_max);
            _channels_max = std::max(channels, _channels_max);
            _decoder->Release();
        }
    }
    // return the reader read pointer to the beginning of the resource
    _reader->reset();
}
#endif
