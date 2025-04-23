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

#include "loaders/audio/audio_read_and_decode.h"

#include <cstring>
#include <iterator>

#include "decoders/audio/audio_decoder_factory.hpp"
#include "decoders/image/decoder_factory.h"

#ifdef ROCAL_AUDIO

Timing
AudioReadAndDecode::GetTiming() {
    Timing t;
    t.decode_time = _decode_time.get_timing();
    t.read_time = _file_load_time.get_timing();
    return t;
}

AudioReadAndDecode::AudioReadAndDecode() : _file_load_time("FileLoadTime", DBG_TIMING),
                                           _decode_time("DecodeTime", DBG_TIMING) {
}

AudioReadAndDecode::~AudioReadAndDecode() {
    _reader = nullptr;
    _decoder.clear();
}

void AudioReadAndDecode::Create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id) {
    // Can initialize it to any decoder types if needed
    _batch_size = batch_size;
    _decoder.resize(_batch_size);
    _decompressed_buff_ptrs.resize(_batch_size);
    _audio_meta_info.resize(_batch_size);
    _decoder_config = decoder_config;
    if ((_decoder_config._type != DecoderType::SKIP_DECODE)) {
        for (int i = 0; i < batch_size; i++) {
            _decoder[i] = create_audio_decoder(decoder_config);
        }
    }
    _num_threads = reader_config.get_cpu_num_threads();
    _reader = create_reader(reader_config);
}

void AudioReadAndDecode::Reset() {
    _reader->reset();
}

size_t
AudioReadAndDecode::Count() {
    return _reader->count_items();
}

size_t AudioReadAndDecode::last_batch_padded_size() {
    return _reader->last_batch_padded_size();
}

LoaderModuleStatus
AudioReadAndDecode::Load(float *audio_buffer,
                         DecodedDataInfo& audio_info,
                         const size_t max_decoded_samples,
                         const size_t max_decoded_channels) {
    if (max_decoded_samples == 0 || max_decoded_channels == 0)
        THROW("Zero audio dimension is not valid")
    if (!audio_buffer)
        THROW("Null pointer passed as output buffer")
    if (_reader->count_items() < _batch_size)
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load audios from the disk and push them into the buff
    unsigned file_counter = 0;
    const size_t audio_size = max_decoded_samples * max_decoded_channels;
    // Decode with the channels and size for a single audio
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();  // Debug timing
    while ((file_counter != _batch_size) && _reader->count_items() > 0) {
        size_t fsize = _reader->open();
        if (fsize == 0) {
            WRN("Opened file " + _reader->id() + " of size 0");
            continue;
        }
        _audio_meta_info[file_counter].file_name = _reader->id();
        _audio_meta_info[file_counter].file_path = _reader->file_path();
        _reader->close();
        file_counter++;
    }
    _file_load_time.end();  // Debug timing
    _decode_time.start();   // Debug timing
    if (_decoder_config._type != DecoderType::SKIP_DECODE) {
        for (size_t i = 0; i < _batch_size; i++) {
            _decompressed_buff_ptrs[i] = audio_buffer + (audio_size * i);
        }
#pragma omp parallel for num_threads(_num_threads)  // default(none) TBD: option disabled in Ubuntu 20.04
        for (size_t i = 0; i < _batch_size; i++) {
            int original_samples, original_channels;
            float original_sample_rate;
            if (_decoder[i]->Initialize(_audio_meta_info[i].file_path.c_str()) != AudioDecoder::Status::OK) {
                THROW("Decoder can't be initialized for file: " + _audio_meta_info[i].file_name.c_str())
            }
            if (_decoder[i]->DecodeInfo(&original_samples, &original_channels, &original_sample_rate) != AudioDecoder::Status::OK) {
                THROW("Unable to fetch decode info for file: " + _audio_meta_info[i].file_name.c_str())
            }
            _audio_meta_info[i].channels = original_channels;
            _audio_meta_info[i].samples = original_samples;
            _audio_meta_info[i].sample_rate = original_sample_rate;
            if (_decoder[i]->Decode(_decompressed_buff_ptrs[i]) != AudioDecoder::Status::OK) {
                THROW("Decoder failed for file: " + _audio_meta_info[i].file_name.c_str())
            }
            _decoder[i]->Release();
        }
        for (size_t i = 0; i < _batch_size; i++) {
            audio_info._data_names[i] = _audio_meta_info[i].file_name;
            audio_info._audio_samples[i] = _audio_meta_info[i].samples;
            audio_info._audio_channels[i] = _audio_meta_info[i].channels;
            audio_info._audio_sample_rates[i] = _audio_meta_info[i].sample_rate;
        }
    }
    _decode_time.end();  // Debug timing
    return LoaderModuleStatus::OK;
}
#endif
