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
 
#include "video_demuxer.h"
#include "roc_pydecode.h"

extern "C" {
    #include <libavformat/avio.h>
    #include <libavutil/file.h>
    #include <libavutil/mem.h>
}


//
// AMD Video Demuxer Python Interface class
//
class PyVideoDemuxer : public VideoDemuxer {

    protected:
        std::shared_ptr <PyPacketData> currentPacket;
        void InitPacket();

    public:
        PyVideoDemuxer(const char *input_file_path) : VideoDemuxer(input_file_path) { InitPacket(); }
        PyVideoDemuxer(VideoDemuxer::StreamProvider *stream_provider) : VideoDemuxer(stream_provider) { InitPacket(); }
        				
        // for python binding
        std::shared_ptr<PyPacketData> DemuxFrame();
        std::shared_ptr<PyPacketData> SeekFrame(int frame_number, int seek_mode, int seek_criteria);
        int GetCodecId();
        uint32_t PyGetBitDepth();
};

/**
 * @brief File stream provider class for demuxing data from memory
 * 
 */
class PyFileStreamProvider : public VideoDemuxer::StreamProvider {

public:
    PyFileStreamProvider(const char *input_file_path) {
        /* slurp file content into buffer */
        int ret = av_file_map(input_file_path, &buf_ptr_, &buffer_size_, 0, NULL);
        if (ret < 0) {
            std::cerr << "Couldn't map input file into memory." << std::endl;
            exit(-1);
        }      
    }

    ~PyFileStreamProvider() {
        av_file_unmap(buf_ptr_, buffer_size_);
    }

    // Fill in the buffer owned by the demuxer
    int GetData(uint8_t *p_buf, int n_buf) {
        // We simply copy from the mapped memory in this example. You may get your data from network or somewhere else
        if (!buffer_size_)
            return AVERROR_EOF;
        memcpy(p_buf, buf_ptr_, n_buf);
        buf_ptr_ += n_buf;
        buffer_size_ -= n_buf;
        return n_buf;
    }

    size_t GetBufferSize() { return buffer_size_; };    

private:
    uint8_t *buf_ptr_ = nullptr;
    size_t buffer_size_ = 0; ///< size left in the buffer
};
