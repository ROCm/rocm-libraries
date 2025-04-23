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

#include <iostream>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #if USE_AVCODEC_GREATER_THAN_58_134
        #include <libavcodec/bsf.h>
    #endif
}

#include "roc_video_dec.h"
#include "roc_pybuffer.h"
  
#include <pybind11/pybind11.h>	 
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eval.h>

namespace py = pybind11;

struct PyPacketData {
    bool      end_of_stream;
    int       pkt_flags;
    int64_t   frame_pts;
    int64_t   frame_size;
    int64_t   bitstream_size;
    uintptr_t frame_adrs;       // yuv frame address
    uintptr_t bitstream_adrs;
    uintptr_t frame_adrs_rgb;   // rgb frame address
    uintptr_t frame_adrs_resized; // new resized yuv frame
    std::vector<std::shared_ptr<BufferInterface>> ext_buf;
    PyPacketData(){
        ext_buf.push_back(std::make_shared<BufferInterface>()); //index[0]: always Y Tensor
        ext_buf.push_back(std::make_shared<BufferInterface>()); //index[1]: UV tensor in case of NV12, otherwise only U tensor when YUV444/P016 is supported
        ext_buf.push_back(std::make_shared<BufferInterface>()); //index[2]: not-used in case of NV12, otherwise V tensor when YUV444/P016 is supported
    }
};

struct ConfigInfo {
    std::string device_name;
    std::string gcn_arch_name;
    int         pci_bus_id;
    int         pci_domain_id;
    int         pci_device_id;
};

// defined in roc_pyvideodemuxer.cpp
void PyVideoDemuxerInitializer(py::module& m);
void PyVideoStreamProviderInitializer(py::module& m);
rocDecVideoCodec ConvertAVCodec2RocDecVideoCodec(int av_codec);
rocDecVideoCodec ConvertAVCodecString2RocDecVideoCodec(std::string codec_name);

// defined in roc_pyvideodecoder.cpp
void PyRocVideoDecoderInitializer(py::module& m);

// defined in roc_pyvideodecodercpu.cpp
void PyRocVideoDecoderCpuInitializer(py::module& m);

// defined in BufferInterface.cpp
void PyExportInitializer(py::module& m);
