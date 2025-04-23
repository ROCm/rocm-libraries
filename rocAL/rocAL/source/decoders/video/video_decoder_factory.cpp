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
#include "decoders/video/video_decoder_factory.h"

#include "decoders/video/ffmpeg_video_decoder.h"
#include "decoders/video/hardware_video_decoder.h"
#include "decoders/video/rocdec_video_decoder.h"
#include "decoders/video/video_decoder.h"

#include "pipeline/commons.h"

#ifdef ROCAL_VIDEO
std::shared_ptr<VideoDecoder> create_video_decoder(DecoderConfig config) {
    switch (config.type()) {
        case DecoderType::FFMPEG_SW_DECODE:
            return std::make_shared<FFmpegVideoDecoder>();
        case DecoderType::FFMPEG_HW_DECODE:
            return std::make_shared<HardWareVideoDecoder>();
#if ENABLE_ROCDECODE
        case DecoderType::ROCDEC_VIDEO_DECODE:
            return std::make_shared<RocDecVideoDecoder>(config.get_hip_stream());
#endif
        default:
            THROW("Unsupported decoder type " + TOSTR(config.type()));
    }
}
#endif
