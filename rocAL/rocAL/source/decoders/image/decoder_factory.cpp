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

#include "decoders/image/decoder_factory.h"

#include "decoders/image/decoder.h"
#include "decoders/image/fused_crop_decoder.h"
#include "decoders/image/hw_jpeg_decoder.h"
#include "decoders/image/open_cv_decoder.h"
#include "decoders/image/turbo_jpeg_decoder.h"
#include "decoders/image/rocjpeg_decoder.h"

#include "pipeline/commons.h"

std::shared_ptr<Decoder> create_decoder(DecoderConfig config) {
    switch (config.type()) {
        case DecoderType::TURBO_JPEG:
            return std::make_shared<TJDecoder>();
            break;
        case DecoderType::FUSED_TURBO_JPEG:
            return std::make_shared<FusedCropTJDecoder>();
            break;
#if ENABLE_OPENCV
        case DecoderType::OPENCV_DEC:
            return std::make_shared<CVDecoder>();
            break;
#endif
#if ROCAL_VIDEO
        case DecoderType::HW_JPEG_DEC:
            return std::make_shared<HWJpegDecoder>();
            break;
#endif
#if ENABLE_ROCJPEG
        case DecoderType::ROCJPEG_DEC:
            return std::make_shared<HWRocJpegDecoder>(config.get_hip_stream());
            break;
#endif
        default:
            THROW("Unsupported decoder type " + TOSTR(config.type()));
    }
}
