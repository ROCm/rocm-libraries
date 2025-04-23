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
#ifdef ROCAL_VIDEO

#include "decoders/image/hw_jpeg_decoder.h"

#include "pipeline/commons.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>

#define AVIO_CONTEXT_BUF_SIZE 32768  // 32K

struct buffer_data {
    uint8_t *ptr;
    size_t size;  ///< size left in the buffer
};

static inline int num_hw_devices() {
    int num_hw_devices = 0;
    FILE *fp = popen("ls -l /dev/dri", "r");
    if (fp == NULL)
        return num_hw_devices;

    char *path = NULL;
    size_t length = 0;
    std::string line;
    while (getline(&path, &length, fp) >= 0) {
        line = std::string(path, length);
        if (line.find("renderD") != std::string::npos)
            num_hw_devices++;
    }
    pclose(fp);
    return num_hw_devices;
}

static enum AVPixelFormat get_vaapi_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;

    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_VAAPI)
            return *p;
    }
    WRN("HardwareJpegDecoder::Unable to decode using VA-API");

    return AV_PIX_FMT_NONE;
}

// ffmpeg helper functions for custom AVIOContex for bitstream reading
static int ReadFunc(void *ptr, uint8_t *buf, int buf_size) {
    struct buffer_data *bd = (struct buffer_data *)ptr;
    buf_size = FFMIN(buf_size, static_cast<int>(bd->size));

    if (!buf_size)
        return AVERROR_EOF;
    // printf("ptr:%p size:%zu\n", bd->ptr, bd->size);

    /* copy internal buffer data to buf */
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;

    return buf_size;
}

void HWJpegDecoder::initialize(int dev_id) {
    int ret = 0;
    char device[128] = "";
    char *pdevice = NULL;
    int num_devices = 1;  // default;
    num_devices = num_hw_devices();
    if (dev_id >= 0) {
        snprintf(device, sizeof(device), "/dev/dri/renderD%d", (128 + (dev_id % num_devices)));
        pdevice = device;
    }
    const char *device_name = pdevice ? pdevice : NULL;

    if ((ret = av_hwdevice_ctx_create(&_hw_device_ctx, AV_HWDEVICE_TYPE_VAAPI, device_name, NULL, 0)) < 0)
        THROW("Couldn't find vaapi device for device_id: " + device_name)
    _hw_type = av_hwdevice_find_type_by_name("vaapi");
    if (_hw_type == AV_HWDEVICE_TYPE_NONE) {
        THROW("HardwareJpegDecoder::Initialize ERROR: vaapi is not supported for this device\n");
    } else
        INFO("HardwareJpegDecoder::Initialize : Found vaapi device for the device\n");
};

Decoder::Status HWJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    struct buffer_data bd = {0};
    int ret = 0;
    bd.ptr = input_buffer;
    bd.size = input_size;

    if (!(_fmt_ctx = avformat_alloc_context())) {
        return Status::NO_MEMORY;
    }

    uint8_t *avio_ctx_buffer = new uint8_t[AVIO_CONTEXT_BUF_SIZE];
    if (!avio_ctx_buffer) {
        return Status::NO_MEMORY;
    }
    _io_ctx = avio_alloc_context(avio_ctx_buffer, AVIO_CONTEXT_BUF_SIZE,
                                 0, &bd, &ReadFunc, NULL, NULL);
    if (!_io_ctx) {
        return Status::NO_MEMORY;
    }

    _fmt_ctx->pb = _io_ctx;
    _fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    // ret = avformat_open_input(&_fmt_ctx, NULL, NULL, NULL);
    ret = avformat_open_input(&_fmt_ctx, NULL, NULL, NULL);
    if (ret < 0) {
        ERR("HardwareJpegDecoder::avformat_open_input failed");
        return Status::HEADER_DECODE_FAILED;
    }
    ret = avformat_find_stream_info(_fmt_ctx, NULL);
    if (ret < 0) {
        ERR("HardwareJpegDecoder::Initialize av_find_stream_info error");
        return Status::HEADER_DECODE_FAILED;
    }
    ret = av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &_decoder, 0);
    if (ret < 0) {
        ERR("HardwareJpegDecoder::Initialize Could not find %s stream in input file " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)));
        return Status::HEADER_DECODE_FAILED;
    }
    _video_stream_idx = ret;

    _video_dec_ctx = avcodec_alloc_context3(_decoder);
    if (!_video_dec_ctx) {
        ERR("HardwareJpegDecoder::Initialize Failed to allocate the " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec context");
        return Status::NO_MEMORY;
    }
    _video_stream = _fmt_ctx->streams[_video_stream_idx];

    if (!_video_stream) {
        ERR("HardwareJpegDecoder::Initialize Could not find video stream in the input, aborting");
        return Status::HEADER_DECODE_FAILED;
    }
    // Copy codec parameters from input stream to output codec context
    if ((ret = avcodec_parameters_to_context(_video_dec_ctx, _video_stream->codecpar)) < 0) {
        ERR("HardwareJpegDecoder::Initialize Failed to copy " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec parameters to decoder context");
        return Status::HEADER_DECODE_FAILED;
    }
    _video_dec_ctx->hw_device_ctx = av_buffer_ref(_hw_device_ctx);
    if (!_video_dec_ctx->hw_device_ctx) {
        ERR("HardwareJpegDecoder:: hardware device reference create failed.\n");
        return Status::NO_MEMORY;
    }
    _video_dec_ctx->get_format = get_vaapi_format;

    // for config for vaapi
    for (int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(_decoder, i);
        if (!config) {
            WRN("HardwareJpegDecoder::Initialize ERROR: decoder " + STR(_decoder->name) + " doesn't support device_type " + STR(av_hwdevice_get_type_name(hw_type)));
            return Status::HEADER_DECODE_FAILED;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == _hw_type) {
            break;
        }
    }

    if (_video_dec_ctx->pix_fmt == AV_PIX_FMT_YUVJ420P)
        _dec_pix_fmt = AV_PIX_FMT_NV12;  // set non-depracated format, vaapi uses NV12 for YUVJ420P
    else if (_video_dec_ctx->pix_fmt == AV_PIX_FMT_YUVJ422P)
        _dec_pix_fmt = AV_PIX_FMT_YUV422P;  // set non-depracated format
    else if (_video_dec_ctx->pix_fmt == AV_PIX_FMT_YUVJ444P)
        _dec_pix_fmt = AV_PIX_FMT_YUV444P;  // set non-depracated format
    else
        _dec_pix_fmt = _video_dec_ctx->pix_fmt;  // correct format will be set after vaapi initialization for hwdec

    // Init the decoders
    if ((ret = avcodec_open2(_video_dec_ctx, _decoder, NULL)) < 0) {
        ERR("HardwareJpegDecoder::Initialize Failed to open " + STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::HEADER_DECODE_FAILED;
    }
    _codec_width = _video_stream->codecpar->width;
    _codec_height = _video_stream->codecpar->height;
    *width = _codec_width;
    *height = _codec_height;

    return Status::OK;
}

Decoder::Status HWJpegDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                      size_t max_decoded_width, size_t max_decoded_height,
                                      size_t original_image_width, size_t original_image_height,
                                      size_t &actual_decoded_width, size_t &actual_decoded_height,
                                      Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size) {
    Decoder::Status status = Status::OK;

    AVPixelFormat out_pix_fmt = AV_PIX_FMT_RGB24;
    int planes = 3;

    switch (desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            out_pix_fmt = AV_PIX_FMT_GRAY8;
            planes = 1;
            break;
        case Decoder::ColorFormat::RGB:
            out_pix_fmt = AV_PIX_FMT_RGB24;
            break;
        case Decoder::ColorFormat::BGR:
            out_pix_fmt = AV_PIX_FMT_BGR24;
            break;
    };
    // Initialize the SwsContext
    SwsContext *swsctx = nullptr;
    if ((max_decoded_width != _codec_width) || (max_decoded_height != _codec_height) || (out_pix_fmt != _dec_pix_fmt)) {
        swsctx = sws_getCachedContext(nullptr, _codec_width, _codec_height, _dec_pix_fmt,
                                      max_decoded_width, max_decoded_height, out_pix_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!swsctx) {
            ERR("HardwareJpegDecoder::Decode Failed to get sws_getCachedContext");
            return Status::CONTENT_DECODE_FAILED;
        }
    }
    AVFrame *dec_frame = av_frame_alloc();
    AVFrame *sw_frame = av_frame_alloc();
    if (!dec_frame || !sw_frame) {
        ERR("HardwareJpegDecoder::Decode couldn't allocate dec_frame");
        return Status::NO_MEMORY;
    }

    bool end_of_stream = false;
    AVPacket pkt;
    uint8_t *dst_data[4] = {0};
    int dst_linesize[4] = {0};
    int image_size = max_decoded_height * max_decoded_width * planes * sizeof(unsigned char);

    do {
        int ret;
        // read packet from input file
        ret = av_read_frame(_fmt_ctx, &pkt);
        if (ret < 0 && ret != AVERROR_EOF) {
            ERR("HardwareJpegDecoder::Decode Failed to read the frame: ret=" + TOSTR(ret));
            status = Status::CONTENT_DECODE_FAILED;
            break;
        }
        if (ret == 0 && pkt.stream_index != _video_stream_idx) continue;
        end_of_stream = (ret == AVERROR_EOF);
        if (end_of_stream) {
            // null packet for bumping process
            pkt.data = nullptr;
            pkt.size = 0;
        }

        // submit the packet to the decoder
        ret = avcodec_send_packet(_video_dec_ctx, &pkt);
        if (ret < 0) {
            ERR("HardWareJpegDecoder::Decode Error while sending packet to the decoder\n");
            status = Status::CONTENT_DECODE_FAILED;
            break;
        }

        // get all the available frames from the decoder
        while (ret >= 0) {
            ret = avcodec_receive_frame(_video_dec_ctx, dec_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                ERR("HardWareJpegDecoder::avcodec_receive_frame failed\n");
                break;
            }
            // retrieve data from GPU to CPU
            if ((av_hwframe_transfer_data(sw_frame, dec_frame, 0)) < 0) {
                ERR("HardWareVideoDecoder::Decode avcodec_receive_frame() failed");
                status = Status::CONTENT_DECODE_FAILED;
                break;
            }
            dst_data[0] = output_buffer;
            dst_linesize[0] = max_decoded_width * planes;
            if (swsctx)
                sws_scale(swsctx, sw_frame->data, sw_frame->linesize, 0, sw_frame->height, dst_data, dst_linesize);
            else {
                // copy from frame to out_buffer
                memcpy(output_buffer, sw_frame->data[0], sw_frame->linesize[0] * max_decoded_height);
            }
            av_packet_unref(&pkt);
            output_buffer += image_size;
        }
    } while (!end_of_stream);

    av_packet_unref(&pkt);
    av_frame_free(&dec_frame);
    av_frame_free(&sw_frame);
    sws_freeContext(swsctx);
    avio_context_free(&_io_ctx);
    // release video_dec_context and fmt_context after each file decoding
    avcodec_free_context(&_video_dec_ctx);
    avformat_close_input(&_fmt_ctx);
    actual_decoded_width = max_decoded_width;
    actual_decoded_height = max_decoded_height;
    return status;
}

void HWJpegDecoder::release() {
    avio_context_free(&_io_ctx);
    if (_video_dec_ctx)
        avcodec_free_context(&_video_dec_ctx);
    if (_fmt_ctx)
        avformat_close_input(&_fmt_ctx);
}

HWJpegDecoder::~HWJpegDecoder() {
    release();
}

#endif