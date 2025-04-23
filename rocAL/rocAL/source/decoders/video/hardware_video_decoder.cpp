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

#include "decoders/video/hardware_video_decoder.h"
#include "pipeline/commons.h"
#include <stdio.h>

#ifdef ROCAL_VIDEO
HardWareVideoDecoder::HardWareVideoDecoder(){};

int HardWareVideoDecoder::SeekFrame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) {
    auto seek_time = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), AV_TIME_BASE_Q);
    int64_t select_frame_pts = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), time_base);
    int ret = av_seek_frame(_fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        ERR("HardWareVideoDecoder::SeekFrame Error in seeking frame. Unable to seek the given frame in a video");
        return ret;
    }
    return select_frame_pts;
}

int HardWareVideoDecoder::hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type, AVBufferRef *hw_device_ctx) {
    int err = 0;
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type, NULL, NULL, 0)) < 0)
        return err;
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    return err;
}

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status HardWareVideoDecoder::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format) {
    VideoDecoder::Status status = Status::OK;

    // Initialize the SwsContext
    SwsContext *swsctx = nullptr;
    if ((out_width != _codec_width) || (out_height != _codec_height) || (out_pix_format != _dec_pix_fmt)) {
        swsctx = sws_getCachedContext(nullptr, _codec_width, _codec_height, _dec_pix_fmt,
                                      out_width, out_height, out_pix_format, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!swsctx) {
            ERR("HardWareVideoDecoder::Decode Failed to get sws_getCachedContext");
            return Status::FAILED;
        }
    }
    int select_frame_pts = SeekFrame(_video_stream->avg_frame_rate, _video_stream->time_base, seek_frame_number);
    if (select_frame_pts < 0) {
        ERR("HardWareVideoDecoder::Decode Error in seeking frame. Unable to seek the given frame in a video");
        return Status::FAILED;
    }
    unsigned frame_count = 0;
    bool end_of_stream = false;
    bool sequence_filled = false;
    uint8_t *dst_data[4] = {0};
    int dst_linesize[4] = {0};
    int image_size = out_height * out_stride * sizeof(unsigned char);
    AVPacket pkt;
    AVFrame *dec_frame = av_frame_alloc();
    AVFrame *sw_frame = av_frame_alloc();
    if (!dec_frame) {
        ERR("HardWareVideoDecoder::Decode Could not allocate dec_frame");
        return Status::NO_MEMORY;
    }
    if (!sw_frame) {
        ERR("HardWareVideoDecoder::Decode Could not allocate sw_frame");
        return Status::NO_MEMORY;
    }
    do {
        int ret;
        // read packet from input file
        ret = av_read_frame(_fmt_ctx, &pkt);
        if (ret < 0 && ret != AVERROR_EOF) {
            ERR("HardWareVideoDecoder::Decode Failed to read the frame: ret=" + TOSTR(ret));
            status = Status::FAILED;
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
            ERR("HardWareVideoDecoder::Decode Error while sending packet to the decoder\n");
            status = Status::FAILED;
            break;
        }

        // get all the available frames from the decoder
        while (ret >= 0) {
            ret = avcodec_receive_frame(_video_dec_ctx, dec_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if ((dec_frame->pts < select_frame_pts) || (ret < 0)) continue;
            if (frame_count % stride == 0) {
                // retrieve data from GPU to CPU
                if ((av_hwframe_transfer_data(sw_frame, dec_frame, 0)) < 0) {
                    ERR("HardWareVideoDecoder::Decode avcodec_receive_frame() failed");
                    return Status::FAILED;
                }

                dst_data[0] = out_buffer;
                dst_linesize[0] = out_stride;
                if (swsctx)
                    sws_scale(swsctx, sw_frame->data, sw_frame->linesize, 0, sw_frame->height, dst_data, dst_linesize);
                else {
                    // copy from frame to out_buffer
                    memcpy(out_buffer, sw_frame->data[0], sw_frame->linesize[0] * out_height);
                }
                out_buffer = out_buffer + image_size;
            }
            ++frame_count;
            av_frame_unref(sw_frame);
            av_frame_unref(dec_frame);
            if (frame_count == sequence_length * stride) {
                sequence_filled = true;
                break;
            }
        }
        av_packet_unref(&pkt);
        if (sequence_filled) break;
    } while (!end_of_stream);
    avcodec_flush_buffers(_video_dec_ctx);
    av_frame_free(&dec_frame);
    av_frame_free(&sw_frame);
    sws_freeContext(swsctx);
    return status;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status HardWareVideoDecoder::Initialize(const char *src_filename, int device_id) {
    VideoDecoder::Status status = Status::OK;
    int ret;
    AVDictionary *opts = NULL;

    // open input file, and initialize the context required for decoding
    _fmt_ctx = avformat_alloc_context();
    _src_filename = src_filename;

    // find if hardware decode is available
    AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
    hw_type = av_hwdevice_find_type_by_name("vaapi");
    if (hw_type == AV_HWDEVICE_TYPE_NONE) {
        ERR("HardWareVideoDecoder::Initialize ERROR: vaapi is not supported for this device\n");
        return Status::FAILED;
    } else
        INFO("HardWareVideoDecoder::Initialize : Found vaapi device for the device\n");

    if (avformat_open_input(&_fmt_ctx, src_filename, NULL, NULL) < 0) {
        ERR("HardWareVideoDecoder::Initialize Couldn't Open video file " + STR(src_filename));
        return Status::FAILED;
    }
    if (avformat_find_stream_info(_fmt_ctx, NULL) < 0) {
        ERR("HardWareVideoDecoder::Initialize av_find_stream_info error");
        return Status::FAILED;
    }
    ret = av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &_decoder, 0);
    if (ret < 0) {
        ERR("HardWareVideoDecoder::Initialize Could not find %s stream in input file " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " " + STR(src_filename));
        return Status::FAILED;
    }
    // for hardware accelerated decoding, find config
    for (int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(_decoder, i);
        if (!config) {
            ERR("HardWareVideoDecoder::Initialize ERROR: decoder " + STR(_decoder->name) + " doesn't support device_type " + STR(av_hwdevice_get_type_name(hw_type)));
            return Status::FAILED;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == hw_type) {
            break;
        }
    }

    _video_stream_idx = ret;
    _video_stream = _fmt_ctx->streams[_video_stream_idx];

    if (!_video_stream) {
        ERR("HardWareVideoDecoder::Initialize Could not find video stream in the input, aborting");
        return Status::FAILED;
    }

    // find decoder for the stream
    _decoder = avcodec_find_decoder(_video_stream->codecpar->codec_id);
    if (!_decoder) {
        ERR("HardWareVideoDecoder::Initialize Failed to find " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::FAILED;
    }

    // Allocate a codec context for the decoder
    _video_dec_ctx = avcodec_alloc_context3(_decoder);
    if (!_video_dec_ctx) {
        ERR("HardWareVideoDecoder::Initialize Failed to allocate the " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec context");
        return Status::NO_MEMORY;
    }

    // Copy codec parameters from input stream to output codec context
    if ((ret = avcodec_parameters_to_context(_video_dec_ctx, _video_stream->codecpar)) < 0) {
        ERR("HardWareVideoDecoder::Initialize Failed to copy " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec parameters to decoder context");
        return Status::FAILED;
    }

    if (hw_decoder_init(_video_dec_ctx, hw_type, hw_device_ctx) < 0) {
        ERR("HardWareVideoDecoder::Initialize ERROR: Failed to create specified HW device");
        return Status::FAILED;
    }
    _dec_pix_fmt = AV_PIX_FMT_NV12;  // nv12 for vaapi

    // Init the decoders
    if ((ret = avcodec_open2(_video_dec_ctx, _decoder, &opts)) < 0) {
        ERR("HardWareVideoDecoder::Initialize Failed to open " +
            STR(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)) + " codec");
        return Status::FAILED;
    }
    _codec_width = _video_stream->codecpar->width;
    _codec_height = _video_stream->codecpar->height;
    return status;
}

void HardWareVideoDecoder::Release() {
    if (_video_dec_ctx)
        avcodec_free_context(&_video_dec_ctx);
    if (_fmt_ctx)
        avformat_close_input(&_fmt_ctx);
}

HardWareVideoDecoder::~HardWareVideoDecoder() {
    Release();
}
#endif
