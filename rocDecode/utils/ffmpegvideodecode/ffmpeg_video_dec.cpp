/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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


#include "ffmpeg_video_dec.h"

/**
 * @brief helper function for inferring AVCodecID from rocDecVideoCodec
 * 
 * @param rocdec_codec 
 * @return AVCodecID 
 */
static inline AVCodecID RocDecVideoCodec2AVCodec(rocDecVideoCodec rocdec_codec) {
    switch (rocdec_codec) {
        case rocDecVideoCodec_MPEG1 : return AV_CODEC_ID_MPEG1VIDEO;
        case rocDecVideoCodec_MPEG2 : return AV_CODEC_ID_MPEG2VIDEO;
        case rocDecVideoCodec_MPEG4 : return AV_CODEC_ID_MPEG4;
        case rocDecVideoCodec_AVC   : return AV_CODEC_ID_H264;
        case rocDecVideoCodec_HEVC  : return AV_CODEC_ID_HEVC;
        case rocDecVideoCodec_VP8   : return AV_CODEC_ID_VP8;
        case rocDecVideoCodec_VP9   : return AV_CODEC_ID_VP9;
        case rocDecVideoCodec_JPEG  : return AV_CODEC_ID_MJPEG;
        case rocDecVideoCodec_AV1   : return AV_CODEC_ID_AV1;
        default                     : return AV_CODEC_ID_NONE;
    }
}

static inline float GetChromaWidthFactor(rocDecVideoSurfaceFormat surface_format) {
    float factor = 0.5;
    switch (surface_format) {
    case rocDecVideoSurfaceFormat_NV12:
    case rocDecVideoSurfaceFormat_P016:
    case rocDecVideoSurfaceFormat_YUV444:
    case rocDecVideoSurfaceFormat_YUV444_16Bit:
        factor = 1.0;
        break;
    case rocDecVideoSurfaceFormat_YUV420:
    case rocDecVideoSurfaceFormat_YUV420_16Bit:
    case rocDecVideoSurfaceFormat_YUV422:
    case rocDecVideoSurfaceFormat_YUV422_16Bit:
        factor = 0.5;
        break;
    }
    return factor;
};


/**
 * @brief helper function for inferring AVCodecID from rocDecVideoSurfaceFormat
 * 
 * @param rocdec_codec 
 * @return AVCodecID 
 */
static inline rocDecVideoSurfaceFormat AVPixelFormat2rocDecVideoSurfaceFormat(AVPixelFormat av_pixel_format) {
    switch (av_pixel_format) {
        case AV_PIX_FMT_YUV420P : 
        case AV_PIX_FMT_YUVJ420P : 
            return rocDecVideoSurfaceFormat_YUV420;
        case AV_PIX_FMT_YUV444P : 
        case AV_PIX_FMT_YUVJ444P : 
            return rocDecVideoSurfaceFormat_YUV444;
        case AV_PIX_FMT_YUV420P10LE :
        case AV_PIX_FMT_YUV420P12LE :
            return rocDecVideoSurfaceFormat_YUV420_16Bit;
        default :
            std::cerr << "ERROR: " << av_get_pix_fmt_name(av_pixel_format) << " pixel_format is not supported!" << std::endl;          
            return rocDecVideoSurfaceFormat_NV12;       // for sanity
    }
}

FFMpegVideoDecoder::FFMpegVideoDecoder(int device_id, OutputSurfaceMemoryType out_mem_type, rocDecVideoCodec codec, bool force_zero_latency,
              const Rect *p_crop_rect, bool extract_user_sei_Message, uint32_t disp_delay, bool no_multithreading, int max_width, int max_height, uint32_t clk_rate) :
              RocVideoDecoder(device_id, out_mem_type, codec, force_zero_latency, p_crop_rect, extract_user_sei_Message, disp_delay, max_width, max_height, clk_rate), no_multithreading_(no_multithreading) {

    if ((out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL) || (out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED)) {
        ROCDEC_THROW("Output Memory Type is not supported", ROCDEC_INVALID_PARAMETER);
    }
    if (rocdec_parser_) {
        rocDecDestroyVideoParser(rocdec_parser_);       // need to do this here since it was already created in the base class
        // create rocdec videoparser
        RocdecParserParams parser_params = {};
        parser_params.codec_type = codec_id_;
        parser_params.max_num_decode_surfaces = 1; // let the parser to determine the decode buffer pool size
        parser_params.clock_rate = clk_rate;
        parser_params.max_display_delay = disp_delay_;
        parser_params.user_data = this;
        parser_params.pfn_sequence_callback = FFMpegHandleVideoSequenceProc;
        parser_params.pfn_decode_picture = FFMpegHandlePictureDecodeProc;
        parser_params.pfn_display_picture = nullptr;  // HandlePictureDisplay will be controlled by FFMpegDecoder
        parser_params.pfn_get_sei_msg = b_extract_sei_message_ ? RocVideoDecoder::HandleSEIMessagesProc : NULL;
        ROCDEC_API_CALL(rocDecCreateVideoParser(&rocdec_parser_, &parser_params));
    }
    if (!no_multithreading_) {
        // start the FFMpeg decoding thread
        ffmpeg_decoder_thread_ = new std::thread(&FFMpegVideoDecoder::DecodeThread, this);
        if (!ffmpeg_decoder_thread_) {
            ROCDEC_THROW("FFMpegVideoDecoder create thread failed", -1);
        }
    }
}


FFMpegVideoDecoder::~FFMpegVideoDecoder() {

    std::lock_guard<std::mutex> lock(mtx_vp_frame_);
    for (auto &p_frame : vp_frames_ffmpeg_) {
        if (p_frame.frame_ptr) {
            if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
                hipError_t hip_status = hipFree(p_frame.frame_ptr);
                if (hip_status != hipSuccess) {
                    std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
                }
            }
        }
    }

    // release frame_q
    while (!dec_frames_.empty()) {
        av_frame_free(&dec_frames_.back());
        dec_frames_.pop_back();
    }
    // free av_packet_data_
    while (!av_packet_data_.empty()) {
        std::pair<uint8_t *, int> *packet_data = &av_packet_data_.back();
        av_freep(&packet_data->first);
        av_packet_data_.pop_back();
    }

    //release av_packets
    while (!av_packets_.empty()) {
        av_packet_free(&av_packets_.back());
        av_packets_.pop_back();
    }

    if (dec_context_) {
        avcodec_free_context(&dec_context_);
    }
}

/* Return value from HandleVideoSequence() are interpreted as   :
*  0: fail, 1: succeeded, > 1: override dpb size of parser (set by RocdecParserParams::max_num_decode_surfaces while creating parser)
*/
int FFMpegVideoDecoder::HandleVideoSequence(RocdecVideoFormat *p_video_format) {
    if (p_video_format == nullptr) {
        ROCDEC_THROW("Rocdec:: Invalid video format in HandleVideoSequence: ", ROCDEC_INVALID_PARAMETER);
        return 0;
    }
    auto start_time = StartTimer();
    input_video_info_str_.str("");
    input_video_info_str_.clear();
    input_video_info_str_ << "Input Video Information" << std::endl
        << "\tCodec        : " << GetCodecFmtName(p_video_format->codec) << std::endl;
        if (p_video_format->frame_rate.numerator && p_video_format->frame_rate.denominator) {
            input_video_info_str_ << "\tFrame rate   : " << p_video_format->frame_rate.numerator << "/" << p_video_format->frame_rate.denominator << " = " << 1.0 * p_video_format->frame_rate.numerator / p_video_format->frame_rate.denominator << " fps" << std::endl;
        }
    input_video_info_str_ << "\tSequence     : " << (p_video_format->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << p_video_format->coded_width << ", " << p_video_format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << p_video_format->display_area.left << ", " << p_video_format->display_area.top << ", "
            << p_video_format->display_area.right << ", " << p_video_format->display_area.bottom << "]" << std::endl
        << "\tBit depth    : " << p_video_format->bit_depth_luma_minus8 + 8
    ;
    input_video_info_str_ << std::endl;

    int num_decode_surfaces = p_video_format->min_num_decode_surfaces;

    // check if the codec is supported in FFMpeg
    // Initialize FFMpeg and find the decoder for codec
    if (!decoder_) decoder_ = avcodec_find_decoder(RocDecVideoCodec2AVCodec(p_video_format->codec));
    if(!decoder_) {
        ROCDEC_THROW("rocDecode<FFMpeg>:: Codec not supported by FFMpeg ", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if (!dec_context_) {
        dec_context_ = avcodec_alloc_context3(decoder_);        //alloc dec_context_
        if (!dec_context_) {
            ROCDEC_THROW("Could not allocate video codec context", ROCDEC_RUNTIME_ERROR);
        }
        // set codec to automatically determine how many threads suits best for the decoding job
        dec_context_->thread_count = 0;

        if (decoder_->capabilities & AV_CODEC_CAP_FRAME_THREADS)
            dec_context_->thread_type = FF_THREAD_FRAME;
        else if (decoder_->capabilities & AV_CODEC_CAP_SLICE_THREADS)
            dec_context_->thread_type = FF_THREAD_SLICE;
        else
            dec_context_->thread_count = 1;

        // open the codec
        if (avcodec_open2(dec_context_, decoder_, NULL) < 0) {
            ROCDEC_THROW("Could not open codec", ROCDEC_RUNTIME_ERROR);
        }
        // get the output pixel format from dec_context_
        decoder_pixel_format_ = (dec_context_->pix_fmt == AV_PIX_FMT_NONE) ? AV_PIX_FMT_YUV420P : dec_context_->pix_fmt;
    }
    // allocate av_frame buffer pool for number of surfaces to be in the decoder pool
    // Note: with multi-threading, av_codec needs (dec_context_->delay + max_num_B_frames) number of av_frames. 
    // max_num_B_frames is assumed to be 4 here: revisit this to fix with the correct number from seq_header
    if (dec_frames_.empty()) {
        for (int i = 0; i < (dec_context_->delay + 4); i++) {
            AVFrame *p_frame = av_frame_alloc();
            dec_frames_.push_back(p_frame);
        }
        av_frame_cnt_ = 0;
    }
    if (av_packet_data_.empty()) {
        for (int i = 0; i < num_decode_surfaces; i++) {
            uint8_t *pkt_data = static_cast<uint8_t *> (av_malloc(MAX_AV_PACKET_DATA_SIZE));
            av_packet_data_.push_back(std::make_pair(pkt_data, MAX_AV_PACKET_DATA_SIZE));
        }
    }

    // allocate av_packets_ for decoding
    if (av_packets_.empty()) {
        for (int i = 0; i < num_decode_surfaces; i++) {
            AVPacket *pkt = av_packet_alloc();
            pkt->data = static_cast<uint8_t *> (av_packet_data_[i].first);
            pkt->size = av_packet_data_[i].second;
            av_packets_.push_back(pkt);
        }
    }

    if (curr_video_format_ptr_ == nullptr) {
        curr_video_format_ptr_ = new RocdecVideoFormat();
    }
    // store current video format: this is required to call reconfigure from application in case of random seek
    if (curr_video_format_ptr_) memcpy(curr_video_format_ptr_, p_video_format, sizeof(RocdecVideoFormat));

    if (coded_width_ && coded_height_) {
        end_of_stream_ = false;
        b_decoder_initialized = false; // reinitialize for reconfigure
        return ReconfigureDecoder(p_video_format);
    }
    // e_codec has been set in the constructor (for parser). Here it's set again for potential correction
    codec_id_ = p_video_format->codec;
    video_chroma_format_ = p_video_format->chroma_format;
    bitdepth_minus_8_ = p_video_format->bit_depth_luma_minus8;
    byte_per_pixel_ = bitdepth_minus_8_ > 0 ? 2 : 1;

    // convert AVPixelFormat to rocDecVideoChromaFormat
    video_surface_format_ = AVPixelFormat2rocDecVideoSurfaceFormat(decoder_pixel_format_);
    coded_width_ = p_video_format->coded_width;
    coded_height_ = p_video_format->coded_height;
    disp_rect_.top = p_video_format->display_area.top;
    disp_rect_.bottom = p_video_format->display_area.bottom;
    disp_rect_.left = p_video_format->display_area.left;
    disp_rect_.right = p_video_format->display_area.right;
    disp_width_ = p_video_format->display_area.right - p_video_format->display_area.left;
    disp_height_ = p_video_format->display_area.bottom - p_video_format->display_area.top;

    // AV1 has max width/height of sequence in sequence header
    if (codec_id_ == rocDecVideoCodec_AV1 && p_video_format->seqhdr_data_length > 0) {
        // dont overwrite if it is already set from cmdline or reconfig.txt
        if (!(max_width_ > p_video_format->coded_width || max_height_ > p_video_format->coded_height)) {
            RocdecVideoFormatEx *vidFormatEx = (RocdecVideoFormatEx *)p_video_format;
            max_width_ = vidFormatEx->max_width;
            max_height_ = vidFormatEx->max_height;
        }
    }
    if (max_width_ < static_cast<int>(p_video_format->coded_width))
        max_width_ = p_video_format->coded_width;
    if (max_height_ < static_cast<int>(p_video_format->coded_height))
        max_height_ = p_video_format->coded_height;

    if (!(crop_rect_.right && crop_rect_.bottom)) {
        target_width_ = (disp_width_ + 1) & ~1;
        target_height_ = (disp_height_ + 1) & ~1;
    } else {
        target_width_ = (crop_rect_.right - crop_rect_.left + 1) & ~1;
        target_height_ = (crop_rect_.bottom - crop_rect_.top + 1) & ~1;
    }

    chroma_height_ = static_cast<int>(ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    chroma_width_ = static_cast<int>(ceil(target_width_ * GetChromaWidthFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    if (video_chroma_format_ == rocDecVideoChromaFormat_Monochrome) num_chroma_planes_ = 0;
    surface_stride_ = target_width_ * byte_per_pixel_;   
    chroma_vstride_ = static_cast<int>(ceil(surface_vstride_ * GetChromaHeightFactor(video_surface_format_)));
    // fill output_surface_info_
    output_surface_info_.output_width = target_width_;
    output_surface_info_.output_height = target_height_;
    output_surface_info_.output_pitch  = surface_stride_;
    output_surface_info_.output_vstride = target_height_;
    output_surface_info_.bit_depth = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_COPIED;
    } else if (out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED){
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
    }
    input_video_info_str_ << "Video Decoding Params:" << std::endl
        << "\tNum Surfaces : " << num_decode_surfaces << std::endl
        << "\tCrop         : [" << disp_rect_.left << ", " << disp_rect_.top << ", "
        << disp_rect_.right << ", " << disp_rect_.bottom << "]" << std::endl
        << "\tResize       : " << target_width_ << "x" << target_height_ << std::endl
    ;
    input_video_info_str_ << std::endl;
    std::cout << input_video_info_str_.str();

    double elapsed_time = StopTimer(start_time);
    AddDecoderSessionOverHead(std::this_thread::get_id(), elapsed_time);
    return num_decode_surfaces;
}

/**
 * @brief function to reconfigure decoder if there is a change in sequence params.
 *
 * @param p_video_format
 * @return int 1: success 0: fail
 */
int FFMpegVideoDecoder::ReconfigureDecoder(RocdecVideoFormat *p_video_format) {
    if (p_video_format->codec != codec_id_) {
        ROCDEC_THROW("Reconfigure Not supported for codec change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if (p_video_format->chroma_format != video_chroma_format_) {
        ROCDEC_THROW("Reconfigure Not supported for chroma format change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if (p_video_format->bit_depth_luma_minus8 != bitdepth_minus_8_){
        ROCDEC_THROW("Reconfigure Not supported for bit depth change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    bool is_decode_res_changed = !(p_video_format->coded_width == coded_width_ && p_video_format->coded_height == coded_height_);
    bool is_display_rect_changed = !(p_video_format->display_area.bottom == disp_rect_.bottom &&
                                     p_video_format->display_area.top == disp_rect_.top &&
                                     p_video_format->display_area.left == disp_rect_.left &&
                                     p_video_format->display_area.right == disp_rect_.right);

    if (!is_decode_res_changed && !is_display_rect_changed && !b_force_recofig_flush_) {
        return 1;
    }

    // Flush and clear internal frame store to reconfigure when either coded size or display size has changed.
    if (p_reconfig_params_ && p_reconfig_params_->p_fn_reconfigure_flush) 
        num_frames_flushed_during_reconfig_ += p_reconfig_params_->p_fn_reconfigure_flush(this, p_reconfig_params_->reconfig_flush_mode, static_cast<void *>(p_reconfig_params_->p_reconfig_user_struct));
    // clear the existing output buffers of different size
    // note that app lose the remaining frames in the vp_frames in case application didn't set p_fn_reconfigure_flush_ callback
    std::lock_guard<std::mutex> lock(mtx_vp_frame_);
    while(!vp_frames_ffmpeg_.empty()) {
        DecFrameBufferFFMpeg *p_frame = &vp_frames_ffmpeg_.back();
        // pop decoded frame
        vp_frames_ffmpeg_.pop_back();
        if (p_frame->frame_ptr) {
            if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
                hipError_t hip_status = hipFree(p_frame->frame_ptr);
                if (hip_status != hipSuccess) std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
            }
        }
        // release associated av_frame
        if (p_frame->av_frame_ptr) {
            av_frame_free(&p_frame->av_frame_ptr);
        }
    }
    output_frame_cnt_ = 0;     // reset frame_count
    if (is_decode_res_changed) {
        coded_width_ = p_video_format->coded_width;
        coded_height_ = p_video_format->coded_height;
    }
    if (is_display_rect_changed) {
        disp_rect_.left = p_video_format->display_area.left;
        disp_rect_.right = p_video_format->display_area.right;
        disp_rect_.top = p_video_format->display_area.top;
        disp_rect_.bottom = p_video_format->display_area.bottom;
        disp_width_ = p_video_format->display_area.right - p_video_format->display_area.left;
        disp_height_ = p_video_format->display_area.bottom - p_video_format->display_area.top;

        if (!(crop_rect_.right && crop_rect_.bottom)) {
            target_width_ = (disp_width_ + 1) & ~1;
            target_height_ = (disp_height_ + 1) & ~1;
        } else {
            target_width_ = (crop_rect_.right - crop_rect_.left + 1) & ~1;
            target_height_ = (crop_rect_.bottom - crop_rect_.top + 1) & ~1;
        }
    }

    surface_stride_ = target_width_ * byte_per_pixel_;
    chroma_height_ = static_cast<int>(std::ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    chroma_width_ = static_cast<int>(ceil(target_width_ * GetChromaWidthFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    if (p_video_format->chroma_format == rocDecVideoChromaFormat_Monochrome) num_chroma_planes_ = 0;
    chroma_vstride_ = static_cast<int>(std::ceil(surface_vstride_ * GetChromaHeightFactor(video_surface_format_)));
    // Fill output_surface_info_
    output_surface_info_.output_width = target_width_;
    output_surface_info_.output_height = target_height_;
    output_surface_info_.output_pitch  = surface_stride_;
    output_surface_info_.output_vstride = (out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL) ? surface_vstride_ : target_height_;
    output_surface_info_.bit_depth = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_COPIED;
    } else if (out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
    }

    // If the coded_width or coded_height hasn't changed but display resolution has changed, then need to update width and height for
    // correct output with cropping. There is no need to reconfigure the decoder.
    if (!is_decode_res_changed && is_display_rect_changed) {
        return 1;
    }

    input_video_info_str_.str("");
    input_video_info_str_.clear();
    input_video_info_str_ << "Input Video Resolution Changed:" << std::endl
        << "\tCoded size   : [" << p_video_format->coded_width << ", " << p_video_format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << p_video_format->display_area.left << ", " << p_video_format->display_area.top << ", "
            << p_video_format->display_area.right << ", " << p_video_format->display_area.bottom << "]" << std::endl;
    input_video_info_str_ << std::endl;
    is_decoder_reconfigured_ = true;
    return 1;
}

/**
 * @brief 
 * 
 * @param pPicParams 
 * @return int 1: success 0: fail
 */
int FFMpegVideoDecoder::HandlePictureDecode(RocdecPicParams *pPicParams) {
    AVPacket *av_pkt = av_packets_[av_pkt_cnt_];
    std::pair<uint8_t *, int> *packet_data = &av_packet_data_[av_pkt_cnt_];
    if (last_packet_.payload_size > packet_data->second) {
        void *new_pkt_data = av_realloc(av_pkt->data, (last_packet_.payload_size + MAX_AV_PACKET_DATA_SIZE));  // add more to avoid frequence reallocation
        if (!new_pkt_data) {
            std::cerr << "ERROR: couldn't allocate packet data" << std::endl;
        }
        packet_data->first   = static_cast<uint8_t *>(new_pkt_data);
        packet_data->second  = (last_packet_.payload_size + MAX_AV_PACKET_DATA_SIZE);
        av_pkt->data = packet_data->first;
    }
    memcpy(av_pkt->data, last_packet_.payload, last_packet_.payload_size);
    av_pkt->size = last_packet_.payload_size;
    av_pkt->flags = 0;
    av_pkt->pts = last_packet_.pts;

    if (no_multithreading_) {
        DecodeAvFrame(av_pkt, dec_frames_[av_frame_cnt_]);
        int num_frames_to_display = decoded_pic_cnt_;
        while (num_frames_to_display) {
            RocdecParserDispInfo dispInfo = {0}; // dispinfo is not used in ffmpeg decoder, so setting it to zero
            HandlePictureDisplay(&dispInfo);
            num_frames_to_display--;
        };
        if (!last_packet_.payload_size && !end_of_stream_) {
            AVPacket pkt = {0};
            DecodeAvFrame(&pkt, dec_frames_[av_frame_cnt_]);
            int num_frames_to_display = decoded_pic_cnt_;
            while (num_frames_to_display) {
                RocdecParserDispInfo dispInfo = {0}; // don't care about this as this will be igonored
                HandlePictureDisplay(&dispInfo);
                num_frames_to_display--;
            };
        }
    } else {
        //push packet into packet q for decoding
        PushPacket(av_pkt);
        if (!last_packet_.payload_size && !end_of_stream_) {
            AVPacket pkt = {0};
            PushPacket(&pkt);
        }
    }
    av_pkt_cnt_ = (av_pkt_cnt_ + 1) % av_packets_.size();
    if (!av_pkt->data || !av_pkt->size) {
        end_of_stream_ = true;
    }

    return 1;
}

/**
 * @brief function to handle display picture
 * 
 * @param pDispInfo 
 * @return int 0:fail 1: success
 */
int FFMpegVideoDecoder::HandlePictureDisplay(RocdecParserDispInfo *pDispInfo) {
    if (b_extract_sei_message_) {
        if (sei_message_display_q_[pDispInfo->picture_index].sei_data) {
            // Write SEI Message
            uint8_t *sei_buffer = static_cast<uint8_t *>(sei_message_display_q_[pDispInfo->picture_index].sei_data);
            uint32_t sei_num_messages = sei_message_display_q_[pDispInfo->picture_index].sei_message_count;
            RocdecSeiMessage *sei_message = sei_message_display_q_[pDispInfo->picture_index].sei_message;
            if (fp_sei_) {
                for (uint32_t i = 0; i < sei_num_messages; i++) {
                    if (codec_id_ == rocDecVideoCodec_AVC || rocDecVideoCodec_HEVC) {
                        switch (sei_message[i].sei_message_type) {
                            case SEI_TYPE_TIME_CODE: {
                                //todo:: check if we need to write timecode
                            }
                            break;
                            case SEI_TYPE_USER_DATA_UNREGISTERED: {
                                fwrite(sei_buffer, sei_message[i].sei_message_size, 1, fp_sei_);
                            }
                            break;
                        }
                    }
                    if (codec_id_ == rocDecVideoCodec_AV1) {
                        fwrite(sei_buffer, sei_message[i].sei_message_size, 1, fp_sei_);
                    }    
                    sei_buffer += sei_message[i].sei_message_size;
                }
            }
            free(sei_message_display_q_[pDispInfo->picture_index].sei_data);
            sei_message_display_q_[pDispInfo->picture_index].sei_data = NULL; // to avoid double free
            free(sei_message_display_q_[pDispInfo->picture_index].sei_message);
            sei_message_display_q_[pDispInfo->picture_index].sei_message = NULL; // to avoid double free
        }
    }
    // this will happen during PopFrame()
    AVFrame *p_av_frame = nullptr;
    if (no_multithreading_ ) {
        if (!av_frame_q_.empty()) {
            p_av_frame = av_frame_q_.front();
            av_frame_q_.pop();
        }
    } else {
        p_av_frame = PopFrame();
    }

    if (p_av_frame == nullptr) {
        std::cerr << "Invalid avframe decode output" << std::endl;
        return 0;
    }
    void* src_ptr[3] = {0};
    int32_t src_pitch[3] = {0};
    src_ptr[0] = p_av_frame->data[0];    
    src_ptr[1] = p_av_frame->data[1];
    src_ptr[2] = p_av_frame->data[2];
    src_pitch[0] = p_av_frame->linesize[0];
    src_pitch[1] = p_av_frame->linesize[1];
    src_pitch[2] = p_av_frame->linesize[2];

    // copy the decoded surface info device or host
    uint8_t *p_dec_frame = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        // if not enough frames in stock, allocate
        if (++output_frame_cnt_ > vp_frames_ffmpeg_.size()) {
            num_alloced_frames_++;
            DecFrameBufferFFMpeg dec_frame = {0};
            if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
                // allocate device memory
                HIP_API_CALL(hipMalloc((void **)&dec_frame.frame_ptr, GetFrameSize()));
            } else {
                dec_frame.frame_ptr = new uint8_t[GetFrameSize()];
            }

            dec_frame.av_frame_ptr = p_av_frame;
            dec_frame.pts = p_av_frame->pts;
            dec_frame.picture_index = -1;     //picture_index is not used here since it is handled within FFMpeg decoder
            vp_frames_ffmpeg_.push_back(dec_frame);
        }
        p_dec_frame = vp_frames_ffmpeg_[output_frame_cnt_ - 1].frame_ptr;
    }

    // Copy luma data
    int dst_pitch = disp_width_ * byte_per_pixel_;
    uint8_t *p_src_ptr_y = static_cast<uint8_t *>(src_ptr[0]) + (disp_rect_.top + crop_rect_.top) * src_pitch[0] + (disp_rect_.left + crop_rect_.left) * byte_per_pixel_;
    uint8_t *p_frame_y = p_dec_frame;
    if (!p_frame_y && !p_src_ptr_y) {
        std::cerr << "HandlePictureDisplay: Invalid Memory address for src/dst" << std::endl;
        return 0;
    }
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        if (src_pitch[0] == dst_pitch) {
            int luma_size = src_pitch[0] * disp_height_;
            HIP_API_CALL(hipMemcpyHtoDAsync(p_frame_y, p_src_ptr_y, luma_size, hip_stream_));
        } else {
            // use 2d copy to copy an ROI
            HIP_API_CALL(hipMemcpy2DAsync(p_frame_y, dst_pitch, p_src_ptr_y, src_pitch[0], dst_pitch, disp_height_, hipMemcpyHostToDevice, hip_stream_));
        }
    } else {
        if (src_pitch[0] == dst_pitch) {
            int luma_size = src_pitch[0] * disp_height_;
            memcpy(p_frame_y, p_src_ptr_y, luma_size);
        } else {
            for (int i = 0; i < disp_height_; i++) {
                memcpy(p_dec_frame, p_src_ptr_y, dst_pitch);
                p_frame_y += dst_pitch;
                p_src_ptr_y += src_pitch[0];
            }
        }
    }
    // Copy chroma plane ( )
    // rocDec output gives pointer to luma and chroma pointers seperated for the decoded frame
    uint8_t *p_frame_uv = p_dec_frame + dst_pitch * disp_height_;
    uint8_t *p_src_ptr_uv = static_cast<uint8_t *>(src_ptr[1]) + ((disp_rect_.top + crop_rect_.top) >> 1) * src_pitch[1] + ((disp_rect_.left + crop_rect_.left)>>1) * byte_per_pixel_ ;
    dst_pitch = chroma_width_ *  byte_per_pixel_;          
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        if (src_pitch[1] == dst_pitch) {
            int chroma_size = chroma_height_ * dst_pitch;
            HIP_API_CALL(hipMemcpyHtoDAsync(p_frame_uv, p_src_ptr_uv, chroma_size, hip_stream_));
        } else {
            // use 2d copy to copy an ROI
            HIP_API_CALL(hipMemcpy2DAsync(p_frame_uv, dst_pitch, p_src_ptr_uv, src_pitch[1], dst_pitch, chroma_height_, hipMemcpyHostToDevice, hip_stream_));
        }
    } else {
        if (src_pitch[1] == dst_pitch) {
            int chroma_size = chroma_height_ * dst_pitch;
            memcpy(p_frame_uv, p_src_ptr_uv, chroma_size);
        } 
        else {
            for (int i = 0; i < chroma_height_; i++) {
                memcpy(p_frame_uv, p_src_ptr_uv, dst_pitch);
                p_frame_uv += dst_pitch;
                p_src_ptr_uv += src_pitch[1];
            }
        }
    }

    if (num_chroma_planes_ == 2) {
        uint8_t *p_frame_v = p_frame_uv + dst_pitch * chroma_height_;
        uint8_t *p_src_ptr_v = static_cast<uint8_t *>(src_ptr[2]) + (disp_rect_.top + crop_rect_.top) * src_pitch[2] + ((disp_rect_.left + crop_rect_.left) >> 1) * byte_per_pixel_;
        if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
            if (src_pitch[2] == dst_pitch) {
                int chroma_size = chroma_height_ * dst_pitch;
                HIP_API_CALL(hipMemcpyDtoDAsync(p_frame_v, p_src_ptr_v, chroma_size, hip_stream_));
            } else {
                // use 2d copy to copy an ROI
                HIP_API_CALL(hipMemcpy2DAsync(p_frame_v, dst_pitch, p_src_ptr_v, src_pitch[2], dst_pitch, chroma_height_, hipMemcpyHostToDevice, hip_stream_));
            }            
        }
        else {
            if (src_pitch[2] == dst_pitch) {
                int chroma_size = chroma_height_ * dst_pitch;
                memcpy(p_frame_v, p_src_ptr_v, chroma_size);
            } 
            else {
                for (int i = 0; i < chroma_height_; i++) {
                    memcpy(p_frame_v, p_src_ptr_v, dst_pitch);
                    p_frame_v += dst_pitch;
                    p_src_ptr_v += src_pitch[1];
                }
            }
        }         
    }
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) HIP_API_CALL(hipStreamSynchronize(hip_stream_));

    return 1;
}


int FFMpegVideoDecoder::DecodeFrame(const uint8_t *data, size_t size, int pkt_flags, int64_t pts, int *num_decoded_pics) {
    output_frame_cnt_ = 0, output_frame_cnt_ret_ = 0;
    decoded_pic_cnt_ = 0;
    last_packet_ = {0};
    last_packet_.payload = data;
    last_packet_.payload_size = size;
    last_packet_.flags = pkt_flags | ROCDEC_PKT_TIMESTAMP;
    last_packet_.pts = pts;
    if (!data || size == 0) {
        last_packet_.flags |= ROCDEC_PKT_ENDOFSTREAM;
    }
    ROCDEC_API_CALL(rocDecParseVideoData(rocdec_parser_, &last_packet_));
    if (last_packet_.flags & ROCDEC_PKT_ENDOFSTREAM) {
        // flush last packet and let FFMpeg decode last frames
        FlushDecoder();
    }

    if (num_decoded_pics) {
        *num_decoded_pics = decoded_pic_cnt_;
    }
    return output_frame_cnt_;
}


uint8_t* FFMpegVideoDecoder::GetFrame(int64_t *pts) {
    if (output_frame_cnt_ > 0) {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        if (vp_frames_ffmpeg_.size() > 0){
            output_frame_cnt_--;
            if (pts) *pts = vp_frames_ffmpeg_[output_frame_cnt_ret_].pts;
            return vp_frames_ffmpeg_[output_frame_cnt_ret_++].frame_ptr;
        }
    }
    return nullptr;
}

bool FFMpegVideoDecoder::ReleaseFrame(int64_t pTimestamp, bool b_flushing) {
    // if not flushing the buffers are re-used, so keep them
    if (out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED || !b_flushing)
        return true;    // nothing to do
    if (!b_flushing)  
        return true;
    else {
        // remove frames in flushing mode
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        DecFrameBufferFFMpeg *fb = &vp_frames_ffmpeg_[0];
        if (pTimestamp != fb->pts) {
            std::cerr << "Decoded Frame is released out of order" << std::endl;
            return false;
        }
        av_frame_free(&fb->av_frame_ptr);
        vp_frames_ffmpeg_.erase(vp_frames_ffmpeg_.begin());     // get rid of the frames from the framestore
    }
    return true;
}

void FFMpegVideoDecoder::InitOutputFrameInfo(AVFrame *p_frame) {
    
    video_surface_format_ = AVPixelFormat2rocDecVideoSurfaceFormat((AVPixelFormat)p_frame->format);
    surface_stride_ = target_width_ * byte_per_pixel_;
    chroma_width_ = static_cast<int>(ceil(target_width_ * GetChromaWidthFactor(video_surface_format_)));
    chroma_height_ = static_cast<int>(ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    // Fill output_surface_info_
    output_surface_info_.output_width = target_width_;
    output_surface_info_.output_height = target_height_;
    output_surface_info_.output_pitch  = surface_stride_;
    output_surface_info_.output_vstride = target_height_;
    output_surface_info_.bit_depth = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_COPIED;
    } else if (out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
    }
}

/**
 * @brief Flush decoder with decoding of last frames
 * 
 * @return int 1: success 0: fail
 */
int FFMpegVideoDecoder::FlushDecoder() {
    AVPacket pkt = {0};
    if (no_multithreading_) {
        DecodeAvFrame(&pkt, dec_frames_[av_frame_cnt_]);
        int num_frames_to_display = decoded_pic_cnt_;
        while (num_frames_to_display) {
            RocdecParserDispInfo dispInfo = {0}; // don't care about this as this will be igonored
            HandlePictureDisplay(&dispInfo);
            num_frames_to_display--;
        };

    } else {
        //push packet into packet q for decoding
        PushPacket(&pkt);
    }
    return 0;
}


void FFMpegVideoDecoder::DecodeThread()
{
    AVPacket *pkt;
    do {
        pkt = PopPacket();
        DecodeAvFrame(pkt, dec_frames_[av_frame_cnt_]);
    } while (!end_of_stream_);
}

int FFMpegVideoDecoder::DecodeAvFrame(AVPacket *av_pkt, AVFrame *p_frame) {
    int status;
    //send packet to av_codec
    status = avcodec_send_packet(dec_context_, av_pkt);
    if (status < 0) {
        std::cout << "Error sending av packet for decoding: status: " << status << std::endl;
    }
    while (status >= 0) {
        status = avcodec_receive_frame(dec_context_, p_frame);
        if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
            end_of_stream_ = (status == AVERROR_EOF);
            return 0;
        }
        else if (status < 0) {
            std::cout << "Error during decoding" << std::endl;
            return 0;
        }
        // for the first frame, initialize OutputsurfaceInfo
        if (!b_decoder_initialized) {
            InitOutputFrameInfo(p_frame);
            b_decoder_initialized = true;
        }
        decoded_pic_cnt_++;
        if (no_multithreading_)
            av_frame_q_.push(p_frame);
        else
            PushFrame(p_frame);  // add frame to the frame_q
        av_frame_cnt_ = (av_frame_cnt_ + 1) % dec_frames_.size();
        p_frame = dec_frames_[av_frame_cnt_]; //advance for next frame decode
    }
    return 0;
}


void FFMpegVideoDecoder::SaveFrameToFile(std::string output_file_name, void *surf_mem, OutputSurfaceInfo *surf_info, size_t rgb_image_size) {
    uint8_t *hst_ptr = nullptr;
    bool is_rgb = (rgb_image_size != 0);
    uint64_t output_image_size = is_rgb ? rgb_image_size : surf_info->output_surface_size_in_bytes;
    if (surf_info->mem_type == OUT_SURFACE_MEM_DEV_COPIED) {
        if (hst_ptr == nullptr) {
            hst_ptr = new uint8_t [output_image_size];
        }
        hipError_t hip_status = hipSuccess;
        hip_status = hipMemcpyDtoH((void *)hst_ptr, surf_mem, output_image_size);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMemcpyDtoH failed! (" << hipGetErrorName(hip_status) << ")" << std::endl;
            delete [] hst_ptr;
            return;
        }
    } else
        hst_ptr = static_cast<uint8_t *> (surf_mem);

    
    if (current_output_filename.empty()) {
        current_output_filename = output_file_name;
    }

    // don't overwrite to the same file if reconfigure is detected for a resolution changes.
    if (is_decoder_reconfigured_) {
        if (fp_out_) {
            fclose(fp_out_);
            fp_out_ = nullptr;
        }
        // Append the width and height of the new stream to the old file name to create a file name to save the new frames
        // do this only if resolution changes within a stream (e.g., decoding a multi-resolution stream using the videoDecode app)
        // don't append to the output_file_name if multiple output file name is provided (e.g., decoding multi-files using the videDecodeMultiFiles)
        if (!current_output_filename.compare(output_file_name)) {
            std::string::size_type const pos(output_file_name.find_last_of('.'));
            extra_output_file_count_++;
            std::string to_append = "_" + std::to_string(surf_info->output_width) + "_" + std::to_string(surf_info->output_height) + "_" + std::to_string(extra_output_file_count_);
            if (pos != std::string::npos) {
                output_file_name.insert(pos, to_append);
            } else {
                output_file_name += to_append;
            }
        }
        is_decoder_reconfigured_ = false;
    } 

    if (fp_out_ == nullptr) {
        fp_out_ = fopen(output_file_name.c_str(), "wb");
    }
    if (fp_out_) {
        if (!is_rgb) {
            uint8_t *tmp_hst_ptr = hst_ptr;
            int img_width = surf_info->output_width;
            int img_height = surf_info->output_height;
            int output_stride =  surf_info->output_pitch;
            if (img_width * surf_info->bytes_per_pixel == output_stride && img_height == surf_info->output_vstride) {
                fwrite(hst_ptr, 1, output_image_size, fp_out_);
            } else {
                uint32_t width = surf_info->output_width * surf_info->bytes_per_pixel;
                if (surf_info->bit_depth <= 16) {
                    for (int i = 0; i < surf_info->output_height; i++) {
                        fwrite(tmp_hst_ptr, 1, width, fp_out_);
                        tmp_hst_ptr += output_stride;
                    }
                    // dump chroma
                    uint32_t chroma_stride = (output_stride >> 1);
                    uint8_t *u_hst_ptr = hst_ptr + output_stride * surf_info->output_height;
                    uint8_t *v_hst_ptr = u_hst_ptr + chroma_stride * chroma_height_;
                    for (int i = 0; i < chroma_height_; i++) {
                        fwrite(u_hst_ptr, 1, chroma_width_, fp_out_);
                        u_hst_ptr += chroma_stride;
                    }
                    if (num_chroma_planes_ == 2) {
                        for (int i = 0; i < chroma_height_; i++) {
                            fwrite(v_hst_ptr, 1, chroma_width_, fp_out_);
                            v_hst_ptr += chroma_stride;
                        }
                    }
                } 
            }
        } else {
            fwrite(hst_ptr, 1, rgb_image_size, fp_out_);
        }
    }

    if (hst_ptr && (surf_info->mem_type != OUT_SURFACE_MEM_HOST_COPIED)) {
        delete [] hst_ptr;
    }
}
