/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iomanip>
#include "decoders/video/rocdec_video_decoder.h"

#ifdef ROCAL_VIDEO
#if ENABLE_HIP && ENABLE_ROCDECODE

// callback function to flush last frames and save it to file when reconfigure happens
int ReconfigureFlushCallback(void *p_viddec_obj, uint32_t flush_mode, void *p_user_struct) {
    int n_frames_flushed = 0;
    if ((p_viddec_obj == nullptr) || (p_user_struct == nullptr)) return n_frames_flushed;

    RocVideoDecoder *viddec = static_cast<RocVideoDecoder *> (p_viddec_obj);
    OutputSurfaceInfo *surf_info;
    if (!viddec->GetOutputSurfaceInfo(&surf_info)) {
        std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
        return n_frames_flushed;
    }

    uint8_t *pframe = nullptr;
    int64_t pts;
    while ((pframe = viddec->GetFrame(&pts))) {
        if (flush_mode != RECONFIG_FLUSH_MODE_NONE) {
            if (flush_mode == ReconfigFlushMode::RECONFIG_FLUSH_MODE_DUMP_TO_FILE) {
                ReconfigDumpFileStruct *p_dump_file_struct = static_cast<ReconfigDumpFileStruct *>(p_user_struct);
                if (p_dump_file_struct->b_dump_frames_to_file) {
                    viddec->SaveFrameToFile(p_dump_file_struct->output_file_name, pframe, surf_info);
                }
            }
        }
        // release and flush frame
        viddec->ReleaseFrame(pts, true);
        n_frames_flushed ++;
    }

    return n_frames_flushed;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status RocDecVideoDecoder::Initialize(const char *src_filename, int device_id) {

    VideoDecoder::Status status = Status::OK;
    _src_filename = src_filename;
    _device_id = device_id;
    
    // create rocDecoder and Demuxer for rocDecode
    OutputSurfaceMemoryType mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;      // set to internal
    _demuxer = std::make_shared<VideoDemuxer>(src_filename);
    rocDecVideoCodec rocdec_codec_id = AVCodec2RocDecVideoCodec(_demuxer->GetCodecID());
    _rocvid_decoder = std::make_shared<RocVideoDecoder>(device_id, mem_type, rocdec_codec_id, 0, nullptr, 0);

    if(!_rocvid_decoder->CodecSupported(device_id, rocdec_codec_id, _demuxer->GetBitDepth())) {
        ERR("GPU doesn't support codec!")
        return VideoDecoder::Status::FAILED;
    }

    return status;
}

// Seeks to the frame_number in the video file and decodes each frame in the sequence.
VideoDecoder::Status RocDecVideoDecoder::Decode(unsigned char *output_buffer_ptr, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format) {
    
    VideoDecoder::Status status = Status::OK;
    VideoSeekContext video_seek_ctx;

    // Reconfig the decoder
    ReconfigDumpFileStruct reconfig_user_struct = { 0 };
    ReconfigParams reconfig_params = { 0 };
    reconfig_params.p_fn_reconfigure_flush = ReconfigureFlushCallback;
    reconfig_params.reconfig_flush_mode = RECONFIG_FLUSH_MODE_NONE;
    reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;
    _rocvid_decoder->SetReconfigParams(&reconfig_params);

    if (!_demuxer || !_rocvid_decoder || !output_buffer_ptr) {
        ERR("Decoder is not initialized");
        return Status::FAILED;        
    }
    if (!output_buffer_ptr || !(sequence_length|stride)) {
        ERR("Invalid parameter passed");
        return Status::FAILED;        
    }

    int64_t pts = 0, requested_frame_pts = 0;
    int n_video_bytes = 0, n_frame_returned = 0, n_frame = 0, pkg_flags = 0;
    OutputSurfaceInfo *surf_info;
    uint8_t *pvideo = nullptr;
    int num_decoded_frames = sequence_length * stride;

    bool b_seek = true;       // seek only for first frame in sequence
    uint32_t image_size = out_height * out_stride * sizeof(uint8_t);
    video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
    video_seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
    VideoPostProcess post_process;
    bool sequence_decoded = false;
    do {
        if (b_seek) {
            video_seek_ctx.seek_frame_ = static_cast<uint64_t>(seek_frame_number);
            video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
            video_seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
            _demuxer->Seek(video_seek_ctx, &pvideo, &n_video_bytes);
            pts = video_seek_ctx.out_frame_pts_;
            requested_frame_pts = video_seek_ctx.requested_frame_pts_;
            b_seek = false;
            _rocvid_decoder->FlushAndReconfigure();
        } else {
            _demuxer->Demux(&pvideo, &n_video_bytes, &pts);
        }
        // Treat 0 bitstream size as end of stream indicator
        if (n_video_bytes == 0) {
            pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
        }
        n_frame_returned = _rocvid_decoder->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts);
        if (!n_frame && !_rocvid_decoder->GetOutputSurfaceInfo(&surf_info)) {
            ERR("Failed to get Output Surface Info!");
            return Status::FAILED;
        }
        // Take the min of sequence length and num frames to avoid out of bounds memory error
        int required_n_frames = std::min(static_cast<int>(sequence_length), n_frame_returned);
        for (int i = 0; i < required_n_frames; i++) {
            uint8_t *pframe = _rocvid_decoder->GetFrame(&pts);
            if (pts >= requested_frame_pts) {
                if (n_frame % stride == 0) {
                    post_process.ColorConvertYUV2RGB(pframe, surf_info, output_buffer_ptr, _output_format, _hip_stream);
                    output_buffer_ptr += image_size;
                }
                n_frame++;
            }
            // release frame
            _rocvid_decoder->ReleaseFrame(pts);
            if (n_frame == num_decoded_frames) {
                sequence_decoded = true;
                break;
            }
        }

        if (sequence_decoded) {
            if (hipStreamSynchronize(_hip_stream) != hipSuccess)
                THROW("hipStreamSynchronize failed: ")
            // needed to flush the frames of the decoder before the start of a new sequence by passing EOS to parser
            n_frame_returned = _rocvid_decoder->DecodeFrame(nullptr, 0, ROCDEC_PKT_ENDOFSTREAM, -1);
            break;
        }
    } while (n_video_bytes);

    return status;
}

RocDecVideoDecoder::~RocDecVideoDecoder() {
    Release();
}

#endif
#endif
