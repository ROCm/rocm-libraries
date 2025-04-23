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

#pragma once

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/pixdesc.h>
    #if USE_AVCODEC_GREATER_THAN_58_134
        #include <libavcodec/bsf.h>
    #endif
}
#include "rocvideodecode/roc_video_dec.h"       // for derived class

#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>


#define MAX_AV_PACKET_DATA_SIZE     4096

typedef struct DecFrameBufferFFMpeg_ {
    AVFrame *av_frame_ptr;      /**< av_frame pointer for the decoded frame */
    uint8_t *frame_ptr;       /**< host/device memory pointer for the decoded frame depending on mem_type*/
    int64_t  pts;             /**<  timestamp for the decoded frame */
    int picture_index;         /**<  surface index for the decoded frame */
} DecFrameBufferFFMpeg;

typedef struct DecPacketBuffer_
{
    AVPacket *av_pckt;
    int av_frame_index;
} DecPacketBuffer;

class FFMpegVideoDecoder: public RocVideoDecoder {
    public:
        /**
         * @brief Construct a new FFMpegVideoDecoder object
         * 
         * @param num_threads : number of cpu threads for the decoder
         * @param out_mem_type : out_mem_type for the decoded surface
         * @param codec : codec type
         * @param force_zero_latency : no support in FFMpeg decoding (false)
         * @param p_crop_rect : to crop output
         * @param extract_user_SEI_Message : enable to extract SEI
         * @param disp_delay : output delayed by #disp_delay surfaces
         * @param max_width : Max. width for the output surface
         * @param max_height : Max. height for the output surface
         * @param clk_rate : FPS clock-rate
         * @param no_multithreading : run FFMpeg decoder in the main thread (no multithreading) 
         */
        FFMpegVideoDecoder(int num_threads,  OutputSurfaceMemoryType out_mem_type, rocDecVideoCodec codec, bool force_zero_latency = false,
                          const Rect *p_crop_rect = nullptr, bool extract_user_SEI_Message = false, uint32_t disp_delay = 0, bool no_multithreading = false, int max_width = 0, int max_height = 0,
                          uint32_t clk_rate = 1000);
        /**
         * @brief destructor
         * 
         */
        ~FFMpegVideoDecoder();

        /**
         * @brief this function decodes a frame and returns the number of frames avalable for display
         * 
         * @param data - pointer to the compressed data buffer that is to be decoded
         * @param size - size of the data buffer in bytes
         * @param pts - presentation timestamp
         * @param flags - video packet flags
         * @param num_decoded_pics - nummber of pictures decoded in this call
         * @return int - num of frames to display
         */
        int DecodeFrame(const uint8_t *data, size_t size, int pkt_flags, int64_t pts = 0, int *num_decoded_pics = nullptr);

        /**
         * @brief This function returns a decoded frame and timestamp. This should be called in a loop fetching all the available frames
         * 
         */
        uint8_t* GetFrame(int64_t *pts);

        /**
         * @brief function to release frame after use by the application: Only used with "OUT_SURFACE_MEM_DEV_INTERNAL"
         * 
         * @param pTimestamp - timestamp of the frame to be released (unmapped)
         * @param b_flushing - true when flushing
         * @return true      - success
         * @return false     - falied
         */
        bool ReleaseFrame(int64_t pTimestamp, bool b_flushing = false);

        /**
         * @brief Helper function to dump decoded output surface to file
         * 
         * @param output_file_name  - Output file name
         * @param dev_mem           - pointer to surface memory
         * @param surf_info         - surface info
         * @param rgb_image_size    - image size for rgb (optional). A non_zero value indicates the surf_mem holds an rgb interleaved image and the entire size will be dumped to file
         */
        void SaveFrameToFile(std::string output_file_name, void *surf_mem, OutputSurfaceInfo *surf_info, size_t rgb_image_size = 0);

        /**
        *   @brief  This function is used to get the current frame size based on pixel format.
        */
        virtual int GetFrameSize() { assert(disp_width_); return ((disp_width_ * disp_height_) + ((chroma_height_ * chroma_width_) * num_chroma_planes_)) * byte_per_pixel_; }

    private:
        /**
         *   @brief  Callback function to be registered for getting a callback when decoding of sequence starts
         */
        static int ROCDECAPI FFMpegHandleVideoSequenceProc(void *p_user_data, RocdecVideoFormat *p_video_format) { return ((FFMpegVideoDecoder *)p_user_data)->HandleVideoSequence(p_video_format); }

        /**
         *   @brief  Callback function to be registered for getting a callback when a decoded frame is ready to be decoded
         */
        static int ROCDECAPI FFMpegHandlePictureDecodeProc(void *p_user_data, RocdecPicParams *p_pic_params) { return ((FFMpegVideoDecoder *)p_user_data)->HandlePictureDecode(p_pic_params); }

        /**
         *   @brief  Callback function to be registered for getting a callback when a decoded frame is available for display
         */
        static int ROCDECAPI FFMpegHandlePictureDisplayProc(void *p_user_data, RocdecParserDispInfo *p_disp_info) { return ((FFMpegVideoDecoder *)p_user_data)->HandlePictureDisplay(p_disp_info); }

        /**
         *   @brief  Callback function to be registered for getting a callback when all the unregistered user SEI Messages are parsed for a frame.
         */
        static int ROCDECAPI FFMpegHandleSEIMessagesProc(void *p_user_data, RocdecSeiMessageInfo *p_sei_message_info) { return ((FFMpegVideoDecoder *)p_user_data)->GetSEIMessage(p_sei_message_info); } 

        /**
         *   @brief  This function gets called when a sequence is ready to be decoded. The function also gets called
             when there is format change
        */
        int HandleVideoSequence(RocdecVideoFormat *p_video_format);

        /**
         *   @brief  This function gets called when a picture is ready to be decoded. rocDecDecodeFrame is called from this function
         *   to decode the picture
         */
        int HandlePictureDecode(RocdecPicParams *p_pic_params);

        /**
         *   @brief  This function gets called after a picture is decoded and available for display. Frames are fetched and stored in 
             internal buffer
        */
        int HandlePictureDisplay(RocdecParserDispInfo *p_disp_info);
        
        /**
         *   @brief  This function gets called when all unregistered user SEI messages are parsed for a frame
         */
        int GetSEIMessage(RocdecSeiMessageInfo *p_sei_message_info) { return RocVideoDecoder::GetSEIMessage(p_sei_message_info);};

        /**
         *   @brief  This function reconfigure decoder if there is a change in sequence params.
         */
        int ReconfigureDecoder(RocdecVideoFormat *p_video_format);

        void DecodeThread();
        int DecodeAvFrame(AVPacket *av_pkt, AVFrame *p_frame);
        void InitOutputFrameInfo(AVFrame *p_frame);
        int FlushDecoder();
        void PushPacket(AVPacket *pkt) {
            {
                std::lock_guard<std::mutex> lock(mtx_pkt_q_);
                av_packet_q_.push(pkt);
            }
            cv_pkt_.notify_one();
        }
        
        AVPacket *PopPacket() {
            AVPacket *pkt;
            std::unique_lock<std::mutex> lock(mtx_pkt_q_);
            cv_pkt_.wait(lock, [&] { return !av_packet_q_.empty(); });
            pkt = av_packet_q_.front();
            av_packet_q_.pop();
            return pkt;
        }

        void PushFrame(AVFrame *av_frame) {
            {
                std::lock_guard<std::mutex> lock(mtx_frame_q_);
                av_frame_q_.push(av_frame);
            }
            cv_frame_.notify_one();
        };

        AVFrame *PopFrame() {
            std::unique_lock<std::mutex> lock(mtx_frame_q_);
            cv_frame_.wait(lock, [&] { return !av_frame_q_.empty() || end_of_stream_; });
            if (end_of_stream_ && av_frame_q_.empty())
                return nullptr;
            AVFrame *p_frame = av_frame_q_.front();
            av_frame_q_.pop();
            return p_frame;
        }

        typedef enum { CMD_ABORT, CMD_DECODE } CommandType;
        typedef enum { STATUS_SUCCESS = 0, STATUS_FAILURE = -1 } StatusType;

        bool no_multithreading_ = false;
        bool b_decoder_initialized = false;
        uint32_t av_frame_cnt_ = 0;
        uint32_t av_pkt_cnt_ = 0;
        RocdecSourceDataPacket last_packet_;
        std::thread *ffmpeg_decoder_thread_ = nullptr;
        std::queue<AVPacket *> av_packet_q_;        // queue for compressed packets
        std::queue<AVFrame *> av_frame_q_;
        std::vector<DecFrameBufferFFMpeg> vp_frames_ffmpeg_;      // vector of decoded frames
        std::vector<AVFrame *> dec_frames_;      // vector of AVFrame * for decoded frames
        std::vector<AVPacket *> av_packets_;    // store of AVPackets for decoding
        std::vector<std::pair<uint8_t *, int>> av_packet_data_;
        std::mutex mtx_pkt_q_, mtx_frame_q_;               //for command and status
        std::condition_variable cv_pkt_, cv_frame_;     //for command and status
        std::atomic<bool> end_of_stream_ = false;
        // Variables for FFMpeg decoding
        AVCodecContext * dec_context_ = nullptr;
        AVPixelFormat decoder_pixel_format_;
#if USE_AVCODEC_GREATER_THAN_58_134
        const AVCodec *decoder_ = nullptr;
#else
        AVCodec *decoder_ = nullptr;
#endif
        AVFormatContext * formatContext = nullptr;
        AVInputFormat * inputFormat = nullptr;
        AVStream *video = nullptr;
};
