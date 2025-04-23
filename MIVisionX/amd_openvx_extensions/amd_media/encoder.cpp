/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "vx_amd_media.h"
#include "kernels.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>
#include <vector>
#include <sstream>

// GPU configuration
#define ENCODE_ENABLE_GPU       1         // enable use of GPU buffers

#if __APPLE__
#include <opencl.h>
#else
#if ENABLE_OPENCL
#include <CL/cl.h>
#elif ENABLE_HIP
#include "hip/hip_runtime.h"
#endif
#endif

#define DEFAULT_MBPS               4.0f
#define DEFAULT_FPS                30.0f
#define DEFAULT_BFRAMES            0
#define DEFAULT_GOPSIZE            60
#define ENCODE_BUFFER_POOL_SIZE    4         // number of buffers in encoder queue: keep it atleast 2

typedef struct {
    vx_uint32 size;
    vx_uint32 type;
} AuxDataContainerHeader;

typedef struct {
    AuxDataContainerHeader h0;
    int outputFrameCount;
    int reserved[3];
    int64_t cpuTimestamp;
} LoomIoMediaEncoderAuxInfo;

class CLoomIoMediaEncoder {
public:
    CLoomIoMediaEncoder(vx_node node, const char ioConfig[], vx_uint32 width, vx_uint32 height, vx_df_image format, vx_uint32 stride, vx_uint32 offset, vx_size input_aux_data_max_size);
    ~CLoomIoMediaEncoder();
    vx_status Initialize();
    vx_status ProcessFrame(vx_image input_image, vx_array input_aux, vx_array output_aux);
    vx_status UpdateBufferGPU(vx_image input_image, vx_array input_aux);
    void SetEnableUserBufferGPUMode(vx_bool bEnable) { m_enableUserBufferGPU = bEnable;};
    vx_bool m_enableUserBufferGPU;

protected:
    typedef enum { cmd_abort, cmd_encode } command;
    void EncodeLoop();
    void PushCommand(command cmd);
    void PushAck(int ack);
    command PopCommand();
    int PopAck();

private:
    vx_node node;
    std::string ioConfig;
    vx_uint32 width, height;
    vx_df_image format;
    AVPixelFormat inputFormat;
    int stride, offset;
    vx_size input_aux_data_max_size;
    AVFormatContext * formatContext;
    AVStream * videoStream;
    AVCodecContext * videoCodecContext;
    const AVCodec * videoCodec;
    const AVOutputFormat *outputFmt;
    SwsContext * conversionContext;
    void* mem[ENCODE_BUFFER_POOL_SIZE];
#if ENABLE_OPENCL
    cl_command_queue cmdq;
#elif ENABLE_HIP
    hipDeviceProp_t hip_dev_prop;
    uint8_t *hostBuffer;
#endif
    AVFrame * videoFrame[ENCODE_BUFFER_POOL_SIZE];
    uint8_t * outputAuxBuffer;
    vx_size outputAuxLength;
    FILE * fpOutput;
    std::mutex mutexCmd, mutexAck;
    std::condition_variable cvCmd, cvAck;
    std::deque<command> queueCmd;
    std::deque<int> queueAck;
    std::thread * thread;
    bool threadTerminated;
    int encodeFrameCount;
    int inputFrameCount;
    float mbps, fps;
    int bframes, gopsize;
};

// helper function for spliting streams.
extern std::vector<std::string> split(const std::string& s, char delimiter);

void CLoomIoMediaEncoder::PushCommand(CLoomIoMediaEncoder::command cmd)
{
    std::unique_lock<std::mutex> lock(mutexCmd);
    queueCmd.push_front(cmd);
    cvCmd.notify_one();
}

CLoomIoMediaEncoder::command CLoomIoMediaEncoder::PopCommand()
{
    std::unique_lock<std::mutex> lock(mutexCmd);
    cvCmd.wait(lock, [=] { return !queueCmd.empty(); });
    command cmd = std::move(queueCmd.back());
    queueCmd.pop_back();
    return cmd;
}

void CLoomIoMediaEncoder::PushAck(int ack)
{
    std::unique_lock<std::mutex> lock(mutexAck);
    queueAck.push_front(ack);
    cvAck.notify_one();
}

int CLoomIoMediaEncoder::PopAck()
{
    std::unique_lock<std::mutex> lock(mutexAck);
    cvAck.wait(lock, [=] { return !queueAck.empty(); });
    int ack = std::move(queueAck.back());
    queueAck.pop_back();
    return ack;
}

CLoomIoMediaEncoder::CLoomIoMediaEncoder(vx_node node_, const char ioConfig_[], vx_uint32 width_, vx_uint32 height_, vx_df_image format_, vx_uint32 stride_, vx_uint32 offset_, vx_size input_aux_data_max_size_)
    : node{ node_ }, ioConfig(ioConfig_), width{ width_ }, height{ height_ }, format{ format_ },
      stride{ static_cast<int>(stride_) }, offset{ static_cast<int>(offset_) }, input_aux_data_max_size{ input_aux_data_max_size_ },
      inputFrameCount{ 0 }, encodeFrameCount{ 0 }, threadTerminated{ false }, inputFormat{ AV_PIX_FMT_UYVY422 }, 
      fpOutput{ nullptr }, formatContext{ nullptr }, videoStream{ nullptr }, outputAuxBuffer{ nullptr }, outputAuxLength{ 0 },
      videoCodecContext{ nullptr }, videoCodec{ nullptr }, conversionContext{ nullptr }, thread{ nullptr },
      mbps{ DEFAULT_MBPS }, fps{ DEFAULT_FPS }, gopsize{ DEFAULT_GOPSIZE }, bframes{ DEFAULT_BFRAMES }
{
    m_enableUserBufferGPU = false;   // use host buffers by default
#if ENABLE_OPENCL
    cmdq = nullptr;
#endif
    memset(mem, 0, sizeof(mem));
    memset(videoFrame, 0, sizeof(videoFrame));
    outputAuxBuffer = new uint8_t[input_aux_data_max_size + sizeof(LoomIoMediaEncoderAuxInfo)]();
    // initialize freq inside GetTimeInMicroseconds()
    GetTimeInMicroseconds();
}

CLoomIoMediaEncoder::~CLoomIoMediaEncoder()
{
    // terminate the thread
    if (thread) {
        PushCommand(cmd_abort);
        while (!threadTerminated) {
            if (PopAck() < 0)
                break;
        }
        thread->join();
        delete thread;
    }
    // close output mediaFile
    if (fpOutput) {
        fclose(fpOutput);
    }
    if (formatContext) {
        av_write_trailer(formatContext);
        av_free(formatContext);
    }
    // release GPU and media resources
    if (m_enableUserBufferGPU) {
    #if ENABLE_OPENCL
        if (cmdq) clReleaseCommandQueue(cmdq);
    #elif ENABLE_HIP
        if (hostBuffer) hipHostFree((void *)hostBuffer);
    #endif
    }

    for (int i = 0; i < ENCODE_BUFFER_POOL_SIZE; i++) {
        if (videoFrame[i]) {
            av_frame_free(&videoFrame[i]);
        }
    }
    if (videoCodecContext) {
        avcodec_free_context(&videoCodecContext);
    }
    if (outputAuxBuffer) delete[] outputAuxBuffer;
}

vx_status CLoomIoMediaEncoder::Initialize()
{
    const char * outFileName = nullptr;
    // check for valid image type support and get stride in bytes (aligned to 16-byte boundary)
    if (format == VX_DF_IMAGE_NV12) {
        inputFormat = AV_PIX_FMT_NV12;
    }
    else if (format == VX_DF_IMAGE_UYVY) {
        inputFormat = AV_PIX_FMT_UYVY422;
    }
    else if (format == VX_DF_IMAGE_YUYV) {
        inputFormat = AV_PIX_FMT_YUYV422;
    }
    else if (format == VX_DF_IMAGE_RGB) {
        inputFormat = AV_PIX_FMT_RGB24;
    }
    else {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_FORMAT, "ERROR: input image format %4.4s not supported", &format);
        return VX_ERROR_INVALID_FORMAT;
    }

    // get media configuration and fileName
    std::vector<std::string> mediainfo = split(ioConfig, ',');
    if (mediainfo.size() < 1) {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: invalid input arguments");
        return VX_ERROR_INVALID_LINK;
    }
    if (mediainfo.size() == 1) {
        outFileName = mediainfo[0].c_str();
    }
    else if (mediainfo.size() == 5) {
        outFileName = mediainfo[0].c_str();
        mbps = atoi(mediainfo[1].c_str());
        fps = atoi(mediainfo[2].c_str());
        bframes = atoi(mediainfo[3].c_str());
        gopsize = atoi(mediainfo[4].c_str());
    } else{
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: invalid input arguments");
        return VX_ERROR_INVALID_LINK;
    }

#if 0 // todo:: runvx has issue taking this string format
    const char * s = ioConfig.c_str();
    if (*s == '{') {
        sscanf(s + 1, "%f,%f,%d,%d", &mbps, &fps, &bframes, &gopsize);
        while (*s && *s != '}')
            s++;
        if (*s == '}') s++;
        if (*s == ',') s++;
        else {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: invalid ioConfig: %s\nERROR: invalid ioConfig: valid syntax: [{mbps,fps,bframes,gopsize},]filename.mp4\n", ioConfig.c_str());
            return VX_ERROR_INVALID_VALUE;
        }
    }
    const char * fileName = s;
#endif

    // open media file and initialize codec
    ERROR_CHECK_STATUS(initialize_ffmpeg());
    ERROR_CHECK_NULLPTR(videoCodec = avcodec_find_encoder(AV_CODEC_ID_H264));
    ERROR_CHECK_NULLPTR(videoCodecContext = avcodec_alloc_context3(videoCodec));
    videoCodecContext->bit_rate = (int)(mbps * 1000000);
    videoCodecContext->width = width;
    videoCodecContext->height = height;
    videoCodecContext->time_base.num = (int)(60000.0f / fps);
    videoCodecContext->time_base.den = 60000;
    videoCodecContext->gop_size = gopsize;
    videoCodecContext->max_b_frames = bframes;
    videoCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
    if (videoCodec->id == AV_CODEC_ID_H264)
        av_opt_set(videoCodecContext->priv_data, "preset", "slow", 0);

    ERROR_CHECK_STATUS(avcodec_open2(videoCodecContext, videoCodec, nullptr));
    ERROR_CHECK_NULLPTR(conversionContext = sws_getContext(width, height, inputFormat, width, height, videoCodecContext->pix_fmt, SWS_BICUBIC, NULL, NULL, NULL));
    for (int i = 0; i < ENCODE_BUFFER_POOL_SIZE; i++) {
        ERROR_CHECK_NULLPTR(videoFrame[i] = av_frame_alloc());
        videoFrame[i]->format = videoCodecContext->pix_fmt;
        videoFrame[i]->width = width;
        videoFrame[i]->height  = height;
        ERROR_CHECK_STATUS(av_frame_get_buffer(videoFrame[i], 0));      // ffmpeg will allocate framebuffer here
        // just one videoFrame[0] is sufficient for GPU mode
        if (m_enableUserBufferGPU)
          break;
    }
    // open output file
    if (strlen(outFileName) > 4 && !strcmp(outFileName + strlen(outFileName) - 4, ".264")) {
        fpOutput = fopen(outFileName, "wb");
        if (!fpOutput) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: CLoomIoMediaEncoder::Initialize: unable to create: %s", outFileName);
            return VX_ERROR_INVALID_LINK;
        }
    }
    else {
        int status = avformat_alloc_output_context2(&formatContext, nullptr, nullptr, outFileName);
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: CLoomIoMediaEncoder::Initialize: avformat_alloc_output_context2(...,%s) failed (%d)", outFileName, status);
            return VX_ERROR_INVALID_LINK;
        }
        ERROR_CHECK_NULLPTR(videoStream = avformat_new_stream(formatContext, videoCodec));
        videoStream->time_base = (AVRational){ 1, (int)fps };
        videoCodecContext->time_base = videoStream->time_base;
        videoStream->id = formatContext->nb_streams - 1;
        if (formatContext->oformat->flags & AVFMT_GLOBALHEADER)
            videoCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        av_dump_format(formatContext, 0, outFileName, 1);
        if (!(formatContext->oformat->flags & AVFMT_NOFILE)) {
            if ((status = avio_open(&formatContext->pb, outFileName, AVIO_FLAG_WRITE)) < 0) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: CLoomIoMediaEncoder::Initialize: avio_open(...,%s) failed (%d)", outFileName, status);
                return VX_ERROR_INVALID_LINK;
            }
        }
        //set parameters for the stream
        ERROR_CHECK_STATUS(avcodec_parameters_from_context(videoStream->codecpar, videoCodecContext));
        ERROR_CHECK_STATUS(avformat_write_header(formatContext, nullptr));
    }

    if (m_enableUserBufferGPU) {
    #if ENABLE_OPENCL
        cl_context context = nullptr;
        ERROR_CHECK_STATUS(vxQueryContext(vxGetContext((vx_reference)node), VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &context, sizeof(context)));
        cl_device_id device_id = nullptr;
        ERROR_CHECK_STATUS(clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, nullptr));
      #if defined(CL_VERSION_2_0)
        cmdq = clCreateCommandQueueWithProperties(context, device_id, 0, nullptr);
      #else
        cmdq = clCreateCommandQueue(context, device_id, 0, nullptr);
      #endif
        ERROR_CHECK_NULLPTR(cmdq);
    #elif ENABLE_HIP
        hostBuffer  = nullptr;
        int hip_device = -1;
        ERROR_CHECK_STATUS(vxQueryContext(vxGetContext((vx_reference)node), VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE, &hip_device, sizeof(hip_device)));
        if (hip_device < 0) {
            return VX_FAILURE;
        }
        int bufHeight = (inputFormat == AV_PIX_FMT_NV12) ? (height + (height>>1)) : height;
        hipError_t err = hipHostMalloc((void **)&hostBuffer, offset + stride * bufHeight, hipHostMallocDefault);
        if (err != hipSuccess) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_NO_MEMORY, "ERROR: CLoomIoMediaEncoder::Memory allocation failed \n");
        }
    #endif
    }

    // start encoder thread
    inputFrameCount = 0;
    encodeFrameCount = 0;
    threadTerminated = false;
    thread = new std::thread(&CLoomIoMediaEncoder::EncodeLoop, this);
    ERROR_CHECK_NULLPTR(thread);

    // debug info
    vxAddLogEntry((vx_reference)node, VX_SUCCESS, "INFO: writing %dx%d %.2fmbps %.2ffps gopsize=%d bframes=%d video into %s", width, height, mbps, fps, gopsize, bframes, outFileName);

    return VX_SUCCESS;
}

#if ENCODE_ENABLE_GPU
vx_status CLoomIoMediaEncoder::UpdateBufferGPU(vx_image input_image, vx_array input_aux)
{
    // update output auxiliary information
    outputAuxLength = 0;
    if (input_aux) {
        ERROR_CHECK_STATUS(vxQueryArray(input_aux, VX_ARRAY_NUMITEMS, &outputAuxLength, sizeof(outputAuxLength)));
        if (outputAuxLength > 0) {
            ERROR_CHECK_STATUS(vxCopyArrayRange(input_aux, 0, outputAuxLength, sizeof(uint8_t), outputAuxBuffer, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        }
    }
    LoomIoMediaEncoderAuxInfo * auxInfo = (LoomIoMediaEncoderAuxInfo *)&outputAuxBuffer[outputAuxLength];
    auxInfo->h0.size = sizeof(*auxInfo);
    auxInfo->h0.type = AMDOVX_KERNEL_AMD_MEDIA_ENCODE;
    auxInfo->outputFrameCount = inputFrameCount;
    auxInfo->cpuTimestamp = GetTimeInMicroseconds();
    outputAuxLength += auxInfo->h0.size;

    // pick the OpenCL buffer frame from pool
    int bufId = inputFrameCount % ENCODE_BUFFER_POOL_SIZE; inputFrameCount++;
    if (m_enableUserBufferGPU) {
    #if ENABLE_OPENCL
        ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &mem[bufId], sizeof(void*)));
    #elif ENABLE_HIP
        ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &mem[bufId], sizeof(void*)));
    #endif
    }
    return VX_SUCCESS;
}
#endif

vx_status CLoomIoMediaEncoder::ProcessFrame(vx_image input_image, vx_array input_aux, vx_array output_aux)
{
    // wait until there is an ACK from encoder thread
    int ack = PopAck();
    if ((ack < 0) || threadTerminated)
    { // nothing to process, so abandon the graph execution
        return VX_ERROR_GRAPH_ABANDONED;
    }

    if (m_enableUserBufferGPU) {        
        UpdateBufferGPU(input_image, input_aux);
       // just submit GPU buffer for encoding
        PushCommand(cmd_encode);
    } else {
        // format convert input image into encode buffer
        int bufId = inputFrameCount % ENCODE_BUFFER_POOL_SIZE; inputFrameCount++;
        int ret = av_frame_make_writable(videoFrame[bufId]);
        if (ret < 0) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::ProcessFrame failed to write data(%d)\n", ret);
            return VX_FAILURE;
        }
        vx_rectangle_t rect = { 0, 0, width, height };
        vx_map_id map_id, map_id1;
        vx_imagepatch_addressing_t addr;
        uint8_t * ptr = nullptr;
        uint8_t *src_data[4] = {0};
        int src_linesize[4] = {0};
        ERROR_CHECK_STATUS(vxMapImagePatch(input_image, &rect, 0, &map_id, &addr, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        src_data[0] = ptr;
        src_linesize[0] = addr.stride_y;
        if (inputFormat == AV_PIX_FMT_NV12) {
            ERROR_CHECK_STATUS(vxMapImagePatch(input_image, &rect, 1, &map_id1, &addr, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
            src_data[1] = ptr;
            src_linesize[1] = addr.stride_y;
        }
        int status = sws_scale(conversionContext, src_data, src_linesize, 0, height, videoFrame[bufId]->data, videoFrame[bufId]->linesize);
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::ProcessFrame: sws_scale() failed (%d)\n", status);
            return VX_FAILURE;
        }
        ERROR_CHECK_STATUS(vxUnmapImagePatch(input_image, map_id));
        if (inputFormat == AV_PIX_FMT_NV12) {
            ERROR_CHECK_STATUS(vxUnmapImagePatch(input_image, map_id1));
        }
        // submit encoding
        PushCommand(cmd_encode);
    }

    // copy input aux data to output, if there is input aux data.
    ERROR_CHECK_STATUS(vxTruncateArray(output_aux, 0));
    if (outputAuxLength > 0) {
        ERROR_CHECK_STATUS(vxAddArrayItems(output_aux, outputAuxLength, outputAuxBuffer, sizeof(uint8_t)));
    }

    return VX_SUCCESS;
}

void CLoomIoMediaEncoder::EncodeLoop()
{
    // initial ACK to inform producer for readiness
    for (int i = 1; i < ENCODE_BUFFER_POOL_SIZE; i++)
        PushAck(0);

    // initialize packet and start encoding
    AVPacket *pkt;
    pkt = av_packet_alloc();
    if (!pkt) return;

    int64_t pts = 0;
    for (command cmd; !threadTerminated && ((cmd = PopCommand()) != cmd_abort);) {
        int status;
        // get the bufId to process
        int bufId = (encodeFrameCount % ENCODE_BUFFER_POOL_SIZE);
        int got_output = 0;
        // format convert input image into encode buffer
        if ( m_enableUserBufferGPU) {
            int mapHeight = (inputFormat == AV_PIX_FMT_NV12)? (height + (height>>1)) : height;
    #if ENABLE_OPENCL
            cl_int err = -1;
            uint8_t * ptr = (uint8_t *)clEnqueueMapBuffer(cmdq, (cl_mem)mem[bufId], CL_TRUE, CL_MAP_READ, offset, mapHeight * stride, 0, nullptr, nullptr, &err);
            if (err < 0 || !ptr) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: clEnqueueMapBuffer() failed (%d)\n", err);
                threadTerminated = true;
                PushAck(-1);
                return;
            }
            clFinish(cmdq);
            uint8_t *src_data[4] = {0};
            int src_linesize[4] = {stride};
            src_data[0] = ptr;
            if (inputFormat == AV_PIX_FMT_NV12) src_data[1] = src_data[0] + height*stride;
            status = sws_scale(conversionContext, src_data, src_linesize, 0, height, videoFrame[0]->data, videoFrame[0]->linesize);

            err = clEnqueueUnmapMemObject(cmdq, (cl_mem)mem[bufId], ptr, 0, nullptr, nullptr);
            if (err < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: clEnqueueUnmapMemObject() failed (%d)\n", err);
                threadTerminated = true;
                PushAck(-1);
                return;
            }
            clFinish(cmdq);
            if (status < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: sws_scale() failed (%d)\n", status);
                threadTerminated = true;
                PushAck(-1);
                return;
            }
    #elif ENABLE_HIP
            hipError_t err;
            err = hipMemcpyDtoH((void *)hostBuffer, ((uint8_t *)mem[bufId] + offset), mapHeight * stride);
            if (err != hipSuccess) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: hipMemcpyDtoH failed => %d\n", err);
                threadTerminated = true;
                PushAck(-1);
                return;
            }
            uint8_t *src_data[4] = {0};
            int src_linesize[4] = {stride};
            src_data[0] = hostBuffer;
            if (inputFormat == AV_PIX_FMT_NV12) src_data[1] = src_data[0] + height*stride;

            status = sws_scale(conversionContext, src_data, src_linesize, 0, height, videoFrame[0]->data, videoFrame[0]->linesize);
            if (status < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: sws_scale() failed (%d)\n", status);
                threadTerminated = true;
                PushAck(-1);
                return;
            }
    #endif
            // reset bufId to zero because only videoFrame[0] is valid
            bufId = 0;
        }
        // encode video frame and write output to file
        videoFrame[bufId]->pts = pts;
        int status_send = avcodec_send_frame(videoCodecContext, videoFrame[bufId]);
        got_output = avcodec_receive_packet(videoCodecContext, pkt);
        status = std::min(status_send, got_output);
        if (status == AVERROR(EAGAIN)) {
            PushAck(0);
            continue;
        }
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: avcodec_send_frame/receive_packet() failed (%4.4s:0x%08x:%d) for frame:%d\n", &status, status, status, encodeFrameCount);
            threadTerminated = true;
            PushAck(-1);
            return;
        }
        if (pkt->size > 0) {
            if (formatContext) {
                av_packet_rescale_ts(pkt, videoCodecContext->time_base, videoStream->time_base);
                pkt->stream_index = videoStream->index;
                status = av_interleaved_write_frame(formatContext, pkt);
                if (status < 0) {
                    vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: av_interleaved_write_frame() failed (%4.4s:0x%08x:%d) for frame:%d\n", &status, status, status, encodeFrameCount);
                    threadTerminated = true;
                    PushAck(-1);
                    return;
                }
            } else if (fpOutput)
            {
                fwrite(pkt->data, 1, pkt->size, fpOutput);
            }
            pts++;
            encodeFrameCount++;
        }
        av_packet_unref(pkt);
        PushAck(0);
    }
    // process the delayed frames
    for (int got_output = !0; got_output;) {
        int status_send = avcodec_send_frame(videoCodecContext, nullptr);
        got_output = avcodec_receive_packet(videoCodecContext, pkt);
        int status = std::min(status_send, got_output);
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: avcodec_send_frame/receive_packet() failed (%4.4s:%d) at the end\n", &status, status);
            threadTerminated = true;
            PushAck(-1);
            return;
        }
        if (pkt->size > 0) {
            if (formatContext) {
                pkt->stream_index = videoStream->index;
                av_packet_rescale_ts(pkt, videoCodecContext->time_base, videoStream->time_base);
                status = av_interleaved_write_frame(formatContext, pkt);
                if (status < 0) {
                    vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: av_interleaved_write_frame() failed (%4.4s:0x%08x:%d) for frame:%d\n", &status, status, status, encodeFrameCount);
                    threadTerminated = true;
                    PushAck(-1);
                    return;
                }
            } else if (fpOutput)
            {
                fwrite(pkt->data, 1, pkt->size, fpOutput);
            }
        }
        av_packet_unref(pkt);
    }
    // mark termination and send ACK
    threadTerminated = true;
    av_packet_free(&pkt);
    PushAck(-1);
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK amd_media_encode_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get encoder and input image
    CLoomIoMediaEncoder * encoder = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &encoder, sizeof(encoder)));
    if (!encoder) return VX_FAILURE;
    return encoder->ProcessFrame((vx_image)parameters[1], (vx_array)parameters[2], (vx_array)parameters[3]);
}

//! \brief The kernel initializer.
static vx_status VX_CALLBACK amd_media_encode_initialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    vx_bool enableUserBufferGPU = false;
    // get input parameters
    char ioConfig[VX_MAX_STRING_BUFFER_SIZE_AMD];
    vx_uint32 width = 0, height = 0, stride = 0, offset = 0;
    vx_df_image format = VX_DF_IMAGE_VIRT;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[0], ioConfig, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE, &stride, sizeof(stride)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET, &offset, sizeof(offset)));
    vx_size input_aux_data_max_size = 0;
    if (parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_CAPACITY, &input_aux_data_max_size, sizeof(input_aux_data_max_size)));
    }
    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &enableUserBufferGPU, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }

    // create and initialize encoder
    if (stride == 0) stride = width;    // stride can't be zero
    CLoomIoMediaEncoder * encoder = new CLoomIoMediaEncoder(node, ioConfig, width, height, format, stride, offset, input_aux_data_max_size);
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &encoder, sizeof(encoder)));
    if (parameters[4]) {
        printf("encoder: GPU mode : %d\n", enableUserBufferGPU);
        encoder->SetEnableUserBufferGPUMode(enableUserBufferGPU);
    }

    ERROR_CHECK_STATUS(encoder->Initialize());

    return VX_SUCCESS;
}

//! \brief The kernel deinitializer.
static vx_status VX_CALLBACK amd_media_encode_deinitialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get encoder
    CLoomIoMediaEncoder * encoder = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &encoder, sizeof(encoder)));

    if (encoder) {
        // release the resources
        delete encoder;
    }

    return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK amd_media_encode_validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // make sure input media filename
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[0], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_STRING_AMD)
        return VX_ERROR_INVALID_FORMAT;
    // make sure input format is UYVY/YUYV/RGB
    vx_uint32 width = 0, height = 0;
    vx_df_image format = VX_DF_IMAGE_VIRT;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if (format != VX_DF_IMAGE_UYVY && format != VX_DF_IMAGE_YUYV && format != VX_DF_IMAGE_RGB && format != VX_DF_IMAGE_NV12)
        return VX_ERROR_INVALID_FORMAT;
    // check input auxiliary data parameter
    if (parameters[2]) {
        // make sure data type is UINT8
        vx_enum itemtype = VX_TYPE_INVALID;
        vx_size capacity = 0;
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
        if (itemtype != VX_TYPE_UINT8)
            return VX_ERROR_INVALID_TYPE;
    }
    // check and set output auxiliary data parameter
    vx_enum itemtype = VX_TYPE_INVALID;
    vx_size capacity = 0;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
    if (itemtype != VX_TYPE_UINT8)
        return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
    vx_bool enableUserBufferGPU = false;
    if (parameters[4]) {
        vx_enum scalar_type;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if(scalar_type != VX_TYPE_BOOL) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &enableUserBufferGPU, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        // do we need to do the following?
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU, &enableUserBufferGPU, sizeof(enableUserBufferGPU)));
//        printf("decoder validate:: set enableUserBufferGPU: %d\n", enableUserBufferGPU);
    }

    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status amd_media_encode_publish(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.amd_media.encode", AMDOVX_KERNEL_AMD_MEDIA_ENCODE,
                            amd_media_encode_kernel, 5, amd_media_encode_validate,
                            amd_media_encode_initialize, amd_media_encode_deinitialize);
    ERROR_CHECK_OBJECT(kernel);

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));  // media config+filename: "[mbps,fps,bframes,gopsize],filename.mp4"
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));   // input image
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));   // input auxiliary data (optional)
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));  // output auxiliary data
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL)); // input: to set enableUserBufferGPU flag

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL amdMediaEncoderNode(vx_graph graph, const char *output_str, vx_image input, vx_array aux_data_in, vx_array aux_data_out, vx_bool enable_gpu_input)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_output = vxCreateScalar(context, VX_TYPE_STRING_AMD, output_str);
        vx_scalar s_enable_gpu_input = vxCreateScalar(context, VX_TYPE_BOOL, &enable_gpu_input);
        vx_reference params[] = {
            (vx_reference)s_output,
            (vx_reference)input,
            (vx_reference)aux_data_in,
            (vx_reference)aux_data_out,
            (vx_reference)s_enable_gpu_input,
        };
        if (vxGetStatus((vx_reference)s_output) == VX_SUCCESS) {
            node = createMediaNode(graph, "com.amd.amd_media.encode", params, sizeof(params) / sizeof(params[0])); // added node to graph
            vxReleaseScalar(&s_output);
            vxReleaseScalar(&s_enable_gpu_input);
        }
    }
    return node;
}
