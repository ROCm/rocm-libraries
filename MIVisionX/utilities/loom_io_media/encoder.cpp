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

#include "vx_loomio_media.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

// OpenCL configuration
#define ENCODE_ENABLE_OPENCL       1         // enable use of OpenCL buffers

#if ENCODE_ENABLE_OPENCL
#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
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
    vx_status UpdateBufferOpenCL(vx_image input_image, vx_array input_aux);
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
    int width;
    int height;
    vx_df_image format;
    AVPixelFormat inputFormat;
    int stride, offset;
    vx_size input_aux_data_max_size;
    AVFormatContext * formatContext;
    AVStream * videoStream;
    AVCodecContext * videoCodecContext;
    AVCodec * videoCodec;
    SwsContext * conversionContext;
#if ENCODE_ENABLE_OPENCL
    cl_command_queue cmdq;
    cl_mem mem[ENCODE_BUFFER_POOL_SIZE];
#endif
    AVFrame * videoFrame[ENCODE_BUFFER_POOL_SIZE];
    uint8_t * outputBuffer;
    int outputBufferSize;
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
    : node{ node_ }, ioConfig(ioConfig_), width{ static_cast<int>(width_) }, height{ static_cast<int>(height_) }, format{ format_ },
      stride{ static_cast<int>(stride_) }, offset{ static_cast<int>(offset_) }, input_aux_data_max_size{ input_aux_data_max_size_ },
      inputFrameCount{ 0 }, encodeFrameCount{ 0 }, threadTerminated{ false }, inputFormat{ AV_PIX_FMT_UYVY422 }, 
      outputBuffer{ nullptr }, outputBufferSize{ 1000000 }, fpOutput{ nullptr }, formatContext{ nullptr }, videoStream{ nullptr }, outputAuxBuffer{ nullptr }, outputAuxLength{ 0 },
      videoCodecContext{ nullptr }, videoCodec{ nullptr }, conversionContext{ nullptr }, thread{ nullptr },
      mbps{ DEFAULT_MBPS }, fps{ DEFAULT_FPS }, gopsize{ DEFAULT_GOPSIZE }, bframes{ DEFAULT_BFRAMES }
{
#if ENCODE_ENABLE_OPENCL
    cmdq = nullptr;
    memset(mem, 0, sizeof(mem));
#endif
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

    // release OpenCL and media resources
    if (outputBuffer) aligned_free(outputBuffer);
#if ENCODE_ENABLE_OPENCL
    if (cmdq) clReleaseCommandQueue(cmdq);
    for (int i = 0; i < ENCODE_BUFFER_POOL_SIZE; i++) {
        if (mem[i]) clReleaseMemObject(mem[i]);
    }
#endif
    for (int i = 0; i < ENCODE_BUFFER_POOL_SIZE; i++) {
        if (videoFrame[i]) {
            if (videoFrame[i]->data[0]) aligned_free(videoFrame[i]->data[0]);
            if (videoFrame[i]->data[1]) aligned_free(videoFrame[i]->data[1]);
            av_frame_free(&videoFrame[i]);
        }
    }
    if (conversionContext) av_free(conversionContext);
    if (videoCodecContext) {
        avcodec_close(videoCodecContext);
        av_free(videoCodecContext);
    }
    if (videoCodec) av_free(videoCodec);
    if (outputAuxBuffer) delete[] outputAuxBuffer;
}

vx_status CLoomIoMediaEncoder::Initialize()
{
    // check for valid image type support and get stride in bytes (aligned to 16-byte boundary)
    if (format == VX_DF_IMAGE_UYVY) {
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
    videoCodecContext->pix_fmt = AV_PIX_FMT_NV12;
    ERROR_CHECK_STATUS(avcodec_open2(videoCodecContext, videoCodec, nullptr));
    ERROR_CHECK_NULLPTR(conversionContext = sws_getContext(width, height, inputFormat, width, height, videoCodecContext->pix_fmt, SWS_BICUBIC, NULL, NULL, NULL));
    for (int i = 0; i < ENCODE_BUFFER_POOL_SIZE; i++) {
        ERROR_CHECK_NULLPTR(videoFrame[i] = av_frame_alloc());
        videoFrame[i]->data[0] = aligned_alloc(width * height);
        videoFrame[i]->data[1] = aligned_alloc(width * height / 2);
        videoFrame[i]->linesize[0] = width;
        videoFrame[i]->linesize[1] = width;
        videoFrame[i]->format = AV_PIX_FMT_NV12;
#if ENCODE_ENABLE_OPENCL
        // just one videoFrame[0] is sufficient
        break;
#endif
    }
    outputBufferSize = 1024*1024;
    ERROR_CHECK_NULLPTR(outputBuffer = aligned_alloc(outputBufferSize));

    // open output file
    if (strlen(fileName) > 4 && !strcmp(fileName + strlen(fileName) - 4, ".264")) {
        fpOutput = fopen(fileName, "wb");
        if (!fpOutput) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: CLoomIoMediaEncoder::Initialize: unable to create: %s", fileName);
            return VX_ERROR_INVALID_LINK;
        }
    }
    else {
        int status = avformat_alloc_output_context2(&formatContext, nullptr, nullptr, fileName);
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: CLoomIoMediaEncoder::Initialize: avformat_alloc_output_context2(...,%s) failed (%d)", fileName, status);
            return VX_ERROR_INVALID_LINK;
        }
        ERROR_CHECK_NULLPTR(videoStream = avformat_new_stream(formatContext, videoCodec));
        videoStream->id = formatContext->nb_streams - 1;
        if (formatContext->oformat->flags & AVFMT_GLOBALHEADER)
            videoCodecContext->flags |= CODEC_FLAG_GLOBAL_HEADER;
        av_dump_format(formatContext, 0, fileName, 1);
        if (!(formatContext->oformat->flags & AVFMT_NOFILE)) {
            if ((status = avio_open(&formatContext->pb, fileName, AVIO_FLAG_WRITE)) < 0) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: CLoomIoMediaEncoder::Initialize: avio_open(...,%s) failed (%d)", fileName, status);
                return VX_ERROR_INVALID_LINK;
            }
        }
        videoStream->codec->width = width;
        videoStream->codec->height = height;
        videoStream->codec->codec_id = videoCodecContext->codec_id;
        videoStream->codec->bit_rate = videoCodecContext->bit_rate;
        videoStream->codec->time_base = videoCodecContext->time_base;
        videoStream->codec->gop_size = videoCodecContext->gop_size;
        videoStream->codec->max_b_frames = videoCodecContext->max_b_frames;
        ERROR_CHECK_STATUS(avformat_write_header(formatContext, nullptr));
    }

#if ENCODE_ENABLE_OPENCL
    // allocate OpenCL encode buffers
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
    for (int i = 0; i < ENCODE_BUFFER_POOL_SIZE; i++) {
        mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, offset + stride * height, nullptr, nullptr);
        ERROR_CHECK_NULLPTR(mem[i]);
    }
#endif

    // start encoder thread
    inputFrameCount = 0;
    encodeFrameCount = 0;
    threadTerminated = false;
    thread = new std::thread(&CLoomIoMediaEncoder::EncodeLoop, this);
    ERROR_CHECK_NULLPTR(thread);

    // debug info
    vxAddLogEntry((vx_reference)node, VX_SUCCESS, "INFO: writing %dx%d %.2fmbps %.2ffps gopsize=%d bframes=%d video into %s", width, height, mbps, fps, gopsize, bframes, fileName);

    return VX_SUCCESS;
}

#if ENCODE_ENABLE_OPENCL
vx_status CLoomIoMediaEncoder::UpdateBufferOpenCL(vx_image input_image, vx_array input_aux)
{
    // wait until there is an ACK from encoder thread
    int ack = PopAck();
    if ((ack < 0) || threadTerminated) {
        // nothing to process, so abandon the graph execution
        return VX_ERROR_GRAPH_ABANDONED;
    }

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
    auxInfo->h0.type = AMDOVX_KERNEL_LOOMIO_MEDIA_ENCODE;
    auxInfo->outputFrameCount = inputFrameCount;
    auxInfo->cpuTimestamp = GetTimeInMicroseconds();
    outputAuxLength += auxInfo->h0.size;

    // pick the OpenCL buffer frame from pool
    int bufId = inputFrameCount % ENCODE_BUFFER_POOL_SIZE; inputFrameCount++;
    ERROR_CHECK_STATUS(vxSetImageAttribute(input_image, VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &mem[bufId], sizeof(cl_mem)));

    return VX_SUCCESS;
}
#endif

vx_status CLoomIoMediaEncoder::ProcessFrame(vx_image input_image, vx_array input_aux, vx_array output_aux)
{
#if ENCODE_ENABLE_OPENCL
    if (threadTerminated)
#else
    // wait until there is an ACK from encoder thread
    int ack = PopAck();
    if ((ack < 0) || threadTerminated)
#endif
    { // nothing to process, so abandon the graph execution
        return VX_ERROR_GRAPH_ABANDONED;
    }

#if ENCODE_ENABLE_OPENCL
    // just submit OpenCL buffer for encoding
    PushCommand(cmd_encode);
#else
    // format convert input image into encode buffer
    int bufId = inputFrameCount % ENCODE_BUFFER_POOL_SIZE; inputFrameCount++;
    vx_rectangle_t rect = { 0, 0, width, height };
    vx_map_id map_id; vx_imagepatch_addressing_t addr;
    uint8_t * ptr = nullptr;
    ERROR_CHECK_STATUS(vxMapImagePatch(input_image, &rect, 0, &map_id, &addr, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
    int status = sws_scale(conversionContext, &ptr, &addr.stride_y, 0, height, videoFrame[bufId]->data, videoFrame[bufId]->linesize);
    ERROR_CHECK_STATUS(vxUnmapImagePatch(input_image, map_id));
    if (status < 0) {
        vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::ProcessFrame: sws_scale() failed (%d)\n", status);
        return VX_FAILURE;
    }
    // submit encoding
    PushCommand(cmd_encode);
#endif

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
    AVPacket pkt = { 0 };
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;
    int64_t pts = 0;
    for (command cmd; !threadTerminated && ((cmd = PopCommand()) != cmd_abort);) {
        int status;
        // get the bufId to process
        int bufId = (encodeFrameCount % ENCODE_BUFFER_POOL_SIZE);
        int got_output = 0;
#if ENCODE_ENABLE_OPENCL
        // format convert input image into encode buffer
        cl_int err = -1;
        uint8_t * ptr = (uint8_t *)clEnqueueMapBuffer(cmdq, mem[bufId], CL_TRUE, CL_MAP_READ, offset, height * stride, 0, nullptr, nullptr, &err);
        if (err < 0 || !ptr) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: clEnqueueMapBuffer() failed (%d)\n", err);
            threadTerminated = true;
            PushAck(-1);
            return;
        }
        clFinish(cmdq);
        status = sws_scale(conversionContext, &ptr, &stride, 0, height, videoFrame[0]->data, videoFrame[0]->linesize);
        err = clEnqueueUnmapMemObject(cmdq, mem[bufId], ptr, 0, nullptr, nullptr);
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
        // reset bufId to zero because only videoFrame[0] is valid
        bufId = 0;
#endif
        // encode video frame and write output to file
        videoFrame[bufId]->pts = pts;
        status = avcodec_encode_video2(videoCodecContext, &pkt, videoFrame[bufId], &got_output);
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: avcodec_encode_video2() failed (%4.4s:0x%08x:%d) for frame:%d\n", &status, status, status, encodeFrameCount);
            threadTerminated = true;
            PushAck(-1);
            return;
        }
        if (got_output && (pkt.size > 0)) {
            if (fpOutput) {
                fwrite(pkt.data, 1, pkt.size, fpOutput);
            }
            if (formatContext) {
                pkt.stream_index = videoStream->index;
                status = av_interleaved_write_frame(formatContext, &pkt);
                if (status < 0) {
                    vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: CLoomIoMediaEncoder::EncodeLoop: av_interleaved_write_frame() failed (%4.4s:0x%08x:%d) for frame:%d\n", &status, status, status, encodeFrameCount);
                    threadTerminated = true;
                    PushAck(-1);
                    return;
                }
            }
        }
        av_free_packet(&pkt);
        // update encode frame count and send ACK
        encodeFrameCount++;
        if (videoStream) {
            pts += av_rescale_q(1, videoStream->codec->time_base, videoStream->time_base);
        }
        else {
            pts += (int64_t)(60000.0f / fps);
        }
        PushAck(0);
    }
    // process the delayed frames
    for (int got_output = !0; got_output;) {
        int status = avcodec_encode_video2(videoCodecContext, &pkt, nullptr, &got_output);
        if (status < 0) {
            vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: avcodec_encode_video2() failed (%4.4s:%d) at the end\n", &status, status);
            threadTerminated = true;
            PushAck(-1);
            return;
        }
        if (got_output && fpOutput && (pkt.size > 0)) {
            fwrite(pkt.data, 1, pkt.size, fpOutput);
        }
        av_free_packet(&pkt);
    }
    // mark termination and send ACK
    threadTerminated = true;
    PushAck(-1);
}

#if ENCODE_ENABLE_OPENCL
//! \brief The kernel execution.
static vx_status VX_CALLBACK loomio_media_encode_opencl_buffer_update_callback(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // get encoder and input image
    CLoomIoMediaEncoder * encoder = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &encoder, sizeof(encoder)));
    if (!encoder) return VX_FAILURE;

    return encoder->UpdateBufferOpenCL((vx_image)parameters[1], (vx_array)parameters[2]);
    return VX_SUCCESS;
}
#endif

//! \brief The kernel execution.
static vx_status VX_CALLBACK loomio_media_encode_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get encoder and input image
    CLoomIoMediaEncoder * encoder = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &encoder, sizeof(encoder)));
    if (!encoder) return VX_FAILURE;

    return encoder->ProcessFrame((vx_image)parameters[1], (vx_array)parameters[2], (vx_array)parameters[3]);
}

//! \brief The kernel initializer.
static vx_status VX_CALLBACK loomio_media_encode_initialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
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
    // create and initialize encoder
    CLoomIoMediaEncoder * encoder = new CLoomIoMediaEncoder(node, ioConfig, width, height, format, stride, offset, input_aux_data_max_size);
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &encoder, sizeof(encoder)));
    ERROR_CHECK_STATUS(encoder->Initialize());

    return VX_SUCCESS;
}

//! \brief The kernel deinitializer.
static vx_status VX_CALLBACK loomio_media_encode_deinitialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
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
static vx_status VX_CALLBACK loomio_media_encode_validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
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
    if (format != VX_DF_IMAGE_UYVY && format != VX_DF_IMAGE_YUYV && format != VX_DF_IMAGE_RGB)
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
    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status loomio_media_encode_publish(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.loomio_media.encode", AMDOVX_KERNEL_LOOMIO_MEDIA_ENCODE, 
                            loomio_media_encode_kernel, 4, loomio_media_encode_validate, 
                            loomio_media_encode_initialize, loomio_media_encode_deinitialize);
    ERROR_CHECK_OBJECT(kernel);

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));  // media config+filename: "[{mbps,fps,bframes,gopsize},]filename.mp4"
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));   // input image
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));   // input auxiliary data (optional)
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));  // output auxiliary data

#if ENCODE_ENABLE_OPENCL
    // register amd_kernel_gpu_buffer_update_callback_f for input image
    AgoKernelGpuBufferUpdateInfo info = { loomio_media_encode_opencl_buffer_update_callback, 1 };
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_UPDATE_CALLBACK, &info, sizeof(info)));
#endif

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}
