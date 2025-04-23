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
#include <vector>
#include <stdlib.h>

// OpenCL configuration
#define DECODE_ENABLE_OPENCL       1         // enable use of OpenCL buffers

#if DECODE_ENABLE_OPENCL
#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#define DECODE_BUFFER_POOL_SIZE    2  // number of buffers in decoder queue: keep it atleast 2

typedef struct {
    vx_uint32 size;
    vx_uint32 type;
} AuxDataContainerHeader;

typedef struct {
    AuxDataContainerHeader h0;
    int outputFrameCount;
    int reserved[3];
    int64_t cpuTimestamp;
} LoomIoMediaDecoderAuxInfo;

class CLoomIoMediaDecoder {
public:
    CLoomIoMediaDecoder(vx_node node, vx_uint32 mediaCount, const char inputMediaFiles[], vx_uint32 width, vx_uint32 height, vx_df_image format, vx_uint32 stride, vx_uint32 offset);
    ~CLoomIoMediaDecoder();
    vx_status Initialize();
    vx_status ProcessFrame(vx_image output, vx_array aux_data);
protected:
    typedef enum { cmd_abort, cmd_decode } command;
    void DecodeLoop(int mediaIndex);
    void PushCommand(int mediaIndex, command cmd);
    void PushAck(int mediaIndex, int ack);
    command PopCommand(int mediaIndex);
    int PopAck(int mediaIndex);
private:
    vx_node node;
    int mediaCount;
    std::string inputMediaFiles;
    int width;
    int height;
    vx_df_image format;
    int decoderImageHeight;
    int stride;
    int offset;
    AVPixelFormat outputFormat;
    vx_uint8 * decodeBuffer[DECODE_BUFFER_POOL_SIZE];
#if DECODE_ENABLE_OPENCL
    cl_mem mem[DECODE_BUFFER_POOL_SIZE];
    cl_command_queue cmdq;
#endif
    std::vector<std::string> inputMediaFileName;
    std::vector<AVFormatContext *> inputMediaFormatContext;
    std::vector<AVInputFormat *> inputMediaFormat;
    std::vector<AVCodecContext *> videoCodecContext;
    std::vector<AVCodec *> videoCodec;
    std::vector<SwsContext *> conversionContext;
    std::vector<AVFrame *> videoFrame;
    std::vector<int> videoStreamIndex;
    std::vector<std::mutex> mutexCmd, mutexAck;
    std::vector<std::condition_variable> cvCmd, cvAck;
    std::vector<std::deque<command>> queueCmd;
    std::vector<std::deque<int>> queueAck;
    std::vector<std::thread *> thread;
    std::vector<bool> eof;
    std::vector<int> decodeFrameCount;
    int outputFrameCount;
};

void CLoomIoMediaDecoder::PushCommand(int mediaIndex, CLoomIoMediaDecoder::command cmd)
{
    std::unique_lock<std::mutex> lock(mutexCmd[mediaIndex]);
    queueCmd[mediaIndex].push_front(cmd);
    cvCmd[mediaIndex].notify_one();
}

CLoomIoMediaDecoder::command CLoomIoMediaDecoder::PopCommand(int mediaIndex)
{
    std::unique_lock<std::mutex> lock(mutexCmd[mediaIndex]);
    cvCmd[mediaIndex].wait(lock, [=] { return !queueCmd[mediaIndex].empty(); });
    command cmd = std::move(queueCmd[mediaIndex].back());
    queueCmd[mediaIndex].pop_back();
    return cmd;
}

void CLoomIoMediaDecoder::PushAck(int mediaIndex, int ack)
{
    std::unique_lock<std::mutex> lock(mutexAck[mediaIndex]);
    queueAck[mediaIndex].push_front(ack);
    cvAck[mediaIndex].notify_one();
}

int CLoomIoMediaDecoder::PopAck(int mediaIndex)
{
    std::unique_lock<std::mutex> lock(mutexAck[mediaIndex]);
    cvAck[mediaIndex].wait(lock, [=] { return !queueAck[mediaIndex].empty(); });
    int ack = std::move(queueAck[mediaIndex].back());
    queueAck[mediaIndex].pop_back();
    return ack;
}

CLoomIoMediaDecoder::CLoomIoMediaDecoder(vx_node node_, vx_uint32 mediaCount_, const char inputMediaFiles_[], vx_uint32 width_, vx_uint32 height_, vx_df_image format_, vx_uint32 stride_, vx_uint32 offset_)
    : node{ node_ }, inputMediaFiles(inputMediaFiles_), mediaCount{ static_cast<int>(mediaCount_) }, width{ static_cast<int>(width_) },
      height{ static_cast<int>(height_) }, format{ format_ }, stride{ static_cast<int>(stride_) }, offset{ static_cast<int>(offset_) },
      decoderImageHeight{ static_cast<int>(height_ / ((mediaCount_ < 1) ? 1 : mediaCount_)) }, outputFormat{ AV_PIX_FMT_UYVY422 }, outputFrameCount{ 0 },
      inputMediaFileName(mediaCount_), inputMediaFormatContext(mediaCount_), inputMediaFormat(mediaCount_), videoCodecContext(mediaCount_),
      videoCodec(mediaCount_), conversionContext(mediaCount_), videoFrame(mediaCount_), videoStreamIndex(mediaCount_),
      mutexCmd(mediaCount_), cvCmd(mediaCount_), queueCmd(mediaCount_), mutexAck(mediaCount_), cvAck(mediaCount_), queueAck(mediaCount_),
      thread(mediaCount_), eof(mediaCount_), decodeFrameCount(mediaCount_)
{
#if DECODE_ENABLE_OPENCL
    memset(mem, 0, sizeof(mem));
    cmdq = nullptr;
#endif
    memset(decodeBuffer, 0, sizeof(decodeBuffer));
    // initialize freq inside GetTimeInMicroseconds()
    GetTimeInMicroseconds();
}

CLoomIoMediaDecoder::~CLoomIoMediaDecoder()
{
    // terminate the thread
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        if (thread[mediaIndex]) {
            PushCommand(mediaIndex, cmd_abort);
            while (!eof[mediaIndex]) {
                if (PopAck(mediaIndex) < 0)
                    break;
            }
            thread[mediaIndex]->join();
            delete thread[mediaIndex];
        }
    }

    // release buffers
#if DECODE_ENABLE_OPENCL
    if (cmdq) clReleaseCommandQueue(cmdq);
#endif
    for (int i = 0; i < DECODE_BUFFER_POOL_SIZE; i++) {
#if DECODE_ENABLE_OPENCL
        if (mem[i]) clReleaseMemObject(mem[i]);
#endif
        if (decodeBuffer[i]) aligned_free(decodeBuffer[i]);
    }

    // release media resources
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        if (videoFrame[mediaIndex]) av_frame_free(&videoFrame[mediaIndex]);
        if (conversionContext[mediaIndex]) av_free(conversionContext[mediaIndex]);
        if (videoCodec[mediaIndex]) av_free(videoCodec[mediaIndex]);
        if (inputMediaFormat[mediaIndex]) av_free(inputMediaFormat[mediaIndex]);
        if (videoCodecContext[mediaIndex]) av_free(videoCodecContext[mediaIndex]);
        if (inputMediaFormatContext[mediaIndex]) av_free(inputMediaFormatContext[mediaIndex]);
    }
}

vx_status CLoomIoMediaDecoder::Initialize()
{
    // check for valid image type support and get stride in bytes (aligned to 16-byte boundary)
    if (format == VX_DF_IMAGE_UYVY) {
        outputFormat = AV_PIX_FMT_UYVY422;
    }
    else if (format == VX_DF_IMAGE_YUYV) {
        outputFormat = AV_PIX_FMT_YUYV422;
    }
    else if (format == VX_DF_IMAGE_RGB) {
        outputFormat = AV_PIX_FMT_RGB24;
    }
    else {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_FORMAT, "ERROR: output image format %4.4s not supported", &format);
        return VX_ERROR_INVALID_FORMAT;
    }
    // check for validity of media count
    if (mediaCount < 1 || (height % mediaCount) != 0) {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: invalid mediaCount (%d) value", mediaCount);
        return VX_ERROR_INVALID_VALUE;
    }

    // get media count and filenames
    if (!inputMediaFiles.compare(inputMediaFiles.size() - 4, 4, ".txt")) {
        // read media filenames from text file
        FILE * fp = fopen(inputMediaFiles.c_str(), "r");
        if (!fp) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: output image format %4.4s not supported", &format);
            return VX_ERROR_INVALID_LINK;
        }
        for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
            char line[4096];
            if (!fgets(line, sizeof(line) - 1, fp)) {
                vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_LINK, "ERROR: not enough media file entries available in %s", inputMediaFiles.c_str());
                return VX_ERROR_INVALID_LINK;
            }
            // remove white space at the ends and add it to inputMediaFileName
            int i = 0;
            while (line[i] && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '\n'))
                i++;
            int j = 0;
            while (line[i] && line[i] != '\r' && line[i] != '\n')
                line[j++] = line[i++];
            line[j] = '\0';
            inputMediaFileName[mediaIndex] = line;
        }
        fclose(fp);
    }
    else if (inputMediaFiles.c_str()[0] == '{') {
        // generate media filenames
        const char * s = inputMediaFiles.c_str() + 1;
        for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
            char line[4096]; int pos = 0;
            for (; *s && *s != ',' && *s != '}'; s++, pos++)
                line[pos] = *s;
            if (*s) s++;
            line[pos] = '\0';
            inputMediaFileName[mediaIndex] = line;
        }
    }
    else {
        // generate media filenames
        for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
            char line[4096]; snprintf(line, sizeof(line), inputMediaFiles.c_str(), mediaIndex);
            inputMediaFileName[mediaIndex] = line;
        }
    }

    // open media file and initialize codec
    ERROR_CHECK_STATUS(initialize_ffmpeg());
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        const char * mediaFileName = inputMediaFileName[mediaIndex].c_str();
        AVFormatContext * formatContext = nullptr;
        AVInputFormat * inputFormat = nullptr;
        int err = avformat_open_input(&formatContext, mediaFileName, inputFormat, nullptr);
        if (err) {
            vx_status status = VX_FAILURE;
            vxAddLogEntry((vx_reference)node, status, "ERROR: avformat_open_input(%s) failed (%d)\n", mediaFileName, err);
            return status;
        }
        inputMediaFormatContext[mediaIndex] = formatContext;
        inputMediaFormat[mediaIndex] = inputFormat;
        err = avformat_find_stream_info(formatContext, nullptr);
        if (err) {
            vx_status status = VX_FAILURE;
            vxAddLogEntry((vx_reference)node, status, "ERROR: avformat_find_stream_info() for %s failed (%d)\n", mediaFileName, err);
            return status;
        }
        AVCodecContext * codecContext = nullptr;
        unsigned int streamIndex = -1;
        for (unsigned int si = 0; si < formatContext->nb_streams; si++) {
            AVCodecContext * vcc = formatContext->streams[si]->codec;
            if (vcc->codec_type == AVMEDIA_TYPE_VIDEO) {
                // pick video stream index with larger dimensions
                if (!codecContext) {
                    codecContext = vcc;
                    streamIndex = si;
                }
                else if ((vcc->width > codecContext->width) && (vcc->height > codecContext->height)) {
                    codecContext = vcc;
                    streamIndex = si;
                }
            }
        }
        if (!codecContext) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: no video found in %s", mediaFileName);
            return VX_ERROR_INVALID_VALUE;
        }
        videoCodecContext[mediaIndex] = codecContext;
        AVCodec * codec = avcodec_find_decoder(codecContext->codec_id);
        if (!codec) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: video codec not supported for %s", mediaFileName);
            return VX_ERROR_INVALID_VALUE;
        }
        videoCodec[mediaIndex] = codec;
        ERROR_CHECK_STATUS(avcodec_open2(codecContext, codec, nullptr));
        if ((codecContext->width != width) || (codecContext->height != decoderImageHeight)) {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_DIMENSION, "ERROR: output image %dx%d in %s has invalid dimensions (%dx%d expected)", codecContext->width, codecContext->height, mediaFileName, width, decoderImageHeight);
            return VX_ERROR_INVALID_DIMENSION;
        }
        SwsContext * swsContext = sws_getContext(width, decoderImageHeight, codecContext->pix_fmt, width, decoderImageHeight, outputFormat, SWS_BICUBIC, NULL, NULL, NULL);
        ERROR_CHECK_NULLPTR(swsContext);
        conversionContext[mediaIndex] = swsContext;
        AVFrame * frame = av_frame_alloc();
        ERROR_CHECK_NULLPTR(frame);
        videoFrame[mediaIndex] = frame;
        // debug log
        vxAddLogEntry((vx_reference)node, VX_SUCCESS, "INFO: reading %dx%d into slice#%d from %s", width, decoderImageHeight, mediaIndex, mediaFileName);
    }

#if DECODE_ENABLE_OPENCL
    // allocate OpenCL decode buffers
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
    for (int i = 0; i < DECODE_BUFFER_POOL_SIZE; i++) {
        mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, offset + stride * height, nullptr, nullptr);
        ERROR_CHECK_NULLPTR(mem[i]);
    }
#endif

    // allocate and align buffer
    for (int i = 0; i < DECODE_BUFFER_POOL_SIZE; i++) {
        decodeBuffer[i] = aligned_alloc(offset + stride * height);
        ERROR_CHECK_NULLPTR(decodeBuffer[i]);
    }

    // start decoder thread and wait until first frame is decoded
    outputFrameCount = 0;
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        decodeFrameCount[mediaIndex] = 0;
        eof[mediaIndex] = false;
    }
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        thread[mediaIndex] = new std::thread(&CLoomIoMediaDecoder::DecodeLoop, this, mediaIndex);
        ERROR_CHECK_NULLPTR(thread[mediaIndex]);
        // initial ACK to inform producer for readiness
        for (int i = 1; i < DECODE_BUFFER_POOL_SIZE; i++)
            PushCommand(mediaIndex, cmd_decode);
    }
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        PopAck(mediaIndex);
    }

    return VX_SUCCESS;
}

vx_status CLoomIoMediaDecoder::ProcessFrame(vx_image output, vx_array aux_data)
{
    // continue decoding another frame
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        PushCommand(mediaIndex, cmd_decode);
    }
    // wait until next frame is available
    for (int mediaIndex = 0; mediaIndex < mediaCount; mediaIndex++) {
        int ack = PopAck(mediaIndex);
        if ((ack < 0) || eof[mediaIndex]) {
            // nothing to process, so abandon the graph execution
            return VX_ERROR_GRAPH_ABANDONED;
        }
    }

    // set aux data
    if (aux_data) {
        // construct aux data
        LoomIoMediaDecoderAuxInfo haux = { 0 };
        haux.h0.size = sizeof(LoomIoMediaDecoderAuxInfo);
        haux.h0.type = AMDOVX_KERNEL_LOOMIO_MEDIA_DECODE;
        haux.outputFrameCount = outputFrameCount;
        haux.cpuTimestamp = GetTimeInMicroseconds();
        // set aux data
        ERROR_CHECK_STATUS(vxTruncateArray(aux_data, 0));
        ERROR_CHECK_STATUS(vxAddArrayItems(aux_data, sizeof(haux), &haux, sizeof(uint8_t)));
    }

    // set the output buffer
    int bufId = outputFrameCount % DECODE_BUFFER_POOL_SIZE; outputFrameCount++;
#if DECODE_ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxSetImageAttribute(output, VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &mem[bufId], sizeof(cl_mem)));
#else
    vx_rectangle_t rect = { 0, 0, width, height };
    vx_imagepatch_addressing_t addr = { 0 };
    addr.stride_x = stride / width;
    addr.stride_y = stride;
    ERROR_CHECK_STATUS(vxCopyImagePatch(output, &rect, 0, &addr, &decodeBuffer[bufId][offset], VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#endif

    return VX_SUCCESS;
}

void CLoomIoMediaDecoder::DecodeLoop(int mediaIndex)
{
    // decode loop
    AVPacket avpkt = { 0 };
    av_init_packet(&avpkt);
    avpkt.data = nullptr;
    avpkt.size = 0;
    for (command cmd; !eof[mediaIndex] && ((cmd = PopCommand(mediaIndex)) != cmd_abort);) {
        int gotPicture = 0;
        while (!gotPicture && !eof[mediaIndex]) {
            for (;;) {
                int status = av_read_frame(inputMediaFormatContext[mediaIndex], &avpkt);
                if (status < 0) {
                    eof[mediaIndex] = true;
                    break;
                }
                if (avpkt.stream_index == videoStreamIndex[mediaIndex])
                    break;
            }
            int status = avcodec_decode_video2(videoCodecContext[mediaIndex], videoFrame[mediaIndex], &gotPicture, &avpkt);
            if (status < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: avcodec_decode_video2() failed (%d)\n", status);
                eof[mediaIndex] = true;
                PushAck(mediaIndex, -1);
                return;
            }
        }
        if (gotPicture) {
            // pick decode frame and perform format conversion for the media slice
            int bufId = (decodeFrameCount[mediaIndex] % DECODE_BUFFER_POOL_SIZE);
            vx_uint8 * decodedSlice = &decodeBuffer[bufId][offset + mediaIndex * decoderImageHeight * stride];
            int status = sws_scale(conversionContext[mediaIndex], videoFrame[mediaIndex]->data, videoFrame[mediaIndex]->linesize, 0, decoderImageHeight, &decodedSlice, &stride);
            if (status < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: sws_scale() failed (%d)\n", status);
                eof[mediaIndex] = true;
                PushAck(mediaIndex, -1);
                return;
            }
#if DECODE_ENABLE_OPENCL
            // copy the buffer slice to OpenCL
            cl_int err = clEnqueueWriteBuffer(cmdq, mem[bufId], CL_TRUE, offset + mediaIndex * decoderImageHeight * stride, decoderImageHeight * stride, decodedSlice, 0, nullptr, nullptr);
            if (err < 0) {
                vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR: clEnqueueWriteBuffer(buf[%d], slice[%d]) failed (%d)\n", bufId, mediaIndex, err);
                eof[mediaIndex] = true;
                PushAck(mediaIndex, -1);
                return;
            }
            clFinish(cmdq);
#endif
            // update decoded frame count and send ACK
            decodeFrameCount[mediaIndex]++;
            PushAck(mediaIndex, 0);
        }
    }
    // mark eof and send ACK
    eof[mediaIndex] = true;
    PushAck(mediaIndex, -1);
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK loomio_media_decode_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get decoder and output image
    CLoomIoMediaDecoder * decoder = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &decoder, sizeof(decoder)));
    if (!decoder) return VX_FAILURE;

    return decoder->ProcessFrame((vx_image)parameters[1], (vx_array)parameters[2]);
}

//! \brief The kernel initializer.
static vx_status VX_CALLBACK loomio_media_decode_initialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get input parameters
    char inputMediaConfig[VX_MAX_STRING_BUFFER_SIZE_AMD];
    vx_uint32 width = 0, height = 0, stride = 0, offset = 0;
    vx_df_image format = VX_DF_IMAGE_VIRT;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[0], inputMediaConfig, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE, &stride, sizeof(stride)));
#if DECODE_ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET, &offset, sizeof(offset)));
#endif

    // create and initialize decoder
    const char * s = inputMediaConfig;
    vx_uint32 mediaCount = atoi(s);
    while (*s && *s != ',') s++;
    if (mediaCount < 1 || *s != ',') {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ERROR: invalid ioConfig: %s\nERROR: invalid ioConfig: valid syntax: <mediaCount>,mediaList.txt|media%%d.mp4|{file1.mp4,file2.mp4,...}\n", inputMediaConfig);
        return VX_ERROR_INVALID_VALUE;
    }
    if (*s == ',') s++;
    CLoomIoMediaDecoder * decoder = new CLoomIoMediaDecoder(node, mediaCount, s, width, height, format, stride, offset);
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &decoder, sizeof(decoder)));
    ERROR_CHECK_STATUS(decoder->Initialize());

    return VX_SUCCESS;
}

//! \brief The kernel deinitializer.
static vx_status VX_CALLBACK loomio_media_decode_deinitialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get decoder
    CLoomIoMediaDecoder * decoder = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &decoder, sizeof(decoder)));

    if (decoder) {
        // release the resources
        delete decoder;
    }

    return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK loomio_media_decode_validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // make sure input scalar contains num cameras and media file name
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[0], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_STRING_AMD)
        return VX_ERROR_INVALID_FORMAT;
    // make sure output format is UYVY/YUYV/RGB
    vx_uint32 width = 0, height = 0;
    vx_df_image format = VX_DF_IMAGE_VIRT;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if (format != VX_DF_IMAGE_UYVY && format != VX_DF_IMAGE_YUYV && format != VX_DF_IMAGE_RGB)
        return VX_ERROR_INVALID_FORMAT;
    // set output image meta
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
#if DECODE_ENABLE_OPENCL
    vx_bool enableUserBufferGPU = vx_true_e;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU, &enableUserBufferGPU, sizeof(enableUserBufferGPU)));
#endif

    // check aux data parameter
    if (parameters[2]) {
        // make sure data type is UINT8
        vx_enum itemtype = VX_TYPE_INVALID;
        vx_size capacity = 0;
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
        ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
        if (itemtype != VX_TYPE_UINT8)
            return VX_ERROR_INVALID_TYPE;
        // set meta for aux data
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE, &itemtype, sizeof(itemtype)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
    }

    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status loomio_media_decode_publish(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.loomio_media.decode", AMDOVX_KERNEL_LOOMIO_MEDIA_DECODE, 
                            loomio_media_decode_kernel, 3, loomio_media_decode_validate, 
                            loomio_media_decode_initialize, loomio_media_decode_deinitialize);
    ERROR_CHECK_OBJECT(kernel);

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // media config+filename: mediaCount,mediaList.txt|media%d.mp4|{file1.mp4,file2.mp4,...}
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED)); // output image
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL)); // output auxiliary data (optional)

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}
