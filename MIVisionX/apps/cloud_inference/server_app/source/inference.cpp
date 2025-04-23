/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "inference.h"
#include "netutil.h"
#include "common.h"
#include <thread>
#include <chrono>
#include <dlfcn.h>
#include <opencv2/opencv.hpp>
#if USE_OPENCV_4
#include <opencv2/highgui.hpp>
#else
#include <highgui.h>
#endif
#include <numeric>

#if USE_SSE_OPTIMIZATION
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#endif

void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}


InferenceEngine::InferenceEngine(int sock_, Arguments * args_, const std::string clientName_, InfComCommand * cmd)
    : sock{ sock_ }, args{ args_ }, clientName{ clientName_ },
      GPUs{ cmd->data[1] },
      dimInput{ cmd->data[2], cmd->data[3], cmd->data[4] },
      dimOutput{ cmd->data[5], cmd->data[6], cmd->data[7] },
      receiveFileNames { (bool)cmd->data[8] }, topK { cmd->data[9] }, 
      detectBoundingBoxes { cmd->data[10] }, decodeMode { cmd->data[11] }, loop { (bool)cmd->data[12] },
      reverseInputChannelOrder{ 0 }, preprocessMpy{ 1, 1, 1 }, preprocessAdd{ 0, 0, 0 },
      moduleHandle{ nullptr }, annCreateGraph{ nullptr }, annAddtoGraph { nullptr},
      deviceLockSuccess{ false }, useShadowFilenames{ false }
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER && !DONOT_RUN_INFERENCE
    , openvx_context{ nullptr }, openvx_graph{ nullptr }, openvx_input{ nullptr }, openvx_output{ nullptr }
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    , threadMasterInputQ{ nullptr },
#if ENABLE_OPENCL
      device_id{ nullptr }, opencl_context{ nullptr }, opencl_cmdq{ nullptr },
      queueDeviceInputMemIdle{ nullptr }, queueDeviceInputMemBusy{ nullptr },
      queueDeviceOutputMemIdle{ nullptr }, queueDeviceOutputMemBusy{ nullptr },
#endif      
      openvx_context{ nullptr }, openvx_graph{ nullptr }, openvx_input{ nullptr }, openvx_output{ nullptr },
      threadDeviceInputCopy{ nullptr }, threadDeviceProcess{ nullptr }, threadDeviceOutputCopy{ nullptr },
      queueDeviceTagQ{ nullptr }, queueDeviceImageQ{ nullptr },
      region { nullptr }, useFp16 { 0 }
#if  USE_ADVANCED_MESSAGE_Q
    , inputQ(MAX_INPUT_QUEUE_DEPTH)
#endif
#endif
{
    // extract model name, options, and module path
    char modelName_[128] = { 0 }, options_[128] = { 0 };
    sscanf(cmd->message, "%s%s", modelName_, options_);
    modelName = modelName_;
    options = options_;
    // configuration
    batchSize = args->getBatchSize();
    if (!args->fp16Inference()) {
        inputSizeInBytes = 4 * dimInput[0] * dimInput[1] * dimInput[2] * batchSize;
        outputSizeInBytes = 4 * dimOutput[0] * dimOutput[1] * dimOutput[2] * batchSize;
    }else
    {
        useFp16 = 1;
        inputSizeInBytes = 2 * dimInput[0] * dimInput[1] * dimInput[2] * batchSize;
        outputSizeInBytes = 2 * dimOutput[0] * dimOutput[1] * dimOutput[2] * batchSize;
        std::cout << "INFO::inferenceserver is running with FP16 inference" << std::endl;
    }
    numDecThreads = args->decThreads();
    if (numDecThreads){
        numDecThreads = (numDecThreads + 1) & ~1;    // make it multiple of 2
        numDecThreads = std::min(numDecThreads, batchSize); // can't be more than batch_size
    }

    if (detectBoundingBoxes)
        region = new CYoloRegion();
    // lock devices
#if ENABLE_OPENCL
    if(!args->lockGpuDevices(GPUs, device_id))
        deviceLockSuccess = true;
#endif
    if (!args->getlocalShadowRootDir().empty() || decodeMode == 1){
        useShadowFilenames = true;
        std::cout << "INFO::inferenceserver is running with LocalShadowFolder and infcom command receiving only filenames" << std::endl;
    }

    PROFILER_INITIALIZE();
}

InferenceEngine::~InferenceEngine()
{
#if ENABLE_OPENCL  
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER && !DONOT_RUN_INFERENCE
    if(openvx_graph) {
        vxReleaseGraph(&openvx_graph);
    }
    if(openvx_input) {
        vxReleaseTensor(&openvx_input);
    }
    if(openvx_output) {
        vxReleaseTensor(&openvx_output);
    }
    if(openvx_context) {
        vxReleaseContext(&openvx_context);
    }
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    // wait for all threads to complete and release all resources
    std::tuple<int,char*,int> endOfSequenceInput(-1,nullptr,0);
    inputQ.enqueue(endOfSequenceInput);
    if(threadMasterInputQ && threadMasterInputQ->joinable()) {
        threadMasterInputQ->join();
    }
    std::tuple<char*,int> endOfSequenceImage(nullptr,0);
    int endOfSequenceTag = -1;
    for(int i = 0; i < GPUs; i++) {
        if(queueDeviceTagQ[i]) {
            queueDeviceTagQ[i]->enqueue(endOfSequenceTag);
        }
        if(queueDeviceImageQ[i]) {
            queueDeviceImageQ[i]->enqueue(endOfSequenceImage);
        }
        if(threadDeviceInputCopy[i] && threadDeviceInputCopy[i]->joinable()) {
            threadDeviceInputCopy[i]->join();
        }
        if(threadDeviceProcess[i] && threadDeviceProcess[i]->joinable()) {
            threadDeviceProcess[i]->join();
        }
        if(threadDeviceOutputCopy[i] && threadDeviceOutputCopy[i]->joinable()) {
            threadDeviceOutputCopy[i]->join();
        }
        while(queueDeviceInputMemIdle[i] && queueDeviceInputMemIdle[i]->size() > 0) {
            cl_mem mem;
            queueDeviceInputMemIdle[i]->dequeue(mem);
            clReleaseMemObject(mem);
        }
        while(queueDeviceOutputMemIdle[i] && queueDeviceOutputMemIdle[i]->size() > 0) {
            cl_mem mem;
            queueDeviceOutputMemIdle[i]->dequeue(mem);
            clReleaseMemObject(mem);
        }
        if(queueDeviceTagQ[i]) {
            delete queueDeviceTagQ[i];
        }
        if(queueDeviceImageQ[i]) {
            delete queueDeviceImageQ[i];
        }
        if(queueDeviceInputMemIdle[i]) {
            delete queueDeviceInputMemIdle[i];
        }
        if(queueDeviceInputMemBusy[i]) {
            delete queueDeviceInputMemBusy[i];
        }
        if(queueDeviceOutputMemIdle[i]) {
            delete queueDeviceOutputMemIdle[i];
        }
        if(queueDeviceOutputMemBusy[i]) {
            delete queueDeviceOutputMemBusy[i];
        }
        if(openvx_graph[i]) {
            vxReleaseGraph(&openvx_graph[i]);
        }
        if(openvx_input[i]) {
            vxReleaseTensor(&openvx_input[i]);
        }
        if(openvx_output[i]) {
            vxReleaseTensor(&openvx_output[i]);
        }
        if(openvx_context[i]) {
            vxReleaseContext(&openvx_context[i]);
        }
        if(opencl_cmdq[i]) {
            clReleaseCommandQueue(opencl_cmdq[i]);
        }
        if(opencl_context[i]) {
            clReleaseContext(opencl_context[i]);
        }
    }
#endif
    // release all device resources
    if(deviceLockSuccess) {
        args->releaseGpuDevices(GPUs, device_id);
    }
    if(moduleHandle) {
        dlclose(moduleHandle);
    }
    if (region) delete region;
    PROFILER_SHUTDOWN();
#endif    
}

vx_status InferenceEngine::DecodeScaleAndConvertToTensor(vx_size width, vx_size height, int size, unsigned char *inp, float *buf, int use_fp16)
{
    int length = width*height;
    cv::Mat matOrig = cv::imdecode(cv::Mat(1, size, CV_8UC1, inp), cv::IMREAD_COLOR);
#if USE_SSE_OPTIMIZATION
    unsigned char *data_resize = nullptr;
    unsigned char * img;
    if ((width == matOrig.cols) && (height == matOrig.rows))
    {
        // no resize required
        img = matOrig.data;
    }else
    {
        unsigned int aligned_size = ((length+width) * 3 + 128)&~127;
        data_resize = new unsigned char[aligned_size];
        RGB_resize(matOrig.data, data_resize, matOrig.cols, matOrig.rows, matOrig.step, width, height);
        img = data_resize;
    }
    PROFILER_START(inference_server_app, workRGBtoTensor);

    __m128i mask_B, mask_G, mask_R;
    if (reverseInputChannelOrder)
    {
        mask_B = _mm_setr_epi8((char)0x0, (char)0x80, (char)0x80, (char)0x80, (char)0x3, (char)0x80, (char)0x80, (char)0x80, (char)0x6, (char)0x80, (char)0x80, (char)0x80, (char)0x9, (char)0x80, (char)0x80, (char)0x80);
        mask_G = _mm_setr_epi8((char)0x1, (char)0x80, (char)0x80, (char)0x80, (char)0x4, (char)0x80, (char)0x80, (char)0x80, (char)0x7, (char)0x80, (char)0x80, (char)0x80, (char)0xA, (char)0x80, (char)0x80, (char)0x80);
        mask_R = _mm_setr_epi8((char)0x2, (char)0x80, (char)0x80, (char)0x80, (char)0x5, (char)0x80, (char)0x80, (char)0x80, (char)0x8, (char)0x80, (char)0x80, (char)0x80, (char)0xB, (char)0x80, (char)0x80, (char)0x80);
    }
    else
    {
        mask_R = _mm_setr_epi8((char)0x0, (char)0x80, (char)0x80, (char)0x80, (char)0x3, (char)0x80, (char)0x80, (char)0x80, (char)0x6, (char)0x80, (char)0x80, (char)0x80, (char)0x9, (char)0x80, (char)0x80, (char)0x80);
        mask_G = _mm_setr_epi8((char)0x1, (char)0x80, (char)0x80, (char)0x80, (char)0x4, (char)0x80, (char)0x80, (char)0x80, (char)0x7, (char)0x80, (char)0x80, (char)0x80, (char)0xA, (char)0x80, (char)0x80, (char)0x80);
        mask_B = _mm_setr_epi8((char)0x2, (char)0x80, (char)0x80, (char)0x80, (char)0x5, (char)0x80, (char)0x80, (char)0x80, (char)0x8, (char)0x80, (char)0x80, (char)0x80, (char)0xB, (char)0x80, (char)0x80, (char)0x80);
    }
    int alignedLength = (length-2)& ~3;
    bool bPreprocess = (preprocessMpy[0] != 1) & (preprocessAdd[0] != 0) ;
    if (!use_fp16) {
        float * B_buf = buf;
        float * G_buf = B_buf + length;
        float * R_buf = G_buf + length;
        int i = 0;

        __m128 fR, fG, fB;
        if (bPreprocess) {
            for (; i < alignedLength; i += 4)
            {
                __m128i pix0 = _mm_loadu_si128((__m128i *) img);
                fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
                fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
                fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
                fB = _mm_mul_ps(fB, _mm_set1_ps(preprocessMpy[0]));
                fG = _mm_mul_ps(fG, _mm_set1_ps(preprocessMpy[1]));
                fR = _mm_mul_ps(fR, _mm_set1_ps(preprocessMpy[2]));
                fB = _mm_add_ps(fB, _mm_set1_ps(preprocessAdd[0]));
                fG = _mm_add_ps(fG, _mm_set1_ps(preprocessAdd[1]));
                fR = _mm_add_ps(fR, _mm_set1_ps(preprocessAdd[2]));
                _mm_storeu_ps(B_buf, fB);
                _mm_storeu_ps(G_buf, fG);
                _mm_storeu_ps(R_buf, fR);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
            for (; i < length; i++, img += 3) {
                *B_buf++ = (img[0] * preprocessMpy[0]) + preprocessAdd[0];
                *G_buf++ = (img[1] * preprocessMpy[1]) + preprocessAdd[1];
                *R_buf++ = (img[2] * preprocessMpy[2]) + preprocessAdd[2];
            }
        }else
        {
            for (; i < alignedLength; i += 4)
            {
                __m128i pix0 = _mm_loadu_si128((__m128i *) img);
                fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
                fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
                fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
                _mm_storeu_ps(B_buf, fB);
                _mm_storeu_ps(G_buf, fG);
                _mm_storeu_ps(R_buf, fR);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
            for (; i < length; i++, img += 3) {
                *B_buf++ = img[0];
                *G_buf++ = img[1];
                *R_buf++ = img[2];
            }
        }
    } else
    {
        unsigned short * B_buf = (unsigned short *)buf;
        unsigned short * G_buf = B_buf + length;
        unsigned short * R_buf = G_buf + length;
        int i = 0;

        __m128 fR, fG, fB;
        __m128i hR, hG, hB;
        if (bPreprocess) {
            for (; i < alignedLength; i += 4)
            {
                __m128i pix0 = _mm_loadu_si128((__m128i *) img);
                fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
                fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
                fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
                fB = _mm_mul_ps(fB, _mm_set1_ps(preprocessMpy[0]));
                fG = _mm_mul_ps(fG, _mm_set1_ps(preprocessMpy[1]));
                fR = _mm_mul_ps(fR, _mm_set1_ps(preprocessMpy[2]));
                fB = _mm_add_ps(fB, _mm_set1_ps(preprocessAdd[0]));
                fG = _mm_add_ps(fG, _mm_set1_ps(preprocessAdd[1]));
                fR = _mm_add_ps(fR, _mm_set1_ps(preprocessAdd[2]));
                // convert to half
                hB = _mm_cvtps_ph(fB, 0xF);
                hG = _mm_cvtps_ph(fG, 0xF);
                hR = _mm_cvtps_ph(fR, 0xF);
                _mm_storel_epi64((__m128i*)B_buf, hB);
                _mm_storel_epi64((__m128i*)G_buf, hG);
                _mm_storel_epi64((__m128i*)R_buf, hR);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
            for (; i < length; i++, img += 3) {
                *B_buf++ = _cvtss_sh((float)((img[0] * preprocessMpy[0]) + preprocessAdd[0]), 1);
                *G_buf++ = _cvtss_sh((float)((img[1] * preprocessMpy[1]) + preprocessAdd[1]), 1);
                *R_buf++ = _cvtss_sh((float)((img[2] * preprocessMpy[2]) + preprocessAdd[2]), 1);
            }
        } else
        {
            for (; i < alignedLength; i += 4)
            {
                __m128i pix0 = _mm_loadu_si128((__m128i *) img);
                fB = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_B));
                fG = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_G));
                fR = _mm_cvtepi32_ps(_mm_shuffle_epi8(pix0, mask_R));
                // convert to half
                hB = _mm_cvtps_ph(fB, 0xF);
                hG = _mm_cvtps_ph(fG, 0xF);
                hR = _mm_cvtps_ph(fR, 0xF);
                _mm_storel_epi64((__m128i*)B_buf, hB);
                _mm_storel_epi64((__m128i*)G_buf, hG);
                _mm_storel_epi64((__m128i*)R_buf, hR);
                B_buf += 4; G_buf += 4; R_buf += 4;
                img += 12;
            }
            for (; i < length; i++, img += 3) {
                *B_buf++ = _cvtss_sh((float)img[0], 1);
                *G_buf++ = _cvtss_sh((float)img[1], 1);
                *R_buf++ = _cvtss_sh((float)img[2], 1);
            }
        }
    }
    PROFILER_STOP(inference_server_app, workRGBtoTensor);
    if (data_resize != nullptr) delete[] data_resize;
#else
    cv::Mat matScaled;
    cv::resize(matOrig, matScaled, cv::Size(width, height));
    float *ptr = buf;
    for (int c = 0; c < 3; c++, ptr += length) {
        float a = preprocessMpy[c], b = preprocessAdd[c];
        unsigned char * img = matScaled.data + (reverseInputChannelOrder ? c : (2 - c));
        for (int i = 0; i < length; i++, img += 3) {
            ptr[i] = *img * a + b;
        }
    }
    matScaled.release();
#endif
    matOrig.release();
    return VX_SUCCESS;
}

#define FP_BITS     16
#define FP_MUL      (1<<FP_BITS)

void InferenceEngine::RGB_resize(unsigned char *Rgb_in, unsigned char *Rgb_out, unsigned int swidth, unsigned int sheight,  unsigned int sstride, unsigned int dwidth, unsigned int dheight)
{
    float xscale = (float)((double)swidth / (double)dwidth);
    float yscale = (float)((double)sheight / (double)dheight);
    int alignW = (dwidth + 15)&~15;
    unsigned int *Xmap = new unsigned int[alignW*2];
    unsigned short *Xf = (unsigned short *)(Xmap + alignW);
    unsigned short *Xf1 = Xf + alignW;

    int xpos = (int)(FP_MUL * (xscale*0.5 - 0.5));
    int xinc = (int)(FP_MUL * xscale);
    int yinc = (int)(FP_MUL * yscale);		// to convert to fixed point
    unsigned int aligned_width = dwidth;
    // generate xmap
    for (unsigned int x = 0; x < dwidth; x++, xpos += xinc)
    {
        int xf;
        int xmap = (xpos >> FP_BITS);
        if (xmap >= (int)(swidth - 8)){
            aligned_width = x;
        }
        if (xmap >= (int)(swidth - 1)){
            Xmap[x] = (swidth - 1)*3;
        }
        else
            Xmap[x] = (xmap<0)? 0: xmap*3;
        xf = ((xpos & 0xffff) + 0x80) >> 8;
        Xf[x] = xf;
        Xf1[x] = (0x100 - xf);
    }
    aligned_width &= ~3;
    int dstride = dwidth * 3;
    unsigned char *pSrcBorder = Rgb_in + (sheight*sstride) - 3;    // points to the last pixel

    int ypos = (int)(FP_MUL * (yscale*0.5 - 0.5));
    for (int y = 0; y < (int)dheight; y++, ypos += yinc)
    {
        int ym, fy, fy1;
        unsigned char *pSrc1, *pSrc2;
        unsigned char *pdst = Rgb_out + y*dstride;

        ym = (ypos >> FP_BITS);
        fy = ((ypos & 0xffff) + 0x80) >> 8;
        fy1 = (0x100 - fy);
        if (ym >= (int)(sheight - 1)){
            pSrc1 = pSrc2 = Rgb_in + (sheight - 1)*sstride;
        }
        else
        {
            pSrc1 = (ym<0)? Rgb_in : (Rgb_in + ym*sstride);
            pSrc2 = pSrc1 + sstride;
        }
        __m128i w_y = _mm_setr_epi32(fy1, fy, fy1, fy);
        const __m128i mm_zeros = _mm_setzero_si128();
        const __m128i mm_round = _mm_set1_epi32((int)0x80);
        __m128i p01, p23, ps01, ps23, pRG1, pRG2, pRG3;
        unsigned int x = 0;
        for (; x < aligned_width; x += 4)
        {
            // load 2 pixels each
            p01 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x]]);
            p23 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+1]]);
            ps01 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x]]);
            ps23 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x + 1]]);
            // unpcklo for p01 and ps01
            p01 = _mm_unpacklo_epi8(p01, ps01);
            p23 = _mm_unpacklo_epi8(p23, ps23);
            p01 = _mm_unpacklo_epi16(p01, _mm_srli_si128(p01, 6));     //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for first pixel
            p23 = _mm_unpacklo_epi16(p23, _mm_srli_si128(p23, 6));      //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for second pixel

            // load xf and 1-xf
            ps01 = _mm_setr_epi32(Xf1[x], Xf1[x], Xf[x], Xf[x]);			// xfxfxf1xf1
            ps01 = _mm_mullo_epi32(ps01, w_y);                      // W0W1W2W3 for first pixel
            ps23 = _mm_setr_epi32(Xf1[x + 1], Xf1[x + 1], Xf[x + 1], Xf[x + 1]);
            ps23 = _mm_mullo_epi32(ps23, w_y);                      // W0W1W2W3 for second pixel
            ps01 = _mm_srli_epi32(ps01, 8);                 // convert to 16bit
            ps23 = _mm_srli_epi32(ps23, 8);                 // convert to 16bit
            ps01 = _mm_packus_epi32(ps01, ps01);                 // convert to 16bit
            ps23 = _mm_packus_epi32(ps23, ps23);                 // convert to 16bit

            // extend to 16bit
            pRG1 = _mm_unpacklo_epi8(p01, mm_zeros);        // R0R1R2R3 and G0G1G2G3
            p01 = _mm_srli_si128(p01, 8);             // B0B1B2B3xxxx
            p01 = _mm_unpacklo_epi32(p01, p23);       // B0B1B2B3 R0R1R2R3: ist and second
            p23 = _mm_srli_si128(p23, 4);             // G0G1G2G3 B0B1B2B3 for second pixel
            p01 = _mm_unpacklo_epi8(p01, mm_zeros);         // B0B1B2B3 R0R1R2R3
            pRG2 = _mm_unpacklo_epi8(p23, mm_zeros);        // G0G1G2G3 B0B1B2B3 for second pixel

            pRG1 = _mm_madd_epi16(pRG1, ps01);                  // (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3)
            pRG2 = _mm_madd_epi16(pRG2, ps23);                  //(W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3) for seond pixel
            ps01 = _mm_unpacklo_epi64(ps01, ps23);
            p01 = _mm_madd_epi16(p01, ps01);                  //(W0*B0+W1*B1), (W2*B2+W3*B3), (W0*R0+W1*R1), (W2*R2+W3*R3) 1st and second pixel

            pRG1 = _mm_hadd_epi32(pRG1, p01);      // R0,G0, B0, R1 (32bit)
            p01 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+2]]);
            p23 = _mm_loadl_epi64((const __m128i*) &pSrc1[Xmap[x+3]]);
            ps01 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+2]]);
            ps23 = _mm_loadl_epi64((const __m128i*) &pSrc2[Xmap[x+3]]);
            pRG1 = _mm_add_epi32(pRG1, mm_round);
            // unpcklo for p01 and ps01
            p01 = _mm_unpacklo_epi8(p01, ps01);
            p01 = _mm_unpacklo_epi16(p01, _mm_srli_si128(p01, 6));     //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for first pixel
            p23 = _mm_unpacklo_epi8(p23, ps23);
            p23 = _mm_unpacklo_epi16(p23, _mm_srli_si128(p23, 6));      //R0R1R2R3 G0G1G2G3 B0B1B2B3 XXXX for second pixel
            // load xf and 1-xf
            ps01 = _mm_setr_epi32(Xf1[x+2], Xf1[x+2], Xf[x+2], Xf[x+2]);			// xfxfxf1xf1
            ps01 = _mm_mullo_epi32(ps01, w_y);                      // W0W1W2W3 for first pixel
            ps23 = _mm_setr_epi32(Xf1[x + 3], Xf1[x + 3], Xf[x + 3], Xf[x + 3]);
            ps23 = _mm_mullo_epi32(ps23, w_y);                      // W0W1W2W3 for second pixel
            ps01 = _mm_srli_epi32(ps01, 8);                 // convert to 16bit
            ps23 = _mm_srli_epi32(ps23, 8);                 // convert to 16bit
            ps01 = _mm_packus_epi32(ps01, ps01);                 // convert to 16bit
            ps23 = _mm_packus_epi32(ps23, ps23);                 // convert to 16bit
            // extend to 16bit
            pRG3 = _mm_unpacklo_epi8(p01, mm_zeros);        // R0R1R2R3 and G0G1G2G3
            p01 = _mm_srli_si128(p01, 8);             // B0B1B2B3xxxx
            p01 = _mm_unpacklo_epi32(p01, p23);       // B0B1B2B3 R0R1R2R3: ist and second
            p23 = _mm_srli_si128(p23, 4);             // G0G1G2G3 B0B1B2B3 for second pixel
            p01 = _mm_unpacklo_epi8(p01, mm_zeros);         // B0B1B2B3 R0R1R2R3
            p23 = _mm_unpacklo_epi8(p23, mm_zeros);        // G0G1G2G3 B0B1B2B3 for second pixel

            pRG3 = _mm_madd_epi16(pRG3, ps01);                  // (W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3)
            p23 = _mm_madd_epi16(p23, ps23);                  //(W0*R0+W1*R1), (W2*R2+W3*R3), (W0*G0+W1*G1), (W2*G2+W3*G3) for seond pixel
            ps01 = _mm_unpacklo_epi64(ps01, ps23);
            p01 = _mm_madd_epi16(p01, ps01);                  //(W0*B0+W1*B1), (W2*B2+W3*B3), (W0*B0+W1*B1), (W2*B2+W3*B3) for seond pixel

            pRG2 = _mm_hadd_epi32(pRG2, pRG3);      // G1, B1, R2,G2 (32bit)
            p01 = _mm_hadd_epi32(p01, p23);      // B2,R3, G3, B3 (32bit)
            pRG2 = _mm_add_epi32(pRG2, mm_round);
            p01 = _mm_add_epi32(p01, mm_round);
            pRG1 = _mm_srli_epi32(pRG1, 8);      // /256
            pRG2 = _mm_srli_epi32(pRG2, 8);      // /256
            p01 = _mm_srli_epi32(p01, 8);      // /256

            // convert to 16bit
            pRG1 = _mm_packus_epi32(pRG1, pRG2); //R0G0B0R1G1B1R2G2
            p01 = _mm_packus_epi32(p01, p01); //B2R3B3G3
            pRG1 = _mm_packus_epi16(pRG1, mm_zeros);
            p01 = _mm_packus_epi16(p01, mm_zeros);
            _mm_storeu_si128((__m128i *)pdst, _mm_unpacklo_epi64(pRG1, p01));
            pdst += 12;
        }

        for (; x < dwidth; x++) {
            int result;
            const unsigned char *p0 = pSrc1 + Xmap[x];
            const unsigned char *p01 = p0 + 3;
            const unsigned char *p1 = pSrc2 + Xmap[x];
            const unsigned char *p11 = p1 + 3;
            if (p0 > pSrcBorder) p0 = pSrcBorder;
            if (p1 > pSrcBorder) p1 = pSrcBorder;
            if (p01 > pSrcBorder) p01 = pSrcBorder;
            if (p11 > pSrcBorder) p11 = pSrcBorder;
            result = ((Xf1[x] * fy1*p0[0]) + (Xf[x] * fy1*p01[0]) + (Xf1[x] * fy*p1[0]) + (Xf[x] * fy*p11[0]) + 0x8000) >> 16;
            *pdst++ = (unsigned char) std::max(0, std::min(result, 255));
            result = ((Xf1[x] * fy1*p0[1]) + (Xf[x] * fy1*p01[1]) + (Xf1[x] * fy*p1[1]) + (Xf[x] * fy*p11[1]) + 0x8000) >> 16;
            *pdst++ = (unsigned char)std::max(0, std::min(result, 255));
            result = ((Xf1[x] * fy1*p0[2]) + (Xf[x] * fy1*p01[2]) + (Xf1[x] * fy*p1[2]) + (Xf[x] * fy*p11[2]) + 0x8000) >> 16;
            *pdst++ = (unsigned char)std::max(0, std::min(result, 255));
        }
    }
    if (Xmap) delete[] Xmap;
}


void InferenceEngine::DecodeScaleAndConvertToTensorBatch(std::vector<std::tuple<char*, int>>& batch_Q, int start, int end, int dim[3], float *tens_buf)
{
    for (int i = start; i <= end; i++)
    {
        std::tuple<char*, int> image = batch_Q[i];
        char * byteStream = std::get<0>(image);
        int size = std::get<1>(image);
        if (byteStream == nullptr || size == 0) {
            break;
        }
        // decode, scale, and format convert into the OpenCL buffer
        float *buf;
        if (useFp16)
            buf = (float *) ((unsigned short *)tens_buf + dim[0] * dim[1] * dim[2] * i);
        else
            buf = (float *) tens_buf + dim[0] * dim[1] * dim[2] * i;
        DecodeScaleAndConvertToTensor(dim[0], dim[1], size, (unsigned char *)byteStream, buf, useFp16);
        delete[] byteStream;
    }
}


int InferenceEngine::run()
{
#if ENABLE_OPENCL  
    //////
    /// make device lock is successful
    ///
    if(!deviceLockSuccess) {
        return error_close(sock, "could not lock %d GPUs devices for inference request from %s", GPUs, clientName.c_str());
    }

    //////
    /// check if server and client are in the same mode for data
    ///
    if (receiveFileNames && !useShadowFilenames)
    {
        return error_close(sock, "client is sending filenames but server is not configured with shadow folder\n");
    }

    //////
    /// check if client is requesting topK which is not supported
    ///
    if (topK > 5)
    {
        return error_close(sock, "Number of topK confidances: %d not supported\n", topK);
    }

    //////
    /// check for model validity
    ///
    bool found = false;
    for(size_t i = 0; i < args->getNumConfigureddModels(); i++) {
        std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float,std::string> info = args->getConfiguredModelInfo(i);
        if(std::get<0>(info) == modelName &&
           std::get<1>(info) == dimInput[0] &&
           std::get<2>(info) == dimInput[1] &&
           std::get<3>(info) == dimInput[2] &&
           std::get<4>(info) == dimOutput[0] &&
           std::get<5>(info) == dimOutput[1] &&
           std::get<6>(info) == dimOutput[2])
        {
            reverseInputChannelOrder = std::get<7>(info);
            preprocessMpy[0] = std::get<8>(info);
            preprocessMpy[1] = std::get<9>(info);
            preprocessMpy[2] = std::get<10>(info);
            preprocessAdd[0] = std::get<11>(info);
            preprocessAdd[1] = std::get<12>(info);
            preprocessAdd[2] = std::get<13>(info);
            modelPath = args->getConfigurationDir() + "/" + std::get<14>(info);
            found = true;
            break;
        }
    }
    if(!found) {
        for(size_t i = 0; i < args->getNumUploadedModels(); i++) {
            std::tuple<std::string,int,int,int,int,int,int,int,float,float,float,float,float,float> info = args->getUploadedModelInfo(i);
            if(std::get<0>(info) == modelName &&
               std::get<1>(info) == dimInput[0] &&
               std::get<2>(info) == dimInput[1] &&
               std::get<3>(info) == dimInput[2] &&
               std::get<4>(info) == dimOutput[0] &&
               std::get<5>(info) == dimOutput[1] &&
               std::get<6>(info) == dimOutput[2])
            {
                reverseInputChannelOrder = std::get<7>(info);
                preprocessMpy[0] = std::get<8>(info);
                preprocessMpy[1] = std::get<9>(info);
                preprocessMpy[2] = std::get<10>(info);
                preprocessAdd[0] = std::get<11>(info);
                preprocessAdd[1] = std::get<12>(info);
                preprocessAdd[2] = std::get<13>(info);
                modelPath = args->getConfigurationDir() + "/" + modelName;
                found = true;
                break;
            }
        }
    }
    if(found) {
        modulePath = modelPath + "/build/" + MODULE_LIBNAME;
        moduleHandle = dlopen(modulePath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if(!moduleHandle) {
            found = false;
            error("could not locate module %s for %s", modulePath.c_str(), clientName.c_str());
        }
        if (args->getModelCompilerPath().empty()) {
            if(!(annCreateGraph = (type_annCreateGraph *) dlsym(moduleHandle, "annCreateGraph"))) {
                found = false;
                error("could not find function annCreateGraph() in module %s for %s", modulePath.c_str(), clientName.c_str());
            }
        }
        else if(!(annAddtoGraph = (type_annAddToGraph *) dlsym(moduleHandle, "annAddToGraph"))) {
            found = false;
            error("could not find function annAddToGraph() in module %s for %s", modulePath.c_str(), clientName.c_str());
        }
    }
    else {
        error("unable to find requested model:%s input:%dx%dx%d output:%dx%dx%d from %s", modelName.c_str(),
              dimInput[2], dimInput[1], dimInput[0], dimOutput[2], dimOutput[1], dimOutput[0], clientName.c_str());
    }
    if(!found) {
        // send and wait for INFCOM_CMD_DONE message
        InfComCommand reply = {
            INFCOM_MAGIC, INFCOM_CMD_DONE, { 0 }, { 0 }
        };
        ERRCHK(sendCommand(sock, reply, clientName));
        ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));
        close(sock);
        return -1;
    }
    info("found requested model:%s input:%dx%dx%d output:%dx%dx%d from %s", modelName.c_str(),
          dimInput[2], dimInput[1], dimInput[0], dimOutput[2], dimOutput[1], dimOutput[0], clientName.c_str());

    // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
    InfComCommand updateCmd = {
        INFCOM_MAGIC, INFCOM_CMD_INFERENCE_INITIALIZATION, { 0 }, "started initialization"
    };
    ERRCHK(sendCommand(sock, updateCmd, clientName));
    ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
    info(updateCmd.message);

#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
#if DONOT_RUN_INFERENCE
    info("InferenceEngine: using NO_INFERENCE_SCHEDULER and DONOT_RUN_INFERENCE");
#else
    { // create OpenVX resources
        info("InferenceEngine: using NO_INFERENCE_SCHEDULER");
        vx_status status;
        openvx_context = vxCreateContext();
        if((status = vxGetStatus((vx_reference)openvx_context)) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateContext() failed (%d)", status);
        vx_size idim[4] = { (vx_size)dimInput[0], (vx_size)dimInput[1], (vx_size)dimInput[2], (vx_size)batchSize };
        vx_size odim[4] = { (vx_size)dimOutput[0], (vx_size)dimOutput[1], (vx_size)dimOutput[2], (vx_size)batchSize };
        openvx_input = vxCreateTensor(openvx_context, 4, idim, VX_TYPE_FLOAT32, 0);
        openvx_output = vxCreateTensor(openvx_context, 4, odim, VX_TYPE_FLOAT32, 0);
        if((status = vxGetStatus((vx_reference)openvx_input)) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensor(input) failed (%d)", status);
        if((status = vxGetStatus((vx_reference)openvx_output)) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensor(output) failed (%d)", status);
        //////
        // load the model
        openvx_graph = annCreateGraph(openvx_context, openvx_input, openvx_output, modelPath.c_str());
        if((status = vxGetStatus((vx_reference)openvx_graph)) != VX_SUCCESS)
            fatal("InferenceEngine: annCreateGraph() failed (%d)", status);

        // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
        updateCmd.data[0] = 80;
        sprintf(updateCmd.message, "completed OpenVX graph");
        ERRCHK(sendCommand(sock, updateCmd, clientName));
        ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
        info(updateCmd.message);
    }
#endif
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    info("InferenceEngine: using LIBRE_INFERENCE_SCHEDULER");
    //////
    /// allocate OpenVX and OpenCL resources
    /// 
    for(int gpu = 0; gpu < GPUs; gpu++) {
        //////
        // create OpenCL context
        cl_context_properties ctxprop[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)args->getPlatformId(),
            0, 0
        };
        cl_int err;
        opencl_context[gpu] = clCreateContext(ctxprop, 1, &device_id[gpu], NULL, NULL, &err);
        if(err)
            fatal("InferenceEngine: clCreateContext(#%d) failed (%d)", gpu, err);
#if defined(CL_VERSION_2_0)
        cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0, 0 };
        opencl_cmdq[gpu] = clCreateCommandQueueWithProperties(opencl_context[gpu], device_id[gpu], properties, &err);
#else
        opencl_cmdq[gpu] = clCreateCommandQueue(opencl_context[gpu], device_id[gpu], 0, &err);
#endif
        if(err) {
            fatal("InferenceEngine: clCreateCommandQueue(device_id[%d]) failed (%d)", gpu, err);
        }

        // create scheduler device queues
#if  USE_ADVANCED_MESSAGE_Q
        queueDeviceTagQ[gpu] = new MessageQueueAdvanced<int>(MAX_DEVICE_QUEUE_DEPTH);
        queueDeviceImageQ[gpu] = new MessageQueueAdvanced<std::tuple<char*,int>>(MAX_INPUT_QUEUE_DEPTH);
#else
        queueDeviceTagQ[gpu] = new MessageQueue<int>();
        queueDeviceTagQ[gpu]->setMaxQueueDepth(MAX_DEVICE_QUEUE_DEPTH);
        queueDeviceImageQ[gpu] = new MessageQueue<std::tuple<char*,int>>();
#endif
        queueDeviceInputMemIdle[gpu] = new MessageQueue<cl_mem>();
        queueDeviceInputMemBusy[gpu] = new MessageQueue<cl_mem>();
        queueDeviceOutputMemIdle[gpu] = new MessageQueue<cl_mem>();
        queueDeviceOutputMemBusy[gpu] = new MessageQueue<cl_mem>();

        // create OpenCL buffers for input/output and add them to queueDeviceInputMemIdle/queueDeviceOutputMemIdle
        cl_mem memInput = nullptr, memOutput = nullptr;
        for(int i = 0; i < INFERENCE_PIPE_QUEUE_DEPTH; i++) {
            cl_int err;
            memInput = clCreateBuffer(opencl_context[gpu], CL_MEM_READ_WRITE, inputSizeInBytes, NULL, &err);
            if(err) {
                fatal("InferenceEngine: clCreateBuffer(#%d,%d) [#%d] failed (%d)", gpu, inputSizeInBytes, i, err);
            }
            memOutput = clCreateBuffer(opencl_context[gpu], CL_MEM_READ_WRITE, outputSizeInBytes, NULL, &err);
            if(err) {
                fatal("InferenceEngine: clCreateBuffer(#%d,%d) [#%d] failed (%d)", gpu, outputSizeInBytes, i, err);
            }
            queueDeviceInputMemIdle[gpu]->enqueue(memInput);
            queueDeviceOutputMemIdle[gpu]->enqueue(memOutput);
        }

        //////
        // create OpenVX context
        vx_status status;
        openvx_context[gpu] = vxCreateContext();
        if((status = vxGetStatus((vx_reference)openvx_context[gpu])) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateContext(#%d) failed (%d)", gpu, status);
        if((status = vxSetContextAttribute(openvx_context[gpu], VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,
                                          &opencl_context[gpu], sizeof(cl_context))) != VX_SUCCESS)
            fatal("InferenceEngine: vxSetContextAttribute(#%d,VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT) failed (%d)", gpu, status);
        vx_size idim[4] = { (vx_size)dimInput[0], (vx_size)dimInput[1], (vx_size)dimInput[2], (vx_size)batchSize };
        vx_size odim[4] = { (vx_size)dimOutput[0], (vx_size)dimOutput[1], (vx_size)dimOutput[2], (vx_size)batchSize };
        if (useFp16) {
            vx_size istride[4] = { 2, (vx_size)2 * dimInput[0], (vx_size)2 * dimInput[0] * dimInput[1], (vx_size)2 * dimInput[0] * dimInput[1] * dimInput[2] };
            vx_size ostride[4] = { 2, (vx_size)2 * dimOutput[0], (vx_size)2 * dimOutput[0] * dimOutput[1], (vx_size)2 * dimOutput[0] * dimOutput[1] * dimOutput[2] };
            openvx_input[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, idim, VX_TYPE_FLOAT16, 0, istride, memInput, VX_MEMORY_TYPE_OPENCL);
            openvx_output[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, odim, VX_TYPE_FLOAT16, 0, ostride, memOutput, VX_MEMORY_TYPE_OPENCL);
            if (openvx_output[gpu] == nullptr)
                printf(" vxCreateTensorFromHandle(output) failed for gpu#%d\n", gpu);
        } else {
            vx_size istride[4] = { 4, (vx_size)4 * dimInput[0], (vx_size)4 * dimInput[0] * dimInput[1], (vx_size)4 * dimInput[0] * dimInput[1] * dimInput[2] };
            vx_size ostride[4] = { 4, (vx_size)4 * dimOutput[0], (vx_size)4 * dimOutput[0] * dimOutput[1], (vx_size)4 * dimOutput[0] * dimOutput[1] * dimOutput[2] };
            openvx_input[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, idim, VX_TYPE_FLOAT32, 0, istride, memInput, VX_MEMORY_TYPE_OPENCL);
            openvx_output[gpu] = vxCreateTensorFromHandle(openvx_context[gpu], 4, odim, VX_TYPE_FLOAT32, 0, ostride, memOutput, VX_MEMORY_TYPE_OPENCL);
        }
        if((status = vxGetStatus((vx_reference)openvx_input[gpu])) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensorFromHandle(input#%d) failed (%d)", gpu, status);
        if((status = vxGetStatus((vx_reference)openvx_output[gpu])) != VX_SUCCESS)
            fatal("InferenceEngine: vxCreateTensorFromHandle(output#%d) failed (%d)", gpu, status);

        //////
        // load the model
        if (annCreateGraph != nullptr) {
            openvx_graph[gpu] = annCreateGraph(openvx_context[gpu], openvx_input[gpu], openvx_output[gpu], modelPath.c_str());
            if((status = vxGetStatus((vx_reference)openvx_graph[gpu])) != VX_SUCCESS)
                fatal("InferenceEngine: annCreateGraph(#%d) failed (%d)", gpu, status);
        }
        else if (annAddtoGraph != nullptr) {
            std::string weightsFile = modelPath + "/weights.bin";
            vxRegisterLogCallback(openvx_context[gpu], log_callback, vx_false_e);
            openvx_graph[gpu] = vxCreateGraph(openvx_context[gpu]);
            status = vxGetStatus((vx_reference)openvx_graph[gpu]);
            if(status) {
                fatal("InferenceEngine: vxCreateGraph(#%d) failed (%d)", gpu, status);
                return -1;
            }
            status = annAddtoGraph(openvx_graph[gpu], openvx_input[gpu], openvx_output[gpu], weightsFile.c_str());
            if(status) {
                fatal("InferenceEngine: annAddToGraph(#%d) failed (%d)", gpu, status);
                return -1;
            }
        }

        // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
        updateCmd.data[0] = 80 * (gpu + 1) / GPUs;
        sprintf(updateCmd.message, "completed OpenVX graph for GPU#%d", gpu);
        ERRCHK(sendCommand(sock, updateCmd, clientName));
        ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
        info(updateCmd.message);
    }
#endif

    //////
    /// start scheduler threads
    ///
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
    // nothing to do
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
    threadMasterInputQ = new std::thread(&InferenceEngine::workMasterInputQ, this);
    for(int gpu = 0; gpu < GPUs; gpu++) {
        threadDeviceInputCopy[gpu] = new std::thread(&InferenceEngine::workDeviceInputCopy, this, gpu);
        threadDeviceProcess[gpu] = new std::thread(&InferenceEngine::workDeviceProcess, this, gpu);
        threadDeviceOutputCopy[gpu] = new std::thread(&InferenceEngine::workDeviceOutputCopy, this, gpu);
    }
#endif

    // send and wait for INFCOM_CMD_INFERENCE_INITIALIZATION message
    updateCmd.data[0] = 100;
    sprintf(updateCmd.message, "inference engine is ready");
    ERRCHK(sendCommand(sock, updateCmd, clientName));
    ERRCHK(recvCommand(sock, updateCmd, clientName, INFCOM_CMD_INFERENCE_INITIALIZATION));
    info(updateCmd.message);

    ////////
    /// \brief keep running the inference in loop
    ///
    bool endOfImageRequested = false;
    for(bool endOfSequence = false; !endOfSequence; ) {
        bool didSomething = false;

        // send all the available results to the client
        int resultCountAvailable = outputQ.size();
        if(resultCountAvailable > 0) {
            didSomething = true;
            while(resultCountAvailable > 0) {
                if (!detectBoundingBoxes){
                    if (topK < 1){
                        int resultCount = std::min(resultCountAvailable, (INFCOM_MAX_IMAGES_FOR_TOP1_PER_PACKET/2));
                        InfComCommand cmd = {
                            INFCOM_MAGIC, INFCOM_CMD_INFERENCE_RESULT, { resultCount, 0 }, { 0 }
                        };
                        for(int i = 0; i < resultCount; i++) {
                            std::tuple<int,int> result;
                            outputQ.dequeue(result);
                            int tag = std::get<0>(result);
                            int label = std::get<1>(result);
                            if(tag < 0) {
                                endOfSequence = true;
                                resultCount = i;
                                break;
                            }
                            cmd.data[2 + i * 2 + 0] = tag; // tag
                            cmd.data[2 + i * 2 + 1] = label; // label
                        }
                        if(resultCount > 0) {
                            cmd.data[0] = resultCount;
                            ERRCHK(sendCommand(sock, cmd, clientName));
                            resultCountAvailable -= resultCount;
                            ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_INFERENCE_RESULT));
                        }
                        if(endOfSequence) {
                            break;
                        }
                    }else {
                        // send topK labels
                        int maxResults = INFCOM_MAX_IMAGES_FOR_TOP1_PER_PACKET/(topK+1);
                        int resultCount = std::min(resultCountAvailable, maxResults);
                        InfComCommand cmd = {
                            INFCOM_MAGIC, INFCOM_CMD_TOPK_INFERENCE_RESULT, { resultCount, topK }, { 0 }
                        };
                        for(int i = 0; i < resultCount; i++) {
                            std::tuple<int,int> result;
                            std::vector<unsigned int> labels;
                            outputQ.dequeue(result);
                            int tag = std::get<0>(result);
                            if(tag < 0) {
                                endOfSequence = true;
                                resultCount = i;
                                break;
                            }
                            outputQTopk.dequeue(labels);
                            cmd.data[2 + i * (topK+1) + 0] = tag; // tag
                            for (int j=0; j<topK; j++){
                                cmd.data[3 + i * (topK+1) + j] = labels[j]; // label[j]
                            }
                            labels.clear();
                        }
                        if(resultCount > 0) {
                            cmd.data[0] = resultCount;
                            ERRCHK(sendCommand(sock, cmd, clientName));
                            resultCountAvailable -= resultCount;
                            ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_TOPK_INFERENCE_RESULT));
                        }
                        if(endOfSequence) {
                            break;
                        }
                    }
                }else
                {
                    // Dequeue the bounding box
                    std::tuple<int,int> result;
                    std::vector<ObjectBB> bounding_boxes;
                    outputQ.dequeue(result);
                    int tag = std::get<0>(result);
                    int label = std::get<1>(result);        // label of first bounding box
                    if(tag < 0) {
                        endOfSequence = true;
                        resultCountAvailable--;
                        break;
                    }else
                    {
                        int numBB = 0;
                        int numMessages = 0;
                        if (label >= 0) {
                            OutputQBB.dequeue(bounding_boxes);
                            numBB = bounding_boxes.size();
                            if (numBB) numMessages = numBB/3;   // max 3 bb per mesasge
                            if (numBB % 3) numMessages++;
                        }
                        if (!numBB) {
                            InfComCommand cmd = {
                                INFCOM_MAGIC, INFCOM_CMD_BB_INFERENCE_RESULT, { tag, 0 }, { 0 }        // no bb detected
                            };
                            ERRCHK(sendCommand(sock, cmd, clientName));
                            ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_BB_INFERENCE_RESULT));
                        } else
                        {
                            ObjectBB *pObj= &bounding_boxes[0];
                            for (int i=0, j=0; (i < numMessages && j < numBB); i++) {
                                int numBB_per_message = std::min((numBB-j), 3);
                                int bb_info = (numBB_per_message & 0xFFFF) | (numBB << 16);
                                InfComCommand cmd = {
                                    INFCOM_MAGIC, INFCOM_CMD_BB_INFERENCE_RESULT, { tag, bb_info }, { 0 }        // 3 bounding boxes in one message
                                };
                                cmd.data[2] = (unsigned int)((pObj->y*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->x*0x7FFF)+0.5);
                                cmd.data[3] = (unsigned int)((pObj->h*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->w*0x7FFF)+0.5);
                                cmd.data[4] = (unsigned int) ((pObj->confidence*0x3FFFFFFF)+0.5);    // convert float to Q30.1
                                cmd.data[5] = pObj->label;
                                pObj++;
                                if (numBB_per_message > 1) {
                                    cmd.data[6] = (unsigned int)((pObj->y*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->x*0x7FFF)+0.5);
                                    cmd.data[7] = (unsigned int)((pObj->h*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->w*0x7FFF)+0.5);
                                    cmd.data[8] = (unsigned int) ((pObj->confidence*0x3FFFFFFF)+0.5);    // convert float to Q30.1
                                    cmd.data[9] = pObj->label;
                                    pObj++;
                                }
                                if (numBB_per_message > 2) {
                                    cmd.data[10] = (unsigned int)((pObj->y*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->x*0x7FFF)+0.5);
                                    cmd.data[11] = (unsigned int)((pObj->h*0x7FFF)+0.5)<<16  | (unsigned int)((pObj->w*0x7FFF)+0.5);
                                    cmd.data[12] = (unsigned int) ((pObj->confidence*0x3FFFFFFF)+0.5);    // convert float to Q30.1;
                                    cmd.data[13] = pObj->label;
                                    pObj++;
                                }
                                ERRCHK(sendCommand(sock, cmd, clientName));
                                ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_BB_INFERENCE_RESULT));
                                j += numBB_per_message;
                            }
                        }
                        resultCountAvailable--;
                    }
                    bounding_boxes.clear();
                }
            }
        }

        // if not endOfImageRequested, request client to send images
        if(!endOfImageRequested) {
            // get number of empty slots in the input queue
            int imageCountRequested = 0;
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
            imageCountRequested = 1;
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
            imageCountRequested = MAX_INPUT_QUEUE_DEPTH - inputQ.size();
#endif
            if(imageCountRequested > 0) {
                didSomething = true;
                // send request for upto INFCOM_MAX_IMAGES_PER_PACKET images
                imageCountRequested = std::min(imageCountRequested, (INFCOM_MAX_IMAGES_FOR_TOP1_PER_PACKET/2));
                InfComCommand cmd = {
                    INFCOM_MAGIC, INFCOM_CMD_SEND_IMAGES, { imageCountRequested }, { 0 }
                };
                ERRCHK(sendCommand(sock, cmd, clientName));
                ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_SEND_IMAGES));

                // check of endOfImageRequested and receive images one at a time
                int imageCountReceived = cmd.data[0];
                if(imageCountReceived < 0) {
                    // submit the endOfSequence indicator to scheduler
#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
                    endOfSequence = true;
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
                    inputQ.enqueue(std::tuple<int,char*,int>(-1,nullptr,0));
#endif
                    endOfImageRequested = true;
                }
                int i = 0;
                for(; i < imageCountReceived; i++) {
                    // get header with tag and size info
                    int header[2] = { 0, 0 };
                    ERRCHK(recvBuffer(sock, &header, sizeof(header), clientName));
                    int tag = header[0];
                    int size = header[1];
                    // do sanity check with unreasonable parameters
                    if(tag < 0 || size <= 0 || size > 50000000) {
                        return error_close(sock, "invalid (tag:%d,size:%d) from %s", tag, size, clientName.c_str());
                    }
                    char * byteStream = 0;
                    if (receiveFileNames)
                    {
                        std::string fileNameDir = args->getlocalShadowRootDir() + "/";
                        char * buff = new char [size];
                        ERRCHK(recvBuffer(sock, buff, size, clientName));
                        fileNameDir.append(std::string(buff, size));
                        FILE * fp = fopen(fileNameDir.c_str(), "rb");
                        if(!fp) {
                            return error_close(sock, "filename %s (incorrect)", fileNameDir.c_str());
                        }
                        fseek(fp,0,SEEK_END);
                        int fsize = ftell(fp);
                        fseek(fp,0,SEEK_SET);
                        byteStream = new char [fsize];
                        size = (int)fread(byteStream, 1, fsize, fp);
                        fclose(fp);
                        delete[] buff;
                        if (size != fsize) {
                            return error_close(sock, "error reading %d bytes from file:%s", fsize, fileNameDir.c_str());
                        }
                    }
                    else
                    {
                        // allocate and receive the image and EOF marker
                        byteStream = new char [size];
                        ERRCHK(recvBuffer(sock, byteStream, size, clientName));
                    }
                    int eofMarker = 0;
                    ERRCHK(recvBuffer(sock, &eofMarker, sizeof(eofMarker), clientName));
                    if(eofMarker != INFCOM_EOF_MARKER) {
                        return error_close(sock, "eofMarker 0x%08x (incorrect)", eofMarker);
                    }

#if INFERENCE_SCHEDULER_MODE == NO_INFERENCE_SCHEDULER
#if DONOT_RUN_INFERENCE
                    // consume the input immediately since there is no scheduler
                    // simulate the input (tag,byteStream,size) processing using a 4ms sleep
                    int label = tag % dimOutput[2];
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    // release byteStream and keep the results in outputQ
                    delete[] byteStream;
                    outputQ.enqueue(std::tuple<int,int>(tag,label));
#else
                    // process the input immediately since there is no scheduler
                    // decode, scale, and format convert into the OpenVX input buffer
                    vx_map_id map_id;
                    vx_size stride[4];
                    float * ptr = nullptr;
                    vx_status status;
                    status = vxMapTensorPatch(openvx_input, 4, NULL, NULL, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxMapTensorPatch(input)) failed(%d)", status);
                    }
                    DecodeScaleAndConvertToTensor(dimInput[0], dimInput[1], size, (unsigned char *)byteStream, ptr, useFp16);
                    status = vxUnmapTensorPatch(openvx_input, map_id);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxUnmapTensorPatch(input)) failed(%d)", status);
                    }
                    // process the graph
                    status = vxProcessGraph(openvx_graph);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxProcessGraph()) failed(%d)", status);
                    }
                    ptr = nullptr;
                    status = vxMapTensorPatch(openvx_output, 4, NULL, NULL, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxMapTensorPatch(output)) failed(%d)", status);
                    }
                    int label = 0;
                    float max_prob = ptr[0];
                    for(int c = 1; c < dimOutput[2]; c++) {
                        float prob = ptr[c];
                        if(prob > max_prob) {
                            label = c;
                            max_prob = prob;
                        }
                    }
                    status = vxUnmapTensorPatch(openvx_output, map_id);
                    if(status != VX_SUCCESS) {
                        fatal("workDeviceProcess: vxUnmapTensorPatch(output)) failed(%d)", status);
                    }
                    // release byteStream and keep the results in outputQ
                    delete[] byteStream;
                    outputQ.enqueue(std::tuple<int,int>(tag,label));
#endif
#elif INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
                    // submit the input (tag,byteStream,size) to scheduler
                    inputQ.enqueue(std::tuple<int,char*,int>(tag,byteStream,size));
#endif
                }
            }
        }

        // if nothing done, wait for sometime
        if(!didSomething && INFERENCE_SERVICE_IDLE_TIME > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(INFERENCE_SERVICE_IDLE_TIME));
        }
    }
    info("runInference: terminated for %s", clientName.c_str());

    // send and wait for INFCOM_CMD_DONE message
    InfComCommand reply = {
        INFCOM_MAGIC, INFCOM_CMD_DONE, { 0 }, { 0 }
    };
    ERRCHK(sendCommand(sock, reply, clientName));
    ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));
#endif
    return 0;
}

#if INFERENCE_SCHEDULER_MODE == LIBRE_INFERENCE_SCHEDULER
void InferenceEngine::workMasterInputQ()
{
    args->lock();
    info("workMasterInputQ: started for %s", clientName.c_str());
    args->unlock();

    int batchSize = args->getBatchSize();
    int totalInputCount = 0;
    int inputCountInBatch = 0, gpu = 0;
    for(;;) {
        PROFILER_START(inference_server_app, workMasterInputQ);
         // get next item from the input queue
        std::tuple<int,char*,int> input;
        inputQ.dequeue(input);
        int tag = std::get<0>(input);
        char * byteStream = std::get<1>(input);
        int size = std::get<2>(input);

        // check for end of input
        if(tag < 0 || byteStream == nullptr || size == 0)
            break;
        totalInputCount++;

        // add the image to selected deviceQ
        std::tuple<char*,int> image(byteStream,size);
        queueDeviceTagQ[gpu]->enqueue(tag);
        queueDeviceImageQ[gpu]->enqueue(image);
        PROFILER_STOP(inference_server_app, workMasterInputQ);

        // at the end of Batch pick another device
        inputCountInBatch++;
        if(inputCountInBatch == batchSize) {
            inputCountInBatch = 0;
            gpu = (gpu + 1) % GPUs;
            for(int i = 0; i < GPUs; i++) {
                if(i != gpu && queueDeviceTagQ[i]->size() < queueDeviceTagQ[gpu]->size()) {
                    gpu = i;
                }
            }
        }
    }

    // send endOfSequence indicator to all scheduler threads
    for(int i = 0; i < GPUs; i++) {
        int endOfSequenceTag = -1;
        std::tuple<char*,int> endOfSequenceImage(nullptr,0);
        queueDeviceTagQ[i]->enqueue(endOfSequenceTag);
        queueDeviceImageQ[i]->enqueue(endOfSequenceImage);
    }
    args->lock();
    info("workMasterInputQ: terminated for %s [scheduled %d images]", clientName.c_str(), totalInputCount);
    args->unlock();
}

void InferenceEngine::workDeviceInputCopy(int gpu)
{
#if ENABLE_OPENCL  
    args->lock();
    info("workDeviceInputCopy: GPU#%d started for %s", gpu, clientName.c_str());
    args->unlock();

    // create OpenCL command-queue
    cl_command_queue cmdq;
    cl_int err;
#if defined(CL_VERSION_2_0)
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    cmdq = clCreateCommandQueueWithProperties(opencl_context[gpu], device_id[gpu], properties, &err);
#else
    cmdq = clCreateCommandQueue(opencl_context[gpu], device_id[gpu], 0, &err);
#endif
    if(err) {
        fatal("workDeviceInputCopy: clCreateCommandQueue(device_id[%d]) failed (%d)", gpu, err);
    }

    int totalBatchCounter = 0, totalImageCounter = 0;
    for(bool endOfSequenceReached = false; !endOfSequenceReached; ) {
        PROFILER_START(inference_server_app, workDeviceInputCopyBatch);
        // get an empty OpenCL buffer and lock the buffer for writing
        cl_mem mem = nullptr;
        queueDeviceInputMemIdle[gpu]->dequeue(mem);
        if(mem == nullptr) {
            fatal("workDeviceInputCopy: unexpected nullptr in queueDeviceInputMemIdle[%d]", gpu);
        }
        cl_int err;
        void * mapped_ptr = (void *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, inputSizeInBytes, 0, NULL, NULL, &err);
        if(err) {
            fatal("workDeviceInputCopy: clEnqueueMapBuffer(#%d) failed (%d)", gpu, err);
        }

        // get next batch of inputs and convert them into tensor and release input byteStream
        // TODO: replace with an efficient implementation
        int inputCount = 0;
        if (numDecThreads > 0) {
            std::vector<std::tuple<char*, int>> batch_q;
            int sub_batch_size = batchSize/numDecThreads;
            std::thread dec_threads[numDecThreads];
            int numT = numDecThreads;
            // dequeue batch
            for (; inputCount<batchSize; inputCount++)
            {
                std::tuple<char*, int> image;
                queueDeviceImageQ[gpu]->dequeue(image);
                char * byteStream = std::get<0>(image);
                int size = std::get<1>(image);
                if(byteStream == nullptr || size == 0) {
                    printf("workDeviceInputCopy:: Eos reached inputCount: %d\n", inputCount);
                    endOfSequenceReached = true;
                    break;
                }
                batch_q.push_back(image);
            }
            if (inputCount){
                PROFILER_START(inference_server_app, workDeviceInputCopyJpegDecode);
                if (inputCount < batchSize)
                {
                    sub_batch_size = (inputCount+numT-1)/numT;
                    numT = (inputCount+(sub_batch_size-1))/sub_batch_size;
                }
                int start = 0; int end = sub_batch_size-1;
                for (unsigned int t = 0; t < (numT - 1); t++)
                {
                    dec_threads[t]  = std::thread(&InferenceEngine::DecodeScaleAndConvertToTensorBatch, this, std::ref(batch_q), start, end, dimInput, (float *)mapped_ptr);
                    start += sub_batch_size;
                    end += sub_batch_size;
                }
                start = std::min(start, (inputCount - 1));
                end = std::min(end, (inputCount-1));
                // do some work in this thread
                DecodeScaleAndConvertToTensorBatch(batch_q, start, end, dimInput, (float *)mapped_ptr);
                for (unsigned int t = 0; t < (numT - 1); t++)
                {
                    dec_threads[t].join();
                }
                PROFILER_STOP(inference_server_app, workDeviceInputCopyJpegDecode);
            }
        } else {
            for(; inputCount < batchSize; inputCount++) {
                // get next item from the input queue and check for end of input
                std::tuple<char*,int> image;
                queueDeviceImageQ[gpu]->dequeue(image);
                char * byteStream = std::get<0>(image);
                int size = std::get<1>(image);
                if(byteStream == nullptr || size == 0) {
                    endOfSequenceReached = true;
                    break;
                }
                // decode, scale, and format convert into the OpenCL buffer
                float *buf;
                if (useFp16)
                    buf = (float *) ((unsigned short *)mapped_ptr + dimInput[0] * dimInput[1] * dimInput[2] * inputCount);
                else
                    buf = (float *) mapped_ptr + dimInput[0] * dimInput[1] * dimInput[2] * inputCount;

                PROFILER_START(inference_server_app, workDeviceInputCopyJpegDecode);
                DecodeScaleAndConvertToTensor(dimInput[0], dimInput[1], size, (unsigned char *)byteStream, buf, useFp16);
                PROFILER_STOP(inference_server_app, workDeviceInputCopyJpegDecode);
                // release byteStream
                delete[] byteStream;
            }
        }
        // unlock the OpenCL buffer to perform the writing
        err = clEnqueueUnmapMemObject(cmdq, mem, mapped_ptr, 0, NULL, NULL);
        if(err) {
            fatal("workDeviceInputCopy: clEnqueueMapBuffer(#%d) failed (%d)", gpu, err);
        }
        err = clFinish(cmdq);
        if(err) {
            fatal("workDeviceInputCopy: clFinish(#%d) failed (%d)", gpu, err);
        }

        if(inputCount > 0) {
            // add the input for processing
            queueDeviceInputMemBusy[gpu]->enqueue(mem);
            // update counters
            totalBatchCounter++;
            totalImageCounter += inputCount;
        }
        else {
            // add the input back to idle queue
            queueDeviceInputMemIdle[gpu]->enqueue(mem);
        }
        PROFILER_STOP(inference_server_app, workDeviceInputCopyBatch);
    }
    // release OpenCL command queue
    clReleaseCommandQueue(cmdq);

    // add the endOfSequenceMarker to next stage
    cl_mem endOfSequenceMarker = nullptr;
    queueDeviceInputMemBusy[gpu]->enqueue(endOfSequenceMarker);

    args->lock();
    info("workDeviceInputCopy: GPU#%d terminated for %s [processed %d batches, %d images]", gpu, clientName.c_str(), totalBatchCounter, totalImageCounter);
    args->unlock();
#endif    
}

void InferenceEngine::workDeviceProcess(int gpu)
{
#if ENABLE_OPENCL  
    args->lock();
    info("workDeviceProcess: GPU#%d started for %s", gpu, clientName.c_str());
    args->unlock();

    int processCounter = 0;
    for(;;) {
        // get a busy OpenCL buffer for input and check for end of sequence marker
        cl_mem input = nullptr;
        queueDeviceInputMemBusy[gpu]->dequeue(input);
        if(!input) {
            break;
        }

        // get an empty OpenCL buffer for output and a busy OpenCL buffer for input
        cl_mem output = nullptr;
        queueDeviceOutputMemIdle[gpu]->dequeue(output);
        if(input == nullptr) {
            fatal("workDeviceProcess: unexpected nullptr in queueDeviceOutputMemIdle[%d]", gpu);
        }

        // process the graph
        vx_status status;
        status = vxSwapTensorHandle(openvx_input[gpu], input, nullptr);
        if(status != VX_SUCCESS) {
            fatal("workDeviceProcess: vxSwapTensorHandle(input#%d) failed(%d)", gpu, status);
        }
        status = vxSwapTensorHandle(openvx_output[gpu], output, nullptr);
        if(status != VX_SUCCESS) {
            fatal("workDeviceProcess: vxSwapTensorHandle(output#%d) failed(%d)", gpu, status);
        }
#if !DONOT_RUN_INFERENCE
        PROFILER_START(inference_server_app, workDeviceProcess);
        status = vxProcessGraph(openvx_graph[gpu]);
        PROFILER_STOP(inference_server_app, workDeviceProcess);
        if(status != VX_SUCCESS) {
            fatal("workDeviceProcess: vxProcessGraph(#%d) failed(%d)", gpu, status);
        }
#else
        info("InferenceEngine:workDeviceProcess DONOT_RUN_INFERENCE mode");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // simulate some work
#endif
        // add the input for idle queue and output to busy queue
        queueDeviceInputMemIdle[gpu]->enqueue(input);
        queueDeviceOutputMemBusy[gpu]->enqueue(output);
        processCounter++;
    }

    // add the endOfSequenceMarker to next stage
    cl_mem endOfSequenceMarker = nullptr;
    queueDeviceOutputMemBusy[gpu]->enqueue(endOfSequenceMarker);

    args->lock();
    info("workDeviceProcess: GPU#%d terminated for %s [processed %d batches]", gpu, clientName.c_str(), processCounter);
    args->unlock();
#endif    
}

void InferenceEngine::workDeviceOutputCopy(int gpu)
{
#if ENABLE_OPENCL  
    args->lock();
    info("workDeviceOutputCopy: GPU#%d started for %s", gpu, clientName.c_str());
    args->unlock();

    // create OpenCL command-queue
    cl_command_queue cmdq;
    cl_int err;
#if defined(CL_VERSION_2_0)
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    cmdq = clCreateCommandQueueWithProperties(opencl_context[gpu], device_id[gpu], properties, &err);
#else
    cmdq = clCreateCommandQueue(opencl_context[gpu], device_id[gpu], 0, &err);
#endif
    if(err) {
        fatal("workDeviceOutputCopy: clCreateCommandQueue(device_id[%d]) failed (%d)", gpu, err);
    }

    int totalBatchCounter = 0, totalImageCounter = 0;
    for(bool endOfSequenceReached = false; !endOfSequenceReached; ) {
        // get an output OpenCL buffer and lock the buffer for reading
        cl_mem mem = nullptr;
        queueDeviceOutputMemBusy[gpu]->dequeue(mem);
        if(mem == nullptr) {
            break;
        }
        PROFILER_START(inference_server_app, workDeviceOutputCopy);
        cl_int err;
        void * mapped_ptr = (float *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_READ, 0, outputSizeInBytes, 0, NULL, NULL, &err);
        if(err) {
            fatal("workDeviceOutputCopy: clEnqueueMapBuffer(#%d) failed (%d)", gpu, err);
        }

        // get next batch of inputs
        int outputCount = 0;
        int useFp16 = args->fp16Inference();
        for(; outputCount < batchSize; outputCount++) {
            // get next item from the tag queue and check for end of input
            int tag;
            queueDeviceTagQ[gpu]->dequeue(tag);
            if(tag < 0) {
                endOfSequenceReached = true;
                break;
            }

            // decode, scale, and format convert into the OpenCL buffer
            void *buf;
            if (!useFp16)
                buf = (float *)mapped_ptr + dimOutput[0] * dimOutput[1] * dimOutput[2] * outputCount;
            else
                buf = (unsigned short *)mapped_ptr + dimOutput[0] * dimOutput[1] * dimOutput[2] * outputCount;

            if (!detectBoundingBoxes)
            {
                if (topK < 1){
                    int label = 0;
                    if (!useFp16) {
                        float *out = (float *)buf;
                        float max_prob = out[0];
                        for(int c = 1; c < dimOutput[2]; c++) {
                            float prob = out[c];
                            if(prob > max_prob) {
                                label = c;
                                max_prob = prob;
                            }
                        }
                    } else {
                        unsigned short *out = (unsigned short *)buf;
                        float max_prob = _cvtsh_ss(out[0]);
                        for(int c = 1; c < dimOutput[2]; c++) {
                            float prob = _cvtsh_ss(out[c]);
                            if(prob > max_prob) {
                                label = c;
                                max_prob = prob;
                            }
                        }
                    }
                    outputQ.enqueue(std::tuple<int,int>(tag,label));
                }else {
                    // todo:: add support for fp16
                    std::vector<float>  prob_vec((float*)buf, (float*)buf + dimOutput[2]);
                    std::vector<size_t> idx(prob_vec.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    sort_indexes(prob_vec, idx);            // sort indeces based on prob
                    std::vector<unsigned int>    labels;
                    outputQ.enqueue(std::tuple<int,int>(tag,idx[0]));
                    int j=0;
                    for (auto i: idx) {
                        // make label which is index and prob
                        int packed_label_prob = (i&0xFFFF)|(((unsigned int)((prob_vec[i]*0x7FFF)+0.5))<<16);   // convert prob to 16bit float and store in MSBs
                        labels.push_back(packed_label_prob);
                        if (++j >= topK) break;
                    }
                    outputQTopk.enqueue(labels);
                }
            }else
            {
                std::vector<ObjectBB> detected_objects;
                region->GetObjectDetections((float *)buf, BB_biases, dimOutput[2], dimOutput[1], dimOutput[0], BOUNDING_BOX_NUMBER_OF_CLASSES, dimInput[0], dimInput[1], BOUNDING_BOX_CONFIDENCE_THRESHHOLD, BOUNDING_BOX_NMS_THRESHHOLD, 13, detected_objects);
                if (detected_objects.size() > 0) {
                    // add it to outputQ
                    outputQ.enqueue(std::tuple<int,int>(tag,detected_objects[0].label));
                    // add detected objects with BB into BoundingBox Q
                    OutputQBB.enqueue(detected_objects);
                } else
                {
                    // add it to outputQ
                    outputQ.enqueue(std::tuple<int,int>(tag,-1));
                }
            }
        }

        // unlock the OpenCL buffer to perform the writing
        err = clEnqueueUnmapMemObject(cmdq, mem, mapped_ptr, 0, NULL, NULL);
        if(err) {
            fatal("workDeviceOutputCopy: clEnqueueMapBuffer(#%d) failed (%d)", gpu, err);
        }
        err = clFinish(cmdq);
        if(err) {
            fatal("workDeviceOutputCopy: clFinish(#%d) failed (%d)", gpu, err);
        }

        // add the output back to idle queue
        queueDeviceOutputMemIdle[gpu]->enqueue(mem);
        PROFILER_STOP(inference_server_app, workDeviceOutputCopy);

        // update counter
        if(outputCount > 0) {
            totalBatchCounter++;
            totalImageCounter += outputCount;
        }
    }

    // release OpenCL command queue
    clReleaseCommandQueue(cmdq);

    // send end of sequence marker to next stage
    outputQ.enqueue(std::tuple<int,int>(-1,-1));
    args->lock();
    info("workDeviceOutputCopy: GPU#%d terminated for %s [processed %d batches, %d images]", gpu, clientName.c_str(), totalBatchCounter, totalImageCounter);
    args->unlock();
#endif    
}
#endif

#if ENABLE_OPENCL
void InferenceEngine::dumpBuffer(cl_command_queue cmdq, cl_mem mem, std::string fileName)
{
    cl_int err;
    size_t size = 0;
    err = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size), &size, NULL);
    if(err) return;
    unsigned char * ptr = (unsigned char *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &err);
    if(err) return;
    err = clFinish(cmdq);
    if(err) return;
    FILE * fp = fopen(fileName.c_str(), "wb");
    if(err) return;
    fwrite(ptr, 1, size, fp);
    fclose(fp);
    err = clEnqueueUnmapMemObject(cmdq, mem, ptr, 0, NULL, NULL);
    if(err) return;
    err = clFinish(cmdq);
    if(err) return;
    printf("OK: dumped %ld bytes into %s\n", size, fileName.c_str());
}
#endif    
