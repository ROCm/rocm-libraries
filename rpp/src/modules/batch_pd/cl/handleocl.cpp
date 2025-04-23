/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <chrono>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <thread>
#include "ocldeviceinfo.hpp"
#include "load_file.hpp"
#include "device_name.hpp"
#include "errors.hpp"
#include "logger.hpp"
#include "handle.hpp"
#include "kernel_cache.hpp"
#include "binary_cache.hpp"

namespace rpp {

#ifndef NDEBUG
void dumpKernel(cl_kernel kern,
                const std::string& kernel_name,
                const std::vector<size_t>& vld,
                const std::vector<size_t>& vgd,
                const std::string& params)
{
    static int dumpOpenCLFileCounter = 0;
    static std::vector<cl_kernel> kernList;
    for(auto it = kernList.begin(); it != kernList.end(); it++)
        if(*it == kern)
            return;
    kernList.push_back(kern);
    std::string work;
    for(size_t i = 0; i < vgd.size(); i++)
    {
        if(i)
            work += ",";
        work += std::to_string(vgd[i]);
    }
    for(size_t i = 0; i < vld.size(); i++)
    {
        work += i ? "," : "/";
        work += std::to_string(vld[i]);
    }
    auto getValueFromParams = [&](const std::string& par, int& value, const char* define) {
        const char* q = strstr(par.c_str(), define);
        if(q)
            value = atoi(q + strlen(define));
    };
    int an = 0, ac = 0, ah = 0, aw = 0, ax = 0, ay = 0, ak = 0, ap = 0, aq = 0, au = 1, av = 1,
        aP = 0, aQ = 0, af = 1;
    getValueFromParams(params, an, "-D MLO_BATCH_SZ=");
    getValueFromParams(params, ac, "-D MLO_N_INPUTS=");
    getValueFromParams(params, ac, "-D MLO_N_IN_CHNLS=");
    getValueFromParams(params, ah, "-D MLO_IN_HEIGHT=");
    getValueFromParams(params, aw, "-D MLO_IN_WIDTH=");
    getValueFromParams(params, ak, "-D MLO_N_OUTPUTS=");
    getValueFromParams(params, ak, "-D MLO_N_OUT_CHNLS=");
    getValueFromParams(params, aP, "-D MLO_OUT_HEIGHT=");
    getValueFromParams(params, aQ, "-D MLO_OUT_WIDTH=");
    getValueFromParams(params, ay, "-D MLO_FILTER_SIZE1=");
    getValueFromParams(params, ax, "-D MLO_FILTER_SIZE0=");
    getValueFromParams(params, ap, "-D MLO_FILTER_PAD1=");
    getValueFromParams(params, aq, "-D MLO_FILTER_PAD0=");
    getValueFromParams(params, av, "-D MLO_FILTER_STRIDE1=");
    getValueFromParams(params, au, "-D MLO_FILTER_STRIDE0=");
    getValueFromParams(params, ay, "-D MLO_FLTR_SZ1=");
    getValueFromParams(params, ax, "-D MLO_FLTR_SZ0=");
    getValueFromParams(params, ap, "-D MLO_FLTR_PAD_SZ1=");
    getValueFromParams(params, aq, "-D MLO_FLTR_PAD_SZ0=");
    getValueFromParams(params, av, "-D MLO_FLTR_STRIDE1=");
    getValueFromParams(params, au, "-D MLO_FLTR_STRIDE0=");
    getValueFromParams(params, af, "-D MLO_DIR_FORWARD=");
    int isize = an * ac * ah * aw * 4;
    int osize = an * ak * aP * aQ * 4;
    int wsize = ak * ac * ay * ax * 4;
    if(!isize || !osize || !wsize)
    {
        if(params.size() > 0)
            printf("dumpKernel: can't dump kernel %s missing macros in params: %s\n",
                   kernel_name.c_str(),
                   params.c_str());
        return;
    }
    dumpOpenCLFileCounter++;
    cl_program prog = nullptr;
    clGetKernelInfo(kern, CL_KERNEL_PROGRAM, sizeof(prog), &prog, nullptr);
    cl_uint num_arg = 0;
    clGetKernelInfo(kern, CL_KERNEL_NUM_ARGS, sizeof(num_arg), &num_arg, nullptr);
    size_t sizeK = 0;
    clGetProgramInfo(prog, CL_PROGRAM_SOURCE, 0, nullptr, &sizeK);
    std::vector<char> bufK(sizeK + 1);
    char* buf   = bufK.data();
    size_t size = 0;
    clGetProgramInfo(prog, CL_PROGRAM_SOURCE, sizeK, buf, &size);
    buf[size] = 0;
    char fileName[1024];
    FILE* fp;
    snprintf(fileName, sizeof(fileName), "dump_%03d_command.txt", dumpOpenCLFileCounter);
    fp = fopen(fileName, "w");
    if(!fp)
    {
        printf("ERROR: unable to create: %s\n", fileName);
    }
    else
    {
        if(af)
        {
            fprintf(fp,
                    "execkern -bo -cl-std=CL2.0 dump_%03d_kernel.cl -k %s if#%d:dump_fwd_in.bin "
                    "if#%d:dump_fwd_wei.bin of#%d:#intmp.bin#/+1e%d/dump_fwd_out_cpu.bin %s %s -- "
                    "comment -n %d -c %d -H %d -W %d -x %d -y %d -k %d -p %d -q %d -u %d -v %d -- "
                    "P %d Q %d",
                    dumpOpenCLFileCounter,
                    kernel_name.c_str(),
                    isize,
                    wsize,
                    osize,
                    af ? -6 : -9,
                    num_arg > 3 ? "iv#0 " : "",
                    work.c_str(),
                    an,
                    ac,
                    ah,
                    aw,
                    ax,
                    ay,
                    ak,
                    ap,
                    aq,
                    au,
                    av,
                    aP,
                    aQ);
        }
        else
        {
            fprintf(fp,
                    "execkern -bo -cl-std=CL2.0 dump_%03d_kernel.cl -k %s if#%d:dump_bwd_out.bin "
                    "if#%d:dump_bwd_wei.bin of#%d:#outtmp.bin#/+1e%d/dump_bwd_in_cpu.bin %s %s -- "
                    "comment -n %d -c %d -H %d -W %d -x %d -y %d -k %d -p %d -q %d -u %d -v %d -- "
                    "P %d Q %d",
                    dumpOpenCLFileCounter,
                    kernel_name.c_str(),
                    isize,
                    wsize,
                    osize,
                    af ? -6 : -9,
                    num_arg > 3 ? "iv#0 " : "",
                    work.c_str(),
                    an,
                    ac,
                    ah,
                    aw,
                    ax,
                    ay,
                    ak,
                    ap,
                    aq,
                    au,
                    av,
                    aP,
                    aQ);
        }
        fclose(fp);
        printf("*** OpenCL kernel %s command dumped into %s with work %s\n",
               kernel_name.c_str(),
               fileName,
               work.c_str());
    }
    snprintf(fileName, sizeof(fileName), "dump_%03d_kernel.cl", dumpOpenCLFileCounter);
    fp = fopen(fileName, "w");
    if(!fp)
    {
        printf("ERROR: unable to create: %s\n", fileName);
    }
    else
    {
        const char* s = params.c_str();
        fprintf(fp, "//[compiler-options] %s\n", s);
        for(const char* t = s; (t = strstr(t, "-D")) != nullptr;)
        {
            t += 2;
            while(*t && (*t == ' ' || *t == '\t'))
                t++;
            fprintf(fp, "#define ");
            while(*t && *t != ' ' && *t != '\t' && *t != '=')
                fprintf(fp, "%c", *t++);
            if(*t == '=')
            {
                fprintf(fp, " ");
                t++;
                while(*t && *t != ' ' && *t != '\t')
                    fprintf(fp, "%c", *t++);
            }
            fprintf(fp, "\n");
        }
        for(const char* p = buf; *p; p++)
            if(*p != '\r')
                fprintf(fp, "%c", *p);
        fclose(fp);
        printf("*** OpenCL kernel %s source dumped into %s with work %s\n",
               kernel_name.c_str(),
               fileName,
               work.c_str());
    }
}
#endif // NDEBUG

void* default_allocator(void* context, size_t sz)
{
    assert(context != nullptr);
    cl_int status = CL_SUCCESS;
    auto result   = clCreateBuffer(
        reinterpret_cast<cl_context>(context), CL_MEM_READ_ONLY, sz, nullptr, &status);
    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status, "OpenCL error creating buffer: " + std::to_string(sz));
    }
    return result;
}

void default_deallocator(void*, void* mem) { clReleaseMemObject(DataCast(mem)); }

struct HandleImpl
{
    using AqPtr = rpp::manage_ptr<typename std::remove_pointer<rppAcceleratorQueue_t>::type, decltype(&clReleaseCommandQueue), &clReleaseCommandQueue>;
    using ContextPtr = rpp::manage_ptr<typename std::remove_pointer<cl_context>::type, decltype(&clReleaseContext), &clReleaseContext>;

    ContextPtr context = nullptr;
    AqPtr queue = nullptr;
    cl_device_id device = nullptr; // NOLINT
    Allocator allocator{};
    KernelCache cache;
    bool enable_profiling = false;
    float profiling_result = 0.0;
    size_t nBatchSize = 1;
    Rpp32u numThreads = 0;
    InitHandle* initHandle = nullptr;

    ContextPtr create_context()
    {
        cl_uint numPlatforms;
        cl_platform_id platform = nullptr;
        if(clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS)
        {
            RPP_THROW("clGetPlatformIDs failed. " + std::to_string(numPlatforms));
        }
        if(0 < numPlatforms)
        {
            std::vector<cl_platform_id> platforms(numPlatforms);
            if(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) != CL_SUCCESS)
            {
                RPP_THROW("clGetPlatformIDs failed.2");
            }
            for(cl_uint i = 0; i < numPlatforms; ++i)
            {
                char pbuf[100];

                if(clGetPlatformInfo(
                       platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, nullptr) != CL_SUCCESS)
                {
                    RPP_THROW("clGetPlatformInfo failed.");
                }

                platform = platforms[i];
                if(strcmp(pbuf, "Advanced Micro Devices, Inc.") == 0)
                {
                    break;
                }
            }
        }

        // Create an OpenCL context
        cl_int status                = 0;
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
        cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;
        ContextPtr result{
            clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status)};
        if(status != CL_SUCCESS)
        {
            RPP_THROW_CL_STATUS(status, "Error: Creating Handle. (clCreateContextFromType)");
        }
        return result;
    }

    ContextPtr create_context_from_queue()
    {
        // FIXME: hack for all the queues on the same context
        // do we need anything special to handle multiple GPUs
        cl_context ctx;
        cl_int status = 0;
        status =
            clGetCommandQueueInfo(queue.get(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
        if(status != CL_SUCCESS)
        {
            RPP_THROW_CL_STATUS(status,
                                   "Error: Creating Handle. Cannot Initialize Handle from Queue");
        }
        clRetainContext(ctx);
        return ContextPtr{ctx};
    }

    void ResetProfilingResult()
    {
        profiling_result = 0.0;
    }

    void AccumProfilingResult(float curr_res)
    {
        profiling_result += curr_res;
    }

    void SetProfilingResult(cl_event& e)
    {
        if(this->enable_profiling)
        {
            size_t st, end;
            clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(size_t), &st, nullptr);
            clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(size_t), &end, nullptr);
            profiling_result = static_cast<float>(end - st) * 1.0e-6; // NOLINT
        }
    }

    void PreInitializeBufferCPU()
    {
        this->initHandle = new InitHandle();

        this->initHandle->nbatchSize = this->nBatchSize;
        this->initHandle->mem.mcpu.srcSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.dstSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.maxSrcSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.maxDstSize = (RppiSize *)malloc(sizeof(RppiSize) * this->nBatchSize);
        this->initHandle->mem.mcpu.roiPoints = (RppiROI *)malloc(sizeof(RppiROI) * this->nBatchSize);
        this->initHandle->mem.mcpu.srcBatchIndex = (Rpp64u *)malloc(sizeof(Rpp64u) * this->nBatchSize);
        this->initHandle->mem.mcpu.dstBatchIndex = (Rpp64u *)malloc(sizeof(Rpp64u) * this->nBatchSize);
        this->initHandle->mem.mcpu.inc = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mcpu.dstInc = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);

        for(int i = 0; i < 10; i++)
        {
            this->initHandle->mem.mcpu.floatArr[i].floatmem = (Rpp32f *)malloc(sizeof(Rpp32f) * this->nBatchSize);
            this->initHandle->mem.mcpu.uintArr[i].uintmem = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
            this->initHandle->mem.mcpu.intArr[i].intmem = (Rpp32s *)malloc(sizeof(Rpp32s) * this->nBatchSize);
            this->initHandle->mem.mcpu.ucharArr[i].ucharmem = (Rpp8u *)malloc(sizeof(Rpp8u) * this->nBatchSize);
            this->initHandle->mem.mcpu.charArr[i].charmem = (Rpp8s *)malloc(sizeof(Rpp8s) * this->nBatchSize);
        }
        this->initHandle->mem.mcpu.scratchBufferHost = (Rpp32f *)malloc(sizeof(Rpp32f) * 99532800 * this->nBatchSize); // 7680 * 4320 * 3
    }

    void PreInitializeBuffer()
    {
        this->initHandle = new InitHandle();
        this->PreInitializeBufferCPU();

        cl_int err;
        auto ctx = this->context.get();
        this->initHandle->mem.mgpu.csrcSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.csrcSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cdstSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cdstSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxSrcSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxSrcSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxDstSize.height = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.cmaxDstSize.width = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.x = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.y = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.roiHeight = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.croiPoints.roiWidth = (Rpp32u *)malloc(sizeof(Rpp32u) * this->nBatchSize);
        this->initHandle->mem.mgpu.srcSize.height = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.srcSize.width = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.dstSize.height = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.dstSize.width = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.maxSrcSize.height = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.maxSrcSize.width = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.maxDstSize.height = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.maxDstSize.width = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.roiPoints.x = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.roiPoints.y = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.roiPoints.roiHeight = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.roiPoints.roiWidth = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.inc = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.dstInc = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);

        this->initHandle->mem.mgpu.srcBatchIndex = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp64u) * this->nBatchSize, NULL, &err);
        this->initHandle->mem.mgpu.dstBatchIndex = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp64u) * this->nBatchSize, NULL, &err);

        for(int i = 0; i < 10; i++)
        {
            this->initHandle->mem.mgpu.floatArr[i].floatmem = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32f) * this->nBatchSize, NULL, &err);
            this->initHandle->mem.mgpu.uintArr[i].uintmem = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32u) * this->nBatchSize, NULL, &err);
            this->initHandle->mem.mgpu.intArr[i].intmem = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp32s) * this->nBatchSize, NULL, &err);
            this->initHandle->mem.mgpu.ucharArr[i].ucharmem = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp8u) * this->nBatchSize, NULL, &err);
            this->initHandle->mem.mgpu.charArr[i].charmem = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(Rpp8s) * this->nBatchSize, NULL, &err);
        }
    }
};

Handle::Handle(size_t batchSize, rppAcceleratorQueue_t stream) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    clRetainCommandQueue(stream);
    impl->queue   = HandleImpl::AqPtr{stream};
    impl->context = impl->create_context_from_queue();

    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBuffer();
}

Handle::Handle(size_t batchSize, Rpp32u numThreads) : impl(new HandleImpl())
{
    impl->nBatchSize = batchSize;
    numThreads = std::min(numThreads, std::thread::hardware_concurrency());
    if(numThreads == 0)
        numThreads = batchSize;
    impl->numThreads = numThreads;
    this->SetAllocator(nullptr, nullptr, nullptr);
    impl->PreInitializeBufferCPU();
}

Handle::~Handle()                 = default;

void Handle::SetStream(rppAcceleratorQueue_t streamID) const
{
    if(streamID == nullptr)
    {
        RPP_THROW("Error setting stream to nullptr");
    }

    clRetainCommandQueue(streamID);
    impl->queue = HandleImpl::AqPtr{streamID};
}

void Handle::rpp_destroy_object_gpu()
{
    this->rpp_destroy_object_host();

    free(this->GetInitHandle()->mem.mgpu.csrcSize.height);
    free(this->GetInitHandle()->mem.mgpu.csrcSize.width);
    free(this->GetInitHandle()->mem.mgpu.cdstSize.height);
    free(this->GetInitHandle()->mem.mgpu.cdstSize.width);
    free(this->GetInitHandle()->mem.mgpu.cmaxSrcSize.height);
    free(this->GetInitHandle()->mem.mgpu.cmaxSrcSize.width);
    free(this->GetInitHandle()->mem.mgpu.cmaxDstSize.height);
    free(this->GetInitHandle()->mem.mgpu.cmaxDstSize.width);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.x);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.y);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.roiHeight);
    free(this->GetInitHandle()->mem.mgpu.croiPoints.roiWidth);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.srcSize.height);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.srcSize.width);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.dstSize.height);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.dstSize.width);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.maxSrcSize.height);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.maxSrcSize.width);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.maxDstSize.height);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.maxDstSize.width);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.roiPoints.x);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.roiPoints.y);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.roiPoints.roiHeight);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.roiPoints.roiWidth);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.inc);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.dstInc);

    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.srcBatchIndex);
    clReleaseMemObject(this->GetInitHandle()->mem.mgpu.dstBatchIndex);

    for(int i = 0; i < 10; i++)
    {
        clReleaseMemObject(this->GetInitHandle()->mem.mgpu.floatArr[i].floatmem);
        clReleaseMemObject(this->GetInitHandle()->mem.mgpu.uintArr[i].uintmem);
        clReleaseMemObject(this->GetInitHandle()->mem.mgpu.intArr[i].intmem);
        clReleaseMemObject(this->GetInitHandle()->mem.mgpu.ucharArr[i].ucharmem);
        clReleaseMemObject(this->GetInitHandle()->mem.mgpu.charArr[i].charmem);
    }
}

void Handle::rpp_destroy_object_host()
{
    free(this->GetInitHandle()->mem.mcpu.srcSize);
    free(this->GetInitHandle()->mem.mcpu.dstSize);
    free(this->GetInitHandle()->mem.mcpu.maxSrcSize);
    free(this->GetInitHandle()->mem.mcpu.maxDstSize);
    free(this->GetInitHandle()->mem.mcpu.roiPoints);
    free(this->GetInitHandle()->mem.mcpu.srcBatchIndex);
    free(this->GetInitHandle()->mem.mcpu.dstBatchIndex);
    free(this->GetInitHandle()->mem.mcpu.inc);
    free(this->GetInitHandle()->mem.mcpu.dstInc);

    for(int i = 0; i < 10; i++)
    {
        free(this->GetInitHandle()->mem.mcpu.floatArr[i].floatmem);
        free(this->GetInitHandle()->mem.mcpu.uintArr[i].uintmem);
        free(this->GetInitHandle()->mem.mcpu.intArr[i].intmem);
        free(this->GetInitHandle()->mem.mcpu.ucharArr[i].ucharmem);
        free(this->GetInitHandle()->mem.mcpu.charArr[i].charmem);
    }

    free(this->GetInitHandle()->mem.mcpu.scratchBufferHost);
}

size_t Handle::GetBatchSize() const
{
    return this->impl->nBatchSize;
}

Rpp32u Handle::GetNumThreads() const
{
    return this->impl->numThreads;
}

void Handle::SetBatchSize(size_t bSize) const
{
    this->impl->nBatchSize = bSize;
}

rppAcceleratorQueue_t Handle::GetStream() const
{
    return impl->queue.get();
}

InitHandle* Handle::GetInitHandle() const
{
    return impl->initHandle;
}

void Handle::SetAllocator(rppAllocatorFunction allocator,
                          rppDeallocatorFunction deallocator,
                          void* allocatorContext) const
{
    if(allocator == nullptr && allocatorContext != nullptr)
    {
        RPP_THROW("Allocator context can not be used with the default allocator");
    }
    this->impl->allocator.allocator   = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;
    this->impl->allocator.context = allocatorContext == nullptr ? this->impl->context.get() : allocatorContext;
}

void Handle::EnableProfiling(bool enable)
{
    this->impl->enable_profiling = enable;
}

void Handle::ResetKernelTime()
{
    this->impl->ResetProfilingResult();
}

void Handle::AccumKernelTime(float curr_time)
{
    this->impl->AccumProfilingResult(curr_time);
}

float Handle::GetKernelTime() const
{
    return this->impl->profiling_result;
}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index,
                               bool is_kernel_str,
                               const std::string& kernel_src)
{

    auto obj = this->impl->cache.AddKernel(*this,
                                           algorithm,
                                           network_config,
                                           program_name,
                                           kernel_name,
                                           vld,
                                           vgd,
                                           params,
                                           cache_index,
                                           is_kernel_str,
                                           kernel_src);
    return this->Run(obj);
}

bool Handle::HasKernel(const std::string& algorithm, const std::string& network_config) const
{
    return this->impl->cache.HasKernels(algorithm, network_config);
}

void Handle::ClearKernels(const std::string& algorithm, const std::string& network_config)
{
    this->impl->cache.ClearKernels(algorithm, network_config);
}

const std::vector<Kernel>& Handle::GetKernelsImpl(const std::string& algorithm, const std::string& network_config)
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel k)
{
    auto q = this->GetStream();
    if(this->impl->enable_profiling)
    {
        return k.Invoke(q,
                        std::bind(&HandleImpl::SetProfilingResult,
                        std::ref(*this->impl),
                        std::placeholders::_1));
    }
    else
    {
        return k.Invoke(q);
    }
}

auto Handle::GetKernels(const std::string& algorithm, const std::string& network_config)
{
    auto kernels = this->GetKernelsImpl(algorithm, network_config);

    std::vector<KernelInvoke> transformedKernels;

    transformedKernels.reserve(kernels.size());

    std::transform(kernels.begin(), kernels.end(), std::back_inserter(transformedKernels),
                   [this](Kernel k) { return this->Run(k); });

    return transformedKernels;
}

KernelInvoke Handle::GetKernel(const std::string& algorithm, const std::string& network_config)
{
    auto ks = this->GetKernelsImpl(algorithm, network_config);
    if(ks.empty())
    {
        RPP_THROW("looking for default kernel (does not exist): " + algorithm + ", " +
                        network_config);
    }
    return this->Run(ks.front());
}

Program Handle::LoadProgram(const std::string& program_name, std::string params, bool is_kernel_str, const std::string& kernel_src)
{
    auto cache_file = rpp::LoadBinary(this->GetDeviceName(), program_name, params, is_kernel_str);
    if(cache_file.empty())
    {
        auto p = rpp::LoadProgram(rpp::GetContext(this->GetStream()),
                                  rpp::GetDevice(this->GetStream()),
                                  program_name,
                                  params,
                                  is_kernel_str,
                                  kernel_src);

        // Save to cache
        auto path = rpp::GetCachePath() / std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        rpp::SaveProgramBinary(p, path.string());
        rpp::SaveBinary(path.string(), this->GetDeviceName(), program_name, params, is_kernel_str);

        return std::move(p);
    }
    else
    {
        return LoadBinaryProgram(rpp::GetContext(this->GetStream()),
                                 rpp::GetDevice(this->GetStream()),
                                 rpp::LoadFile(cache_file));
    }
}

void Handle::Finish() const
{
    clFinish(this->GetStream());
}

void Handle::Flush() const
{
    clFlush(this->GetStream());
}

bool Handle::IsProfilingEnabled() const
{
    return this->impl->enable_profiling;
}

std::size_t Handle::GetLocalMemorySize()
{
    return rpp::GetDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE>(rpp::GetDevice(this->GetStream()));
}

std::size_t Handle::GetGlobalMemorySize()
{
    return rpp::GetDeviceInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(rpp::GetDevice(this->GetStream()));
}

std::string Handle::GetDeviceName()
{
    std::string name = rpp::GetDeviceInfo<CL_DEVICE_NAME>(rpp::GetDevice(this->GetStream()));
    ParseDevName(name);
    return GetDeviceNameFromMap(name);
}

std::ostream& Handle::Print(std::ostream& os) const
{
    return os;
}

std::size_t Handle::GetMaxMemoryAllocSize()
{
    if(m_MaxMemoryAllocSizeCached == 0)
        m_MaxMemoryAllocSizeCached = rpp::GetDeviceInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(rpp::GetDevice(this->GetStream()));
    return m_MaxMemoryAllocSizeCached;
}

std::size_t Handle::GetMaxComputeUnits()
{
    return rpp::GetDeviceInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(rpp::GetDevice(this->GetStream()));
}

Allocator::ManageDataPtr Handle::Create(std::size_t sz)
{
    this->Finish();
    return this->impl->allocator(sz);
}

Allocator::ManageDataPtr& Handle::WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    cl_int status = clEnqueueWriteBuffer(this->GetStream(), ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status, "OpenCL error writing to buffer: " + std::to_string(sz));
    }
    return ddata;
}

void Handle::ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = clEnqueueReadBuffer(this->GetStream(), ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status, "OpenCL error reading from buffer: " + std::to_string(sz));
    }
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size)
{
    this->Finish();
    auto status = clEnqueueCopyBuffer(this->GetStream(), src, dest, 0, 0, size, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        RPP_THROW_CL_STATUS(status, "OpenCL error copying buffer: " + std::to_string(size));
    }
}

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size)
{
    struct region
    {
        std::size_t origin;
        std::size_t size;
    };
    cl_int error = 0;
    auto r = region{offset, size};
    auto mem = clCreateSubBuffer(data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &error);
    return {mem, manage_deleter<decltype(&clReleaseMemObject), &clReleaseMemObject>{}};
}

} // namespace rpp
