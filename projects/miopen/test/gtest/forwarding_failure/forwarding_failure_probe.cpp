// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Makes the MIOpen calls named on its command line, in order, and prints one
// "step=result" line per call to stdout. It links only the public library, so it
// sees what an application sees. Every switch the forwarding failure tests vary
// is read once per process, which is why script/test_forwarding_failure_modes.py
// runs this once per environment instead of testing in-process.
//
//  conv_forward        miopenConvolutionForward; prints status
//  error_string        miopenGetErrorString of last conv_forward status
//  last_forwarded      status miopenGetLastForwardedError reports
//  last_forwarded_null miopenGetLastForwardedError(NULL); prints returned status
//  get_version         miopenGetVersion, which MIOpen always serves; prints status
//  destroy_after_forward
//                      miopenCreate, then conv_forward on that handle, then
//                      miopenDestroy; prints all three statuses. The forwarded call
//                      gives the handle a hipDNN handle for miopenDestroy to
//                      destroy. Needs a GPU, so stops after printing a failed
//                      miopenCreate.
//  destroy_after_forward_on_stream
//                      destroy_after_forward on a handle made by
//                      miopenCreateWithStream on a new HIP stream, so the forwarded
//                      call also sets that stream on the hipDNN handle. Prints the
//                      hipStreamCreate result as stream_create first, and stops
//                      after it if it failed.
//  exit_after_forward  destroy_after_forward without the miopenDestroy, so the
//                      handle and its hipDNN handle are still alive at exit
//  batchnorm          batchnorm forward training, forward inference with
//                      estimates, and backward; prints bn_train, bn_infer and
//                      bn_backward statuses
//  repeat N            runs the previous step N more times

#include <miopen/hipdnn_forwarding.h>
#include <miopen/miopen.h>

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct ConvProblem
{
    miopenTensorDescriptor_t x         = nullptr;
    miopenTensorDescriptor_t w         = nullptr;
    miopenTensorDescriptor_t y         = nullptr;
    miopenConvolutionDescriptor_t conv = nullptr;
};

bool Check(miopenStatus_t status, const char* what)
{
    if(status == miopenStatusSuccess)
        return true;
    std::fprintf(
        stderr, "forwarding_failure_probe: %s failed with %d\n", what, static_cast<int>(status));
    return false;
}

bool MakeProblem(ConvProblem& p)
{
    return Check(miopenCreateTensorDescriptor(&p.x), "create x") &&
           Check(miopenCreateTensorDescriptor(&p.w), "create w") &&
           Check(miopenCreateTensorDescriptor(&p.y), "create y") &&
           Check(miopenCreateConvolutionDescriptor(&p.conv), "create conv") &&
           Check(miopenSet4dTensorDescriptor(p.x, miopenFloat, 1, 1, 4, 4), "set x") &&
           Check(miopenSet4dTensorDescriptor(p.w, miopenFloat, 1, 1, 3, 3), "set w") &&
           Check(miopenSet4dTensorDescriptor(p.y, miopenFloat, 1, 1, 2, 2), "set y") &&
           Check(miopenInitConvolutionDescriptor(p.conv, miopenConvolution, 0, 0, 1, 1, 1, 1),
                 "init conv");
}

void FreeProblem(ConvProblem& p)
{
    if(p.conv != nullptr)
        miopenDestroyConvolutionDescriptor(p.conv);
    for(miopenTensorDescriptor_t desc : {p.x, p.w, p.y})
        if(desc != nullptr)
            miopenDestroyTensorDescriptor(desc);
}

// The descriptors are real so the MIOpen route gets far enough to reject a
// null handle with a status: it reads the convolution descriptor before its
// error handling starts, so a null one would crash. The hipDNN route, with
// hipDNN unavailable, fails before reading any argument. Neither needs a GPU.
miopenStatus_t ConvForward(const ConvProblem& p, miopenHandle_t handle = nullptr)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    return miopenConvolutionForward(handle,
                                    &alpha,
                                    p.x,
                                    nullptr,
                                    p.w,
                                    nullptr,
                                    p.conv,
                                    miopenConvolutionFwdAlgoGEMM,
                                    &beta,
                                    p.y,
                                    nullptr,
                                    nullptr,
                                    0);
}

void ExitAfterForward(const ConvProblem& p)
{
    miopenHandle_t handle        = nullptr;
    const miopenStatus_t created = miopenCreate(&handle);
    std::printf("create=%d\n", static_cast<int>(created));
    if(created == miopenStatusSuccess)
        std::printf("forward_with_handle=%d\n", static_cast<int>(ConvForward(p, handle)));
    std::fflush(stdout);
}

void DestroyAfterForward(const ConvProblem& p, bool onStream)
{
    hipStream_t stream = nullptr;
    if(onStream)
    {
        const hipError_t streamCreated = hipStreamCreate(&stream);
        std::printf("stream_create=%d\n", static_cast<int>(streamCreated));
        if(streamCreated != hipSuccess)
            return;
    }
    miopenHandle_t handle = nullptr;
    const miopenStatus_t created =
        onStream ? miopenCreateWithStream(&handle, stream) : miopenCreate(&handle);
    std::printf("create=%d\n", static_cast<int>(created));
    if(created == miopenStatusSuccess)
    {
        // Data pointers are never read: the forwarded call fails before it runs
        // anything, after it has tried to give the handle its hipDNN state.
        std::printf("forward_with_handle=%d\n", static_cast<int>(ConvForward(p, handle)));
        std::fflush(stdout);
        std::printf("destroy=%d\n", static_cast<int>(miopenDestroy(handle)));
    }
    if(stream != nullptr)
        static_cast<void>(hipStreamDestroy(stream));
}

// x's descriptor serves every activation tensor, a 1x1x1x1 one every per-channel
// one. As for ConvForward, neither route reads the data pointers: the MIOpen
// route rejects the null handle, and the hipDNN route, with hipDNN unavailable,
// fails first. Inference gets non-null estimates, as without them it runs
// training instead.
bool BatchNorm(const ConvProblem& p)
{
    miopenTensorDescriptor_t stats = nullptr;
    if(!Check(miopenCreateTensorDescriptor(&stats), "create stats") ||
       !Check(miopenSet4dTensorDescriptor(stats, miopenFloat, 1, 1, 1, 1), "set stats"))
    {
        if(stats != nullptr)
            miopenDestroyTensorDescriptor(stats);
        return false;
    }
    float alpha    = 1.0f;
    float beta     = 0.0f;
    float estimate = 0.0f;

    const miopenStatus_t train = miopenBatchNormalizationForwardTraining(nullptr,
                                                                         miopenBNSpatial,
                                                                         &alpha,
                                                                         &beta,
                                                                         p.x,
                                                                         nullptr,
                                                                         p.x,
                                                                         nullptr,
                                                                         stats,
                                                                         nullptr,
                                                                         nullptr,
                                                                         0.0,
                                                                         nullptr,
                                                                         nullptr,
                                                                         1e-5,
                                                                         nullptr,
                                                                         nullptr);
    std::printf("bn_train=%d\n", static_cast<int>(train));
    const miopenStatus_t infer = miopenBatchNormalizationForwardInference(nullptr,
                                                                          miopenBNSpatial,
                                                                          &alpha,
                                                                          &beta,
                                                                          p.x,
                                                                          nullptr,
                                                                          p.x,
                                                                          nullptr,
                                                                          stats,
                                                                          nullptr,
                                                                          nullptr,
                                                                          &estimate,
                                                                          &estimate,
                                                                          1e-5);
    std::printf("bn_infer=%d\n", static_cast<int>(infer));
    const miopenStatus_t backward = miopenBatchNormalizationBackward(nullptr,
                                                                     miopenBNSpatial,
                                                                     &alpha,
                                                                     &beta,
                                                                     &alpha,
                                                                     &beta,
                                                                     p.x,
                                                                     nullptr,
                                                                     p.x,
                                                                     nullptr,
                                                                     p.x,
                                                                     nullptr,
                                                                     stats,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     1e-5,
                                                                     nullptr,
                                                                     nullptr);
    std::printf("bn_backward=%d\n", static_cast<int>(backward));
    miopenDestroyTensorDescriptor(stats);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    ConvProblem problem;
    bool haveProblem              = false;
    miopenStatus_t lastConvStatus = miopenStatusSuccess;
    std::string previous;
    int exitCode = 0;

    const auto ensureProblem = [&] {
        if(!haveProblem)
            haveProblem = MakeProblem(problem);
        return haveProblem;
    };

    const auto run = [&](const std::string& step) {
        if(step == "conv_forward")
        {
            if(!ensureProblem())
                return false;
            lastConvStatus = ConvForward(problem);
            std::printf("conv_forward=%d\n", static_cast<int>(lastConvStatus));
        }
        else if(step == "destroy_after_forward" || step == "destroy_after_forward_on_stream")
        {
            if(!ensureProblem())
                return false;
            DestroyAfterForward(problem, step == "destroy_after_forward_on_stream");
        }
        else if(step == "exit_after_forward")
        {
            if(!ensureProblem())
                return false;
            ExitAfterForward(problem);
        }
        else if(step == "batchnorm")
        {
            if(!ensureProblem() || !BatchNorm(problem))
                return false;
        }
        else if(step == "error_string")
        {
            std::printf("error_string=%s\n", miopenGetErrorString(lastConvStatus));
        }
        else if(step == "last_forwarded")
        {
            miopenStatus_t forwarded = miopenStatusUnknownError;
            if(!Check(miopenGetLastForwardedError(&forwarded), "miopenGetLastForwardedError"))
                return false;
            std::printf("last_forwarded=%d\n", static_cast<int>(forwarded));
        }
        else if(step == "last_forwarded_null")
        {
            std::printf("last_forwarded_null=%d\n",
                        static_cast<int>(miopenGetLastForwardedError(nullptr)));
        }
        else if(step == "get_version")
        {
            size_t major = 0;
            size_t minor = 0;
            size_t patch = 0;
            std::printf("get_version=%d\n",
                        static_cast<int>(miopenGetVersion(&major, &minor, &patch)));
        }
        else
        {
            std::fprintf(stderr, "forwarding_failure_probe: unknown step '%s'\n", step.c_str());
            return false;
        }
        return true;
    };

    for(int i = 1; i < argc && exitCode == 0; ++i)
    {
        if(std::strcmp(argv[i], "repeat") == 0)
        {
            if(i + 1 == argc || previous.empty())
            {
                std::fprintf(
                    stderr,
                    "forwarding_failure_probe: repeat needs a count and a step before it\n");
                exitCode = 2;
                break;
            }
            const int count = std::atoi(argv[++i]);
            for(int n = 0; n < count && exitCode == 0; ++n)
                if(!run(previous))
                    exitCode = 2;
            continue;
        }
        previous = argv[i];
        if(!run(previous))
            exitCode = 2;
    }

    FreeProblem(problem);
    return exitCode;
}
