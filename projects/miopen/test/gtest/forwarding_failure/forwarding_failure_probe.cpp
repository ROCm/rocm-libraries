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
//  repeat N            runs the previous step N more times

#include <miopen/hipdnn_forwarding.h>
#include <miopen/miopen.h>

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

// The descriptors are real so the MIOpen route gets far enough to reject the
// null handle with a status: it reads the convolution descriptor before its
// error handling starts, so a null one would crash. The hipDNN route, with
// hipDNN unavailable, fails before reading any argument. Neither needs a GPU.
miopenStatus_t ConvForward(const ConvProblem& p)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    return miopenConvolutionForward(nullptr,
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

} // namespace

int main(int argc, char** argv)
{
    ConvProblem problem;
    bool haveProblem              = false;
    miopenStatus_t lastConvStatus = miopenStatusSuccess;
    std::string previous;
    int exitCode = 0;

    const auto run = [&](const std::string& step) {
        if(step == "conv_forward")
        {
            if(!haveProblem)
            {
                if(!MakeProblem(problem))
                    return false;
                haveProblem = true;
            }
            lastConvStatus = ConvForward(problem);
            std::printf("conv_forward=%d\n", static_cast<int>(lastConvStatus));
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
