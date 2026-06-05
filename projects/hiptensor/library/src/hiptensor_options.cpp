/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hiptensor_options.hpp"
#include "util.hpp"

#include <algorithm>
#include <cctype>

namespace hiptensor
{
    /* static */ std::unique_ptr<HiptensorOptions> const& HiptensorOptions::instance()
    {
        static auto sInstance = std::make_unique<HiptensorOptions>();
        return sInstance;
    }

    HiptensorOptions::HiptensorOptions()
        : mOstream()
        , mLogOstream()
        , mOmitSkipped(false)
        , mOmitFailed(false)
        , mOmitPassed(false)
        , mOmitCout(false)
        , mUsingDefaultParams(true)
        , mValidate(true)
        , mColMajorStrides(HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR)
        , mHotRuns(1)
        , mColdRuns(0)
        , mInputFilename("")
        , mOutputFilename("")
        , mLogFilename("")
    {
        // Override HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR with environment variable if present
        auto stride_env = getEnvironmentVariable("HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR");
        if(stride_env.has_value())
        {
            mColMajorStrides
                = checkEnvironmentVariableEnabled("HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR");
        }
    }

    void HiptensorOptions::setOstream(std::string file)
    {
        mOstream.initializeStream(file);
    }

    void HiptensorOptions::setLogOstream(std::string file)
    {
        mLogOstream.initializeStream(file);
    }

    void HiptensorOptions::setOmits(int mask)
    {
        if(mask & 1)
        {
            mOmitSkipped = true;
        }
        else
        {
            mOmitSkipped = false;
        }

        if(mask & 2)
        {
            mOmitFailed = true;
        }
        else
        {
            mOmitFailed = false;
        }

        if(mask & 4)
        {
            mOmitPassed = true;
        }
        else
        {
            mOmitPassed = false;
        }

        if(mask & 8)
        {
            mOmitCout = true;
        }
        else
        {
            mOmitCout = false;
        }
    }

    void HiptensorOptions::setDefaultParams(bool val)
    {
        mUsingDefaultParams = val;
    }

    void HiptensorOptions::setValidation(std::string val)
    {
        std::transform(val.begin(), val.end(), val.begin(), ::toupper);
        if(val.compare("ON") == 0)
        {
            mValidate = true;
        }
        else if(val.compare("OFF") == 0)
        {
            mValidate = false;
        }
    }

    void HiptensorOptions::setHotRuns(int runs)
    {
        mHotRuns = runs;
    }

    void HiptensorOptions::setColdRuns(int runs)
    {
        mColdRuns = runs;
    }

    void HiptensorOptions::setInputYAMLFilename(std::string file)
    {
        mInputFilename = file;
    }

    void HiptensorOptions::setOutputStreamFilename(std::string file)
    {
        mOutputFilename = file;
    }

    void HiptensorOptions::setLogStreamFilename(std::string file)
    {
        mLogFilename = file;
    }

    HiptensorOStream& HiptensorOptions::ostream()
    {
        return mOstream;
    }

    HiptensorOStream& HiptensorOptions::logOstream()
    {
        return mLogOstream;
    }

    bool HiptensorOptions::omitSkipped()
    {
        return mOmitSkipped;
    }

    bool HiptensorOptions::omitFailed()
    {
        return mOmitFailed;
    }

    bool HiptensorOptions::omitPassed()
    {
        return mOmitPassed;
    }

    bool HiptensorOptions::omitCout()
    {
        return mOmitCout;
    }

    bool HiptensorOptions::usingDefaultConfig()
    {
        return mUsingDefaultParams;
    }

    bool HiptensorOptions::performValidation()
    {
        return mValidate;
    }

    int32_t HiptensorOptions::hotRuns()
    {
        return mHotRuns;
    }

    int32_t HiptensorOptions::coldRuns()
    {
        return mColdRuns;
    }

    std::string HiptensorOptions::inputFilename()
    {
        return mInputFilename;
    }

    std::string HiptensorOptions::outputFilename()
    {
        return mOutputFilename;
    }

    std::string HiptensorOptions::logFilename()
    {
        return mLogFilename;
    }

    bool HiptensorOptions::isColMajorStrides()
    {
        return mColMajorStrides;
    }

} // namespace hiptensor
