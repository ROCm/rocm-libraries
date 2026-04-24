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

#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

#include <hiptensor/hiptensor.h>

#include "hiptensor_ostream.hpp"

namespace hiptensor
{
    struct HIPTENSOR_EXPORT HiptensorOptions
    {
        // For static initialization
        friend std::unique_ptr<HiptensorOptions> std::make_unique<HiptensorOptions>();

        // Defined in hiptensor_options.cpp (not a template) so that exactly one static instance exists
        // across all modules. A template function (e.g. via LazySingleton<HiptensorOptions>) would produce
        // a separate static per module on Windows, giving distinct HiptensorOptions objects in the DLL and
        // in any binary that includes this header directly.
        static std::unique_ptr<HiptensorOptions> const& instance();

    private: // No public instantiation except make_unique.
             // No copy
        HiptensorOptions();
        HiptensorOptions(HiptensorOptions const&)            = delete;
        HiptensorOptions& operator=(HiptensorOptions const&) = delete;

    public:
        HiptensorOptions(HiptensorOptions&&) = default;
        ~HiptensorOptions()                  = default;

        void setOstream(std::string file);
        void setLogOstream(std::string file);
        void setOmits(int mask);
        void setDefaultParams(bool val);
        void setValidation(std::string val);
        void setHotRuns(int runs);
        void setColdRuns(int runs);
        void setInputYAMLFilename(std::string file);
        void setOutputStreamFilename(std::string file);
        void setLogStreamFilename(std::string file);

        HiptensorOStream& ostream();
        HiptensorOStream& logOstream();

        bool omitSkipped();
        bool omitFailed();
        bool omitPassed();
        bool omitCout();
        bool usingDefaultConfig();
        bool performValidation();
        bool isColMajorStrides();

        int32_t hotRuns();
        int32_t coldRuns();

        std::string inputFilename();
        std::string outputFilename();
        std::string logFilename();

    protected:
        HiptensorOStream mOstream, mLogOstream;

        bool mOmitSkipped, mOmitFailed, mOmitPassed, mOmitCout;
        bool mUsingDefaultParams;
        bool mValidate;
        bool mColMajorStrides;

        int32_t mHotRuns, mColdRuns;

        std::string mInputFilename, mOutputFilename, mLogFilename;
    };

} // namespace hiptensor
