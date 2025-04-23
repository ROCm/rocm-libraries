/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_OPTIONS
#define HIPTENSOR_OPTIONS

#include <stdlib.h>

#include "hiptensor_ostream.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    struct HiptensorOptions : public LazySingleton<HiptensorOptions>
    {
        // For static initialization
        friend std::unique_ptr<HiptensorOptions> std::make_unique<HiptensorOptions>();

    private: // No public instantiation except make_unique.
             // No copy
        HiptensorOptions();
        HiptensorOptions(HiptensorOptions const&)            = delete;
        HiptensorOptions& operator=(HiptensorOptions const&) = delete;

    public:
        HiptensorOptions(HiptensorOptions&&) = default;
        ~HiptensorOptions()                  = default;

        void setOstream(std::string file);
        void setOmits(int mask);
        void setDefaultParams(bool val);
        void setValidation(std::string val);
        void setHotRuns(int runs);
        void setColdRuns(int runs);
        void setInputYAMLFilename(std::string file);
        void setOutputStreamFilename(std::string file);

        HiptensorOStream& ostream();

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

    protected:
        HiptensorOStream mOstream;

        bool mOmitSkipped, mOmitFailed, mOmitPassed, mOmitCout;
        bool mUsingDefaultParams;
        bool mValidate;
        bool mColMajorStrides;

        int32_t mHotRuns, mColdRuns;

        std::string mInputFilename, mOutputFilename;
    };

} // namespace hiptensor

#endif // HIPTENSOR_OPTIONS
