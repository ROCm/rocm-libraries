/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_OPTIONS_HPP
#define ROCWMMA_OPTIONS_HPP

#include "rocwmma/rocwmma-version.hpp"
#include "rocwmma_ostream.hpp"
#include "singleton.hpp"
#include <stdlib.h>

namespace rocwmma
{
    enum struct EmulationOption
    {
        NONE,
        SMOKE,
        REGRESSION,
        EXTENDED
    };

    struct RocwmmaOptions : public LazySingleton<RocwmmaOptions>
    {
        // For static initialization
        friend std::unique_ptr<RocwmmaOptions> std::make_unique<RocwmmaOptions>();

    private: // No public instantiation except make_unique.
        // No copy
        RocwmmaOptions(RocwmmaOptions const&)            = delete;
        RocwmmaOptions& operator=(RocwmmaOptions const&) = delete;

    public:
        RocwmmaOptions(RocwmmaOptions&&) = default;
        ~RocwmmaOptions()                = default;

        RocwmmaOptions()
            : mOstream()
            , mOmitSkipped(false)
            , mOmitFailed(false)
            , mOmitPassed(false)
            , mOmitCout(false)
            , mEmulationOption(EmulationOption::NONE)
        {
        }

        void setOmits(int mask)
        {
            if(mask & 1)
                mOmitSkipped = true;
            if(mask & 2)
                mOmitFailed = true;
            if(mask & 4)
                mOmitPassed = true;
            if(mask & 8)
                mOmitCout = true;
        }

        void setEmulationOption(EmulationOption value)
        {
            mEmulationOption = value;
        }

        void parseOptions(int argc, char** argv)
        {
            const std::vector<std::string> args(argv + 1, argv + argc);
            std::string                    fileName;

            for(auto i = 0; i < argc - 1; i++)
            {
                if(args[i] == "-v" || args[i] == "--version")
                {
                    std::cout << "rocWMMA Version: " << rocwmma_get_version() << std::endl;
                    continue;
                }
                if(args[i] == "-os" || args[i] == "--output_stream")
                {
                    if(i + 2 >= argc)
                    {
                        std::cerr << "Missing output stream\n";
                        std::cerr << "Usage: -os || --output_stream *file.csv*\n";
                        exit(EXIT_FAILURE);
                    }
                    fileName = args[i + 1];
                    i++;
                    continue;
                }
                if(args[i] == "--omit")
                {
                    if(i + 2 >= argc)
                    {
                        std::cerr << "Missing omit integer mask\n";
                        std::cerr << "Usage: --omit *integer_mask*\n";
                        exit(EXIT_FAILURE);
                    }
                    setOmits(std::stoi(args[i + 1]));
                    i++;
                    continue;
                }
                if(args[i] == "--emulation")
                {
                    if(i + 2 >= argc)
                    {
                        std::cerr << "Missing emulation option\n";
                        std::cerr << "Usage: --emulation [smoke|regression|extended]\n";
                        exit(EXIT_FAILURE);
                    }
                    auto emulationValue = parseEmulationOption(args[i + 1]);
                    if(emulationValue == EmulationOption::NONE)
                    {
                        std::cerr << "Invalid emulation option: " << args[i + 1] << "\n";
                        std::cerr << "Usage: --emulation [smoke|regression|extended]\n";
                        exit(EXIT_FAILURE);
                    }
                    setEmulationOption(emulationValue);
                    i++;
                    continue;
                }
            }

            mOstream.initializeStream(fileName);
        }

        rocwmmaOStream& ostream()
        {
            return mOstream;
        }

        bool omitSkipped()
        {
            return mOmitSkipped;
        }

        bool omitFailed()
        {
            return mOmitFailed;
        }

        bool omitPassed()
        {
            return mOmitPassed;
        }

        bool omitCout()
        {
            return mOmitCout;
        }

        EmulationOption emulationOption()
        {
            return mEmulationOption;
        }

    private:
        EmulationOption parseEmulationOption(std::string const& value)
        {
            std::string lowercase_value = value;
            std::transform(
                lowercase_value.begin(), lowercase_value.end(), lowercase_value.begin(), ::tolower);

            if(lowercase_value == "smoke")
            {
                return EmulationOption::SMOKE;
            }
            else if(lowercase_value == "regression")
            {
                return EmulationOption::REGRESSION;
            }
            else if(lowercase_value == "extended")
            {
                return EmulationOption::EXTENDED;
            }
            else
            {
                return EmulationOption::NONE;
            }
        }

        rocwmmaOStream mOstream;

        bool mOmitSkipped, mOmitFailed, mOmitPassed, mOmitCout;

        EmulationOption mEmulationOption;
    };
}

#endif // ROCWMMA_OPTIONS_HPP
