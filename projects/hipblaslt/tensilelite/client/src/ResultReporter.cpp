/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ResultReporter.hpp"

namespace TensileLite
{
    namespace Client
    {
        std::string ToString(LogLevel level)
        {
            switch(level)
            {
            case LogLevel::Error:
                return "Error";
            case LogLevel::Terse:
                return "Terse";
            case LogLevel::Normal:
                return "Normal";
            case LogLevel::Verbose:
                return "Verbose";
            case LogLevel::Debug:
                return "Debug";

            case LogLevel::Count:;
            }

            return "Invalid";
        }

        std::ostream& operator<<(std::ostream& stream, LogLevel level)
        {
            return stream << ToString(level);
        }

        std::istream& operator>>(std::istream& stream, LogLevel& level)
        {
            std::string str;
            stream >> str;

            for(int i = 0; i < (int)LogLevel::Count; i++)
            {
                auto myVal = (LogLevel)(i);
                auto myStr = ToString(myVal);
                if(str == myStr)
                {
                    level = myVal;
                    break;
                }
            }

            return stream;
        }
    } // namespace Client
} // namespace TensileLite
