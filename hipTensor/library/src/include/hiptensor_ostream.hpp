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

#ifndef HIPTENSOR_IOSTREAM
#define HIPTENSOR_IOSTREAM

#include <fstream>
#include <iostream>
#include <utility>

namespace hiptensor
{
    struct HiptensorOStream
    {
    public:
        void initializeStream(std::string fileName)
        {
            if(fileName.length() > 0)
            {
                mOstream.open(fileName);
            }
        }

        bool isOpen()
        {
            return mOstream.is_open();
        }

        // Default output for non-enumeration types
        template <typename T, std::enable_if_t<!std::is_enum<std::decay_t<T>>{}, int> = 0>
        HiptensorOStream& operator<<(T&& x)
        {
            if(mOstream.is_open())
                mOstream << std::forward<T>(x);
        }

        // Default output for enumeration types
        template <typename T, std::enable_if_t<std::is_enum<std::decay_t<T>>{}, int> = 0>
        HiptensorOStream& operator<<(T&& x)
        {
            if(mOstream.is_open())
            {
                mOstream << std::underlying_type_t<std::decay_t<T>>(x);
            }
        }

        std::ofstream& fstream()
        {
            return mOstream;
        }

    protected:
        std::ofstream mOstream;
    };

} // namespace hiptensor

#endif // HIPTENSOR_IOSTREAM
