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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <cstring>
#include <ctime>

#include "include/platform.hpp"

void safeLocaltime(char* buff, size_t buffSize, const time_t* t)
{
    struct tm tmInfo;
#ifdef _WIN32
    if(localtime_s(&tmInfo, t) != 0)
    {
        strncpy_s(buff, buffSize, "TIMESTAMP-ERROR", _TRUNCATE);
        return;
    }
#else
    if(localtime_r(t, &tmInfo) == nullptr)
    {
        strncpy(buff, "TIMESTAMP-ERROR", buffSize - 1);
        buff[buffSize - 1] = '\0';
        return;
    }
#endif
    strftime(buff, buffSize, "%Y-%m-%d %H:%M:%S", &tmInfo);
}

FILE* safeFopen(const char* filename, const char* mode)
{
#ifdef _WIN32
    FILE* file = nullptr;
    if(fopen_s(&file, filename, mode) == 0)
    {
        return file;
    }
    return nullptr;
#else
    return fopen(filename, mode);
#endif
}
