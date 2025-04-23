/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#include <iostream>
#if DBGINFO
#define INFO(X) std::clog << "[INF] "                 \
                          << " {" << __func__ << "} " \
                          << " " << X << std::endl;
#else
#define INFO(X) ;
#endif
#if DBGLOG
#define LOG(X) std::clog << "[LOG] "                 \
                         << " {" << __func__ << "} " \
                         << " " << X << std::endl;
#else
#define LOG(X) ;
#endif
#define ERR(X) std::cerr << "[ERR] "                 \
                         << " {" << __func__ << "} " \
                         << " " << X << std::endl;
#if WRNLOG
#define WRN(X) std::clog << "[WRN] "                 \
                         << " {" << __func__ << "} " \
                         << " " << X << std::endl;
#else
#define WRN(X) ;
#endif
