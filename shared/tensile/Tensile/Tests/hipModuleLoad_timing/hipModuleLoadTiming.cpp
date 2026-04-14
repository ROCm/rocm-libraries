/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>

using namespace std;
using namespace chrono;

/****************************
* Usage: ./hipModuleLoadTiming.out filename
*****************************/
int main(int argc, char** argv)
{

    if(argc < 2)
    {
        cout << "Usage: ./hipModuleLoadTiming.out filename" << std::endl;
        return 1;
    }

    const char* filename = argv[1];

    hipModule_t module;
    {
        time_point<system_clock> startTime = system_clock::now();
        hipModuleLoad(&module, filename);
        time_point<system_clock> endTime    = system_clock::now();
        duration<float>          difference = endTime - startTime;
        std::cout << "hipModuleLoad() took " << difference.count() << " seconds" << std::endl;
    }

    {
        time_point<system_clock> startTime = system_clock::now();
        std::ifstream            file(filename, std::ifstream::binary);
        if(file)
        {
            //Get length
            file.seekg(0, file.end);
            size_t length = file.tellg();
            file.seekg(0, file.beg);

            char* buffer = new char[length];

            file.read(buffer, length);
            time_point<system_clock> endTime    = system_clock::now();
            duration<float>          difference = endTime - startTime;
            std::cout << "Loading from filesystem to memory took " << difference.count()
                      << " seconds" << std::endl;
        }
        else
            cout << "File " << filename << " not found" << endl;
    }
}
