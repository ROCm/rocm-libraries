// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "test_common.hpp"

size_t        size = 1000;
float*        data_0;
unsigned int* data_1;

HIP_CHECK(hipMalloc(&data_0, sizeof(*data_0) * size));
HIP_CHECK(hipMalloc(&data_1, sizeof(*data_1) * size));

hipGraph_t graph;

hipStream_t stream;
HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

rocrand_generator generator;
rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
rocrand_set_stream(generator, stream);
rocrand_initialize_generator(generator);

HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

rocrand_generate_normal(generator, data_0, size, 10.0F, 2.0F);
rocrand_generate_poisson(generator, data_1, size, 3);

HIP_CHECK(hipStreamEndCapture(stream, &graph));

hipGraphExec_t instance;
HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

HIP_CHECK(hipGraphLaunch(instance, stream));
HIP_CHECK(hipStreamSynchronize(stream));

HIP_CHECK(hipGraphExecDestroy(instance));
rocrand_destroy_generator(generator);
HIP_CHECK(hipStreamDestroy(stream));
HIP_CHECK(hipGraphDestroy(graph));
HIP_CHECK(hipFree(data_1));
HIP_CHECK(hipFree(data_0));
