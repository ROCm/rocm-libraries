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

#pragma once

#include <Tensile.h>

namespace OldTensile
{
    using TensileGEMM = TensileStatus (*)(float*       dataD,
                                          const float* dataC,
                                          const float* dataA,
                                          const float* dataB,
                                          float        alpha,
                                          float        beta,
                                          unsigned int strideC1J,
                                          unsigned int strideC2K,
                                          unsigned int strideA1L,
                                          unsigned int strideA2K,
                                          unsigned int strideB1L,
                                          unsigned int strideB2K,
                                          unsigned int sizeI,
                                          unsigned int sizeJ,
                                          unsigned int sizeK,
                                          unsigned int sizeL,
                                          hipStream_t  stream,
                                          unsigned int numInputEvents,
                                          hipEvent_t*  inputEvents,
                                          hipEvent_t*  outputEvent);

    inline TensileGEMM GetTensileGEMM(bool transA, bool transB)
    {
        if(!transA && !transB)
            return tensile_Cijk_Ailk_Bljk_SB;
        if(!transA && transB)
            return tensile_Cijk_Ailk_Bjlk_SB;
        if(transA && !transB)
            return tensile_Cijk_Alik_Bljk_SB;
        if(transA && transB)
            return tensile_Cijk_Alik_Bjlk_SB;
        return nullptr;
    }

    inline TensileStatus CallOldTensile(bool         transA,
                                        bool         transB,
                                        float*       dataD,
                                        const float* dataC,
                                        const float* dataA,
                                        const float* dataB,
                                        float        alpha,
                                        float        beta,
                                        unsigned int strideC1J,
                                        unsigned int strideC2K,
                                        unsigned int strideA1L,
                                        unsigned int strideA2K,
                                        unsigned int strideB1L,
                                        unsigned int strideB2K,
                                        unsigned int sizeI,
                                        unsigned int sizeJ,
                                        unsigned int sizeK,
                                        unsigned int sizeL,
                                        hipStream_t  stream,
                                        unsigned int numInputEvents,
                                        hipEvent_t*  inputEvents,
                                        hipEvent_t*  outputEvent)
    {
        auto fn = GetTensileGEMM(transA, transB);

        return fn(dataD,
                  dataC,
                  dataA,
                  dataB,
                  alpha,
                  beta,
                  strideC1J,
                  strideC2K,
                  strideA1L,
                  strideA2K,
                  strideB1L,
                  strideB2K,
                  sizeI,
                  sizeJ,
                  sizeK,
                  sizeL,
                  stream,
                  numInputEvents,
                  inputEvents,
                  outputEvent);
    }
} // namespace OldTensile
