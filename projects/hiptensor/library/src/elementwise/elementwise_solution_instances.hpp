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

#pragma once

#include <memory>

#include "elementwise_solution_registry.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class ElementwiseSolutionInstances : public ElementwiseSolutionRegistry,
                                         public LazySingleton<ElementwiseSolutionInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<ElementwiseSolutionInstances>
            std::make_unique<ElementwiseSolutionInstances>();

        ~ElementwiseSolutionInstances() = default;

    private:
        void ElementwiseBinarySolution2DDoubleInstances();
        void ElementwiseBinarySolution2DFloatInstances();
        void ElementwiseBinarySolution2DHalfInstances();
        void ElementwiseBinarySolution3DDoubleInstances();
        void ElementwiseBinarySolution3DFloatInstances();
        void ElementwiseBinarySolution3DHalfInstances();
        void ElementwiseBinarySolution4DDoubleInstances();
        void ElementwiseBinarySolution4DFloatInstances();
        void ElementwiseBinarySolution4DHalfInstances();
        void ElementwiseBinarySolution5DDoubleInstances();
        void ElementwiseBinarySolution5DFloatInstances();
        void ElementwiseBinarySolution5DHalfInstances();
        void ElementwiseBinarySolution6DDoubleInstances();
        void ElementwiseBinarySolution6DFloatInstances();
        void ElementwiseBinarySolution6DHalfInstances();
        void ElementwiseTrinarySolution2DDoubleInstances();
        void ElementwiseTrinarySolution2DFloatInstances();
        void ElementwiseTrinarySolution2DHalfInstances();
        void ElementwiseTrinarySolution3DDoubleInstances();
        void ElementwiseTrinarySolution3DFloatInstances();
        void ElementwiseTrinarySolution3DHalfInstances();
        void ElementwiseTrinarySolution4DDoubleInstances();
        void ElementwiseTrinarySolution4DFloatInstances();
        void ElementwiseTrinarySolution4DHalfInstances();
        void ElementwiseTrinarySolution5DDoubleInstances();
        void ElementwiseTrinarySolution5DFloatInstances();
        void ElementwiseTrinarySolution5DHalfInstances();
        void ElementwiseTrinarySolution6DDoubleInstances();
        void ElementwiseTrinarySolution6DFloatInstances();
        void ElementwiseTrinarySolution6DHalfInstances();
        void ElementwisePermuteSolution2DFloatInstances();
        void ElementwisePermuteSolution2DFloatNoopInstances();
        void ElementwisePermuteSolution2DHalfInstances();
        void ElementwisePermuteSolution2DHalfNoopInstances();
        void ElementwisePermuteSolution3DFloatInstances();
        void ElementwisePermuteSolution3DFloatNoopInstances();
        void ElementwisePermuteSolution3DHalfInstances();
        void ElementwisePermuteSolution3DHalfNoopInstances();
        void ElementwisePermuteSolution4DFloatInstances();
        void ElementwisePermuteSolution4DFloatNoopInstances();
        void ElementwisePermuteSolution4DHalfInstances();
        void ElementwisePermuteSolution4DHalfNoopInstances();
        void ElementwisePermuteSolution5DFloatInstances();
        void ElementwisePermuteSolution5DFloatNoopInstances();
        void ElementwisePermuteSolution5DHalfInstances();
        void ElementwisePermuteSolution5DHalfNoopInstances();
        void ElementwisePermuteSolution6DFloatInstances();
        void ElementwisePermuteSolution6DFloatNoopInstances();
        void ElementwisePermuteSolution6DHalfInstances();
        void ElementwisePermuteSolution6DHalfNoopInstances();
        // Singleton: only one instance
        ElementwiseSolutionInstances();
        ElementwiseSolutionInstances(ElementwiseSolutionInstances const&)            = delete;
        ElementwiseSolutionInstances(ElementwiseSolutionInstances&&)                 = delete;
        ElementwiseSolutionInstances& operator=(ElementwiseSolutionInstances const&) = delete;
        ElementwiseSolutionInstances& operator=(ElementwiseSolutionInstances&&)      = delete;
    };

} // namespace hiptensor
