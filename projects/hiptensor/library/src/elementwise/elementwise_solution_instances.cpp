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

#include "elementwise_solution_instances.hpp"
#include "elementwise_solution.hpp"

// Ensure access to
#include "device/hiptensor_elementwise_scale_instances.hpp"

namespace hiptensor
{
    ElementwiseSolutionInstances::ElementwiseSolutionInstances()
    {
        ElementwiseBinarySolution2DDoubleInstances();
        ElementwiseBinarySolution3DDoubleInstances();
        ElementwiseBinarySolution4DDoubleInstances();
        ElementwiseBinarySolution4DFloatInstances();
        ElementwiseBinarySolution4DHalfInstances();
        ElementwiseBinarySolution5DDoubleInstances();
        ElementwiseBinarySolution5DFloatInstances();
        ElementwiseBinarySolution5DHalfInstances();
        ElementwiseBinarySolution6DDoubleInstances();
        ElementwiseBinarySolution6DFloatInstances();
        ElementwiseBinarySolution6DHalfInstances();
        ElementwiseTrinarySolution2DDoubleInstances();
        ElementwiseTrinarySolution2DFloatInstances();
        ElementwiseTrinarySolution2DHalfInstances();
        ElementwiseTrinarySolution3DDoubleInstances();
        ElementwiseTrinarySolution3DFloatInstances();
        ElementwiseTrinarySolution3DHalfInstances();
        ElementwiseTrinarySolution4DDoubleInstances();
        ElementwiseTrinarySolution4DFloatInstances();
        ElementwiseTrinarySolution4DHalfInstances();
        ElementwiseTrinarySolution5DDoubleInstances();
        ElementwiseTrinarySolution5DFloatInstances();
        ElementwiseTrinarySolution5DHalfInstances();
        ElementwiseTrinarySolution6DDoubleInstances();
        ElementwiseTrinarySolution6DFloatInstances();
        ElementwiseTrinarySolution6DHalfInstances();
        ElementwisePermuteSolution4DFloatInstances();
        ElementwisePermuteSolution4DFloatNoopInstances();
        ElementwisePermuteSolution4DHalfInstances();
        ElementwisePermuteSolution4DHalfNoopInstances();
        ElementwisePermuteSolution5DFloatInstances();
        ElementwisePermuteSolution5DFloatNoopInstances();
        ElementwisePermuteSolution5DHalfInstances();
        ElementwisePermuteSolution5DHalfNoopInstances();
        ElementwisePermuteSolution6DFloatInstances();
        ElementwisePermuteSolution6DFloatNoopInstances();
        ElementwisePermuteSolution6DHalfInstances();
        ElementwisePermuteSolution6DHalfNoopInstances();
        ElementwiseBinarySolution2DFloatInstances();
        ElementwiseBinarySolution2DHalfInstances();
        ElementwiseBinarySolution3DFloatInstances();
        ElementwiseBinarySolution3DHalfInstances();
        ElementwisePermuteSolution2DFloatInstances();
        ElementwisePermuteSolution2DFloatNoopInstances();
        ElementwisePermuteSolution2DHalfInstances();
        ElementwisePermuteSolution2DHalfNoopInstances();
        ElementwisePermuteSolution3DFloatInstances();
        ElementwisePermuteSolution3DFloatNoopInstances();
        ElementwisePermuteSolution3DHalfInstances();
        ElementwisePermuteSolution3DHalfNoopInstances();
    }
} // namespace hiptensor
