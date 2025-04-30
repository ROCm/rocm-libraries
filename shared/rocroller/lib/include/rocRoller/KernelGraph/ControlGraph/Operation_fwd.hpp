/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        struct Assign;
        struct Barrier;
        struct ComputeIndex;
        struct ConditionalOp;
        struct AssertOp;
        struct Deallocate;
        struct DoWhileOp;
        struct Exchange;
        struct ForLoopOp;
        struct Kernel;
        struct LoadLDSTile;
        struct LoadLinear;
        struct LoadVGPR;
        struct LoadSGPR;
        struct LoadTiled;
        struct Multiply;
        struct NOP;
        struct Block;
        struct Scope;
        struct SetCoordinate;
        struct StoreLDSTile;
        struct LoadTileDirect2LDS;
        struct StoreLinear;
        struct StoreTiled;
        struct StoreVGPR;
        struct StoreSGPR;
        struct TensorContraction;
        struct UnrollOp;
        struct WaitZero;
        struct SeedPRNG;

        using Operation = std::variant<Assign,
                                       Barrier,
                                       ComputeIndex,
                                       ConditionalOp,
                                       AssertOp,
                                       Deallocate,
                                       DoWhileOp,
                                       Exchange,
                                       ForLoopOp,
                                       Kernel,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       LoadSGPR,
                                       LoadTileDirect2LDS,
                                       Multiply,
                                       NOP,
                                       Block,
                                       Scope,
                                       SetCoordinate,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR,
                                       StoreSGPR,
                                       TensorContraction,
                                       UnrollOp,
                                       WaitZero,
                                       SeedPRNG>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;

        template <typename T>
        concept CConcreteOperation = (COperation<T> && !std::same_as<Operation, T>);
    }
}
