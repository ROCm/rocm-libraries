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

#pragma once

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Predicates.hpp>

#include <vector>

namespace TensileLite
{
    namespace Predicates
    {
        /**
 * \addtogroup Predicates
 * @{
 */
        /**
 * @brief GPU Predicates
 */
        namespace GPU
        {
            struct ProcessorEqual : public Predicate_CRTP<ProcessorEqual, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                AMDGPU::Processor value;

                ProcessorEqual() = default;
                ProcessorEqual(AMDGPU::Processor p)
                    : value(p)
                {
                }

                static std::string Type()
                {
                    return "Processor";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    return gpu.processor == value;
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    return debugEvalCmp(gpu, stream, "prob", gpu.processor, "==", "sol", value);
                }
            };

            struct CUCountEqual : public Predicate_CRTP<CUCountEqual, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                int value;

                CUCountEqual() = default;
                CUCountEqual(int val)
                    : value(val)
                {
                }

                static std::string Type()
                {
                    return "CUCount";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    return gpu.computeUnitCount == value;
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    return debugEvalCmp(gpu, stream, "prob", gpu.computeUnitCount, "==", "sol", value);
                }
            };

            struct RunsKernelTargeting : public Predicate_CRTP<RunsKernelTargeting, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                AMDGPU::Processor value;

                RunsKernelTargeting() = default;
                RunsKernelTargeting(AMDGPU::Processor p)
                    : value(p)
                {
                }

                static std::string Type()
                {
                    return "TargetProcessor";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    return gpu.runsKernelTargeting(value);
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    bool rv = (*this)(gpu);
                    stream << rv << ": " << this->type() << std::endl;
                    return rv;
                }
            };
        } // namespace GPU

        /**
 * @}
 */
    } // namespace Predicates
} // namespace TensileLite
