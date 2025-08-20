/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/PropertyMatching.hpp>

#include <cstddef>

namespace TensileLite
{
    /**
 * \addtogroup PropertyClasses
 * @{
 */
    namespace Contraction
    {
        struct FreeSizeA : public Property_CRTP<FreeSizeA, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "FreeSizeA";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.freeSizeA(index);
            }
        };

        struct FreeSizeB : public Property_CRTP<FreeSizeB, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "FreeSizeB";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.freeSizeB(index);
            }
        };

        struct BatchSize : public Property_CRTP<BatchSize, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "BatchSize";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.batchSize(index);
            }
        };

        struct BoundSize : public Property_CRTP<BoundSize, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "BoundSize";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.boundSize(index);
            }
        };

        struct AStride : public Property_CRTP<AStride, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "AStride";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.a().strides()[index];
            }
        };

        struct BStride : public Property_CRTP<BStride, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "BStride";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.b().strides()[index];
            }
        };

        struct CStride : public Property_CRTP<CStride, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "CStride";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.c().strides()[index];
            }
        };

        struct DStride : public Property_CRTP<DStride, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = true,
                HasValue = false
            };
            size_t index;

            static std::string Type()
            {
                return "DStride";
            }

            virtual size_t operator()(ContractionProblemGemm const& problem) const
            {
                return problem.d().strides()[index];
            }
        };

        struct OperationIdentifier
            : public Property_CRTP<OperationIdentifier, ContractionProblemGemm, std::string>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };

            static std::string Type()
            {
                return "OperationIdentifier";
            }

            virtual std::string operator()(ContractionProblemGemm const& problem) const
            {
                return problem.operationIdentifier();
            }
        };
    } // namespace Contraction

    /**
 * @}
 */
} // namespace TensileLite
