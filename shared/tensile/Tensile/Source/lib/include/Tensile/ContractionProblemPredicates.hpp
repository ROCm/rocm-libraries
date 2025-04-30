/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/Predicates.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        /**
 * \addtogroup Predicates
 * @{
 */
        /**
 * @brief ContractionProblem predicates
 */
        namespace Contraction
        {
            struct Free0SizeMultiple : public Predicate_CRTP<Free0SizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                Free0SizeMultiple() = default;
                Free0SizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "Free0SizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (!problem.transposeC01() ? problem.freeSizeA(index)
                                                    : problem.freeSizeB(index))
                               % value
                           == 0;
                }
            };

            struct Free1SizeMultiple : public Predicate_CRTP<Free1SizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                Free1SizeMultiple() = default;
                Free1SizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "Free1SizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (!problem.transposeC01() ? problem.freeSizeB(index)
                                                    : problem.freeSizeA(index))
                               % value
                           == 0;
                }
            };

            struct BatchSizeMultiple : public Predicate_CRTP<BatchSizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                BatchSizeMultiple() = default;
                BatchSizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BatchSizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.batchSize(index) % value == 0;
                }
            };

            struct BatchSizeEqual : public Predicate_CRTP<BatchSizeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                BatchSizeEqual() = default;
                BatchSizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BatchSizeEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.batchSize(index) == value;
                }
            };

            struct BoundSizeMultiple : public Predicate_CRTP<BoundSizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                int64_t index;
                size_t  value;

                BoundSizeMultiple() = default;
                BoundSizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BoundSizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    if(index < 0)
                        return problem.boundSize(problem.boundIndices().size() + index) % value
                               == 0;
                    else
                        return problem.boundSize(index) % value == 0;
                }
            };

            struct ProblemSizeEqual : public Predicate_CRTP<ProblemSizeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                ProblemSizeEqual() = default;
                ProblemSizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "ProblemSizeEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.problemSizes()[index] == value;
                }
            };

            struct MaxProblemSizeGreaterThan
                : public Predicate_CRTP<MaxProblemSizeGreaterThan, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                MaxProblemSizeGreaterThan() = default;
                MaxProblemSizeGreaterThan(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "MaxProblemSizeGreaterThan";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.maxProblemSize() > value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.maxProblemSize() << " > " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            // If the tensor contains no free dimensions, then the first batch dimension
            // serves as the leading free size
            struct LeadingFree0SizesGreaterOrEqual
                : public Predicate_CRTP<LeadingFree0SizesGreaterOrEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                LeadingFree0SizesGreaterOrEqual() = default;
                LeadingFree0SizesGreaterOrEqual(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "LeadingFree0SizesGreaterOrEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    assert(problem.batchIndices().size() <= 1);
                    // TODO: this is not quite right since it assumes batchSize(0) is lowest
                    // order in index assignments
                    //   If tensor contains multiple batch dims this may not be true.
                    //   Really should modify Contractions.py to select SizeN >= value, based on
                    //   desired index requirement
                    return (problem.freeIndicesA().size() ? problem.freeSizeA(0) >= value
                                                          : problem.batchSize(0) >= value);
                }
                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": ("
                           << (problem.freeIndicesA().size() ? "freeA0:" : "batchA0:")
                           << (problem.freeIndicesA().size() ? problem.freeSizeA(0)
                                                             : problem.batchSize(0))
                           << " >= " << value << ") == " << rv;

                    return rv;
                }
            };

            struct LeadingFree1SizesGreaterOrEqual
                : public Predicate_CRTP<LeadingFree1SizesGreaterOrEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                LeadingFree1SizesGreaterOrEqual() = default;
                LeadingFree1SizesGreaterOrEqual(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "LeadingFree1SizesGreaterOrEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    assert(problem.batchIndices().size() <= 1);
                    // TODO: this is not quite right since it assumes batchSize(0) is lowest
                    // order in index assignments
                    //   If tensor contains multiple batch dims this may not be true.
                    //   Really should modify Contractions.py to select SizeN >= value, based on
                    //   desired index requirement
                    return (problem.freeIndicesB().size() ? problem.freeSizeB(0) >= value
                                                          : problem.batchSize(0) >= value);
                }
                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": ("
                           << (problem.freeIndicesB().size() ? "freeB0:" : "batchB0:")
                           << (problem.freeIndicesB().size() ? problem.freeSizeB(0)
                                                             : problem.batchSize(0))
                           << " >= " << value << ") == " << rv;

                    return rv;
                }
            };

            struct SizeEqual : public Predicate_CRTP<SizeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeEqual() = default;
                SizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.size(index) == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.size(index) << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct SizeGreaterThan : public Predicate_CRTP<SizeGreaterThan, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeGreaterThan() = default;
                SizeGreaterThan(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeGreaterThan";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (problem.size(index) > value);
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.size(index) << " > " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct SizeLessThan : public Predicate_CRTP<SizeLessThan, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeLessThan() = default;
                SizeLessThan(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeLessThan";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (problem.size(index) < value);
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.size(index) << " < " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct SizeMultiple : public Predicate_CRTP<SizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeMultiple() = default;
                SizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return ((problem.size(index) % value) == 0);
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.size(index) << " % " << value
                           << " == 0) == " << rv;

                    return rv;
                }
            };

            struct Range
            {
                size_t min = 0;
                size_t max = std::numeric_limits<size_t>::max();
            };

            inline std::ostream& operator<<(std::ostream& stream, Range const& range)
            {
                return stream << "min: " << range.min << ", max: " << range.max;
            }

            struct SizeInRange : public Predicate_CRTP<SizeInRange, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                Range  value;

                SizeInRange() = default;
                SizeInRange(size_t index, Range value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeInRange";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (problem.size(index) >= value.min) && (problem.size(index) < value.max);
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);
                    stream << *this << ": (" << value.min << " <= " << problem.size(index)
                           << " <= " << value.max << ") == " << rv;

                    return rv;
                }
            };

            struct StrideAEqual : public Predicate_CRTP<StrideAEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideAEqual() = default;
                StrideAEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideAEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.a().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.a().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideBEqual : public Predicate_CRTP<StrideBEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideBEqual() = default;
                StrideBEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideBEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.b().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.b().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideCEqual : public Predicate_CRTP<StrideCEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideCEqual() = default;
                StrideCEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideCEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.c().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideDEqual : public Predicate_CRTP<StrideDEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideDEqual() = default;
                StrideDEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideDEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.d().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.d().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct LDCEqualsLDD : public Predicate_CRTP<LDCEqualsLDD, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                static std::string Type()
                {
                    return "LDCEqualsLDD";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides()[1] == problem.d().strides()[1];
                }
            };

            struct CEqualsD : public Predicate_CRTP<CEqualsD, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                static std::string Type()
                {
                    return "CEqualsD";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.cEqualsD();
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << this->type() << "(problem.cEqualsD() == True) == " << rv;

                    return rv;
                }
            };

            struct BetaZero : public Predicate_CRTP<BetaZero, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                BetaZero() = default;

                static std::string Type()
                {
                    return "BetaZero";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.beta() == 0.0;
                }
            };

            struct BetaOne : public Predicate_CRTP<BetaOne, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                BetaOne() = default;

                static std::string Type()
                {
                    return "BetaOne";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.beta() == 1.0;
                }
            };

            struct HighPrecisionAccumulateEqual
                : public Predicate_CRTP<HighPrecisionAccumulateEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                HighPrecisionAccumulateEqual() = default;
                HighPrecisionAccumulateEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "HighPrecisionAccumulate";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.highPrecisionAccumulate() == value;
                }
            };

            struct KernelLanguageCompatible
                : public Predicate_CRTP<KernelLanguageCompatible, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                KernelLanguage value;

                KernelLanguageCompatible() = default;
                KernelLanguageCompatible(KernelLanguage value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "KernelLanguageCompatible";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.kernelLanguage() == value
                           || problem.kernelLanguage() == KernelLanguage::Any;
                }
            };

            struct DeterministicModeEqual
                : public Predicate_CRTP<DeterministicModeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                DeterministicModeEqual() = default;
                DeterministicModeEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "DeterministicMode";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.deterministicMode() == value;
                }
            };

            struct ArithmeticUnitCompatible
                : public Predicate_CRTP<ArithmeticUnitCompatible, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                ArithmeticUnit value;

                ArithmeticUnitCompatible() = default;
                ArithmeticUnitCompatible(ArithmeticUnit value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "ArithmeticUnitCompatible";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.arithmeticUnit() == value
                           || problem.arithmeticUnit() == ArithmeticUnit::Any;
                }
            };

            struct AIGreaterThanEqual
                : public Predicate_CRTP<AIGreaterThanEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                double value;

                AIGreaterThanEqual() = default;
                AIGreaterThanEqual(double value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "AIGreaterThanEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.arithmeticIntensity() >= value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.arithmeticIntensity() << " >= " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct AILessThanEqual : public Predicate_CRTP<AILessThanEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                double value;

                AILessThanEqual() = default;
                AILessThanEqual(double value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "AILessThanEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.arithmeticIntensity() <= value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.arithmeticIntensity() << " <= " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct AlphaValue : public Predicate_CRTP<AlphaValue, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                ScalarValue value;

                AlphaValue() = default;
                AlphaValue(ScalarValue value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "AlphaValue";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.alphaRestriction() == value || value == ScalarValue::Any;
                }
            };

            struct BetaValue : public Predicate_CRTP<BetaValue, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                ScalarValue value;

                BetaValue() = default;
                BetaValue(ScalarValue value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BetaValue";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.betaRestriction() == value || value == ScalarValue::Any;
                }
            };

            struct TypesEqual : public Predicate_CRTP<TypesEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                TypesEqual() = default;

                std::array<DataType, 4> value;

                static std::string Type()
                {
                    return "TypesEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.a().dataType() == value[0] && problem.b().dataType() == value[1]
                           && problem.c().dataType() == value[2]
                           && problem.d().dataType() == value[3];
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(),
                                       "(a:",
                                       value[0],
                                       ", b:",
                                       value[1],
                                       ", c:",
                                       value[2],
                                       ", d:",
                                       value[3],
                                       ")");
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << this->type() << "(a:" << problem.a().dataType() << " == " << value[0]
                           << "&& b:" << problem.b().dataType() << " == " << value[1]
                           << "&& c:" << problem.c().dataType() << " == " << value[2]
                           << "&& d:" << problem.d().dataType() << " == " << value[3]
                           << "): " << rv;

                    return rv;
                }
            };

            struct OperationIdentifierEqual
                : public Predicate_CRTP<OperationIdentifierEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                OperationIdentifierEqual() = default;

                std::string value;

                static std::string Type()
                {
                    return "OperationIdentifierEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.operationIdentifier() == value;
                }
            };

            struct BufferLoadOffsetLimitCheck
                : public Predicate_CRTP<BufferLoadOffsetLimitCheck, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                BufferLoadCheckPacket value;

                BufferLoadOffsetLimitCheck() = default;
                BufferLoadOffsetLimitCheck(BufferLoadCheckPacket value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BufferLoadOffsetLimitCheck";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    const uint64_t TWO_POW_32 = 4294967296;
                    return (problem.a().strides()[1] * value.depthUorMT0 + value.shiftPtrElemA)
                                   * problem.a().elementBytes()
                               < TWO_POW_32
                           && (problem.b().strides()[1] * value.depthUorMT1 + value.shiftPtrElemB)
                                      * problem.b().elementBytes()
                                  < TWO_POW_32;
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(),
                                       "(DU/MT0:",
                                       value.depthUorMT0,
                                       ", DU/MT1:",
                                       value.depthUorMT1,
                                       ", ShiftPtrPadElementA:",
                                       value.shiftPtrElemA,
                                       ", ShiftPtrPadElementB:",
                                       value.shiftPtrElemB,
                                       ")");
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": ("
                           << " (" << problem.a().strides()[1] << " * " << value.depthUorMT0
                           << " + " << value.shiftPtrElemA << ") * " << problem.a().elementBytes()
                           << " < 4294967296 && "
                           << " (" << problem.b().strides()[1] << " * " << value.depthUorMT1
                           << " + " << value.shiftPtrElemB << ") * " << problem.b().elementBytes()
                           << " < 4294967296"
                           << ") == " << rv;

                    return rv;
                }
            };

            struct BufferStoreOffsetLimitCheck
                : public Predicate_CRTP<BufferStoreOffsetLimitCheck, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                BufferStoreOffsetLimitCheck() = default;
                BufferStoreOffsetLimitCheck(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "BufferStoreOffsetLimitCheck";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    const uint64_t TWO_POW_32 = 4294967296;
                    return problem.d().strides()[1] * problem.d().elementBytes() * value
                           < TWO_POW_32;
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(), "(MT1:", value, ")");
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.d().strides()[1] << " * "
                           << problem.d().elementBytes() << " * " << value << " < 4294967296"
                           << ") == " << rv;

                    return rv;
                }
            };

            struct PersistentKernelCheck
                : public Predicate_CRTP<PersistentKernelCheck, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                PersistentKernelCheck() = default;

                static std::string Type()
                {
                    return "PersistentKernelCheck";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.getPersistentKernelEligibility();
                }
            };

            struct GlobalSplitUCheckMinK
                : public Predicate_CRTP<GlobalSplitUCheckMinK, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                size_t value;

                GlobalSplitUCheckMinK() = default;
                GlobalSplitUCheckMinK(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "GlobalSplitUCheckMinK";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.boundSize(0) >= value;
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(), "(value:", value, ")");
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.boundSize(0) << " >= " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct CDStridesEqual : public Predicate_CRTP<CDStridesEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                CDStridesEqual() = default;

                static std::string Type()
                {
                    return "CDStridesEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides() == problem.d().strides();
                }
            };

            struct StridedBatchedEqual
                : public Predicate_CRTP<StridedBatchedEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                StridedBatchedEqual() = default;
                StridedBatchedEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "StridedBatched";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.stridedBatched() == value;
                }
            };

            struct CUEfficiency : public Predicate_CRTP<CUEfficiency, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                CUEfficiency() = default;

                static std::string Type()
                {
                    return "CUEfficiency";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    if(problem.performanceMetric() == PerformanceMetric::CUEfficiency)
                    {
                        return true;
                    }
                    else if(problem.performanceMetric() == PerformanceMetric::Auto)
                    {
                        // True if total flops below a constant threshold
                        // Current threshold chosen naively as the flops for a
                        // 1500x1500 square matrix multiply
                        return problem.flopCount() < size_t(1500ll * 1500ll * 1500ll * 2);
                    }
                    else
                    {
                        return false;
                    }
                }
            };

            struct ExperimentalGrid : public Predicate_CRTP<ExperimentalGrid, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                ExperimentalGrid() = default;

                static std::string Type()
                {
                    return "ExperimentalGrid";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (problem.performanceMetric() == PerformanceMetric::ExperimentalGrid);
                }
            };

            struct ExperimentalDTree : public Predicate_CRTP<ExperimentalDTree, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                ExperimentalDTree() = default;

                static std::string Type()
                {
                    return "ExperimentalDTree";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return (problem.performanceMetric() == PerformanceMetric::ExperimentalDTree);
                }
            };

            struct Fp16AltImpl : public Predicate_CRTP<Fp16AltImpl, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                Fp16AltImpl() = default;

                static std::string Type()
                {
                    return "Fp16AltImpl";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.fp16AltImpl();
                }
            };

            struct Fp16AltImplRound : public Predicate_CRTP<Fp16AltImplRound, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                Fp16AltImplRound() = default;

                static std::string Type()
                {
                    return "Fp16AltImplRound";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.fp16AltImplRound();
                }
            };

            struct StochasticRoundingEqual
                : public Predicate_CRTP<StochasticRoundingEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true // false?
                };
                bool value;

                StochasticRoundingEqual() = default;
                StochasticRoundingEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "StochasticRounding";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.stochasticRounding() == value;
                }
            };

            struct F32XdlMathOpEqual : public Predicate_CRTP<F32XdlMathOpEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                DataType value;

                F32XdlMathOpEqual() = default;
                F32XdlMathOpEqual(DataType value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "F32XdlMathOp";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.f32XdlMathOp() == value;
                }
            };

            struct EqualityMatching : public Predicate_CRTP<EqualityMatching, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };

                EqualityMatching() = default;

                static std::string Type()
                {
                    return "EqualityMatching";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return true;
                }
            };
        } // namespace Contraction

        /**
 * @}
 */
    } // namespace Predicates
} // namespace Tensile
