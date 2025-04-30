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

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Predicates.hpp>

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Predicates.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>,
                  Predicates::Predicate<ContractionProblem>,
                  IO>
        {
            using Self = SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>;
            using Base = DefaultSubclassMappingTraits<
                SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>,
                Predicates::Predicate<ContractionProblem>,
                IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            using Generic = PredicateMappingTraits<ContractionProblem, IO>;

            static SubclassMap GetSubclasses()
            {
                SubclassMap rv({
                    Base::template Pair<Predicates::Contraction::Free0SizeMultiple>(),
                    Base::template Pair<Predicates::Contraction::Free1SizeMultiple>(),
                    Base::template Pair<Predicates::Contraction::BatchSizeMultiple>(),
                    Base::template Pair<Predicates::Contraction::BatchSizeEqual>(),
                    Base::template Pair<Predicates::Contraction::BoundSizeMultiple>(),
                    Base::template Pair<Predicates::Contraction::MaxProblemSizeGreaterThan>(),
                    Base::template Pair<Predicates::Contraction::LeadingFree0SizesGreaterOrEqual>(),
                    Base::template Pair<Predicates::Contraction::LeadingFree1SizesGreaterOrEqual>(),
                    Base::template Pair<Predicates::Contraction::SizeEqual>(),
                    Base::template Pair<Predicates::Contraction::SizeGreaterThan>(),
                    Base::template Pair<Predicates::Contraction::SizeLessThan>(),
                    Base::template Pair<Predicates::Contraction::SizeMultiple>(),
                    Base::template Pair<Predicates::Contraction::SizeInRange>(),
                    Base::template Pair<Predicates::Contraction::StrideAEqual>(),
                    Base::template Pair<Predicates::Contraction::StrideBEqual>(),
                    Base::template Pair<Predicates::Contraction::StrideCEqual>(),
                    Base::template Pair<Predicates::Contraction::StrideDEqual>(),
                    Base::template Pair<Predicates::Contraction::LDCEqualsLDD>(),
                    Base::template Pair<Predicates::Contraction::CEqualsD>(),
                    Base::template Pair<Predicates::Contraction::AIGreaterThanEqual>(),
                    Base::template Pair<Predicates::Contraction::AILessThanEqual>(),
                    Base::template Pair<Predicates::Contraction::AlphaValue>(),
                    Base::template Pair<Predicates::Contraction::BetaValue>(),
                    Base::template Pair<Predicates::Contraction::BetaZero>(),
                    Base::template Pair<Predicates::Contraction::BetaOne>(),
                    Base::template Pair<Predicates::Contraction::HighPrecisionAccumulateEqual>(),
                    Base::template Pair<Predicates::Contraction::KernelLanguageCompatible>(),
                    Base::template Pair<Predicates::Contraction::DeterministicModeEqual>(),
                    Base::template Pair<Predicates::Contraction::ArithmeticUnitCompatible>(),
                    Base::template Pair<Predicates::Contraction::TypesEqual>(),
                    Base::template Pair<Predicates::Contraction::OperationIdentifierEqual>(),
                    Base::template Pair<Predicates::Contraction::BufferLoadOffsetLimitCheck>(),
                    Base::template Pair<Predicates::Contraction::BufferStoreOffsetLimitCheck>(),
                    Base::template Pair<Predicates::Contraction::PersistentKernelCheck>(),
                    Base::template Pair<Predicates::Contraction::GlobalSplitUCheckMinK>(),
                    Base::template Pair<Predicates::Contraction::CDStridesEqual>(),
                    Base::template Pair<Predicates::Contraction::StridedBatchedEqual>(),
                    Base::template Pair<Predicates::Contraction::CUEfficiency>(),
                    Base::template Pair<Predicates::Contraction::ExperimentalGrid>(),
                    Base::template Pair<Predicates::Contraction::ExperimentalDTree>(),
                    Base::template Pair<Predicates::Contraction::Fp16AltImpl>(),
                    Base::template Pair<Predicates::Contraction::Fp16AltImplRound>(),
                    Base::template Pair<Predicates::Contraction::F32XdlMathOpEqual>(),
                    Base::template Pair<Predicates::Contraction::EqualityMatching>(),
                    Base::template Pair<Predicates::Contraction::StochasticRoundingEqual>(),
                });

                auto gmap = Generic::GetSubclasses();
                rv.insert(gmap.begin(), gmap.end());

                return rv;
            }
        };

        template <typename IO>
        using ContractionProblemPredicateSMT
            = SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>;

        template <typename IO>
        const typename ContractionProblemPredicateSMT<IO>::SubclassMap
            SubclassMappingTraits<Predicates::Predicate<ContractionProblem>, IO>::subclasses
            = ContractionProblemPredicateSMT<IO>::GetSubclasses();

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::Free0SizeMultiple, IO>
            : public AutoMappingTraits<Predicates::Contraction::Free0SizeMultiple, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::Free1SizeMultiple, IO>
            : public AutoMappingTraits<Predicates::Contraction::Free1SizeMultiple, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BatchSizeMultiple, IO>
            : public AutoMappingTraits<Predicates::Contraction::BatchSizeMultiple, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BatchSizeEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::BatchSizeEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BoundSizeMultiple, IO>
            : public AutoMappingTraits<Predicates::Contraction::BoundSizeMultiple, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::MaxProblemSizeGreaterThan, IO>
            : public AutoMappingTraits<Predicates::Contraction::MaxProblemSizeGreaterThan, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::LeadingFree0SizesGreaterOrEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::LeadingFree0SizesGreaterOrEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::LeadingFree1SizesGreaterOrEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::LeadingFree1SizesGreaterOrEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::SizeEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::SizeEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::SizeGreaterThan, IO>
            : public AutoMappingTraits<Predicates::Contraction::SizeGreaterThan, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::SizeLessThan, IO>
            : public AutoMappingTraits<Predicates::Contraction::SizeLessThan, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::SizeMultiple, IO>
            : public AutoMappingTraits<Predicates::Contraction::SizeMultiple, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::StrideAEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::StrideAEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::StrideBEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::StrideBEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::StrideCEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::StrideCEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::StrideDEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::StrideDEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::LDCEqualsLDD, IO>
            : public AutoMappingTraits<Predicates::Contraction::LDCEqualsLDD, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::CEqualsD, IO>
            : public AutoMappingTraits<Predicates::Contraction::CEqualsD, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::AIGreaterThanEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::AIGreaterThanEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::AILessThanEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::AILessThanEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::AlphaValue, IO>
            : public AutoMappingTraits<Predicates::Contraction::AlphaValue, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BetaValue, IO>
            : public AutoMappingTraits<Predicates::Contraction::BetaValue, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BetaZero, IO>
            : public AutoMappingTraits<Predicates::Contraction::BetaZero, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BetaOne, IO>
            : public AutoMappingTraits<Predicates::Contraction::BetaOne, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::HighPrecisionAccumulateEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::HighPrecisionAccumulateEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::ArithmeticUnitCompatible, IO>
            : public AutoMappingTraits<Predicates::Contraction::ArithmeticUnitCompatible, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::KernelLanguageCompatible, IO>
            : public AutoMappingTraits<Predicates::Contraction::KernelLanguageCompatible, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::DeterministicModeEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::DeterministicModeEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::TypesEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::TypesEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::OperationIdentifierEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::OperationIdentifierEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BufferLoadOffsetLimitCheck, IO>
            : public AutoMappingTraits<Predicates::Contraction::BufferLoadOffsetLimitCheck, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::BufferStoreOffsetLimitCheck, IO>
            : public AutoMappingTraits<Predicates::Contraction::BufferStoreOffsetLimitCheck, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::PersistentKernelCheck, IO>
            : public AutoMappingTraits<Predicates::Contraction::PersistentKernelCheck, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::GlobalSplitUCheckMinK, IO>
            : public AutoMappingTraits<Predicates::Contraction::GlobalSplitUCheckMinK, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::CDStridesEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::CDStridesEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::StridedBatchedEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::StridedBatchedEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::CUEfficiency, IO>
            : public AutoMappingTraits<Predicates::Contraction::CUEfficiency, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::ExperimentalGrid, IO>
            : public AutoMappingTraits<Predicates::Contraction::ExperimentalGrid, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::ExperimentalDTree, IO>
            : public AutoMappingTraits<Predicates::Contraction::ExperimentalDTree, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::Fp16AltImpl, IO>
            : public AutoMappingTraits<Predicates::Contraction::Fp16AltImpl, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::Fp16AltImplRound, IO>
            : public AutoMappingTraits<Predicates::Contraction::Fp16AltImplRound, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::F32XdlMathOpEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::F32XdlMathOpEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::EqualityMatching, IO>
            : public AutoMappingTraits<Predicates::Contraction::EqualityMatching, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::StochasticRoundingEqual, IO>
            : public AutoMappingTraits<Predicates::Contraction::StochasticRoundingEqual, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::Range, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, Predicates::Contraction::Range& range)
            {
                iot::mapOptional(io, "min", range.min);
                iot::mapOptional(io, "max", range.max);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<Predicates::Contraction::SizeInRange, IO>
            : public AutoMappingTraits<Predicates::Contraction::SizeInRange, IO>
        {
        };
    } // namespace Serialization
} // namespace Tensile
