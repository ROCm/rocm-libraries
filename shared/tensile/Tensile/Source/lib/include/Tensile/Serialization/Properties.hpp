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

#include <Tensile/Serialization/Base.hpp>

#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/PropertyMatching.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Serialization
    {

        template <typename Object, typename Value, typename IO>
        struct MappingTraits<std::shared_ptr<Property<Object, Value>>, IO>
            : public BaseClassMappingTraits<Property<Object, Value>, IO, true>
        {
        };

        template <typename IO>
        struct SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>,
                  Property<ContractionProblem, size_t>,
                  IO>
        {
            using Self = SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>;
            using Base = DefaultSubclassMappingTraits<
                SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>,
                Property<ContractionProblem, size_t>,
                IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap({Base::template Pair<Contraction::FreeSizeA>(),
                                    Base::template Pair<Contraction::FreeSizeB>(),
                                    Base::template Pair<Contraction::BatchSize>(),
                                    Base::template Pair<Contraction::BoundSize>(),
                                    Base::template Pair<Contraction::AStride>(),
                                    Base::template Pair<Contraction::BStride>(),
                                    Base::template Pair<Contraction::CStride>(),
                                    Base::template Pair<Contraction::DStride>()});
            }
        };

        template <typename IO>
        struct SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>,
                  Property<ContractionProblem, std::string>,
                  IO>
        {
            using Self = SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>;
            using Base = DefaultSubclassMappingTraits<
                SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>,
                Property<ContractionProblem, std::string>,
                IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap({Base::template Pair<Contraction::OperationIdentifier>()});
            }
        };

        template <typename IO>
        const typename SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>::SubclassMap
            SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>::subclasses
            = SubclassMappingTraits<Property<ContractionProblem, size_t>, IO>::GetSubclasses();

        template <typename IO>
        const typename SubclassMappingTraits<Property<ContractionProblem, std::string>,
                                             IO>::SubclassMap
            SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>::subclasses
            = SubclassMappingTraits<Property<ContractionProblem, std::string>, IO>::GetSubclasses();

        template <typename IO>
        struct MappingTraits<Contraction::FreeSizeA, IO>
            : public AutoMappingTraits<Contraction::FreeSizeA, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::FreeSizeB, IO>
            : public AutoMappingTraits<Contraction::FreeSizeB, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::BatchSize, IO>
            : public AutoMappingTraits<Contraction::BatchSize, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::BoundSize, IO>
            : public AutoMappingTraits<Contraction::BoundSize, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::AStride, IO>
            : public AutoMappingTraits<Contraction::AStride, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::BStride, IO>
            : public AutoMappingTraits<Contraction::BStride, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::CStride, IO>
            : public AutoMappingTraits<Contraction::CStride, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::DStride, IO>
            : public AutoMappingTraits<Contraction::DStride, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<Contraction::OperationIdentifier, IO>
            : public AutoMappingTraits<Contraction::OperationIdentifier, IO>
        {
        };
    } // namespace Serialization
} // namespace Tensile
