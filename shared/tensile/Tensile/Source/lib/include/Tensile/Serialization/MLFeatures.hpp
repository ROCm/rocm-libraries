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

#include <Tensile/MLFeatures.hpp>
#include <Tensile/PropertyMatching.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<MLFeatures::CUGranularityScaleFactors, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, MLFeatures::CUGranularityScaleFactors& cugsf)
            {
                iot::mapRequired(io, "mt0", cugsf.mt0Scale);
                iot::mapRequired(io, "mt1", cugsf.mt1Scale);
                iot::mapRequired(io, "cus", cugsf.cuScale);
            }

            const static bool flow = true;
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::WaveGranularityScaleFactors, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, MLFeatures::WaveGranularityScaleFactors& wgsf)
            {
                iot::mapRequired(io, "mt0", wgsf.cuFactors.mt0Scale);
                iot::mapRequired(io, "mt1", wgsf.cuFactors.mt1Scale);
                iot::mapRequired(io, "cus", wgsf.cuFactors.cuScale);
                iot::mapRequired(io, "ws", wgsf.waveScale);
            }

            const static bool flow = true;
        };

        // Set Flow
        template <typename IO>
        struct MappingTraits<std::shared_ptr<MLFeatures::MLFeature<ContractionProblem>>, IO>
            : public BaseClassMappingTraits<MLFeatures::MLFeature<ContractionProblem>, IO, true>
        {
        };

        template <typename IO>
        struct SubclassMappingTraits<MLFeatures::MLFeature<ContractionProblem>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<MLFeatures::MLFeature<ContractionProblem>, IO>,
                  MLFeatures::MLFeature<ContractionProblem>,
                  IO>
        {
            using Self = SubclassMappingTraits<MLFeatures::MLFeature<ContractionProblem>, IO>;
            using Base = DefaultSubclassMappingTraits<
                SubclassMappingTraits<MLFeatures::MLFeature<ContractionProblem>, IO>,
                MLFeatures::MLFeature<ContractionProblem>,
                IO>;
            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static typename Base::SubclassMap GetSubclasses()
            {
                return SubclassMap({Base::template Pair<MLFeatures::FreeSizeA>(),
                                    Base::template Pair<MLFeatures::FreeSizeB>(),
                                    Base::template Pair<MLFeatures::BatchSize>(),
                                    Base::template Pair<MLFeatures::BoundSize>(),
                                    Base::template Pair<MLFeatures::Tile0Granularity>(),
                                    Base::template Pair<MLFeatures::Tile1Granularity>(),
                                    Base::template Pair<MLFeatures::CUGranularity>(),
                                    Base::template Pair<MLFeatures::WavesPerSIMD>(),
                                    Base::template Pair<MLFeatures::ProblemScaleMeasure>()});
            }
        };

        template <typename IO>
        using ContractionProblemFeatureSMT
            = SubclassMappingTraits<MLFeatures::MLFeature<ContractionProblem>, IO>;

        template <typename IO>
        const typename ContractionProblemFeatureSMT<IO>::SubclassMap
            ContractionProblemFeatureSMT<IO>::subclasses
            = ContractionProblemFeatureSMT<IO>::GetSubclasses();

        template <typename IO>
        struct MappingTraits<MLFeatures::FreeSizeA, IO>
            : public AutoMappingTraits<MLFeatures::FreeSizeA, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::FreeSizeB, IO>
            : public AutoMappingTraits<MLFeatures::FreeSizeB, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::BatchSize, IO>
            : public AutoMappingTraits<MLFeatures::BatchSize, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::BoundSize, IO>
            : public AutoMappingTraits<MLFeatures::BoundSize, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::Tile0Granularity, IO>
            : public AutoMappingTraits<MLFeatures::Tile0Granularity, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::ProblemScaleMeasure, IO>
            : public AutoMappingTraits<MLFeatures::ProblemScaleMeasure, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::CUGranularity, IO>
            : public AutoMappingTraits<MLFeatures::CUGranularity, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::WavesPerSIMD, IO>
            : public AutoMappingTraits<MLFeatures::WavesPerSIMD, IO>
        {
        };

        template <typename IO>
        struct MappingTraits<MLFeatures::Tile1Granularity, IO>
            : public AutoMappingTraits<MLFeatures::Tile1Granularity, IO>
        {
        };
    } // namespace Serialization
} // namespace Tensile
