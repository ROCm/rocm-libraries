/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Properties.hpp>

#include <cstddef>

namespace TensileLite
{
    /**
     * \addtogroup Properties
     *
     * @brief An MLFeature is a Property whose value is of type `float`.
     *
     * This allows it to be used as an input to ML models.
     */
    namespace MLFeatures
    {
        /* Scale factors used for partially calculating granularities */
        struct CUGranularityScaleFactors
        {
            float mt0Scale; // 1/mt0
            float mt1Scale; // 1/mt1
            float cuScale;
        };

        struct WaveGranularityScaleFactors
        {
            CUGranularityScaleFactors cuFactors;
            float                     waveScale;
        };

        float tilesPerCU(ContractionProblemGemm const&    problem,
                         CUGranularityScaleFactors const& cuFactors);

        std::ostream& operator<<(std::ostream& stream, CUGranularityScaleFactors const& cugsf);
        std::ostream& operator<<(std::ostream& stream, WaveGranularityScaleFactors const& wgsf);

        /**
         * @brief A Property whose value is of type `float`.
         */
        template <typename Object>
        using MLFeature = Property<Object, float>;

        /**
         * \copydoc TensileLite::Property_CRTP
         */
        template <typename Class, typename Object>
        using MLFeature_CRTP = Property_CRTP<Class, Object, float>;

        /**
         * \ingroup Properties
         * \defgroup MLFeatures MLFeature Classes
         *
         * @brief Individual MLFeature classes.
         */

        /**
         * \addtogroup MLFeatures
         * @{
         */
        struct FreeSizeA : public MLFeature_CRTP<FreeSizeA, ContractionProblemGemm>
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

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                return (float)problem.freeSizeA(index);
            }
        };

        struct FreeSizeB : public MLFeature_CRTP<FreeSizeB, ContractionProblemGemm>
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

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                return (float)problem.freeSizeB(index);
            }
        };

        struct BatchSize : public MLFeature_CRTP<BatchSize, ContractionProblemGemm>
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

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                return (float)problem.batchSize(index);
            }
        };

        struct BoundSize : public MLFeature_CRTP<BoundSize, ContractionProblemGemm>
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

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                return (float)problem.boundSize(index);
            }
        };

        struct Tile0Granularity : public MLFeature_CRTP<Tile0Granularity, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            float value; // 1/mt0

            static std::string Type()
            {
                return "Tile0Granularity";
            }

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                float numTiles = problem.freeSizeA(0) * value;
                return ContractionSolution::computeGranularity(numTiles);
            }
        };

        struct Tile1Granularity : public MLFeature_CRTP<Tile1Granularity, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            float value; // 1/mt1

            static std::string Type()
            {
                return "Tile1Granularity";
            }

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                float numTiles = problem.freeSizeB(0) * value;
                return ContractionSolution::computeGranularity(numTiles);
            }
        };

        struct CUGranularity : public MLFeature_CRTP<CUGranularity, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = false,
                HasValue = true,

            };
            CUGranularityScaleFactors value;

            static std::string Type()
            {
                return "CUGranularity";
            }

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                return ContractionSolution::computeGranularity(tilesPerCU(problem, value));
            }
        };

        struct WavesPerSIMD : public MLFeature_CRTP<WavesPerSIMD, ContractionProblemGemm>
        {
            enum
            {
                HasIndex = false,
                HasValue = true,

            };
            WaveGranularityScaleFactors value;

            static std::string Type()
            {
                return "WavesPerSIMD";
            }

            virtual float operator()(ContractionProblemGemm const& problem) const
            {
                return ceil(tilesPerCU(problem, value.cuFactors)) * value.waveScale;
            }
        };

        /**
         * @}
         */
    } // namespace MLFeatures
} // namespace TensileLite
