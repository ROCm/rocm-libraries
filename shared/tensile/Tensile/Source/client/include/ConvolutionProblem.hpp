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

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor_fwd.hpp>
#include <vector>

namespace Tensile
{

    class TENSILE_API ConvolutionProblem
    {
    public:
        static const size_t MaxNumSpatialDims = 3;
        static const size_t InvalidPos;
        enum class TensorFormat
        {
            NCHW,
            NHWC,
            CNHW, // Activation Formats
            KCYX,
            CKYX // Weight Formats
        };

        class TENSILE_API BaseFormat
        {
        public:
            std::string formatIdentifier() const
            {
                return m_formatIdentifier;
            };
            TensorFormat format() const
            {
                return m_format;
            };

        protected:
            std::string  m_formatIdentifier;
            TensorFormat m_format;
        };

        class TENSILE_API ActivationFormat : public BaseFormat
        {
        public:
            ActivationFormat();
            void        FromIdentifier(std::string          identifier,
                                       size_t               formatNumSpatialDims,
                                       size_t               numSpatialDims,
                                       std::vector<size_t>* filters);
            std::string description() const;

            size_t batchPosition() const
            {
                return m_batchPosition;
            };
            size_t channelPosition() const
            {
                return m_channelPosition;
            };

            const std::vector<size_t> filterPositions() const
            {
                return m_filterPositions;
            };

            // 0,1,2 order is X,Y,Z
            // size is number of spatial dims, no extra InvalidPos values
            const std::vector<size_t> spatialPositions() const
            {
                return m_spatialPositions;
            };

        private:
            size_t m_batchPosition;
            size_t m_channelPosition;

            //! 0,1,2 order is X,Y,Z
            //! size is number of spatial dims, no extra InvalidPos values
            std::vector<size_t> m_filterPositions;
            std::vector<size_t> m_spatialPositions;
        };

        class TENSILE_API WeightFormat : public BaseFormat
        {
        public:
            WeightFormat();
            void FromIdentifier(std::string          identifier,
                                bool                 transposeCK,
                                size_t               numSpatialDims,
                                std::vector<size_t>* filters);

            std::string description() const;
            size_t      coutPosition() const
            {
                return m_coutPosition;
            };
            size_t cinPosition() const
            {
                return m_cinPosition;
            };
            const std::vector<size_t> filterPositions() const
            {
                return m_filterPositions;
            };

        private:
            size_t m_cinPosition;
            size_t m_coutPosition;
            //! 0,1,2 order is X,Y,Z
            //! always MaxNumSpatialDims elements.  Positions may be InvalidPos
            std::vector<size_t> m_filterPositions;
        };

        //! Use for formatB and output which can take either format
        struct ComboFormat
        {
            const ActivationFormat& activation() const
            {
                return m_activation;
            };
            const WeightFormat& weights() const
            {
                return m_weights;
            };
            ActivationFormat& activationW()
            {
                return m_activation;
            };
            WeightFormat& weightsW()
            {
                return m_weights;
            };

        private:
            ActivationFormat m_activation;
            WeightFormat     m_weights;
        };

        struct TENSILE_API LoopCounts
        {
            LoopCounts()
                : scount(MaxNumSpatialDims, 1){};
            void                    setupForData(ConvolutionProblem const& convProblem,
                                                 ContractionProblem const& problem);
            void                    setupFormat(ConvolutionProblem const& convProblem);
            const ActivationFormat& formatA() const
            {
                return m_formatA;
            };
            const ComboFormat& formatB() const
            {
                return m_formatB;
            };
            const ComboFormat& formatD() const
            {
                return m_formatD;
            };
            std::string         description() const;
            size_t              batchCount;
            size_t              cinCount;
            size_t              coutCount;
            std::vector<size_t> scount;
            std::vector<size_t> spatialCount; //whd (original convolution input tensor)
            std::vector<size_t> filterCount; //xyz
            std::vector<size_t> strideCount; //vu#
            std::vector<size_t> dilationCount; //jl^
            std::vector<size_t> padStartCount; //qp$
            std::vector<size_t> padEndCount; //q_p_$_

            ActivationFormat m_formatA;
            ComboFormat      m_formatB;
            ComboFormat      m_formatD; // output tensor
        };

        ConvolutionProblem() {}

        void             FromIdentifier(std::string identifier);
        void             setupFormat(std::vector<size_t>& filters);
        TensorDescriptor setupDataActivation(LoopCounts const&         counts,
                                             ContractionProblem const& problem) const;
        TensorDescriptor setupDataOutput(LoopCounts const&         counts,
                                         ContractionProblem const& problem) const;
        TensorDescriptor setupForwardWeights(LoopCounts const&         counts,
                                             ContractionProblem const& problem) const;
        void             validate(const ContractionProblem&             problem,
                                  const ConvolutionProblem::LoopCounts& counts) const;

        //! Number of spatial dims after packing.
        size_t numSpatialDims() const
        {
            return m_numSpatialDims;
        }

        std::string        description() const;
        std::string const& operationIdentifier() const
        {
            return m_operationIdentifier;
        }
        std::string const& AIdentifier() const
        {
            return m_AIdentifier;
        }
        std::string const& BIdentifier() const
        {
            return m_BIdentifier;
        }
        std::string const& DIdentifier() const
        {
            return m_DIdentifier;
        }
        size_t numFormatSpatialDims() const
        {
            return m_numFormatSpatialDims;
        }

    private:
        //! ConvolutionForward, ConvolutionBackwardData, ConvolutionBackwardWeights
        std::string m_operationIdentifier;
        std::string m_AIdentifier;
        std::string m_BIdentifier;
        std::string m_DIdentifier;

        size_t m_groups = 1;

        size_t m_numSpatialDims       = 0;
        size_t m_numFormatSpatialDims = 0;
    };

    TENSILE_API std::ostream& operator<<(std::ostream&             stream,
                                         ConvolutionProblem const& convolution);
} // namespace Tensile
