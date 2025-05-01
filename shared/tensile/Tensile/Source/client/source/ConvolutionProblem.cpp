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

#include <ConvolutionProblem.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>

namespace Tensile
{
    const size_t ConvolutionProblem::InvalidPos = -1;
    ConvolutionProblem::ActivationFormat::ActivationFormat() {}
    void ConvolutionProblem::ActivationFormat::FromIdentifier(std::string identifier,
                                                              size_t      formatNumSpatialDims,
                                                              size_t      numSpatialDims,
                                                              std::vector<size_t>* filters)
    {
        // summation dimensions immediately follow the spatial dim(s)
        m_formatIdentifier = identifier;
        m_filterPositions.clear();
        m_spatialPositions.clear();
        if(identifier == "NCHW")
        {
            assert(formatNumSpatialDims == 2);
            m_format = TensorFormat::NCHW;

            size_t position = 0;

            if(filters)
                for(int fi = 0; fi < filters->size(); fi++)
                {
                    if((*filters)[fi] != 1)
                        m_filterPositions.push_back(position++);
                    else
                        m_filterPositions.push_back(InvalidPos);
                }
            for(auto si = 0; si < numSpatialDims; si++)
                m_spatialPositions.push_back(position++);

            m_channelPosition = position++;
            m_batchPosition   = position++;
        }
        else if(identifier == "NHWC")
        {
            assert(formatNumSpatialDims == 2);
            m_format = TensorFormat::NHWC;

            m_channelPosition = 0;

            size_t position = m_channelPosition + 1;
            if(filters)
                for(int fi = 0; fi < filters->size(); fi++)
                {
                    if((*filters)[fi] != 1)
                        m_filterPositions.push_back(position++);
                    else
                        m_filterPositions.push_back(InvalidPos);
                }

            // assume spatial dimensions are collapsed here:
            std::cout << "FIXME\n";
            for(auto si = 0; si < numSpatialDims; si++)
                m_spatialPositions.push_back(position++);
            m_batchPosition = position++;
        }
        else if(identifier == "CNHW")
        {
            assert(formatNumSpatialDims == 2);
            m_format = TensorFormat::CNHW;

            size_t position = 0;
            if(filters)
                for(int fi = 0; fi < filters->size(); fi++)
                {
                    if((*filters)[fi] != 1)
                        m_filterPositions.push_back(position++);
                    else
                        m_filterPositions.push_back(InvalidPos);
                }
            // assume spatial dimensions are collapsed here:
            std::cout << "FIXME\n";
            for(auto si = 0; si < numSpatialDims; si++)
                m_spatialPositions.push_back(position++);
            m_batchPosition   = position++;
            m_channelPosition = position++;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid tensor format in convolution identifier:")
                                     + identifier);
        }
    }

    std::string ConvolutionProblem::ActivationFormat::description() const
    {
        std::ostringstream rv;
        rv << m_formatIdentifier << "_"
           << " batchPosition=" << m_batchPosition << " channelPosition=" << m_channelPosition;
        rv << " spatialPositions[]=";
        for(auto i = 0; i < m_spatialPositions.size(); i++)
        {
            if(i != 0)
                rv << ",";
            rv << static_cast<int>(m_spatialPositions[i]);
        }

        rv << " filterPositions[]=";
        for(auto i = 0; i < m_filterPositions.size(); i++)
        {
            if(i != 0)
                rv << ",";
            rv << static_cast<int>(m_filterPositions[i]);
        }

        return rv.str();
    }

    ConvolutionProblem::WeightFormat::WeightFormat() {}

    void ConvolutionProblem::WeightFormat::FromIdentifier(std::string          identifier,
                                                          bool                 transposeCK,
                                                          size_t               formatNumSpatialDims,
                                                          std::vector<size_t>* filters)
    {
        m_formatIdentifier = identifier;
        if(identifier == "KCYX")
            m_format = TensorFormat::KCYX;
        else if(identifier == "CKYX")
            m_format = TensorFormat::CKYX;
        else
            throw std::runtime_error(std::string("Invalid weight format in convolution identifier:")
                                     + identifier);

        m_filterPositions.clear();
        if((identifier == "KCYX" && !transposeCK) || (identifier == "CKYX" && transposeCK))
        {

            assert(formatNumSpatialDims == 2);

            size_t position = 0;
            if(filters)
                // Weight dims are assigned in reverse order for optimal Tensile summation
                // processing
                for(int fi = 0; fi < filters->size(); fi++)
                {
                    if((*filters)[fi] != 1)
                        m_filterPositions.push_back(position++);
                    else
                        m_filterPositions.push_back(InvalidPos);
                }
            m_cinPosition  = position++;
            m_coutPosition = position;
        }
        else if((identifier == "CKYX" && !transposeCK) || (identifier == "KCYX" && transposeCK))
        {
            assert(formatNumSpatialDims == 2);
            size_t position = 0;
            if(filters)
                for(int fi = 0; fi < filters->size(); fi++)
                {
                    if((*filters)[fi] != 1)
                        m_filterPositions.push_back(position++);
                    else
                        m_filterPositions.push_back(InvalidPos);
                }
            m_coutPosition = position;
            m_cinPosition  = m_coutPosition + 1;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid weight format in convolution identifier:")
                                     + identifier);
        }
    }
    std::string ConvolutionProblem::WeightFormat::description() const
    {
        std::ostringstream rv;
        rv << m_formatIdentifier << "_"
           << " coutPosition=" << m_coutPosition << " cinPosition=" << m_cinPosition
           << " filterPositions[]=";
        for(auto i = 0; i < m_filterPositions.size(); i++)
        {
            if(i != 0)
                rv << ",";
            rv << static_cast<int>(m_filterPositions[i]);
        }
        return rv.str();
    }
    void ConvolutionProblem::LoopCounts::setupFormat(ConvolutionProblem const& convProblem)
    {
        std::string operationIdentifier = convProblem.operationIdentifier();
        if(operationIdentifier == "ConvolutionForward"
           || operationIdentifier == "ConvolutionBackwardData")
        {
            m_formatA.FromIdentifier(convProblem.AIdentifier(),
                                     convProblem.numFormatSpatialDims(),
                                     convProblem.numSpatialDims(),
                                     &filterCount);
            m_formatB.weightsW().FromIdentifier(convProblem.BIdentifier(),
                                                operationIdentifier == "ConvolutionBackwardData",
                                                convProblem.numFormatSpatialDims(),
                                                &filterCount);
            m_formatD.activationW().FromIdentifier(convProblem.DIdentifier(),
                                                   convProblem.numFormatSpatialDims(),
                                                   convProblem.numSpatialDims(),
                                                   nullptr);
        }
        else if(operationIdentifier == "ConvolutionBackwardWeights")
        {
        }
        else
        {
            throw std::runtime_error(std::string("Invalid operation identifier:")
                                     + operationIdentifier);
        }
    }

    // Setup for forward or backward data
    void ConvolutionProblem::LoopCounts::setupForData(ConvolutionProblem const& convProblem,
                                                      ContractionProblem const& problem)
    {
        size_t                        numSpatial       = convProblem.numFormatSpatialDims();
        std::vector<size_t>           convProblemSizes = problem.convProblemSizes();
        std::vector<size_t>::iterator it               = convProblemSizes.begin();

        //convolution problem size must be six times of numSpatial
        assert(convProblemSizes.size() == 6 * numSpatial);

        spatialCount.assign(it, it + numSpatial);
        filterCount.assign(it += numSpatial, it + numSpatial);
        strideCount.assign(it += numSpatial, it + numSpatial);
        dilationCount.assign(it += numSpatial, it + numSpatial);
        padStartCount.assign(it += numSpatial, it + numSpatial);
        padEndCount.assign(it += numSpatial, it + numSpatial);

        setupFormat(convProblem);

        batchCount = problem.a().sizes()[m_formatA.batchPosition()];
        cinCount   = problem.a().sizes()[m_formatA.channelPosition()];
        coutCount  = problem.b().sizes()[m_formatB.weights().coutPosition()];
        for(int si = 0; si < m_formatA.spatialPositions().size(); si++)
        {
            auto       spatialPositionA   = m_formatA.spatialPositions()[si];
            auto const problemSpatialSize = problem.a().sizes()[spatialPositionA];
            scount[si]                    = problemSpatialSize;
        }
    }

    std::string ConvolutionProblem::LoopCounts::description() const
    {
        std::ostringstream rv;
        rv << "  formatA = " << m_formatA.description() << std::endl;
        rv << "  formatB = " << m_formatB.weights().description() << std::endl;
        rv << "  batchCount=" << batchCount << " coutCount=" << coutCount
           << " scalarCount_dhw=" << scount[2] << "x" << scount[1] << "x" << scount[0]
           << " cinCount=" << cinCount << std::endl;

        rv << "  spatialCount ";
        for(auto i = spatialCount.begin(); i != spatialCount.end(); ++i)
            rv << *i << ',';
        rv << "filterCount ";
        for(auto i = filterCount.begin(); i != filterCount.end(); ++i)
            rv << *i << ',';
        rv << "strideCount ";
        for(auto i = strideCount.begin(); i != strideCount.end(); ++i)
            rv << *i << ',';
        rv << "dilationCount ";
        for(auto i = dilationCount.begin(); i != dilationCount.end(); ++i)
            rv << *i << ',';
        rv << "padStartCount ";
        for(auto i = padStartCount.begin(); i != padStartCount.end(); ++i)
            rv << *i << ',';
        rv << "padEndCount ";
        for(auto i = padEndCount.begin(); i != padEndCount.end(); ++i)
            rv << *i << ',';
        return rv.str();
    }

    void ConvolutionProblem::FromIdentifier(std::string identifier)
    {
        // example identifier:
        // ConvolutionForward_NCHW_KCHW_NCHW_filter:3x3x1_stride:1x1x1_dilation:1x1x1_groups:1
        std::vector<std::string> parts;
        boost::split(parts, identifier, boost::algorithm::is_any_of("_"));

        if(parts.size() < 4)
            // id, formatA, formatB, outputTensor
            throw std::runtime_error(
                std::string("Invalid convolution identifier- must have at least 3 sections:")
                + identifier);

        m_operationIdentifier       = parts[0];
        m_AIdentifier               = parts[1];
        m_BIdentifier               = parts[2];
        m_DIdentifier               = parts[3];
        size_t formatNumSpatialDims = parts[1].size() - 2;
        m_numFormatSpatialDims      = formatNumSpatialDims;

        for(auto part = parts.begin() + 4; part != parts.end(); part++)
        {
            std::vector<std::string> flags;
            boost::split(flags, *part, boost::algorithm::is_any_of(":"));
            assert(flags.size() == 2); // must be key:value pair

            if(flags[0] == "spatialDims")
                m_numSpatialDims = boost::lexical_cast<size_t>(flags[1]);
            else if(flags[0] == "indices")
            {
            }
            else if(flags[0] == "groups")
            {
                m_groups = boost::lexical_cast<int>(flags[1]);
                assert(m_groups == 1); // not supported yet
            }
            else
                throw std::runtime_error(std::string("Invalid flag in convolution identifier:")
                                         + flags[0]);

            std::cout << flags[0] << ":::" << flags[1] << "\n";
        };
    }

    TensorDescriptor
        ConvolutionProblem::setupDataActivation(ConvolutionProblem::LoopCounts const& counts,
                                                ContractionProblem const&             problem) const
    {
        // Mimic the expected dimension order in formatA:
        std::vector<size_t>  activationDims;
        std::vector<int64_t> activationStri;
        int64_t              batchStride;
        int64_t              realSpatialSize = 1;

        auto formatA = counts.formatA();
        switch(formatA.format())
        {
        case ConvolutionProblem::TensorFormat::NCHW:
            for(int fi = 0; fi < counts.filterCount.size(); fi++)
                if(formatA.filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                {
                    activationDims.push_back(counts.filterCount[fi]);
                    activationStri.push_back(fi == 0 ? counts.dilationCount[fi]
                                                     : counts.dilationCount[fi]
                                                           * counts.spatialCount[fi - 1]);
                }
            for(int si = 0; si < formatA.spatialPositions().size(); si++)
            {
                activationDims.push_back(counts.scount[si]);
                activationStri.push_back(si == 0 ? counts.strideCount[si]
                                                 : counts.strideCount[si]
                                                       * counts.spatialCount[si - 1]);
            }
            for(int si = 0; si < m_numFormatSpatialDims; si++)
                realSpatialSize *= counts.spatialCount[si];
            activationDims.push_back(problem.a().sizes()[formatA.channelPosition()]);
            activationStri.push_back(realSpatialSize);
            activationDims.push_back(problem.a().sizes()[formatA.batchPosition()]);
            batchStride = realSpatialSize * problem.a().sizes()[formatA.channelPosition()];
            activationStri.push_back(batchStride);
            break;
        case ConvolutionProblem::TensorFormat::NHWC:
            assert(0); // need strides
            activationDims.push_back(problem.a().sizes()[formatA.channelPosition()]);
            for(int fi = 0; fi < counts.filterCount.size(); fi++)
                if(formatA.filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                    activationDims.push_back(counts.filterCount[fi]);
            for(int si = 0; si < formatA.spatialPositions().size(); si++)
                activationDims.push_back(counts.scount[si]);
            activationDims.push_back(problem.a().sizes()[formatA.batchPosition()]);
        case ConvolutionProblem::TensorFormat::CNHW:
            assert(0); // need strides
            for(int fi = 0; fi < counts.filterCount.size(); fi++)
                if(formatA.filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                    activationDims.push_back(counts.filterCount[fi]);
            for(int si = 0; si < formatA.spatialPositions().size(); si++)
                activationDims.push_back(counts.scount[si]);
            activationDims.push_back(problem.a().sizes()[formatA.batchPosition()]);
            activationDims.push_back(problem.a().sizes()[formatA.channelPosition()]);
            break;
        default:
            throw std::runtime_error("unknown formatA");
        };
        TensorDescriptor rv(problem.a().dataType(),
                            activationDims.begin(),
                            activationDims.end(),
                            activationStri.begin(),
                            activationStri.end(),
                            problem.a().offset());
        return rv;
    }

    TensorDescriptor
        ConvolutionProblem::setupDataOutput(ConvolutionProblem::LoopCounts const& counts,
                                            ContractionProblem const&             problem) const
    {
        std::vector<size_t> outputDims;
        auto                formatA = counts.formatA();
        auto                formatD = counts.formatD();
        switch(formatD.activation().format())
        {
        case ConvolutionProblem::TensorFormat::NCHW:
            for(int si = 0; si < formatA.spatialPositions().size(); si++)
                outputDims.push_back(counts.scount[si]);
            outputDims.push_back(problem.d().sizes()[formatD.activation().channelPosition()]);
            outputDims.push_back(problem.d().sizes()[formatD.activation().batchPosition()]);
            break;
        case ConvolutionProblem::TensorFormat::NHWC:
            outputDims.push_back(problem.d().sizes()[formatD.activation().channelPosition()]);
            for(int si = 0; si < formatA.spatialPositions().size(); si++)
                outputDims.push_back(counts.scount[si]);
            outputDims.push_back(problem.d().sizes()[formatD.activation().batchPosition()]);
            break;
        case ConvolutionProblem::TensorFormat::CNHW:
            for(int si = 0; si < formatA.spatialPositions().size(); si++)
                outputDims.push_back(counts.scount[si]);
            outputDims.push_back(problem.d().sizes()[formatD.activation().batchPosition()]);
            outputDims.push_back(problem.d().sizes()[formatD.activation().channelPosition()]);
            break;
        default:
            throw std::runtime_error("unknown formatD");
        };
        TensorDescriptor rv(
            problem.d().dataType(), outputDims.begin(), outputDims.end(), problem.d().offset());
        return rv;
    }

    TensorDescriptor
        ConvolutionProblem::setupForwardWeights(ConvolutionProblem::LoopCounts const& counts,
                                                ContractionProblem const&             problem) const
    {
        std::vector<size_t> filterDims;
        auto                formatB = counts.formatB();
        switch(formatB.weights().format())
        {
        case ConvolutionProblem::TensorFormat::KCYX:
            for(int fi = 0; fi < counts.filterCount.size(); fi++)
                if(formatB.weights().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                    filterDims.push_back(counts.filterCount[fi]);
            filterDims.push_back(problem.b().sizes()[formatB.weights().cinPosition()]);
            filterDims.push_back(problem.b().sizes()[formatB.weights().coutPosition()]);
            break;
        case ConvolutionProblem::TensorFormat::CKYX:
            for(int fi = 0; fi < counts.filterCount.size(); fi++)
                if(formatB.weights().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                    filterDims.push_back(counts.filterCount[fi]);
            filterDims.push_back(problem.b().sizes()[formatB.weights().coutPosition()]);
            filterDims.push_back(problem.b().sizes()[formatB.weights().cinPosition()]);
            break;
        default:
            throw std::runtime_error("unknown formatB");
        };
        TensorDescriptor rv(
            problem.b().dataType(), filterDims.begin(), filterDims.end(), problem.b().offset());
        return rv;
    }

    void ConvolutionProblem::validate(const ContractionProblem&             problem,
                                      const ConvolutionProblem::LoopCounts& counts) const
    {
        if(1)
        {
            std::cout << "validate::\n";

            std::cout << "  freeAIndices: ";
            for(auto i : problem.freeIndicesA())
                std::cout << i << ",";

            std::cout << "\n  freeBIndices";
            for(auto i : problem.freeIndicesB())
                std::cout << i << ",";

            std::cout << "\n  batchIndices: ";
            for(auto i : problem.batchIndices())
                std::cout << i << ",";

            std::cout << "\n  summationIndicies: ";
            for(auto i : problem.boundIndices())
                std::cout << i << ",";

            std::cout << "\n";

            if(m_operationIdentifier != "ConvolutionForward")
            {
                throw std::runtime_error(std::string("Unsupported operation identifier for check")
                                         + m_operationIdentifier);
            }
        }

        // Ensure positions are where we expect them to be in the convolution tensor
        // description:
        auto formatA = counts.formatA();
        auto formatB = counts.formatB();
        auto formatD = counts.formatD();

        assert(problem.batchIndices().end()
               != std::find_if(problem.batchIndices().begin(),
                               problem.batchIndices().end(),
                               [formatA](const ContractionProblem::BatchIndex& bi) {
                                   return bi.a == formatA.batchPosition();
                               }));

        assert(problem.boundIndices().end()
               != std::find_if(problem.boundIndices().begin(),
                               problem.boundIndices().end(),
                               [formatA](const ContractionProblem::BoundIndex& bi) {
                                   return bi.a == formatA.channelPosition();
                               }));
        if(m_operationIdentifier == "ConvolutionForward")
        {
            for(int i = 0; i < formatA.filterPositions().size(); i++)
            {
                auto const filterPositionA = formatA.filterPositions()[i];
                if(filterPositionA != ConvolutionProblem::InvalidPos)
                    assert(problem.boundIndices().end()
                           != std::find_if(
                               problem.boundIndices().begin(),
                               problem.boundIndices().end(),
                               [filterPositionA](const ContractionProblem::BoundIndex& bi) {
                                   return bi.a == filterPositionA;
                               }));
            }
            for(int i = 0; i < formatB.weights().filterPositions().size(); i++)
            {
                auto const filterPositionB = formatB.weights().filterPositions()[i];
                if(filterPositionB != ConvolutionProblem::InvalidPos)
                    assert(problem.boundIndices().end()
                           != std::find_if(
                               problem.boundIndices().begin(),
                               problem.boundIndices().end(),
                               [filterPositionB](const ContractionProblem::BoundIndex& bi) {
                                   return bi.b == filterPositionB;
                               }));
            }
            for(auto s : formatA.spatialPositions())
                assert(problem.freeIndicesA().end()
                       != std::find_if(problem.freeIndicesA().begin(),
                                       problem.freeIndicesA().end(),
                                       [s](const ContractionProblem::FreeIndex& bi) {
                                           return bi.isA && bi.i == s;
                                       }));
        }
        else
            throw std::runtime_error(std::string("Unsupported operation identifier for check")
                                     + m_operationIdentifier);
    }

    template <typename T>
    static std::string delimitedVector(const std::vector<T>& v, const std::string& delimiter)
    {
        std::ostringstream rv;

        std::string delim;
        for(auto e : v)
        {
            rv << delim << e;
            delim = delimiter;
        }
        return rv.str();
    }

    std::string ConvolutionProblem::description() const
    {
        std::ostringstream rv;

        rv << operationIdentifier();

        return rv.str();
    }

    TENSILE_API std::ostream& operator<<(std::ostream&             stream,
                                         ConvolutionProblem const& convolution)
    {
        return stream << convolution.description();
    }

} // namespace Tensile
