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

#include <regex>

#include "client/GEMMParameters.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            std::string TypeParameters::kernelNamePart() const
            {
                std::ostringstream rv;

                rv << "GEMM_" << toString(transA) << toString(transB);

                if(scaleA != rocRoller::Operations::ScaleMode::None)
                {
                    rv << "_mx" << typeA;
                    rv << "_st" << scaleTypeA;
                    if(scaleA == rocRoller::Operations::ScaleMode::Separate)
                        rv << "_bs" << scaleBlockSize;
                }
                else
                {
                    rv << "_" << typeA;
                }

                if(scaleB != rocRoller::Operations::ScaleMode::None)
                {
                    rv << "_mx" << typeB;
                    rv << "_st" << scaleTypeB;
                    if(scaleB == rocRoller::Operations::ScaleMode::Separate)
                        rv << "_bs" << scaleBlockSize;
                }
                else
                {
                    rv << "_" << typeB;
                }

                for(auto const& t : {typeC, typeD, typeAcc})
                    rv << "_" << t;

                if(scaleSkipPermlane)
                {
                    rv << "_PreSW_AB";
                }

                return rv.str();
            }

            std::string SolutionParameters::generateKernelName() const
            {
                std::ostringstream rv;
                rv << types.kernelNamePart();

                rv << "_MT";
                rocRoller::streamJoin(rv, std::vector{macM, macN, macK}, "x");

                rv << "_WG";
                rocRoller::streamJoin(rv, std::vector{workgroupSizeX, workgroupSizeY}, "x");

                if(workgroupMappingDim != -1)
                {
                    rv << "_WGM" << workgroupMappingDim;
                }

                rv << "_WGMXCC";
                rocRoller::streamJoin(rv, std::vector{workgroupRemapXCC}, "");
                if(workgroupRemapXCC && workgroupRemapXCCValue > 0)
                {
                    rocRoller::streamJoin(rv, std::vector{workgroupRemapXCCValue}, "");
                }

                rv << "_LA" << loadPathA;
                rv << "_LB" << loadPathB;

                rv << "_SD" << storeLDSD;

                rv << "_SLDS";
                rocRoller::streamJoin(rv, std::vector{loadLDSScaleA, loadLDSScaleB}, "");

                rv << "_UNROLL";
                rocRoller::streamJoin(rv, std::vector{unrollX, unrollY}, "x");

                rv << "_SwizzleScale" << swizzleScale << prefetchScale;
                rv << "_SwizzleTileSize" << swizzleTileSize;

                if(prefetch)
                {
                    rv << "_PF";
                    rocRoller::streamJoin(
                        rv, std::vector{prefetchInFlight, prefetchLDSFactor}, "x");
                    rv << "m" << prefetchMixMemOps;
                }

                rv << "_MI";
                rocRoller::streamJoin(
                    rv, std::vector{waveM, waveN, waveK, (waveB < 0 ? -waveB : waveB)}, "x");

                rv << "_" << scheduler;

                if(streamK)
                {
                    rv << "_SK";
                    if(streamKTwoTileDPFirst)
                        rv << "2TDPFirst";
                    else if(streamKTwoTile)
                        rv << "2T";
                }

                return rv.str();
            }

            std::string toString(TransposeType trans)
            {
                switch(trans)
                {
                case TransposeType::T:
                    return "T";
                case TransposeType::N:
                    return "N";
                default:
                    rocRoller::Throw<rocRoller::FatalError>("Unknown transpose type");
                }
            }

            std::ostream& operator<<(std::ostream& s, TransposeType const& x)
            {
                s << toString(x);
                return s;
            }

            std::string toString(MNKTuple x)
            {
                return fmt::format("{}x{}x{}", x.m, x.n, x.k);
            }

            std::ostream& operator<<(std::ostream& s, MNKTuple const& x)
            {
                s << toString(x);
                return s;
            }

            std::string toString(MNKBTuple x)
            {
                return fmt::format("{}x{}x{}x{}", x.m, x.n, x.k, x.b);
            }

            std::ostream& operator<<(std::ostream& s, MNKBTuple const& x)
            {
                s << toString(x);
                return s;
            }

            std::string toString(MKNLTuple x)
            {
                return fmt::format("{}x{}X{}x{}", x.m, x.k, x.n, x.l);
            }

            std::ostream& operator<<(std::ostream& s, MKNLTuple const& x)
            {
                s << toString(x);
                return s;
            }

            std::ostream& operator<<(std::ostream& s, TypeParameters const& x)
            {
                s << "Type:      A:" << x.typeA << " B:" << x.typeB << " C:" << x.typeC
                  << " D:" << x.typeD << " ACC:" << x.typeAcc << std::endl;
                s << "Transpose: " << toString(x.transA) << toString(x.transB) << std::endl;
                s << "Scaling:   A:" << x.scaleA << "(" << x.scaleTypeA << ")";
                s << " B:" << x.scaleB << "(" << x.scaleTypeB << ")";
                if(x.scaleA == rocRoller::Operations::ScaleMode::Separate
                   or x.scaleB == rocRoller::Operations::ScaleMode::Separate)
                {
                    s << " BlockSize:" << x.scaleBlockSize;
                }
                s << std::endl;
                return s;
            }

            std::ostream& operator<<(std::ostream& s, ProblemParameters const& x)
            {
                s << "MxNxK:     " << x.m << "x" << x.n << "x" << x.k << std::endl;
                s << x.types;
                return s;
            }

            std::ostream& operator<<(std::ostream& s, SolutionParameters const& x)
            {
                s << "Version:         " << x.version << std::endl;
                s << "Arch:            " << x.architecture.toString() << std::endl;
                if(x.streamK)
                {
                    s << "Algorithm:       StreamK twoTile:" << x.streamKTwoTile
                      << "(DPFirst:" << x.streamKTwoTileDPFirst << ")" << std::endl;
                }
                else
                {
                    s << "Algorithm:       DataParallel" << std::endl;
                }
                s << "Tiling:          " << x.macM << "x" << x.macN << "x" << x.macK << std::endl;
                s << "MI:              " << x.waveM << "x" << x.waveN << "x" << x.waveK << "x"
                  << x.waveB << std::endl;
                s << "SwizzleScale:    " << x.swizzleScale << std::endl;
                s << "PrefetchScale:   " << x.prefetchScale << std::endl;
                s << "SwizzleTileSize: " << x.swizzleTileSize << std::endl;
                s << "Load A:          " << x.loadPathA << std::endl;
                s << "Load B:          " << x.loadPathB << std::endl;
                s << "Store D LDS:     " << x.storeLDSD << std::endl;
                s << "LSDScale:        " << x.loadLDSScaleA << x.loadLDSScaleB << std::endl;
                s << "Prefetch:        "
                  << "enabled:" << x.prefetch << " inflight:" << x.prefetchInFlight
                  << " LDS:" << x.prefetchLDSFactor << " mixMemOps: " << x.prefetchMixMemOps
                  << std::endl;
                s << "Unroll:          X:" << x.unrollX << " Y:" << x.unrollY << std::endl;
                s << "Scheduler:       " << x.scheduler << std::endl;
                s << "WG size:         " << x.workgroupSizeX * x.workgroupSizeY << std::endl;
                s << "WG Mapping Dim:  " << x.workgroupMappingDim << std::endl;
                s << "WG XCC Remap:    " << x.workgroupRemapXCC;
                if(x.workgroupRemapXCC)
                {
                    if(x.workgroupRemapXCCValue != -1)
                    {
                        s << " value: " << x.workgroupRemapXCCValue;
                    }
                    else
                    {
                        s << " default";
                    }
                }
                s << std::endl;
                s << x.types;
                return s;
            }

        }
    }
}

namespace rocRoller::Client::GEMMClient::CLI
{
    bool ParseMNKB(const std::string& arg, rocRoller::Client::GEMMClient::MNKBTuple& x)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        x.b = 1;

        std::regex  pattern(R"((\d+)x(\d+)x(\d+)(x(\d+))?)");
        std::smatch match;

        bool matched = std::regex_match(arg, match, pattern);
        if(matched)
        {
            x.m = std::stoi(match[1]);
            x.n = std::stoi(match[2]);
            x.k = std::stoi(match[3]);
            if(match[5].matched)
                x.b = std::stoi(match[5]);
        }

        if(not matched or (x.m < 0) or (x.k < 0) or (x.n < 0) or (x.b < 0))
        {
            std::cerr << "Invalid format for MxNxKxB tuple.\n" << std::endl;
            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }

    bool ParseMNK(const std::string& arg, rocRoller::Client::GEMMClient::MNKTuple& x)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        std::regex  pattern(R"((\d+)x(\d+)x(\d+))");
        std::smatch match;

        bool matched = std::regex_match(arg, match, pattern);
        if(matched)
        {
            x.m = std::stoi(match[1]);
            x.n = std::stoi(match[2]);
            x.k = std::stoi(match[3]);
        }

        if(not matched or (x.m < 0) or (x.k < 0) or (x.n < 0))
        {
            std::cerr << "Invalid format for MxNxK tuple.\n" << std::endl;
            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }

    bool ParseMKNL(const std::string& arg, rocRoller::Client::GEMMClient::MKNLTuple& x)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        std::regex  pattern(R"((\d+)x(\d+)[/X](\d+)x(\d+))");
        std::smatch match;

        bool matched = std::regex_match(arg, match, pattern);
        if(matched)
        {
            x.m = std::stoi(match[1]);
            x.k = std::stoi(match[2]);
            x.n = std::stoi(match[3]);
            x.l = std::stoi(match[4]);
        }

        if(not matched or (x.m < 0) or (x.k < 0) or (x.n < 0) or (x.l < 0))
        {
            std::cerr << "Invalid format for MxK/NxL tuple.\n" << std::endl;
            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }
}
