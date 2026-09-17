// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <regex>

#include "client/GEMMParameters.hpp"
#include <common/SourceMatcher.hpp>

#include <functional>

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            std::string TypeParameters::kernelNamePart() const
            {
                std::ostringstream rv;

                // TODO: Use abbreviated type names.  Currently the
                // types are strings.  If we change them to DataType
                // we can use shorter names.

                rv << toString(transA) << toString(transB);

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

                if(scaleSkipPermlane != rocRoller::ScaleSkipPermlaneMode::None)
                {
                    rv << "_PreSW" << toString(scaleSkipPermlane);
                }

                if(!scalePretileA.empty())
                {
                    rv << "_PTA";
                    rocRoller::streamJoin(rv, scalePretileA, "x");
                }

                if(!scalePretileB.empty())
                {
                    rv << "_PTB";
                    rocRoller::streamJoin(rv, scalePretileB, "x");
                }

                return rv.str();
            }

            KernelNames SolutionParameters::generateKernelName() const
            {
                auto constexpr maxLength = 196;

                std::ostringstream fullName;

                fullName << "RRGEMM_";
                fullName << types.kernelNamePart();
                fullName << "_WGTS";
                rocRoller::streamJoin(fullName, std::vector{macM, macN, macK}, "x");
                fullName << "_WGS";
                rocRoller::streamJoin(fullName, std::vector{workgroupSizeX, workgroupSizeY}, "x");

                if(workgroupMappingDim != -1)
                {
                    fullName << "_WGM" << workgroupMappingDim;
                }

                fullName << "_WGMXCC";
                rocRoller::streamJoin(fullName, std::vector{workgroupRemapXCC}, "");
                if(workgroupRemapXCC && workgroupRemapXCCValue > 0)
                {
                    rocRoller::streamJoin(fullName, std::vector{workgroupRemapXCCValue}, "");
                }

                fullName << "_LA" << loadPathA;
                fullName << "_LB" << loadPathB;

                fullName << "_SD" << storePath;

                fullName << "_LSA" << loadPathAScale;
                fullName << "_LSB" << loadPathBScale;

                fullName << "_SwizzleScale" << swizzleScale << prefetchScale;
                fullName << "_SwizzleTileSize" << swizzleTileSize;

                if(prefetch)
                {
                    fullName << "_PF";
                    rocRoller::streamJoin(
                        fullName, std::vector{prefetchInFlight, prefetchLDSFactor}, "x");
                    fullName << "m" << prefetchMixMemOps;
                }

                fullName << "_MI";
                rocRoller::streamJoin(
                    fullName, std::vector{waveM, waveN, waveK, (waveB < 0 ? -waveB : waveB)}, "x");

                fullName << "_" << scheduler;

                if(streamK != StreamKMode::None)
                {
                    fullName << "_SK";
                    if(streamK == StreamKMode::TwoTileDPFirst)
                        fullName << "2TDPFirst";
                    else if(streamK == StreamKMode::TwoTile)
                        fullName << "2T";
                }

                auto fullNameStr  = fullName.str();
                auto shortNameStr = fullNameStr;

                // Truncate and append hash if necessary
                if(shortNameStr.length() > maxLength)
                {
                    auto hashedValue = std::hash<std::string>{}(fullNameStr);
                    shortNameStr     = fmt::format(
                        "{}_{:08x}", shortNameStr.substr(0, maxLength - 9), hashedValue);
                }

                return KernelNames{fullNameStr, shortNameStr};
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

            std::string toString(XYTuple x)
            {
                return fmt::format("{}x{}", x.x, x.y);
            }

            std::ostream& operator<<(std::ostream& s, XYTuple const& x)
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

            std::ostream& operator<<(std::ostream& s, std::pair<int, int> const& x)
            {
                s << fmt::format("{},{}", x.first, x.second);
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
                    s << " BlockSize: " << x.scaleBlockSize;
                    s << " Pretile A: " << x.scalePretileA;
                    s << " Pretile B: " << x.scalePretileB;
                }
                s << std::endl;
                return s;
            }

            std::ostream& operator<<(std::ostream& s, ProblemParameters const& x)
            {
                s << "MxNxK:     " << x.m << "x" << x.n << "x" << x.k << std::endl;
                s << "alpha:     " << x.alpha << std::endl;
                s << "beta:      " << x.beta << std::endl;
                s << x.types;
                s << "ScaleVals: A: " << x.scaleValueA << " B: " << x.scaleValueB << std::endl;
                s << "InitModes: A: " << x.initModeA << " B: " << x.initModeB
                  << " C: " << x.initModeC << std::endl;
                return s;
            }

            std::ostream& operator<<(std::ostream& s, SolutionParameters const& x)
            {
                s << "Version:         " << x.version << std::endl;
                s << "Arch:            " << x.architecture.toString() << std::endl;
                if(x.streamK != StreamKMode::None)
                {
                    s << "Algorithm:       StreamK(" << x.streamK << ")" << std::endl;
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
                s << "LDS Padding A:   " << x.padLDSA << std::endl;
                s << "Load B:          " << x.loadPathB << std::endl;
                s << "LDS Padding B:   " << x.padLDSB << std::endl;
                s << "Store D Path:    " << x.storePath << std::endl;
                s << "Load AScale:     " << x.loadPathAScale << std::endl;
                s << "Load BScale:     " << x.loadPathBScale << std::endl;
                s << "Prefetch:        "
                  << "enabled:" << x.prefetch << " inflight:" << x.prefetchInFlight
                  << " LDS:" << x.prefetchLDSFactor << " mixMemOps: " << x.prefetchMixMemOps
                  << std::endl;
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
                s << "WG Cluster size: " << x.workgroupClusterSizeX << "x"
                  << x.workgroupClusterSizeY << "x" << x.workgroupClusterSizeZ << std::endl;
                s << std::endl;
                s << x.types;
                return s;
            }

        }
    }
}

namespace rocRoller::Client::GEMMClient::CLI
{
    bool ParseXY(const std::string& arg, XYTuple& x)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        std::regex  pattern(R"((\d+)x(\d+))");
        std::smatch match;

        bool matched = std::regex_match(arg, match, pattern);
        if(matched)
        {
            x.x = std::stoi(match[1]);
            x.y = std::stoi(match[2]);
        }
        else
        {
            std::cerr << "Invalid format for XxY pair.\n" << std::endl;
            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }

    bool ParseIntPair(const std::string& arg, std::pair<int, int>& x)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        std::regex  pattern(R"((-?\d+),(-?\d+))");
        std::smatch match;

        bool matched = std::regex_match(arg, match, pattern);
        if(matched)
        {
            x.first  = std::stoi(match[1]);
            x.second = std::stoi(match[2]);
        }
        else
        {
            std::cerr << "Invalid format for X,Y pair.\n" << std::endl;
            return PARSE_FAILURE;
        }

        return PARSE_SUCCESS;
    }

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

    bool ParseInitMode(const std::string& arg, DGen::DataInitMode& result)
    {
        if(arg.empty())
            return PARSE_FAILURE;

        bool               fail = false;
        std::istringstream iss(arg);
        std::string        token;

        if(arg == "Bounded")
            result = DGen::DataInitMode(DGen::Bounded{});
        else if(arg == "BoundedAlternatingSign")
            result = DGen::DataInitMode(DGen::BoundedAlternatingSign{});
        else if(arg == "Unbounded")
            result = DGen::DataInitMode(DGen::Unbounded{});
        else if(arg == "Identity")
            result = DGen::DataInitMode(DGen::Identity{});
        else if(arg == "Ones")
            result = DGen::DataInitMode(DGen::Ones{});
        else if(arg == "Zeros")
            result = DGen::DataInitMode(DGen::Zeros{});
        else if(arg == "TrigonometricFromFloat")
            result = DGen::DataInitMode(DGen::TrigonometricFromFloat{});
        else if(startsWith("NormalFromFloat", arg.begin(), arg.end()))
        {
            try
            {
                iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit
                               | std::ifstream::badbit);
                std::getline(iss, token, '(');
                std::getline(iss, token, ',');
                double mean = std::stod(token);
                std::getline(iss, token, ')');
                double std_dev = std::stod(token);

                result = DGen::DataInitMode(DGen::NormalFromFloat{mean, std_dev});
            }
            catch(const std::invalid_argument&)
            {
                fail = true;
            }
            catch(const std::ios_base::failure&)
            {
                fail = true;
            }
            if(fail)
            {
                std::cerr << "Invalid format for Init Mode." << std::endl;
                std::cerr << "Expected: NormalFromFloat(<mean>, <std_dev>)" << std::endl;
                std::cerr << "For example: --initMode_A=\"NormalFromFloat(0.0, 1.0)\"" << std::endl;
                return PARSE_FAILURE;
            }
        }

        return PARSE_SUCCESS;
    }
}
