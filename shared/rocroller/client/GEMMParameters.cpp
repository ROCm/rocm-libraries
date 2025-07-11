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

#include "./include/GEMMParameters.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            std::string SolutionParameters::generateKernelName() const
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

                rv << "_MT";
                rocRoller::streamJoin(rv, std::vector{macM, macN, macK}, "x");

                rv << "_WG";
                rocRoller::streamJoin(rv, std::vector{workgroupSizeX, workgroupSizeY}, "x");

                if(workgroupMapping.first != -1)
                {
                    rv << "_WGM";
                    rocRoller::streamJoin(
                        rv, std::vector{workgroupMapping.first, workgroupMapping.second}, "");
                }

                rv << "_WGMXCC";
                rocRoller::streamJoin(rv, std::vector{workgroupRemapXCC}, "");
                if(workgroupRemapXCC && workgroupRemapXCCValue > 0)
                {
                    rocRoller::streamJoin(rv, std::vector{workgroupRemapXCCValue}, "");
                }

                rv << "_LDS";
                rocRoller::streamJoin(rv, std::vector{loadLDSA, loadLDSB, storeLDSD}, "");

                rv << "_SLDS";
                rocRoller::streamJoin(rv, std::vector{loadLDSScaleA, loadLDSScaleB}, "");

                rv << "_Direct2LDS";
                rocRoller::streamJoin(rv, std::vector{direct2LDSA, direct2LDSB}, "");

                rv << "_UNROLL";
                rocRoller::streamJoin(rv, std::vector{unrollX, unrollY}, "x");

                rv << "_SwizzleScale" << swizzleScale << prefetchScale;

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
                    if(streamKTwoTile)
                    {
                        rv << "2T";
                    }
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

            std::ostream& operator<<(std::ostream& s, ProblemParameters const& x)
            {
                s << "MxNxK:     " << x.m << "x" << x.n << "x" << x.k << std::endl;
                s << "Type:      A:" << x.typeA << " B:" << x.typeB << " C:" << x.typeC
                  << " D:" << x.typeD << " ACC:" << x.typeAcc << std::endl;
                s << "Transpose: " << toString(x.transA) << toString(x.transB) << std::endl;
                return s;
            }

            std::ostream& operator<<(std::ostream& s, SolutionParameters const& x)
            {
                s << "Arch:      " << x.architecture.toString() << std::endl;
                if(x.streamK)
                {
                    s << "Algorithm: StreamK twoTile:" << x.streamKTwoTile << std::endl;
                }
                else
                {
                    s << "Algorithm: DataParallel" << std::endl;
                }
                s << "Tiling:    " << x.macM << "x" << x.macN << "x" << x.macK << std::endl;
                s << "MI:        " << x.waveM << "x" << x.waveN << "x" << x.waveK << "x" << x.waveB
                  << std::endl;
                s << "Scaling:   A:" << x.scaleA << "(" << x.scaleTypeA << ")";
                s << " B:" << x.scaleB << "(" << x.scaleTypeB << ")";
                if(x.scaleA == rocRoller::Operations::ScaleMode::Separate
                   or x.scaleB == rocRoller::Operations::ScaleMode::Separate)
                {
                    s << " BlockSize:" << x.scaleBlockSize;
                }
                s << std::endl;
                s << "SwizzleScale:        " << x.swizzleScale << std::endl;
                s << "LDS:       " << x.loadLDSA << x.loadLDSB << x.storeLDSD << std::endl;
                s << "Direct2LDS:       " << x.direct2LDSA << x.direct2LDSB << std::endl;
                s << "LSDScale:  " << x.loadLDSScaleA << x.loadLDSScaleB << std::endl;
                s << "Prefetch:  "
                  << "enabled:" << x.prefetch << " inflight:" << x.prefetchInFlight
                  << " LDS:" << x.prefetchLDSFactor << std::endl;
                s << "Unroll:    X:" << x.unrollX << " Y:" << x.unrollY << std::endl;
                s << "Scheduler: " << x.scheduler << std::endl;
                s << "WG size:   " << x.workgroupSizeX * x.workgroupSizeY << std::endl;
                if(x.workgroupMapping.first != -1)
                {
                    s << "WG Mapping: " << x.workgroupMapping.first << ","
                      << x.workgroupMapping.second << std::endl;
                }
                s << "WG XCC Remap: " << x.workgroupRemapXCC;
                if(x.workgroupRemapXCC)
                {
                    if(x.workgroupRemapXCCValue != -1)
                    {
                        s << " value: " << x.workgroupRemapXCCValue;
                    }
                    else
                    {
                        s << " Default";
                    }
                }
                s << std::endl;
                s << "Type:      A:" << x.typeA << " B:" << x.typeB << " C:" << x.typeC
                  << " D:" << x.typeD << " ACC:" << x.typeAcc << std::endl;
                s << "Transpose: " << toString(x.transA) << toString(x.transB) << std::endl;
                s << "Version:   " << x.version << std::endl;
                return s;
            }

        }
    }
}
