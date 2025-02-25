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

                rv << "_";
                for(auto const& t : {typeA, typeB, typeC, typeD, typeAcc})
                    rv << t.substr(0, 1);

                rv << "_MT";
                rocRoller::streamJoin(rv, std::vector{macM, macN, macK}, "x");

                rv << "_WG";
                rocRoller::streamJoin(rv, std::vector{workgroupSizeX, workgroupSizeY}, "x");

                rv << "_LDS";
                rocRoller::streamJoin(rv, std::vector{loadLDSA, loadLDSB, storeLDSD}, "");

                rv << "_UNROLL";
                rocRoller::streamJoin(rv, std::vector{unrollX, unrollY}, "x");

                if(prefetch)
                {
                    rv << "_PF";
                    rocRoller::streamJoin(
                        rv, std::vector{prefetchInFlight, prefetchLDSFactor}, "x");
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
                s << "Tranpose:  " << toString(x.transA) << toString(x.transB) << std::endl;
                return s;
            }

            std::ostream& operator<<(std::ostream& s, SolutionParameters const& x)
            {
                if(x.scheduler == "TENSILE_ASM")
                {
                    s << "Algorithm: TENSILE" << std::endl;
                    return s;
                }
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
                s << "LDS:       " << x.loadLDSA << x.loadLDSB << x.storeLDSD << std::endl;
                s << "Prefetch:  "
                  << "enabled:" << x.prefetch << " inflight:" << x.prefetchInFlight
                  << " LDS:" << x.prefetchLDSFactor << std::endl;
                s << "Unroll:    X:" << x.unrollX << " Y:" << x.unrollY << std::endl;
                s << "Scheduler: " << x.scheduler << std::endl;
                s << "WG size:   " << x.workgroupSizeX * x.workgroupSizeY << std::endl;
                s << "Type:      A:" << x.typeA << " B:" << x.typeB << " C:" << x.typeC
                  << " D:" << x.typeD << " ACC:" << x.typeAcc << std::endl;
                s << "Tranpose:  " << toString(x.transA) << toString(x.transB) << std::endl;
                s << "Version:   " << x.version << std::endl;
                return s;
            }

        }
    }
}
