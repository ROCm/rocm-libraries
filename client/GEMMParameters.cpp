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
                rv << "GEMM_" << toString(problemParams.transA) << toString(problemParams.transB);

                rv << "_";
                for(auto const& t : {problemParams.typeA,
                                     problemParams.typeB,
                                     problemParams.typeC,
                                     problemParams.typeD,
                                     problemParams.typeAcc})
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

                rv << "_" << scheduler;

                if(streamK)
                {
                    rv << "_SK";
                    if(streamKTwoTile)
                    {
                        rv << "2T";
                    }
                    rv << numWGs;
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
        }
    }
}
