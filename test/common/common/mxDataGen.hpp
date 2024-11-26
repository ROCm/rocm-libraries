#include <DataGenerator.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>

using namespace DGen;

namespace rocRoller
{
    DGen::DataGeneratorOptions
        setOptions(const float min, const float max, int blockScaling, const DataPattern pattern);

    template <typename T>
    struct rrDT2DGenDT
    {
        typedef T type;
    };

    template <>
    struct rrDT2DGenDT<FP8>
    {
        typedef DGen::ocp_e4m3_mxfp8 type;
    };

    template <>
    struct rrDT2DGenDT<BF8>
    {
        typedef DGen::ocp_e5m2_mxfp8 type;
    };

    template <>
    struct rrDT2DGenDT<Half>
    {
        typedef DGen::fp16 type;
    };

    template <>
    struct rrDT2DGenDT<BFloat16>
    {
        typedef DGen::bf16 type;
    };

    template <>
    struct rrDT2DGenDT<float>
    {
        typedef DGen::f32 type;
    };

    template <typename rrDT>
    DGen::DataGenerator<typename rrDT2DGenDT<rrDT>::type>
        getDataGenerator(const int         dim1,
                         const int         dim2,
                         const float       min,
                         const float       max,
                         const uint32_t    seed,
                         const int         blockScaling,
                         const DataPattern pattern)
    {
        std::vector<int> size{dim1, dim2};
        std::vector<int> stride{dim2, 1};
        auto             opts = setOptions(min, max, blockScaling, pattern);
        using DGenDT          = typename rrDT2DGenDT<rrDT>::type;
        DGen::DataGenerator<DGenDT> dgen;
        dgen.setSeed(seed);
        return dgen.generate(size, stride, opts);
    }

    template <typename rrDT>
    std::vector<rrDT> DGenVector(const int         dim1,
                                 const int         dim2,
                                 const float       min          = -1.f,
                                 const float       max          = 1.f,
                                 const uint32_t    seed         = 1713573849,
                                 const int         blockScaling = 1,
                                 const DataPattern pattern      = DataPattern::Bounded)
    {
        auto dgen = getDataGenerator<rrDT>(dim1, dim2, min, max, seed, blockScaling, pattern);
        if constexpr(std::is_same_v<rrDT, FP8> or std::is_same_v<rrDT, BF8>)
        {
            // The random values (FP8/BF8) returned by data generator consist
            // of DataBytes and ScaleBytes, while rocRoller does not separate
            // data and scale (just need a 8-bit representation). To handle this,
            // we ask data generator to return the values in float and
            // cast them into corresponding types.
            auto              refFloat = dgen.getReferenceFloat();
            std::vector<rrDT> rrData;
            std::transform(refFloat.begin(),
                           refFloat.end(),
                           std::back_inserter(rrData),
                           [](auto value) { return rrDT(value); });
            return rrData;
        }
        std::vector<uint8_t> dataByte = dgen.getDataBytes();
        std::vector<rrDT>&   rrData   = reinterpret_cast<std::vector<rrDT>&>(dataByte);
        return rrData;
    }

    template <typename TA, typename TB, typename TC>
    void DGenInput(std::vector<TA>& A,
                   std::vector<TB>& B,
                   std::vector<TC>& C,
                   const int        M,
                   const int        N,
                   const int        K,
                   const uint32_t   seed,
                   int              dim = 2)
    {
        if(dim == 2)
        {
#pragma omp parallel sections
            {
#pragma omp section
                {
                    A = DGenVector<TA>(M, K, -1.f, 1.f, seed + 1);
                }

#pragma omp section
                {
                    B = DGenVector<TB>(K, N, -1.f, 1.f, seed + 2);
                }

#pragma omp section
                {
                    C = DGenVector<TC>(M, N, -1.f, 1.f, seed + 3);
                }
            }
        }
        else if(dim == 1)
        {
#pragma omp parallel sections
            {
#pragma omp section
                {
                    A = DGenVector<TA>(1, M, -1.f, 1.f, seed + 1);
                }

#pragma omp section
                {
                    B = DGenVector<TB>(1, N, -1.f, 1.f, seed + 2);
                }

#pragma omp section
                {
                    C = DGenVector<TC>(1, K, -1.f, 1.f, seed + 3);
                }
            }
        }
        else
        {
            Throw<FatalError>("Invalid number of dimensions, the data has to be 1-d or 2-d.");
        }
    }
}
