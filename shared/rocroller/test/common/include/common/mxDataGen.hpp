// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <mxDataGenerator/DataGenerator.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/TensorDescriptor.hpp>

using namespace DGen;

namespace rocRoller
{
    void setOptions(DataGeneratorOptions& opts,
                    const float           min,
                    const float           max,
                    int                   blockScaling,
                    const DataInitMode    initMode);

    template <typename T>
    struct rrDT2DGenDT
    {
        typedef T type;
    };

    template <>
    struct rrDT2DGenDT<FP4>
    {
        typedef DGen::ocp_e2m1_mxfp4 type;
    };

    template <>
    struct rrDT2DGenDT<FP6>
    {
        typedef DGen::ocp_e2m3_mxfp6 type;
    };

    template <>
    struct rrDT2DGenDT<BF6>
    {
        typedef DGen::ocp_e3m2_mxfp6 type;
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
        getDataGenerator(TensorDescriptor const& desc,
                         const float             min,
                         const float             max,
                         const uint32_t          seed,
                         const index_t           blockScaling = 1,
                         const DataInitMode      initMode     = DataInitMode(Bounded{}))
    {
        auto sizes   = desc.sizes();
        auto strides = desc.strides();

        DataGeneratorOptions opts;
        setOptions(opts, min, max, blockScaling, initMode);
        using DGenDT = typename rrDT2DGenDT<rrDT>::type;
        DGen::DataGenerator<DGenDT> dgen;
        dgen.setSeed(seed);
        std::vector<index_t> dgen_sizes(sizes.begin(), sizes.end());
        std::vector<index_t> dgen_strides(strides.begin(), strides.end());
        return dgen.generate(dgen_sizes, dgen_strides, opts);
    }

    template <typename rrDT>
    std::vector<typename PackedTypeOf<rrDT>::type>
        getRandomVector(const DGen::DataGenerator<typename rrDT2DGenDT<rrDT>::type>& dgen,
                        bool                                                         hasScale)
    {
        using UDT = typename PackedTypeOf<rrDT>::type;

        std::vector<uint8_t> dataByte = dgen.getDataBytes();

        if constexpr(std::is_same_v<rrDT,
                                    FP6> || std::is_same_v<rrDT, BF6> || std::is_same_v<rrDT, FP4>)
        {

            return reinterpret_cast<std::vector<UDT>&>(dataByte);
        }

        if constexpr(std::is_same_v<rrDT, FP8> || std::is_same_v<rrDT, BF8>)
        {
            if(hasScale)
                return reinterpret_cast<std::vector<rrDT>&>(dataByte);
            else
            {
                auto              refFloat = dgen.getReferenceFloat();
                std::vector<rrDT> rrData;
                std::transform(refFloat.begin(),
                               refFloat.end(),
                               std::back_inserter(rrData),
                               [](auto value) { return rrDT(value); });
                return rrData;
            }
        }

        if constexpr(std::is_same_v<
                         rrDT,
                         float> || std::is_same_v<rrDT, Half> || std::is_same_v<rrDT, BFloat16>)
        {
            std::vector<rrDT>& rrData = reinterpret_cast<std::vector<rrDT>&>(dataByte);
            return rrData;
        }

        Throw<FatalError>("Unsupported data type");

        return {};
    }

    template <typename rrDT>
    std::vector<typename PackedTypeOf<rrDT>::type> DGenVector(TensorDescriptor&  desc,
                                                              const float        min  = -1.f,
                                                              const float        max  = 1.f,
                                                              const uint32_t     seed = 1713573849,
                                                              bool               hasScale = false,
                                                              const int          blockScaling = 1,
                                                              const DataInitMode initMode
                                                              = DataInitMode(Bounded{}))
    {
        if(hasScale)
            AssertFatal(blockScaling == 32, "Invalid scale block size: ", ShowValue(blockScaling));
        auto dgen = getDataGenerator<rrDT>(desc, min, max, seed, blockScaling, initMode);
        return getRandomVector<rrDT>(dgen, hasScale);
    }

    template <typename TA, typename TB, typename TC>
    void DGenInput(const uint32_t          seed,
                   std::vector<TA>&        hostA,
                   TensorDescriptor const& descA,
                   std::vector<TB>&        hostB,
                   TensorDescriptor const& descB,
                   std::vector<TC>&        hostC,
                   TensorDescriptor const& descC,
                   std::vector<uint8_t>&   hostScaleA,
                   std::vector<uint8_t>&   hostScaleB,
                   bool                    hasScaleA      = false,
                   bool                    hasScaleB      = false,
                   float                   min            = -1.f,
                   float                   max            = 1.f,
                   const uint              scaleBlockSize = 32,
                   DataInitMode            initModeA      = DataInitMode(Bounded{}),
                   DataInitMode            initModeB      = DataInitMode(Bounded{}),
                   DataInitMode            initModeC      = DataInitMode(Bounded{})

    )
    {
        auto blockScalingA = (hasScaleA) ? scaleBlockSize : 1;
        auto blockScalingB = (hasScaleB) ? scaleBlockSize : 1;
        using STA          = typename SegmentedTypeOf<TA>::type;
        using STB          = typename SegmentedTypeOf<TB>::type;

        {
            auto dgenA = getDataGenerator<STA>(descA, min, max, seed + 1, blockScalingA, initModeA);
            hostA      = getRandomVector<STA>(dgenA, hasScaleA);
            if(hasScaleA)
                hostScaleA = dgenA.getScaleBytes();
        }

        {
            auto dgenB = getDataGenerator<STB>(descB, min, max, seed + 2, blockScalingB, initModeB);
            hostB      = getRandomVector<STB>(dgenB, hasScaleB);
            if(hasScaleB)
                hostScaleB = dgenB.getScaleBytes();
        }

        {
            auto dgenC = getDataGenerator<TC>(descC, min, max, seed, 1, initModeC);
            hostC      = getRandomVector<TC>(dgenC, false);
        }
    }

    template <typename TA, typename TB, typename TC>
    void DGenInput(const uint32_t    seed,
                   std::vector<TA>&  hostA,
                   TensorDescriptor& descA,
                   std::vector<TB>&  hostB,
                   TensorDescriptor& descB,
                   std::vector<TC>&  hostC,
                   TensorDescriptor& descC,
                   float             min       = -1.f,
                   float             max       = 1.f,
                   DataInitMode      initModeA = DataInitMode(Bounded{}),
                   DataInitMode      initModeB = DataInitMode(Bounded{}),
                   DataInitMode      initModeC = DataInitMode(Bounded{}))
    {
        std::vector<uint8_t> defaultHostScaleA;
        std::vector<uint8_t> defaultHostScaleB;
        DGenInput(seed,
                  hostA,
                  descA,
                  hostB,
                  descB,
                  hostC,
                  descC,
                  defaultHostScaleA,
                  defaultHostScaleB,
                  false,
                  false,
                  min,
                  max,
                  32,
                  initModeA,
                  initModeB,
                  initModeC);
    }
}
