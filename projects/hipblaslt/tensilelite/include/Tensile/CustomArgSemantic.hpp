// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>

#include <Tensile/DataTypes.hpp>
#include <Tensile/EnumStrings.hpp>
#include <tensilelitehost/export.h>

namespace TensileLite
{
    // Every semantic a custom kernel argument can name, one X_MACRO(name) each. The enum
    // value of a semantic is its position in this list, and it is what the logic files and
    // msgpack store by name, so append new entries and keep the existing ones in place.
    // The enum and its EnumStringTraits table below are all expanded from it.
    #define CustomArgSemantic_MACRO \
        /* Core GEMM problem args */ \
        X_MACRO(SizeFree0) \
        X_MACRO(SizeFree1) \
        X_MACRO(SizeFree2) \
        X_MACRO(SizeFree3) \
        X_MACRO(SizeSum) \
        X_MACRO(SizeSumDiv2) \
        X_MACRO(SizeSum1) \
        X_MACRO(SizeSum2) \
        X_MACRO(StrideA0) \
        X_MACRO(StrideA1) \
        X_MACRO(StrideA2) \
        X_MACRO(StrideB0) \
        X_MACRO(StrideB1) \
        X_MACRO(StrideB2) \
        X_MACRO(StrideC0) \
        X_MACRO(StrideC1) \
        X_MACRO(StrideC2) \
        X_MACRO(StrideD0) \
        X_MACRO(StrideD1) \
        X_MACRO(StrideD2) \
        X_MACRO(StrideE0) \
        X_MACRO(StrideE1) \
        X_MACRO(StrideScaleA0) \
        X_MACRO(StrideScaleA1) \
        X_MACRO(StrideScaleB0) \
        X_MACRO(StrideScaleB1) \
        X_MACRO(StrideA0Bytes) \
        X_MACRO(StrideB0Bytes) \
        X_MACRO(StrideC0Bytes) \
        X_MACRO(StrideD0Bytes) \
        X_MACRO(StrideMetadata0) \
        X_MACRO(StrideMetadata1) \
        X_MACRO(StrideCK) \
        X_MACRO(Alpha) \
        X_MACRO(Beta) \
        X_MACRO(ComputeUnits) \
        X_MACRO(SplitK) \
        X_MACRO(OutputBF16) \
        X_MACRO(Padding) \
        X_MACRO(ConstantZero) \
        X_MACRO(ConstantOne) \
        X_MACRO(DebugPattern) \
        /* Pointer args */ \
        X_MACRO(AddressA) \
        X_MACRO(AddressB) \
        X_MACRO(AddressC) \
        X_MACRO(AddressD) \
        X_MACRO(AddressE) \
        X_MACRO(AddressMetadata) \
        X_MACRO(AddressWorkspace) \
        X_MACRO(AddressFlags) \
        X_MACRO(AddressSynchronizer) \
        X_MACRO(AddressTD) \
        X_MACRO(AddressScaleA) \
        X_MACRO(AddressScaleB) \
        X_MACRO(AddressScaleC) \
        X_MACRO(AddressScaleD) \
        X_MACRO(AddressMXScaleA) \
        X_MACRO(AddressMXScaleB) \
        X_MACRO(AddressScaleAlphaVec) \
        X_MACRO(AddressBias) \
        X_MACRO(AddressAmaxOut) \
        X_MACRO(AmaxWS) \
        X_MACRO(AmaxSync) \
        X_MACRO(Synchronizer) \
        X_MACRO(DebugBuffer) \
        /* Kernel metadata args */ \
        X_MACRO(GemmInfo) \
        X_MACRO(GemmCount) \
        X_MACRO(InternalArgs) \
        X_MACRO(InternalArgs1) \
        X_MACRO(TensileInternalArg0) \
        X_MACRO(TensileInternalArg1) \
        X_MACRO(NumWorkGroups) \
        /* Persistent and legacy scheduling args */ \
        X_MACRO(ItersPerTile) \
        /* Static StreamK */ \
        X_MACRO(MagicNumberItersPerTile) \
        X_MACRO(MagicShiftItersPerTile) \
        X_MACRO(TotalIters) \
        X_MACRO(SKItersPerWG) \
        X_MACRO(SKGrid) \
        X_MACRO(SKTilesAndSplit) \
        X_MACRO(MagicNumberSizeAlg1) \
        X_MACRO(MagicShiftSizeAlg1) \
        X_MACRO(MagicNumberSizeAlg2) \
        X_MACRO(MagicShiftSizeAlg2) \
        /* Epilogue control args */ \
        X_MACRO(BiasType) \
        X_MACRO(StrideBias) \
        X_MACRO(FactorDim) \
        X_MACRO(ActivationTypeArg) \
        X_MACRO(ActivationArg) \
        X_MACRO(GSUSync) \
        /* Random seed args */ \
        X_MACRO(RNDSeed) \
        /* The whole fixed-size fused GEMM-A2A kernarg segment */ \
        X_MACRO(FusedA2ASegment) \
        /* Appended to preserve existing CustomArgSemantic values. */ \
        X_MACRO(PersistentGrid) \
        X_MACRO(BatchOffsetD) \
        X_MACRO(BatchOffsetC) \
        X_MACRO(BatchOffsetA) \
        X_MACRO(BatchOffsetB) \
        X_MACRO(AddressGateResidual) \
        X_MACRO(GateResidualType) \
        X_MACRO(StrideGate0) \
        X_MACRO(StrideGate1) \
        X_MACRO(StrideGate2) \
        X_MACRO(StrideE2) \
        X_MACRO(StrideScaleA2) \
        X_MACRO(StrideScaleB2) \
        X_MACRO(StrideMetadata2) \
        /* rocRoller libdivide workgroup-mapping kernargs. The RR_ prefix keeps \
           them distinct from another generator's WorkgroupMapping. */ \
        X_MACRO(RR_WorkgroupMapping) \
        X_MACRO(RR_MagicMultipleWgm) \
        X_MACRO(RR_MagicShiftAndSignWgm) \
        X_MACRO(RR_MagicMultipleNumTilesN) \
        X_MACRO(RR_MagicShiftAndSignNumTilesN) \
        X_MACRO(RR_QuotientTilesMByWgm) \
        X_MACRO(RR_MagicMultipleWgmTail) \
        X_MACRO(RR_MagicShiftAndSignWgmTail) \
        X_MACRO(RR_QuotientTilesByBlock) \
        X_MACRO(RR_MagicMultipleWgmMainBlock) \
        X_MACRO(RR_MagicShiftAndSignWgmMainBlock) \
        X_MACRO(AddressScaleZeroA) \
        X_MACRO(ScaleBlockSizeA) \
        X_MACRO(ESMRuntimeSupported) \
        X_MACRO(DeltaPeerD) \
        /* rocRoller Stream-K kernargs, appended after DeltaPeerD. The partial
           tile scratch is AddressWorkspace and the flag region is
           AddressSynchronizer. See RocRollerStreamKKernargs.hpp. */ \
        X_MACRO(RR_SKNumTilesN) \
        X_MACRO(RR_SKNumWGs) \
        X_MACRO(RR_SKNumSKTiles) \
        X_MACRO(RR_SKNumSKTilesPerWG) \
        X_MACRO(RR_SKNumDPTiles) \
        X_MACRO(RR_SKNumDPTilesPerWG) \
        X_MACRO(RR_SKMagicMultipleIters) \
        X_MACRO(RR_SKMagicShiftsIters) \
        X_MACRO(RR_SKMagicMultipleNumTilesN) \
        X_MACRO(RR_SKMagicShiftsNumTilesN) \
        /* int64 ceil(size/tile) for a non-power-of-two macrotile. Other RR_* kernargs are 32-bit. */ \
        X_MACRO(RR_NumTilesM) \
        X_MACRO(RR_NumTilesN) \
        /* K / mxBlockA, truncated: the K extent of A's MX scale tensor. */ \
        X_MACRO(SizeSumScaleA)

    enum class CustomArgSemantic
    {
        #define X_MACRO(name) name,
        CustomArgSemantic_MACRO
        #undef X_MACRO
        COUNT,
    };

    template <>
    struct EnumStringTraits<CustomArgSemantic>
    {
        static constexpr std::string_view typeName = "CustomArgSemantic";
        static constexpr auto entries = std::to_array<EnumStringEntry<CustomArgSemantic>>(
            {
#define X_MACRO(name) {CustomArgSemantic::name, #name},
                CustomArgSemantic_MACRO
#undef X_MACRO
            });
    };

    static_assert(EnumStrings<CustomArgSemantic>::size()
                  == static_cast<std::size_t>(CustomArgSemantic::COUNT));

    enum CustomGridSize
    {
        One,
        TilesX,
        TilesY,
        Batch,
        TilesXY,
        TilesXYBatch,
        StreamKWithBatch,
        StreamKNoBatch,
        TilesXYBatchGSU,
        PersistentGrid,
        PersistentWithBatch,
        PersistentNoBatch,
        TilesYGSU,
        ComputeUnits,
        // Unscoped, so this cannot also be named RocRollerStreamK: that
        // enumerator belongs to CustomWorkspaceType. Logic files still
        // serialize it as "RocRollerStreamK".
        RocRollerStreamKGrid,
        CustomGridSize_Count,
    };

    template <>
    struct EnumStringTraits<CustomGridSize>
    {
        static constexpr std::string_view typeName = "CustomGridSize";
        static constexpr auto entries = std::to_array<EnumStringEntry<CustomGridSize>>(
            {
                {CustomGridSize::One, "One"},
                {CustomGridSize::TilesX, "TilesX"},
                {CustomGridSize::TilesY, "TilesY"},
                {CustomGridSize::Batch, "Batch"},
                {CustomGridSize::TilesXY, "TilesXY"},
                {CustomGridSize::TilesXYBatch, "TilesXYBatch"},
                {CustomGridSize::StreamKWithBatch, "StreamKWithBatch"},
                {CustomGridSize::StreamKNoBatch, "StreamKNoBatch"},
                {CustomGridSize::TilesXYBatchGSU, "TilesXYBatchGSU"},
                {CustomGridSize::PersistentGrid, "PersistentGrid"},
                {CustomGridSize::PersistentWithBatch, "PersistentWithBatch"},
                {CustomGridSize::PersistentNoBatch, "PersistentNoBatch"},
                {CustomGridSize::TilesYGSU, "TilesYGSU"},
                {CustomGridSize::ComputeUnits, "ComputeUnits"},
                {CustomGridSize::RocRollerStreamKGrid, "RocRollerStreamK"},
            });
    };

    static_assert(EnumStrings<CustomGridSize>::size()
                  == static_cast<std::size_t>(CustomGridSize_Count));

    enum CustomWorkspaceType
    {
        None,
        SplitK,
        StreamK,
        StreamKWithReduction,
        // rocRoller partials: one tile per workgroup, including an even split.
        RocRollerStreamK,
        CustomWorkspaceType_Count,
    };

    template <>
    struct EnumStringTraits<CustomWorkspaceType>
    {
        static constexpr std::string_view typeName = "CustomWorkspaceType";
        static constexpr auto entries = std::to_array<EnumStringEntry<CustomWorkspaceType>>(
            {
                {CustomWorkspaceType::None, "None"},
                {CustomWorkspaceType::SplitK, "SplitK"},
                {CustomWorkspaceType::StreamK, "StreamK"},
                {CustomWorkspaceType::StreamKWithReduction, "StreamKWithReduction"},
                {CustomWorkspaceType::RocRollerStreamK, "RocRollerStreamK"},
            });
    };

    static_assert(EnumStrings<CustomWorkspaceType>::size()
                  == static_cast<std::size_t>(CustomWorkspaceType_Count));

    struct CustomArgDefinition
    {
        CustomArgType type;
        CustomArgSemantic semantic;
        size_t padding = 0;
        size_t index   = 0;
    };

    inline std::string toString(CustomArgDefinition arg)
    {
        auto s = toString(arg.type) + " " + toString(arg.semantic);
        if(arg.padding > 0)
            s += " padding:" + std::to_string(arg.padding);
        return s;
    }

    inline std::ostream& operator<<(std::ostream& stream, const CustomArgDefinition& arg)
    {
        return stream << toString(arg);
    }

    inline std::istream& operator>>(std::istream& stream, CustomArgDefinition& arg)
    {
        std::string strType;
        std::string strSemantic;
        stream >> strType >> strSemantic;
        arg.type     = enumFromString<CustomArgType>(strType);
        arg.semantic = enumFromString<CustomArgSemantic>(strSemantic);
        return stream;
    }
}
