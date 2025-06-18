// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FUNCTION_MAP_KEY_H
#define FUNCTION_MAP_KEY_H

#include <array>
#include <tuple>
#include <vector>

#include "../../../shared/precision_type.h"
#include "compute_scheme.h"
#include "data_descriptor.h"
#include "enum_printer.h"
#include "twiddles.h"

struct KernelConfig
{
    bool                use_3steps_large_twd  = false;
    bool                half_lds              = false;
    bool                direct_to_from_reg    = false;
    bool                intrinsic_buffer_inst = false;
    unsigned int        transforms_per_block  = 0;
    int                 workgroup_size        = 0;
    std::array<int, 2>  threads_per_transform = {0, 0};
    std::vector<size_t> factors               = {0};
    // above data is what we can tune
    //
    // the followings are other information of this kernel.
    // not tunable values, they come from the tuned problem nodes.
    // When pre-building kernels to AOT-cache, they can tell
    // us what exact kernels to pre-build, avoding other variants,
    // hence reduce cache sizes.
    EmbeddedType ebType = EmbeddedType::NONE;

    // a little different for direction
    // When prebuilding AOT kernels:
    // For real-transform: build with the exact direction (since R2C & C2R are different)
    //     cmplx-transform: still build -1 & 1, since they share same solution.
    int direction = -1;
    // NB:
    //  when generating candidates in tuning process,
    //  we haven't done buffer-assignment and CollapseContiguousDims
    //  but we should be able to know it when export solutions.
    int               static_dim = 0;
    PlacementCode     placement  = PC_UNSET;
    rocfft_array_type iAryType   = rocfft_array_type_complex_interleaved;
    rocfft_array_type oAryType   = rocfft_array_type_complex_interleaved;

    KernelConfig()                    = default;
    KernelConfig(const KernelConfig&) = default;

    KernelConfig(bool                  use_3steps,
                 std::vector<size_t>&& factors,
                 int                   tpb,
                 int                   wgs,
                 std::array<int, 2>&&  tpt,
                 bool                  half_lds              = false,
                 bool                  direct_to_from_reg    = false,
                 bool                  intrinsic_buffer_inst = false,
                 EmbeddedType          ebType                = EmbeddedType::NONE,
                 int                   direction             = -1,
                 int                   static_dim            = 0,
                 PlacementCode         placement             = PC_UNSET,
                 rocfft_array_type     iAryType = rocfft_array_type_complex_interleaved,
                 rocfft_array_type     oAryType = rocfft_array_type_complex_interleaved)
        : use_3steps_large_twd(use_3steps)
        , half_lds(half_lds)
        , direct_to_from_reg(direct_to_from_reg)
        , intrinsic_buffer_inst(intrinsic_buffer_inst)
        , transforms_per_block(tpb)
        , workgroup_size(wgs)
        , threads_per_transform(tpt)
        , factors(factors)
        , ebType(ebType)
        , direction(direction)
        , static_dim(static_dim)
        , placement(placement)
        , iAryType(iAryType)
        , oAryType(oAryType)
    {
    }

    KernelConfig& operator=(const KernelConfig&) = default;

    bool operator==(const KernelConfig& rhs) const
    {
        return std::tie(use_3steps_large_twd,
                        half_lds,
                        direct_to_from_reg,
                        intrinsic_buffer_inst,
                        transforms_per_block,
                        workgroup_size,
                        threads_per_transform,
                        factors)
               == std::tie(rhs.use_3steps_large_twd,
                           rhs.half_lds,
                           rhs.direct_to_from_reg,
                           rhs.intrinsic_buffer_inst,
                           rhs.transforms_per_block,
                           rhs.workgroup_size,
                           rhs.threads_per_transform,
                           rhs.factors);
    }

    bool operator<(const KernelConfig& rhs) const
    {
        return std::tie(use_3steps_large_twd,
                        half_lds,
                        direct_to_from_reg,
                        intrinsic_buffer_inst,
                        transforms_per_block,
                        workgroup_size,
                        threads_per_transform,
                        factors)
               < std::tie(rhs.use_3steps_large_twd,
                          rhs.half_lds,
                          rhs.direct_to_from_reg,
                          rhs.intrinsic_buffer_inst,
                          rhs.transforms_per_block,
                          rhs.workgroup_size,
                          rhs.threads_per_transform,
                          rhs.factors);
    }

    std::string Print() const
    {
        std::stringstream ss;
        ss << "KernelConfig: {";

        ss << "3steps: " << (use_3steps_large_twd ? "true" : "false")
           << ", half_lds: " << (half_lds ? "true" : "false")
           << ", direct_reg: " << (direct_to_from_reg ? "true" : "false")
           << ", try_use_buf_inst: " << (intrinsic_buffer_inst ? "true" : "false")
           << ", tpb: " << transforms_per_block << ", wgs: " << workgroup_size << ", tpt: ["
           << threads_per_transform[0] << "," << threads_per_transform[1] << "], factors: [";

        std::string COMMA = "";
        for(auto factor : factors)
        {
            ss << COMMA << factor;
            COMMA = ", ";
        }
        ss << "]";

        ss << "}";

        return ss.str();
    }

    static KernelConfig EmptyConfig()
    {
        static KernelConfig empty;
        return empty;
    }
};

namespace std
{
    // hash function of FMKey requires the definition of hash function of kernel-config
    template <>
    struct hash<KernelConfig>
    {
        size_t operator()(const KernelConfig& config) const noexcept
        {
            size_t h = 0;
            h ^= std::hash<bool>{}(config.use_3steps_large_twd);
            h ^= std::hash<bool>{}(config.half_lds);
            h ^= std::hash<bool>{}(config.direct_to_from_reg);
            h ^= std::hash<bool>{}(config.intrinsic_buffer_inst);
            h ^= std::hash<unsigned int>{}(config.transforms_per_block);
            h ^= std::hash<int>{}(config.workgroup_size);
            for(auto& v : config.threads_per_transform)
                h ^= std::hash<int>{}(v);

            // twiddle_factors defines a TWIDDLES_MAX_RADICES = 8
            // which means the maximal factorization pass is 8
            auto factors_max_len = config.factors;
            factors_max_len.resize(TWIDDLES_MAX_RADICES);

            for(auto& v : factors_max_len)
                h ^= std::hash<size_t>{}(v);
            return h;
        }
    };
}

// Implementing the ToString / FromString (data_descriptor.h)
// for writing-to/reading-from texted-format solution map
template <>
struct ToString<KernelConfig>
{
    std::string print(const KernelConfig& value) const
    {
        std::string      str = "{";
        std::vector<int> tpt = {value.threads_per_transform[0], value.threads_per_transform[1]};

        str += FieldDescriptor<bool>().describe("use_3steps", value.use_3steps_large_twd) + ",";
        str += FieldDescriptor<bool>().describe("half_lds", value.half_lds) + ",";
        str += FieldDescriptor<bool>().describe("dir_reg", value.direct_to_from_reg) + ",";
        str += FieldDescriptor<bool>().describe("buffer_inst", value.intrinsic_buffer_inst) + ",";
        str += FieldDescriptor<unsigned int>().describe("tpb", value.transforms_per_block) + ",";
        str += FieldDescriptor<int>().describe("wgs", value.workgroup_size) + ",";
        str += VectorFieldDescriptor<int>().describe("tpt", tpt) + ",";
        str += VectorFieldDescriptor<size_t>().describe("factors", value.factors) + ",";
        // below: not tunable data, for AOT cache
        str += FieldDescriptor<std::string>().describe("ebtype", PrintEBType(value.ebType)) + ",";
        str += FieldDescriptor<int>().describe("direction", value.direction) + ",";
        str += FieldDescriptor<int>().describe("static_dim", value.static_dim) + ",";
        str += FieldDescriptor<std::string>().describe("placement",
                                                       PrintPlacementCode(value.placement))
               + ",";
        str += FieldDescriptor<std::string>().describe("iAryType", PrintArrayType(value.iAryType))
               + ",";
        str += FieldDescriptor<std::string>().describe("oAryType", PrintArrayType(value.oAryType));

        str += "}";
        return str;
    }
};

template <>
struct FromString<KernelConfig>
{
    void Get(KernelConfig& ret, std::sregex_token_iterator& current) const
    {
        std::vector<int> tpt;
        size_t           tpb;
        std::string      ebTypeStr, placementStr, iAryTypeStr, oAryTypeStr;

        FieldParser<bool>().parse("use_3steps", ret.use_3steps_large_twd, current);
        FieldParser<bool>().parse("half_lds", ret.half_lds, current);
        FieldParser<bool>().parse("dir_reg", ret.direct_to_from_reg, current);
        FieldParser<bool>().parse("buffer_inst", ret.intrinsic_buffer_inst, current);
        FieldParser<size_t>().parse("tpb", tpb, current);

        FieldParser<int>().parse("wgs", ret.workgroup_size, current);
        VectorFieldParser<int>().parse("tpt", tpt, current);
        VectorFieldParser<size_t>().parse("factors", ret.factors, current);

        if(DescriptorFormatVersion::UsingVersion < 2)
        {
            ret.static_dim = 0;
            ret.direction  = -1;
            ebTypeStr      = PrintEBType(EmbeddedType::NONE);
            placementStr   = PrintPlacementCode(PC_UNSET);
            iAryTypeStr    = PrintArrayType(rocfft_array_type_complex_interleaved);
            oAryTypeStr    = PrintArrayType(rocfft_array_type_complex_interleaved);
        }
        else
        {
            FieldParser<std::string>().parse("ebtype", ebTypeStr, current);
            FieldParser<int>().parse("direction", ret.direction, current);
            FieldParser<int>().parse("static_dim", ret.static_dim, current);
            FieldParser<std::string>().parse("placement", placementStr, current);
            FieldParser<std::string>().parse("iAryType", iAryTypeStr, current);
            FieldParser<std::string>().parse("oAryType", oAryTypeStr, current);
        }

        ret.ebType                   = StrToEBType(ebTypeStr);
        ret.placement                = StrToPlacementCode(placementStr);
        ret.iAryType                 = StrToArrayType(iAryTypeStr);
        ret.oAryType                 = StrToArrayType(oAryTypeStr);
        ret.transforms_per_block     = tpb;
        ret.threads_per_transform[0] = tpt[0];
        ret.threads_per_transform[1] = tpt[1];
    }
};

// Base function map key
struct FMKeyBase
{
    FMKeyBase(std::array<size_t, 3> lengths,
              rocfft_precision      precision,
              ComputeScheme         scheme = CS_NONE)
        : lengths(lengths)
        , precision(precision)
        , scheme(scheme)
    {
    }

    FMKeyBase()                 = default;
    FMKeyBase(const FMKeyBase&) = default;

    virtual ~FMKeyBase(){};

    // LDS size this config is intended for.  This is not a key
    // field, and multiple configs that differ only in LDS size are
    // allowed to coexist in the function pool.
    size_t lds_size_bytes = 0;

    std::array<size_t, 3> lengths;

    rocfft_precision precision;
    ComputeScheme    scheme;
};

// length, precision, scheme are theose fundemantal information of a kernel;
// SBRC_TRANS is also neccessary for SBRC or SBRC_3D, but for non-SBRC, it is just NONE
// And the newly added KernerlConfig is the key to supporting the "multi-configurations".
// KernelConfig denotes what parameters we can alter to "generate and tune" a kernel
//
// NB:
//    Since we didn't have the KernelConfig before, so, when getting the default kernels
//    from the function_pool, the kernel_config "variable" would be a default EmptyConfig().
//    But actually, the config is defined in the kernel-generator.py, so we are still able to
//    know how the "EmptyConfig" can be mapped to a non-empty config (in kernel-gerator.py)
//    (And that is what exactly "fuction_pool::insert_default_entry()" and
//                               "function_pool::get_actual_key()"" is doing
//

// Function map key for regular kernels, derived from FMKeyBase.
struct FMKey : public FMKeyBase
{
    SBRC_TRANSPOSE_TYPE sbrcTrans     = NONE;
    KernelConfig        kernel_config = KernelConfig::EmptyConfig();

    FMKey()             = default;
    FMKey(const FMKey&) = default;

    // simple constructor for 1d-kernel
    FMKey(size_t              length0,
          rocfft_precision    precision,
          ComputeScheme       scheme        = CS_KERNEL_STOCKHAM,
          SBRC_TRANSPOSE_TYPE transpose     = NONE,
          KernelConfig        kernel_config = KernelConfig::EmptyConfig())
        : FMKeyBase({length0, 0, 0}, precision, scheme)
        , sbrcTrans(transpose)
        , kernel_config(kernel_config)

    {
    }

    // with every data
    FMKey(size_t              length0,
          size_t              length1,
          rocfft_precision    precision,
          ComputeScheme       scheme        = CS_KERNEL_2D_SINGLE,
          SBRC_TRANSPOSE_TYPE transpose     = NONE,
          KernelConfig        kernel_config = KernelConfig::EmptyConfig())
        : FMKeyBase({length0, length1, 0}, precision, scheme)
        , sbrcTrans(transpose)
        , kernel_config(kernel_config)
    {
    }

    FMKey& operator=(const FMKey&) = default;

    bool operator==(const FMKey& rhs) const
    {
        return std::tie(lengths, precision, scheme, sbrcTrans, kernel_config)
               == std::tie(
                   rhs.lengths, rhs.precision, rhs.scheme, rhs.sbrcTrans, rhs.kernel_config);
    }

    bool operator!=(const FMKey& rhs) const
    {
        return !((*this) == rhs);
    }

    bool operator<(const FMKey& rhs) const
    {
        return std::tie(lengths, precision, scheme, sbrcTrans, kernel_config)
               < std::tie(rhs.lengths, rhs.precision, rhs.scheme, rhs.sbrcTrans, rhs.kernel_config);
    }

    static FMKey EmptyFMKey()
    {
        static FMKey empty;
        return empty;
    }
};

static std::vector<FMKey> EmptyFMKeyVec = {};

// add an alternative kernel with different kernel config from base FMKey
static FMKey get_alternative_FMKey(const FMKey& base_FMKey, const KernelConfig& alt_config)
{
    FMKey ret_key(base_FMKey);
    ret_key.kernel_config = alt_config;

    return ret_key;
}

static void GetKernelToken(const FMKey& key, std::string& min_token)
{
    min_token = "kernel";

    min_token += "_len";
    min_token += std::to_string(key.lengths[0]);
    if(key.scheme == CS_KERNEL_2D_SINGLE)
        min_token += "x" + std::to_string(key.lengths[1]);

    min_token += "_" + PrintPrecision(key.precision);
    min_token += "_" + PrintKernelSchemeAbbr(key.scheme);

    // NB: KernelToken is used when tuning the kernel configuration,
    //     But when we try different setting of TPB, the SBRCTransType
    //     would not be the same value. So we should not keep the SBRCTransType
    //     in the token, and all the SBRC kernels in that solution-vec may have
    //     specify the real type.
    // min_token += "_" + PrintSBRCTransposeType(key.sbrcTrans);
}

template <>
struct ToString<FMKey>
{
    std::string print(const FMKey& value) const
    {
        std::string         str     = "{";
        std::vector<size_t> lengths = {value.lengths[0], value.lengths[1]};

        str += VectorFieldDescriptor<size_t>().describe("lengths", lengths) + ",";
        str += FieldDescriptor<std::string>().describe("precision", PrintPrecision(value.precision))
               + ",";
        str += FieldDescriptor<std::string>().describe("scheme", PrintScheme(value.scheme)) + ",";
        str += FieldDescriptor<std::string>().describe("sbrc_trans",
                                                       PrintSBRCTransposeType(value.sbrcTrans))
               + ",";
        str += FieldDescriptor<KernelConfig>().describe("kernelConfig", value.kernel_config);
        str += "}";
        return str;
    }
};

template <>
struct FromString<FMKey>
{
    void Get(FMKey& ret, std::sregex_token_iterator& current) const
    {
        std::vector<size_t> len;
        std::string         precStr, schemeStr, sbrcTransStr;
        KernelConfig        config;

        VectorFieldParser<size_t>().parse("lengths", len, current);
        FieldParser<std::string>().parse("precision", precStr, current);
        FieldParser<std::string>().parse("scheme", schemeStr, current);
        FieldParser<std::string>().parse("sbrc_trans", sbrcTransStr, current);
        FieldParser<KernelConfig>().parse("kernelConfig", config, current);

        ret.lengths       = {len[0], len[1]};
        ret.precision     = StrToPrecision(precStr);
        ret.scheme        = StrToComputeScheme(schemeStr);
        ret.sbrcTrans     = StrToSBRCTransType(sbrcTransStr);
        ret.kernel_config = config;
    }
};

// Hash function for FMKey.
struct SimpleHash
{
    size_t operator()(const FMKey& p) const noexcept
    {
        size_t h = 0;
        for(auto& v : p.lengths)
            h ^= std::hash<int>{}(v);
        h ^= std::hash<rocfft_precision>{}(p.precision);
        h ^= std::hash<ComputeScheme>{}(p.scheme);
        h ^= std::hash<SBRC_TRANSPOSE_TYPE>{}(p.sbrcTrans);
        h ^= std::hash<KernelConfig>{}(p.kernel_config);

        return h;
    }
};

// Function map key for partial-pass kernels.
// Derived from FMKeyBase, but adds two kernel configurations.
// Partial pass kernels are linked, as opposed to regular
// kernels that can act completely independent, and require
// a separate key type.
// When querying the function pool with this key, we can
// return the two partial pass kernels that can handle the
// specified FFT, e.g., a 3D FFT with two partial pass kernels.
struct PPFMKey : public FMKeyBase
{
    KernelConfig kernel_config_1 = KernelConfig::EmptyConfig();
    KernelConfig kernel_config_2 = KernelConfig::EmptyConfig();

    PPFMKey()               = default;
    PPFMKey(const PPFMKey&) = default;

    // with every data
    PPFMKey(size_t           length0,
            size_t           length1,
            size_t           length2,
            rocfft_precision precision,
            ComputeScheme    scheme          = CS_3D_PP,
            KernelConfig     kernel_config_1 = KernelConfig::EmptyConfig(),
            KernelConfig     kernel_config_2 = KernelConfig::EmptyConfig())
        : FMKeyBase({length0, length1, length2}, precision, scheme)
        , kernel_config_1(kernel_config_1)
        , kernel_config_2(kernel_config_2)
    {
    }

    PPFMKey& operator=(const PPFMKey&) = default;

    bool operator==(const PPFMKey& rhs) const
    {
        return std::tie(lengths, precision, scheme, kernel_config_1, kernel_config_2)
               == std::tie(rhs.lengths,
                           rhs.precision,
                           rhs.scheme,
                           rhs.kernel_config_1,
                           rhs.kernel_config_2);
    }

    bool operator!=(const PPFMKey& rhs) const
    {
        return !((*this) == rhs);
    }

    bool operator<(const PPFMKey& rhs) const
    {
        return std::tie(lengths, precision, scheme, kernel_config_1, kernel_config_2)
               < std::tie(rhs.lengths,
                          rhs.precision,
                          rhs.scheme,
                          rhs.kernel_config_1,
                          rhs.kernel_config_2);
    }

    static PPFMKey EmptyPPFMKey()
    {
        static PPFMKey empty;
        return empty;
    }

    // TODO: Implement this
    bool base_lds_usage_fits(unsigned int lds_size) const
    {
        return true;
    }
};

// Hash function for PPFMKey.
struct SimpleHashPP
{
    size_t operator()(const PPFMKey& p) const noexcept
    {
        size_t h = 0;
        for(auto& v : p.lengths)
            h ^= std::hash<int>{}(v);
        h ^= std::hash<rocfft_precision>{}(p.precision);
        h ^= std::hash<ComputeScheme>{}(p.scheme);
        h ^= std::hash<KernelConfig>{}(p.kernel_config_1);
        h ^= std::hash<KernelConfig>{}(p.kernel_config_2);

        return h;
    }
};

#endif
