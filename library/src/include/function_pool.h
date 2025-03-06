/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef FUNCTION_POOL_H
#define FUNCTION_POOL_H

#include "../../../shared/rocfft_complex.h"
#include "../device/kernels/common.h"
#include "function_map_key.h"
#include <sstream>
#include <unordered_map>

inline std::string PrintMissingKernelInfo(const FMKey& key)
{
    std::stringstream msg;
    msg << "Kernel not found: \n"
        << "\tlength: " << key.lengths[0] << "," << key.lengths[1] << "\n"
        << "\tprecision: " << key.precision << "\n"
        << "\tscheme: " << PrintScheme(key.scheme) << "\n"
        << "\tSBRC Transpose type: " << PrintSBRCTransposeType(key.sbrcTrans) << std::endl;

    return msg.str();
}

struct FFTKernel
{
    std::vector<size_t> factors;
    // NB:
    //    Some abbrevs for namings that we can follow (tpb/wgs/tpt)
    // number of transforms performed by one threadblock (tpb)
    unsigned int transforms_per_block = 0;
    // workgroup size： number of threads per block (wgs) = tpt * tpb
    int workgroup_size = 0;
    // number of threads to perform single transform (tpt)
    // 2D_SINGLE specifies separate threads for each dimension;
    // otherwise second dim's threads will be 0
    std::array<int, 2> threads_per_transform = {0, 0};
    bool               use_3steps_large_twd  = false;
    bool               half_lds              = false;
    bool               direct_to_from_reg    = false;
    // true if this kernel is compiled ahead of time (i.e. at library
    // build time), using runtime compilation.
    bool aot_rtc = false;

    FFTKernel()                 = default;
    FFTKernel(const FFTKernel&) = default;

    FFTKernel& operator=(const FFTKernel&) = default;

    FFTKernel(bool                  use_3steps,
              std::vector<size_t>&& factors,
              int                   tpb,
              int                   wgs,
              std::array<int, 2>&&  tpt,
              bool                  half_lds           = false,
              bool                  direct_to_from_reg = false,
              bool                  aot_rtc            = false)
        : factors(factors)
        , transforms_per_block(tpb)
        , workgroup_size(wgs)
        , threads_per_transform(tpt)
        , use_3steps_large_twd(use_3steps)
        , half_lds(half_lds)
        , direct_to_from_reg(direct_to_from_reg)
        , aot_rtc(aot_rtc)
    {
    }

    FFTKernel(const KernelConfig& config)
        : factors(config.factors)
        , transforms_per_block(config.transforms_per_block)
        , workgroup_size(config.workgroup_size)
        , threads_per_transform(config.threads_per_transform)
        , use_3steps_large_twd(config.use_3steps_large_twd)
        , half_lds(config.half_lds)
        , direct_to_from_reg(config.direct_to_from_reg)
    {
    }

    KernelConfig get_kernel_config() const
    {
        KernelConfig config;
        config.transforms_per_block  = transforms_per_block;
        config.workgroup_size        = workgroup_size;
        config.threads_per_transform = threads_per_transform;
        config.use_3steps_large_twd  = use_3steps_large_twd;
        config.half_lds              = half_lds;
        config.direct_to_from_reg    = direct_to_from_reg;
        config.factors               = factors;

        return config;
    }
};

struct function_pool_data
{
    // when AOT generator adds a default key-kernel,
    // we get the keys of two version: empty-config vs full-config
    // make the pair as an entry in a map so that we know they are the same things
    std::unordered_map<FMKey, FMKey, SimpleHash>     def_key_pool;
    std::unordered_map<FMKey, FFTKernel, SimpleHash> function_map;

    function_pool_data();

    static function_pool_data& get_function_pool_data()
    {
        static function_pool_data data;
        return data;
    }
};

class function_pool
{
    unsigned int                                      max_lds_bytes;
    std::unordered_map<FMKey, FMKey, SimpleHash>&     def_key_pool;
    std::unordered_map<FMKey, FFTKernel, SimpleHash>& function_map;

    const FMKey& get_actual_key(const FMKey& key) const
    {
        // - for keys that we are querying with no/empty kernel-config, actually we are refering to
        //   the default kernel-configs in kernel-generator.py. So get the actual keys to look-up
        //   the pool.
        // - if not in the def_key_pool, then we simply use itself (for dynamically added kernel)
        if(def_key_pool.count(key) > 0)
            return def_key_pool.at(key);
        else
            return key;
    }

public:
    function_pool(unsigned int max_lds_bytes)
        : max_lds_bytes(max_lds_bytes)
        , def_key_pool(function_pool_data::get_function_pool_data().def_key_pool)
        , function_map(function_pool_data::get_function_pool_data().function_map)
    {
        // We would only see zero if we received a
        // default-constructed device prop struct, which means
        // someone forgot to initialize the struct somewhere.
        if(max_lds_bytes == 0)
            throw std::runtime_error("function_pool: max_lds_bytes not initialized");
    }

    function_pool(const hipDeviceProp_t& prop)
        : function_pool(prop.sharedMemPerBlock)
    {
    }

    function_pool(function_pool& p) = delete;
    function_pool& operator=(const function_pool&) = delete;

    ~function_pool() = default;

    // add a new kernel in runtime
    bool add_new_kernel(const FMKey& new_key)
    {
        // already has this kernel
        if(has_function(new_key))
            return true;

        return std::get<1>(function_map.emplace(new_key, FFTKernel(new_key.kernel_config)));
    }

    bool has_function(const FMKey& key) const
    {
        auto real_key = get_actual_key(key);
        if(!real_key.base_lds_usage_fits(max_lds_bytes))
            return false;
        return function_map.count(real_key) > 0;
    }

    size_t get_largest_length(rocfft_precision precision) const
    {
        auto supported = get_lengths(precision, CS_KERNEL_STOCKHAM);
        auto itr       = std::max_element(supported.cbegin(), supported.cend());
        if(itr != supported.cend())
            return *itr;
        return 0;
    }

    std::vector<size_t> get_lengths(rocfft_precision precision, ComputeScheme scheme) const
    {
        std::vector<size_t> lengths;
        for(auto const& kv : function_map)
        {
            if(!kv.first.base_lds_usage_fits(max_lds_bytes))
                continue;
            if(kv.first.lengths[1] == 0 && kv.first.precision == precision
               && kv.first.scheme == scheme && kv.first.sbrcTrans == NONE)
            {
                lengths.push_back(kv.first.lengths[0]);
            }
        }

        return lengths;
    }

    FFTKernel get_kernel(const FMKey& key) const
    {
        auto real_key = get_actual_key(key);
        if(!real_key.base_lds_usage_fits(max_lds_bytes))
            throw std::out_of_range("kernel not found in map");
        return function_map.at(real_key);
    }

    // helper for common used
    bool has_SBCC_kernel(size_t length, rocfft_precision precision) const
    {
        return has_function(FMKey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
    }

    bool has_SBRC_kernel(size_t              length,
                         rocfft_precision    precision,
                         SBRC_TRANSPOSE_TYPE trans_type = TILE_ALIGNED) const
    {
        return has_function(FMKey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_RC, trans_type));
    }

    bool has_SBCR_kernel(size_t length, rocfft_precision precision) const
    {
        return has_function(FMKey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_CR));
    }

    const auto& get_map() const
    {
        return function_map;
    }
};

// Insert a key-kernel pair for AOT generator. This function is called in
// N (set via CMake) separate files, generated by kernel-generator.py for parallel compiling.
// That is, the default kernel-config we set in the kernel-generator.py we save a pair as
// <key-empty-config, key-actual-config> that allows us to use
// the empty-config key to get the default kernel
static bool insert_default_entry(const FMKey&                                      def_key,
                                 const FFTKernel&                                  kernel,
                                 std::unordered_map<FMKey, FMKey, SimpleHash>&     def_key_pool,
                                 std::unordered_map<FMKey, FFTKernel, SimpleHash>& function_map)
{
    // simple_key means the same thing as def_key, but we just remove kernel-config
    // so we don't need to know the exact config when we're lookin' for the default kernel
    FMKey simple_key(def_key);
    simple_key.kernel_config = KernelConfig::EmptyConfig();

    def_key_pool.emplace(simple_key, def_key);

    // still use the detailed key with config to maintain the function map
    return std::get<1>(function_map.emplace(def_key, kernel));
}

#endif // FUNCTION_POOL_H
