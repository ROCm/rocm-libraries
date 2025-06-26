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

#include "../../../shared/arithmetic.h"
#include "../../../shared/rocfft_complex.h"
#include "../device/kernels/common.h"
#include "function_map_key.h"
#include <functional>
#include <optional>
#include <sstream>
#include <unordered_map>

inline std::string PrintMissingKernelInfoBase(const FMKeyBase& key)
{
    std::stringstream msg;
    msg << "Kernel not found: \n"
        << "\tlength: " << key.lengths[0] << "," << key.lengths[1] << "\n"
        << "\tprecision: " << key.precision << "\n"
        << "\tscheme: " << PrintScheme(key.scheme) << "\n";
    return msg.str();
}

inline std::string PrintMissingKernelInfo(const PPFMKey& key)
{
    std::stringstream msg;
    msg << PrintMissingKernelInfoBase(key);
    return msg.str();
}

inline std::string PrintMissingKernelInfo(const FMKey& key)
{
    std::stringstream msg;
    msg << PrintMissingKernelInfoBase(key)
        << "\tSBRC Transpose type: " << PrintSBRCTransposeType(key.sbrcTrans) << std::endl;
    return msg.str();
}

struct PartialPassParams
{
    PartialPassParams() = default;

    PartialPassParams(const ComputeScheme&             scheme,
                      const unsigned int&              current_dim,
                      const unsigned int&              off_dim,
                      const std::vector<unsigned int>& pp_factors_curr,
                      const std::vector<unsigned int>& pp_factors_other)
        : scheme(scheme)
        , current_dim(current_dim)
        , off_dim(off_dim)
        , pp_factors_curr(pp_factors_curr)
        , pp_factors_other(pp_factors_other)
    {
    }

    ComputeScheme             scheme      = CS_NONE;
    unsigned int              current_dim = 0;
    unsigned int              off_dim     = 0;
    std::vector<unsigned int> pp_factors_curr;
    std::vector<unsigned int> pp_factors_other;
};

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

    PartialPassParams pp_params;

    FFTKernel()                 = default;
    FFTKernel(const FFTKernel&) = default;

    FFTKernel& operator=(const FFTKernel&) = default;

    FFTKernel(bool                        use_3steps,
              std::vector<size_t>&&       factors,
              int                         tpb,
              int                         wgs,
              std::array<int, 2>&&        tpt,
              bool                        half_lds           = false,
              bool                        direct_to_from_reg = false,
              bool                        aot_rtc            = false,
              ComputeScheme               scheme             = CS_NONE,
              unsigned int                current_dim        = 0,
              unsigned int                off_dim            = 0,
              std::vector<unsigned int>&& pp_factors_curr    = std::vector<unsigned int>(),
              std::vector<unsigned int>&& pp_factors_other   = std::vector<unsigned int>())
        : factors(factors)
        , transforms_per_block(tpb)
        , workgroup_size(wgs)
        , threads_per_transform(tpt)
        , use_3steps_large_twd(use_3steps)
        , half_lds(half_lds)
        , direct_to_from_reg(direct_to_from_reg)
        , aot_rtc(aot_rtc)
        , pp_params(scheme, current_dim, off_dim, pp_factors_curr, pp_factors_other)
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

typedef std::unordered_multimap<FMKey, FMKey, SimpleHash>       FPKeyMap;
typedef std::unordered_multimap<PPFMKey, PPFMKey, SimpleHashPP> PPFPKeyMap;

typedef std::unordered_multimap<FMKey, FFTKernel, SimpleHash>                    FPMap;
typedef std::unordered_multimap<PPFMKey, std::array<FFTKernel, 2>, SimpleHashPP> PPFPMap;

struct function_pool_data
{
    // when AOT generator adds a default key-kernel,
    // we get the keys of two version: empty-config vs full-config
    // make the pair as an entry in a map so that we know they are the same things
    std::tuple<FPKeyMap, PPFPKeyMap> def_keys;
    std::tuple<FPMap, PPFPMap>       function_maps;

    function_pool_data();

    static function_pool_data& get_function_pool_data()
    {
        static function_pool_data data;
        return data;
    }
};

class function_pool
{
    unsigned int max_lds_bytes;
    FPKeyMap&    def_key_pool;
    PPFPKeyMap&  def_pp_key_pool;

    FPMap&   function_map;
    PPFPMap& pp_function_map;

    // look in the specified map for the specified key, returning an
    // iterator to the item that fits best into the available LDS
    template <typename Tmap, typename TKey>
    typename Tmap::const_iterator find_key_in_map(const Tmap& fmap, const TKey& key) const
    {
        auto   range    = fmap.equal_range(key);
        auto   best     = fmap.end();
        size_t best_lds = 0;
        for(; range.first != range.second; ++range.first)
        {
            if(range.first->first.lds_size_bytes <= max_lds_bytes
               && range.first->first.lds_size_bytes > best_lds)
            {
                best     = range.first;
                best_lds = best->first.lds_size_bytes;
            }
        }
        return best;
    }

    const FMKey& get_actual_key(const FMKey& key) const
    {
        // - for keys that we are querying with no/empty kernel-config, actually we are refering to
        //   the default kernel-configs in kernel-generator.py. So get the actual keys to look-up
        //   the pool.
        // - if not in the def_key_pool, then we simply use itself (for dynamically added kernel)
        auto it = find_key_in_map(def_key_pool, key);
        if(it != def_key_pool.end())
            return it->second;
        else
            return key;
    }

    const PPFMKey& get_actual_key(const PPFMKey& key) const
    {
        auto it = find_key_in_map(def_pp_key_pool, key);
        if(it != def_pp_key_pool.end())
            return it->second;
        else
            return key;
    }

public:
    function_pool(unsigned int max_lds_bytes)
        : max_lds_bytes(max_lds_bytes)
        , def_key_pool(std::get<0>(function_pool_data::get_function_pool_data().def_keys))
        , def_pp_key_pool(std::get<1>(function_pool_data::get_function_pool_data().def_keys))
        , function_map(std::get<0>(function_pool_data::get_function_pool_data().function_maps))
        , pp_function_map(std::get<1>(function_pool_data::get_function_pool_data().function_maps))
    {
        // We would only see zero if we received a
        // default-constructed device prop struct, which means
        // someone forgot to initialize the struct somewhere.
        if(max_lds_bytes == 0)
            throw std::runtime_error("function_pool: max_lds_bytes not initialized");
    }

    function_pool(const hipDeviceProp_t& prop)
        : max_lds_bytes(prop.sharedMemPerBlock)
        , def_key_pool(std::get<0>(function_pool_data::get_function_pool_data().def_keys))
        , def_pp_key_pool(std::get<1>(function_pool_data::get_function_pool_data().def_keys))
        , function_map(std::get<0>(function_pool_data::get_function_pool_data().function_maps))
        , pp_function_map(std::get<1>(function_pool_data::get_function_pool_data().function_maps))
        , deviceProp(prop)
    {
        // We would only see zero if we received a
        // default-constructed device prop struct, which means
        // someone forgot to initialize the struct somewhere.
        if(max_lds_bytes == 0)
            throw std::runtime_error("function_pool: max_lds_bytes not initialized");
    }

    function_pool(function_pool& p) = delete;
    function_pool& operator=(const function_pool&) = delete;

    ~function_pool() = default;

    // add a new kernel in runtime
    void add_new_kernel(const FMKey& new_key)
    {
        // already has this kernel
        if(has_function(new_key))
            return;

        FMKey new_key_with_lds          = new_key;
        new_key_with_lds.lds_size_bytes = max_lds_bytes;

        function_map.emplace(new_key_with_lds, FFTKernel(new_key_with_lds.kernel_config));
    }

    bool has_function(const FMKey& key) const
    {
        auto real_key = get_actual_key(key);
        return find_key_in_map(function_map, real_key) != function_map.end();
    }

    bool has_function(const PPFMKey& key) const
    {
        auto real_key = get_actual_key(key);
        return find_key_in_map(pp_function_map, real_key) != pp_function_map.end();
    }

    size_t get_largest_pow2_length(rocfft_precision precision) const
    {
        auto supported
            = get_lengths(precision, CS_KERNEL_STOCKHAM, [](size_t len) { return IsPo2(len); });
        auto itr = std::max_element(supported.cbegin(), supported.cend());
        if(itr != supported.cend())
            return *itr;
        return 0;
    }

    // Optional filter can be specified to only get lengths where the filter returns true
    std::vector<size_t> get_lengths(rocfft_precision            precision,
                                    ComputeScheme               scheme,
                                    std::function<bool(size_t)> filter = {}) const
    {
        std::vector<size_t> lengths;
        for(auto const& kv : function_map)
        {
            if(kv.first.lds_size_bytes > max_lds_bytes)
                continue;
            if(kv.first.lengths[1] == 0 && kv.first.precision == precision
               && kv.first.scheme == scheme && kv.first.sbrcTrans == NONE)
            {
                if(!filter || filter(kv.first.lengths[0]))
                {
                    lengths.push_back(kv.first.lengths[0]);
                }
            }
        }

        return lengths;
    }

    FFTKernel get_kernel(const FMKey& key) const
    {
        auto real_key = get_actual_key(key);
        auto it       = find_key_in_map(function_map, real_key);
        if(it == function_map.end())
            throw std::out_of_range("kernel not found in map");
        return it->second;
    }

    FFTKernel get_kernel(const PPFMKey& key, ComputeScheme scheme) const
    {
        auto real_key = get_actual_key(key);
        auto it       = find_key_in_map(pp_function_map, real_key);
        if(it == pp_function_map.end())
            throw std::out_of_range("kernel not found in partial-pass map");

        auto kernel_list = it->second;

        auto scheme_0 = kernel_list[0].pp_params.scheme;
        auto scheme_1 = kernel_list[1].pp_params.scheme;

        if(scheme == scheme_0)
            return kernel_list[0];
        else if(scheme == scheme_1)
            return kernel_list[1];
        else
            throw std::out_of_range("kernel not found in partial-pass map");
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

    // Device properties that the pool was initialized with.  This can
    // be nullopt_t if the pool was only initialized with an LDS size
    // and no actual device is known.
    const std::optional<hipDeviceProp_t> deviceProp;
};

// Insert a key-kernel pair for AOT generator. This function is called in
// N (set via CMake) separate files, generated by kernel-generator.py for parallel compiling.
// That is, the default kernel-config we set in the kernel-generator.py we save a pair as
// <key-empty-config, key-actual-config> that allows us to use
// the empty-config key to get the default kernel
static void insert_default_entry(const FMKey&     def_key,
                                 const FFTKernel& kernel,
                                 FPKeyMap&        def_key_pool,
                                 FPMap&           function_map,
                                 size_t           lds_size_bytes)
{
    FMKey def_key_with_lds          = def_key;
    def_key_with_lds.lds_size_bytes = lds_size_bytes;

    // simple_key means the same thing as def_key, but we just remove kernel-config
    // so we don't need to know the exact config when we're lookin' for the default kernel
    FMKey simple_key{def_key_with_lds};
    simple_key.kernel_config = KernelConfig::EmptyConfig();

    def_key_pool.emplace(simple_key, def_key_with_lds);

    // still use the detailed key with config to maintain the function map
    function_map.emplace(def_key_with_lds, kernel);
}

static void insert_default_entry(const PPFMKey&   def_key,
                                 const FFTKernel& kernel_0,
                                 const FFTKernel& kernel_1,
                                 PPFPKeyMap&      def_key_pool,
                                 PPFPMap&         function_map,
                                 size_t           lds_size_bytes)
{
    PPFMKey def_key_with_lds        = def_key;
    def_key_with_lds.lds_size_bytes = lds_size_bytes;

    PPFMKey simple_key(def_key_with_lds);

    simple_key.kernel_config_1 = KernelConfig::EmptyConfig();
    simple_key.kernel_config_2 = KernelConfig::EmptyConfig();

    def_key_pool.emplace(simple_key, def_key_with_lds);

    std::array<FFTKernel, 2> kernels = {kernel_0, kernel_1};

    // still use the detailed key with config to maintain the function map
    function_map.emplace(def_key_with_lds, kernels);
}

#endif // FUNCTION_POOL_H
