// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tuning_kernel_tuner.h"
#include "../../shared/arithmetic.h"
#include "../../shared/environment.h"
#include "function_pool.h"
#include "logging.h"
#include "rocfft/rocfft.h"
#include "solution_map.h"
#include "tuning_helper.h"
#include "twiddles.h"

#include <algorithm>
#include <iterator>
#include <random>
#include <regex>
#include <set>

static const char* candidates_folder = "TuningCandidates";

static const std::vector<size_t> supported_factors = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17};

// TODO- support half precision
static const size_t LDS_BYTE_LIMIT    = 32 * 1024;
static const size_t BYTES_PER_FLOAT2  = sizeof(float) * 2;
static const size_t BYTES_PER_DOUBLE2 = sizeof(double) * 2;

// use_ltwd_3steps: if use_ltwd_3steps and ltwd_base < 8, then ltwd table will take some lds ,
// tpt: threads_per_transform
// wgs_bound: upper bound of workgroup_size
// return: transfroms_per_block: maximal value within LDS_LIMIT
size_t DeriveMaxTPB(size_t length,
                    bool   is_single,
                    bool   half_lds,
                    bool   use_ltwd_3steps,
                    size_t large1D,
                    size_t tpt,
                    size_t wgs_bound)
{
    size_t bytes_per_elem  = (is_single) ? BYTES_PER_FLOAT2 : BYTES_PER_DOUBLE2;
    size_t bytes_per_batch = length * bytes_per_elem;

    if(half_lds)
        bytes_per_batch /= 2;

    if(use_ltwd_3steps)
    {
        size_t ltwd_base, ltwd_steps;
        get_large_twd_base_steps(large1D, use_ltwd_3steps, ltwd_base, ltwd_steps);
        // only in this condition we put the ltwd table in lds, and the
        // #elem = (1 << base) * 3
        if(ltwd_base < 8)
            bytes_per_batch += ((1 << ltwd_base) * 3) * bytes_per_elem;
    }

    size_t tpb = LDS_BYTE_LIMIT / bytes_per_batch;
    while(tpt * tpb > wgs_bound)
        --tpb;

    // this value is not returned
    // wgs_bound = tpt * tpb;

    return tpb;
}

size_t ConservativeMaxTPB(size_t length, bool is_single)
{
    size_t bytes_per_elem  = (is_single) ? BYTES_PER_FLOAT2 : BYTES_PER_DOUBLE2;
    size_t bytes_per_batch = length * bytes_per_elem;

    // [reduce search space]:
    //  theoretically, using half_lds can make the tpb double,
    //  but from observation, in the EDGE case that tpb EXACTLY fits in LDS_BYTE_LIMIT, (occu = 2)
    //  enable half_lds can double the tpb to make lds fits in LDS_BYTE_LIMIT, (occu = 2)
    //  but it's very likely making the occu down to 1
    //  so a conservation way is still to use a non-half-lds TPB value as the upper_bound

    // So we can discard any configuration with TPB > bound, even for half_lds
    size_t conservative_max_tpb = LDS_BYTE_LIMIT / bytes_per_batch;

    if(length >= 1024)
        conservative_max_tpb += 1;

    return conservative_max_tpb;
}

// recursively find all unique factorizations of given length.  each
// factorization is a vector of ints, sorted so they're uniquified in
// a set.
std::set<std::vector<size_t>> Factorize(size_t length)
{
    std::set<std::vector<size_t>> ret;
    for(auto factor : supported_factors)
    {
        if(length % factor == 0)
        {
            size_t remain = length / factor;
            if(remain == 1)
                ret.insert({factor});
            else
            {
                // recurse into remainder
                auto remain_factorization = Factorize(remain);
                for(auto& remain_factors : remain_factorization)
                {
                    std::vector<size_t> factors{factor};
                    std::copy(
                        remain_factors.begin(), remain_factors.end(), std::back_inserter(factors));
                    std::sort(factors.begin(), factors.end());
                    ret.insert(factors);
                }
            }
        }
    }
    return ret;
}

size_t GetMaxRadicesSize(const std::set<std::vector<size_t>>& all_factors_set)
{
    size_t min_size = TWIDDLES_MAX_RADICES + 1;

    for(auto factors : all_factors_set)
    {
        min_size = std::min(factors.size(), min_size);
    }

    // don't try kernels with too many radices
    return min_size + 2;
    // return min_size + 3;
}

// recursively return power set of a range of ints
std::set<std::vector<size_t>> PowerSet(std::vector<size_t>::const_iterator begin,
                                       std::vector<size_t>::const_iterator end)
{
    std::set<std::vector<size_t>> ret;
    // either include the front element in the output, or don't
    if(std::distance(begin, end) == 1)
    {
        ret.insert({*begin});
        ret.insert({});
    }
    else
    {
        // recurse into the remainder
        auto remain = PowerSet(begin + 1, end);
        for(auto r : remain)
        {
            ret.insert(r);
            r.push_back(*begin);
            ret.insert(r);
        }
    }
    return ret;
}

std::set<size_t> SupportedThreadsPerTransform(const std::vector<size_t>& factorization)
{
    std::set<size_t> tpts;
    auto             tpt_candidates = PowerSet(factorization.begin(), factorization.end());
    for(auto tpt : tpt_candidates)
    {
        if(tpt.empty())
            continue;
        tpts.insert(product(tpt.begin(), tpt.end()));
    }
    return tpts;
}

std::vector<double>
    GetUtilizationRate(size_t length, const std::vector<size_t>& factors, size_t tpt)
{
    std::vector<double> ret; // [height_1, height_2,..., height_n, average]

    double util_rate = 0;
    for(auto width : factors)
    {
        double height = static_cast<double>(length) / width / tpt;
        ret.push_back(height);
        util_rate += height;
    }
    util_rate /= factors.size();
    ret.push_back(util_rate);

    return ret;
}

void PrintRejectionMsg(const std::string& msg, bool print)
{
    if(LOG_TUNING_ENABLED() && print)
        (*LogSingleton::GetInstance().GetTuningOS()) << msg;
}

std::string FactorsToString(const std::vector<size_t>& factors)
{
    // factors as string
    std::string factors_str = "[";
    std::string COMMA       = "";
    for(auto factor : factors)
    {
        factors_str += COMMA + std::to_string(factor);
        COMMA = ", ";
    }
    factors_str += "]";

    return factors_str;
}

void StringToFactors(const std::string& factors_str, std::vector<size_t>& factors)
{
    static const std::regex vector_delims{"[^,\\[\\]\\s]+", std::regex_constants::optimize};

    for(std::sregex_token_iterator tok{factors_str.begin(), factors_str.end(), vector_delims, 0};
        tok != std::sregex_token_iterator();
        ++tok)
    {
        factors.push_back(std::stoi(tok->str()));
    }
}

// [reduce search space]
// using permutation or shifting...
std::set<std::vector<size_t>>
    GetAllFactorizationsForPhase1(std::set<std::vector<size_t>>& target_factors)
{
    std::set<std::vector<size_t>> ret;
    for(auto good_factors : target_factors)
    {
        // std::cout << "for um-permuted good_factor: " << FactorsToString(good_factors) << std::endl;

        // try to peek the permutation results, if the num-of-permu is too large
        // we might want to cut down the candidates.
        std::vector<std::vector<size_t>> permutations;

        std::vector<size_t> permuting_factors = good_factors;
        while(std::next_permutation(permuting_factors.begin(), permuting_factors.end()))
        {
            permutations.push_back(permuting_factors);
            // std::cout << "get permutation: " << FactorsToString(permuting_factors) << std::endl;
        }
        // std::cout << "generated " << permutations.size() << " permutaions\n" << std::endl;

        if(permutations.size() > 6)
        {
            // std::cout << "too many permutaions: " << permutations.size() << ", try use shifting. "
            //           << std::endl;
            permutations.clear();

            size_t              factor_len = good_factors.size();
            std::vector<size_t> reversed   = good_factors; // [a,b,c,...]
            std::reverse(reversed.begin(), reversed.end()); // [...,c,b,a]

            good_factors.insert(good_factors.end(),
                                good_factors.begin(),
                                good_factors.end()); // [a,b,c,....] -> [a,b,c,..,a,b,c...]
            reversed.insert(reversed.end(),
                            reversed.begin(),
                            reversed.end()); // [...,c,b,a] -> [...,c,b,a,..,c,b,a]
            for(size_t i = 0; i < factor_len; ++i)
            {
                std::vector<size_t> shifted_ori(good_factors.begin() + i,
                                                good_factors.begin() + i + factor_len);
                std::vector<size_t> shifted_rev(reversed.begin() + i,
                                                reversed.begin() + i + factor_len);
                permutations.push_back(shifted_ori);
                permutations.push_back(shifted_rev);
                // std::cout << "get shifted: " << FactorsToString(shifted_ori) << std::endl;
                // std::cout << "get shifted: " << FactorsToString(shifted_rev) << std::endl;
            }
            // std::cout << "generated " << permutations.size() << " shifting\n" << std::endl;
        }

        std::copy(permutations.begin(), permutations.end(), std::inserter(ret, ret.end()));
    }

    // std::cout << "all factors in phase 1:" << std::endl;
    // for(auto it : ret)
    //     std::cout << FactorsToString(it) << std::endl;

    return ret;
}

std::set<KernelConfig> Supported2DKernelConfigs(size_t len0, size_t len1, size_t node_id)
{
    std::set<KernelConfig> configs;

    // we reserve some configurations with wgs greater than MAX_WGS
    // in case the MAX_WGS yield nothing.
    // the vector stores configurations with (wgs < 512), (wgs < 768), (wgs < 1024)
    std::vector<std::set<KernelConfig>> configs_larger_wgs(3);

    auto   factorizations_0   = Factorize(len0);
    auto   factorizations_1   = Factorize(len1);
    size_t max_radices_size_0 = GetMaxRadicesSize(factorizations_0);
    size_t max_radices_size_1 = GetMaxRadicesSize(factorizations_1);

    bool        print_reject = !rocfft_getenv("PRINT_REJECT_REASON").empty();
    std::string max_wgs_str  = rocfft_getenv("MAX_WGS");
    size_t      max_wgs      = max_wgs_str.empty() ? 768 : std::atoi(max_wgs_str.c_str());

    // if PHASE-0 : Run without permutating the factors
    // if PHASE-1 : Permute a few good factorizations passed from PHASE-0
    bool  is_phase1      = TuningBenchmarker::GetSingleton().GetPacket()->tuning_phase == 1;
    auto& target_factors = TuningBenchmarker::GetSingleton().GetPacket()->target_factors;
    std::set<std::vector<size_t>> umpermuted_target_factors_d0, umpermuted_target_factors_d1;
    if(is_phase1)
    {
        assert(!target_factors.empty());

        // convert the "factors_strings" set to "target_factors" set
        // and then separate the factors_2kernels to two factors (one for each dim)
        for(const auto& factors_str : target_factors[node_id])
        {
            std::vector<size_t> factors_2kernels, factors_d0, factors_d1;
            StringToFactors(factors_str, factors_2kernels);

            // decompose the factors to 2 radices set
            int    count               = 0;
            size_t cummulative_product = 1;
            while(cummulative_product != len0)
                cummulative_product *= factors_2kernels[count++];

            factors_d0.insert(
                factors_d0.cbegin(), factors_2kernels.cbegin(), factors_2kernels.cbegin() + count);
            factors_d1.insert(
                factors_d1.cbegin(), factors_2kernels.cbegin() + count, factors_2kernels.cend());
            // ---

            umpermuted_target_factors_d0.insert(factors_d0);
            umpermuted_target_factors_d1.insert(factors_d1);
        }

        // This includes only the permuted factors
        factorizations_0 = GetAllFactorizationsForPhase1(umpermuted_target_factors_d0);
        factorizations_1 = GetAllFactorizationsForPhase1(umpermuted_target_factors_d1);

        // we need to do this for 2D-kernel, that is, also include the factors we've already seen in phase 0
        // even if the factors (of dim1 or dim2) are seen before, but the combination might be new.
        factorizations_0.insert(umpermuted_target_factors_d0.begin(),
                                umpermuted_target_factors_d0.end());
        factorizations_1.insert(umpermuted_target_factors_d1.begin(),
                                umpermuted_target_factors_d1.end());
    }

    for(auto factors_0 : factorizations_0)
    {
        if(factors_0.size() > max_radices_size_0)
        {
            PrintRejectionMsg("reject: using too many radices in factors_0\n", print_reject);
            continue;
        }

        // for each factors, we try uwide / wide mode only.
        size_t tpt0_uwide = len0 / (*std::min_element(factors_0.begin(), factors_0.end()));
        size_t tpt0_wide  = len0 / (*std::max_element(factors_0.begin(), factors_0.end()));

        for(auto factors_1 : factorizations_1)
        {
            if(factors_1.size() > max_radices_size_1)
            {
                PrintRejectionMsg("reject: using too many radices in factors_1\n", print_reject);
                continue;
            }

            if(is_phase1)
            {
                // in phase-1, we can skip the factorizations we've already tuned in phase-0
                // check this by testing if both factors are umpermuted
                if(umpermuted_target_factors_d0.count(factors_0) != 0
                   && umpermuted_target_factors_d1.count(factors_1) != 0)
                    continue;
            }

            // for each factors, we try uwide / wide mode only.
            size_t tpt1_uwide = len1 / (*std::min_element(factors_1.begin(), factors_1.end()));
            size_t tpt1_wide  = len1 / (*std::max_element(factors_1.begin(), factors_1.end()));

            for(size_t tpt0 : {tpt0_uwide, tpt0_wide})
            {
                for(size_t tpt1 : {tpt1_uwide, tpt1_wide})
                {
                    // wgs_0 = tpt0 * len1; wgs_1 = tpt1 * len0
                    size_t wgs = max(tpt0 * len1, tpt1 * len0);

                    KernelConfig config;
                    config.direct_to_from_reg    = false;
                    config.intrinsic_buffer_inst = false;
                    config.half_lds              = false;
                    config.threads_per_transform = {(int)tpt0, (int)tpt1};
                    config.transforms_per_block  = 1;
                    config.workgroup_size        = wgs;
                    config.use_3steps_large_twd  = false;
                    config.factors               = factors_0;
                    // combined two factors vectors as one
                    std::copy(
                        factors_1.begin(), factors_1.end(), std::back_inserter(config.factors));

                    if(wgs > max_wgs)
                    {
                        PrintRejectionMsg("reject: wgs(" + std::to_string(wgs) + ") > MAX_WGS("
                                              + std::to_string(max_wgs) + ")\n",
                                          print_reject);

                        // save in preparation pool
                        if(wgs <= 512)
                            configs_larger_wgs[0].insert(config);
                        else if(wgs <= 768)
                            configs_larger_wgs[1].insert(config);
                        else if(wgs <= 1024)
                            configs_larger_wgs[2].insert(config);

                        continue;
                    }

                    configs.insert(config);
                }
            }
        }
    }

    // if no valid configs, it may caused by MAX_WGS,
    // so we check if we can draw some from preparation pool
    if(configs.empty())
    {
        for(auto& set : configs_larger_wgs)
        {
            if(!set.empty())
            {
                if(LOG_TUNING_ENABLED())
                    (*LogSingleton::GetInstance().GetTuningOS())
                        << "no valid candidate meets MAX_WGS, extend the limitation." << std::endl;
                std::copy(set.begin(), set.end(), std::inserter(configs, configs.end()));
                break;
            }
        }
    }

    return configs;
}

std::set<KernelConfig> SupportedKernelConfigs(size_t length,
                                              size_t node_id,
                                              bool   is_single,
                                              bool   is_sbcc,
                                              bool   is_sbrc,
                                              bool   is_sbcr,
                                              size_t large1D)
{
    std::set<KernelConfig> configs;

    auto   factorizations   = Factorize(length);
    size_t max_radices_size = GetMaxRadicesSize(factorizations);
    bool   has_ltwd_mul     = is_sbcc && (large1D > 0);
    // so far our kernel-gen implements intrinsic mode only on these two type
    bool   can_do_intrinsic = is_sbcc || is_sbcr;
    size_t conservative_tpb = ConservativeMaxTPB(length, is_single);

    // [reduce search space]:
    // we will remove those configs(A) with tpt = length,
    // and remove other configs having the same tpb as configs(A)
    std::set<size_t> tpbs_to_remove;
    std::set<size_t> tpts_with_bad_util_rate;
    std::set<size_t> all_tpts;

    bool        print_reject = !rocfft_getenv("PRINT_REJECT_REASON").empty();
    std::string min_wgs_str  = rocfft_getenv("MIN_WGS");
    std::string max_wgs_str  = rocfft_getenv("MAX_WGS");
    size_t      min_wgs      = min_wgs_str.empty() ? 64 : std::atoi(min_wgs_str.c_str());
    size_t      max_wgs      = max_wgs_str.empty() ? 512 : std::atoi(max_wgs_str.c_str());

    // if min_wgs is greater than length, then we lower it.
    min_wgs = (length < min_wgs) ? length : min_wgs;
    min_wgs = (min_wgs % 64 == 0) ? min_wgs : std::max((size_t)0, min_wgs - (min_wgs % 64));
    max_wgs = (max_wgs % 64 == 0) ? max_wgs : max_wgs - (max_wgs % 64);

    auto& target_factors_strs = TuningBenchmarker::GetSingleton().GetPacket()->target_factors;
    bool  is_phase0           = (target_factors_strs.empty());
    bool  no_permutation      = is_phase0;

    // avoid 336 from expanding to 5 factors
    if(length == 336)
        --max_radices_size;

    if(is_phase0 == false)
    {
        // manually permute in phase-1
        no_permutation = true;

        // convert the "factors_strings" set to "target_factors" set
        std::set<std::vector<size_t>> target_factors;
        for(const auto& factors_str : target_factors_strs[node_id])
        {
            std::vector<size_t> factors;
            StringToFactors(factors_str, factors);
            target_factors.insert(factors);
        }

        factorizations = GetAllFactorizationsForPhase1(target_factors);
    }

    for(auto factorization : factorizations)
    {
        // [reduce search space]:
        // don't try kernels with too many radices
        if(is_phase0 && (factorization.size() > max_radices_size))
        {
            PrintRejectionMsg("reject: using too many radices in factors\n", print_reject);
            continue;
        }

        auto tpts = SupportedThreadsPerTransform(factorization);

        // [reduce search space]
        // by utilization rate
        for(auto tpt = tpts.begin(), last = tpts.end(); tpt != last;)
        {
            std::vector<double> util_rates = GetUtilizationRate(length, factorization, *tpt);

            auto max_rates = std::max_element(util_rates.begin(), util_rates.end());
            auto avg_rate  = util_rates.back();
            // if average rate < 1.0 or > 8.0 , or any of heights > 8.0, then it's bad
            if(avg_rate < 1.0 || *max_rates > 8.0)
                tpts_with_bad_util_rate.insert(*tpt);
            ++tpt;
        }

        // go through all permutations of the factors
        do
        {
            for(size_t wgs = min_wgs; wgs <= max_wgs; wgs += 64)
            {
                for(const auto tpt : tpts)
                {
                    if(tpt < wgs)
                    {
                        for(bool half_lds : {true, false})
                        {
                            if(half_lds && (is_sbcr || is_sbrc))
                            {
                                PrintRejectionMsg("reject: only sbrr and sbcc support half-lds\n",
                                                  print_reject);
                                continue;
                            }

                            for(bool use_ltwd_3steps : {true, false})
                            {
                                // skip ltwd_3steps if not needed
                                if(!has_ltwd_mul && use_ltwd_3steps)
                                    continue;

                                // Get the actual value of wgs and trans_per_block
                                size_t max_tpb = DeriveMaxTPB(length,
                                                              is_single,
                                                              half_lds,
                                                              use_ltwd_3steps,
                                                              large1D,
                                                              tpt,
                                                              wgs);

                                // this tpt and tpb will be reject
                                if(tpt == length)
                                    tpbs_to_remove.insert(max_tpb);

                                // [reduce search space]
                                size_t num_tpb_try = (tpt * max_tpb == wgs) ? 1 : 2;
                                for(size_t t = 0; t < num_tpb_try; ++t)
                                {
                                    size_t tpb       = max_tpb + t;
                                    size_t final_wgs = tpt * tpb;
                                    if(final_wgs > max_wgs)
                                        continue;
                                    if(final_wgs <= (wgs - 64))
                                        continue;

                                    // [reduce search space]
                                    if(tpb > conservative_tpb)
                                    {
                                        PrintRejectionMsg("reject: tpb > conservation max tpb\n",
                                                          print_reject);
                                        continue;
                                    }

                                    // [reduce search space]:
                                    // prune some bad configurations
                                    if(length >= 64 && final_wgs < 64)
                                    {
                                        PrintRejectionMsg("reject: (length >= 64) and (wgs < 64)\n",
                                                          print_reject);
                                        continue;
                                    }
                                    if(IsPo2(length) && (length % final_wgs != 0))
                                    {
                                        PrintRejectionMsg("reject: require wgs divisable to length "
                                                          "for Pow2 length\n",
                                                          print_reject);
                                        continue;
                                    }

                                    // [reduce search space]
                                    // from current benchmark result, dir-reg mode always ranks high
                                    for(bool direct_to_from_reg : {true /*, false*/})
                                    {
                                        // half lds currently requires direct to/from reg
                                        if(half_lds && !direct_to_from_reg)
                                        {
                                            PrintRejectionMsg(
                                                "reject: half_lds requires direct to/from reg\n",
                                                print_reject);
                                            continue;
                                        }

                                        for(bool intrinsic : {true, false})
                                        {
                                            // intrinsic currently requires direct to/from reg
                                            if(intrinsic && !direct_to_from_reg)
                                            {
                                                PrintRejectionMsg("reject: intrinsic requires "
                                                                  "direct to/from reg\n",
                                                                  print_reject);
                                                continue;
                                            }

                                            // intrinsic currently is supported on sbcc/sbcr
                                            if(intrinsic && !can_do_intrinsic)
                                            {
                                                PrintRejectionMsg("reject: intrinsic mode is "
                                                                  "supported only on sbcc/sbcr\n",
                                                                  print_reject);
                                                continue;
                                            }

                                            all_tpts.insert(tpt);

                                            KernelConfig config;
                                            config.direct_to_from_reg    = direct_to_from_reg;
                                            config.intrinsic_buffer_inst = intrinsic;
                                            config.half_lds              = half_lds;
                                            config.threads_per_transform = {(int)tpt, 0};
                                            config.transforms_per_block  = tpb;
                                            config.workgroup_size        = final_wgs;
                                            config.use_3steps_large_twd  = use_ltwd_3steps;
                                            config.factors               = factorization;

                                            configs.insert(config);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } while(no_permutation == false
                && std::next_permutation(factorization.begin(), factorization.end()));
    }

    // [reduce search space]
    // if we have other options than tpt == length,
    // then we can remove all configs with tpt==length
    if(all_tpts.size() >= 2 && tpbs_to_remove.size() > 0)
    {
        for(auto config = configs.begin(), last = configs.end(); config != last;)
        {
            // remove bad tpt
            if(config->threads_per_transform[0] == (int)length)
            {
                PrintRejectionMsg("reject: tpt == length\n" + config->Print() + "\n\n",
                                  print_reject);
                config = configs.erase(config);
            }
            // remove bad tpbs
            else if(tpbs_to_remove.count((size_t)config->transforms_per_block) > 0)
            {
                PrintRejectionMsg("reject: tpb is considered bad\n" + config->Print() + "\n\n",
                                  print_reject);
                config = configs.erase(config);
            }
            else
                ++config;
        }
        all_tpts.erase(length);
    }

    // [reduce search space]
    // remove bad tpts that have bad util_rate
    // (we compare the size to make sure we won't have nothing after doing this)
    if(all_tpts.size() > tpts_with_bad_util_rate.size())
    {
        for(auto config = configs.begin(), last = configs.end(); config != last;)
        {
            // remove bad tpt
            size_t tpt = config->threads_per_transform[0];
            if(tpts_with_bad_util_rate.count(tpt))
            {
                PrintRejectionMsg("reject: removing tpt " + std::to_string(tpt)
                                      + " due to bad util_avg_rate\n" + config->Print() + "\n\n",
                                  print_reject);
                config = configs.erase(config);
                all_tpts.erase(tpt);
            }
            else
                ++config;
        }
    }

    // [reduce search space]
    // we can remove the largest 33% tpt, since they are always in low perf.
    if(all_tpts.size() > 0 && is_phase0)
    {
        size_t num_tpts_to_remove = (all_tpts.size() - 1) / 2;
        if(num_tpts_to_remove > 0)
        {
            std::set<size_t>    tpts_to_remove;
            std::vector<size_t> tpts_vec(all_tpts.begin(), all_tpts.end());
            std::sort(tpts_vec.begin(), tpts_vec.end());
            // the largest #-num_tpts_to_remove tpts will be removed
            for(size_t i = 0; i < num_tpts_to_remove; ++i)
            {
                tpts_to_remove.insert(tpts_vec.back());
                tpts_vec.pop_back();
            }

            for(auto config = configs.begin(), last = configs.end(); config != last;)
            {
                // remove bad tpt
                if(tpts_to_remove.count((size_t)(config->threads_per_transform[0])) > 0)
                {
                    PrintRejectionMsg("reject: tpt is considered bad\n" + config->Print() + "\n\n",
                                      print_reject);
                    config = configs.erase(config);
                }
                else
                    ++config;
            }
        }
    }

    return configs;
}

void EnumerateKernelConfigs(const ExecPlan& execPlan)
{
    auto        tuningPacket = TuningBenchmarker::GetSingleton().GetPacket();
    std::string archName     = get_arch_name(execPlan.deviceProp);

    tuningPacket->tuning_arch_name = archName;
    tuningPacket->numCUs           = execPlan.deviceProp.multiProcessorCount;
    tuningPacket->total_nodes      = execPlan.execSeq.size();
    tuningPacket->total_candidates.resize(tuningPacket->total_nodes);
    tuningPacket->tuning_kernel_tokens.resize(tuningPacket->total_nodes);
    tuningPacket->is_builtin_kernel.resize(tuningPacket->total_nodes);

    // get kernel_config permutation for each node
    std::string kernel_token;
    for(size_t node_id = 0; node_id < execPlan.execSeq.size(); node_id++)
    {
        auto& curNode = execPlan.execSeq[node_id];

        size_t bench_ssn = 0;
        bool   check_dup = false;
        bool   is_single = execPlan.rootPlan->precision == rocfft_precision_single;
        size_t len       = curNode->length[0];
        bool   is_2D     = curNode->scheme == CS_KERNEL_2D_SINGLE;
        bool   is_sbcc   = curNode->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC;
        bool   is_sbrc   = curNode->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC;
        bool   is_sbcr   = curNode->scheme == CS_KERNEL_STOCKHAM_BLOCK_CR;
        size_t large1D   = curNode->large1D;
        auto   base_key  = curNode->GetKernelKey();

        // if this kernel is an internal built-in one, we are not tunining it yet, (transpose..etc)
        // but we will plan to tune it in the future.
        if(base_key == FMKey::EmptyFMKey())
        {
            check_dup = true;
            ProblemKey built_in_kernel_key(archName, solution_map::KERNEL_TOKEN_BUILTIN_KERNEL);
            TuningBenchmarker::GetSingleton().GetBindingSolutionMap()->add_solution(
                built_in_kernel_key, FMKey::EmptyFMKey(), check_dup);

            tuningPacket->tuning_kernel_tokens[node_id] = solution_map::KERNEL_TOKEN_BUILTIN_KERNEL;
            tuningPacket->is_builtin_kernel[node_id]    = true;
            tuningPacket->total_candidates[node_id]     = bench_ssn;
            continue;
        }

        // In init stage, save the default kernel token (without extra info)
        GetKernelToken(base_key, kernel_token);
        tuningPacket->tuning_kernel_tokens[node_id] = kernel_token;
        tuningPacket->is_builtin_kernel[node_id]    = false;

        // modify the token: append extra candidate info
        kernel_token += "_leafnode_" + std::to_string(node_id);
        kernel_token += "_phase_" + std::to_string(tuningPacket->tuning_phase);
        ProblemKey probKey_kernel(archName, kernel_token);

        // enumerate !
        auto kernel_configs = (is_2D)
                                  ? Supported2DKernelConfigs(len, curNode->length[1], node_id)
                                  : SupportedKernelConfigs(
                                      len, node_id, is_single, is_sbcc, is_sbrc, is_sbcr, large1D);
        for(KernelConfig config : kernel_configs)
        {
            // We can set the ebType and direction here. But we still don't know static_dim, aryType,
            // placement until buffer-assignment and collapse-dim. We'll get them later. (PowX.cpp)
            config.ebType    = execPlan.execSeq[node_id]->ebtype;
            config.direction = execPlan.execSeq[node_id]->direction;

            // NB:
            //  add_solution will append a solution to the solution-vec under the probKey
            FMKey alt_key = get_alternative_FMKey(base_key, config);

            // NB:
            //     A very important part for SBRCTranspose type !
            //     the SBRC-Trans-Type in the BaseKey of the default kernel is not always right
            //     for all the configurations. Since TPB is changed, so we should also update
            //     the SBRC-Trans-Type according to config and node dimenstion.
            alt_key.sbrcTrans
                = execPlan.execSeq[node_id]->sbrc_transpose_type(config.transforms_per_block);

            // when init tuning: output bunches of candidate kernels,
            // actually we need to check if there is an duplcation, but in this case it'll never happen
            // and the return option-id is irrelavent here. so check_dup=false, primary=true
            TuningBenchmarker::GetSingleton().GetBindingSolutionMap()->add_solution(
                probKey_kernel, alt_key, check_dup);
            bench_ssn++;
        }
        tuningPacket->total_candidates[node_id] = bench_ssn;
    }

    if(tuningPacket->dump_candidates)
    {
        std::string filename         = tuningPacket->tuning_problem_name + "_tuning.dat";
        std::string workspace_folder = "";
        workspace_folder             = rocfft_getenv("TUNING_WORKSPACE");

        fs::path dump_path(workspace_folder.c_str());
        dump_path /= candidates_folder;
        dump_path /= filename.c_str();

        // dump the solution candidates to file (from primary_map), sort=false
        TuningBenchmarker::GetSingleton().GetBindingSolutionMap()->write_solution_map_data(
            dump_path, false);
    }
}
