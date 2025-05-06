/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>

#include <cmath>

#include "test_header.hpp"
#include "bitwise_repro/bwr_utils.hpp"

using ReproducibilityTestParams = ::testing::Types<
    Params<thrust::device_vector<int>, std::decay_t<decltype(thrust::hip::par_det)>>, 
    Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det)>>,
    Params<thrust::device_vector<double>, std::decay_t<decltype(thrust::hip::par_det)>>,
    Params<thrust::device_vector<int>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>, 
    Params<thrust::device_vector<float>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>,
    Params<thrust::device_vector<double>, std::decay_t<decltype(thrust::hip::par_det_nosync)>>>;

TESTS_DEFINE(ReproducibilityTests, ReproducibilityTestParams);

// Delay the operator by a semi-random amount to increase the likelyhood
// of changing the number of lookback steps between the runs.
template <typename F>
struct eepy_scan_op
{
    bool enable_sleep;
    F    scan_op;

    eepy_scan_op(bool enable_sleep, F scan_op = F())
        : enable_sleep(enable_sleep)
        , scan_op(scan_op)
    {
    }

    template <typename T, typename U>
    __device__ auto operator()(const T& a, const U& b) -> decltype(scan_op(a, b))
    {
        if(this->enable_sleep)
        {
            for(unsigned int i = 0; i < blockIdx.x * 3001 % 64; ++i)
            {
                __builtin_amdgcn_s_sleep(63);
            }
        }
        return scan_op(a, b);
    }
};

template <typename T>
void assert_reproducible(const thrust::device_vector<T>& d_a, const thrust::device_vector<T>& d_b)
{
    thrust::host_vector<T> h_a = d_a;
    thrust::host_vector<T> h_b = d_b;
    ASSERT_NO_FATAL_FAILURE(assert_bit_eq(h_a, h_b));
}

void check_bwr_match(const bwr_utils::TokenHelper& token_helper)
{
    if (inter_run_bwr::enabled && inter_run_bwr::db)
        ASSERT_TRUE(inter_run_bwr::db->match(token_helper.get_input_token(), token_helper.get_output_token()));
}

TYPED_TEST(ReproducibilityTests, Scan)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    using ScanOp = eepy_scan_op<thrust::plus<T>>;

    bwr_utils::TokenHelper token_helper;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Policy policy;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_input
                = get_random_data<T>(size, static_cast<T>(-100), static_cast<T>(100), seed);

            Vector d_input = h_input;

            Vector d_output_0(size);
            Vector d_output_1(size);

            // inclusive
            thrust::inclusive_scan(
                policy, d_input.begin(), d_input.end(), d_output_0.begin(), ScanOp(false));
            
            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_input;

                token_helper.build_input_token(
                    "thrust::inclusive_scan",
                    h_data.begin(),
                    h_data.end(),
                    {bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::inclusive_scan(
                policy, d_input.begin(), d_input.end(), d_output_1.begin(), ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;
                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);

            // exclusive
            thrust::exclusive_scan(
                policy, d_input.begin(), d_input.end(), d_output_0.begin(), T {42}, ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_input;

                token_helper.build_input_token(
                    "thrust::exclusive_scan",
                    h_data.begin(),
                    h_data.end(),
                    {bwr_utils::get_scalar_token(T {42}), bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::exclusive_scan(
                policy, d_input.begin(), d_input.end(), d_output_1.begin(), T {42}, ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);
        }
    }
}

TYPED_TEST(ReproducibilityTests, ScanByKey)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    using ScanOp = eepy_scan_op<thrust::plus<T>>;

    hipDeviceProp_t attributes;
    HIP_CHECK(hipGetDeviceProperties(&attributes, 0));

    // Disable int test case for Navi3X because of known issue
    if (std::is_same_v<T, int> && attributes.major == 11){
        GTEST_SKIP();
    }

    bwr_utils::TokenHelper token_helper;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Policy policy;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<int>      h_keys(size);
            thrust::default_random_engine rng(seed);
            const auto                    r = static_cast<size_t>(std::sqrt(size));
            for(size_t i = 0, k = 0, l = 0; i < size; i++)
            {
                if(l == 0)
                {
                    l = 1 + rng() % r;
                    ++k;
                }
                --l;
                h_keys[i] = k;
            }

            thrust::device_vector<int> d_keys = h_keys;

            thrust::host_vector<T> h_input
                = get_random_data<T>(size, static_cast<T>(-100), static_cast<T>(100), seed);

            Vector d_input = h_input;

            Vector d_output_0(size);
            Vector d_output_1(size);

            // inclusive
            thrust::inclusive_scan_by_key(policy,
                                          d_keys.begin(),
                                          d_keys.end(),
                                          d_input.begin(),
                                          d_output_0.begin(),
                                          thrust::equal_to<T> {},
                                          ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_keys = d_keys;
                thrust::host_vector<T> h_input = d_input;

                token_helper.build_input_token(
                    "thrust::inclusive_scan_by_key",
                    h_keys.begin(),
                    h_keys.end(),
                    h_input.begin(),
                    {bwr_utils::get_functor_token<T>("thrust::equal_to"),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::inclusive_scan_by_key(policy,
                                          d_keys.begin(),
                                          d_keys.end(),
                                          d_input.begin(),
                                          d_output_1.begin(),
                                          thrust::equal_to<T> {},
                                          ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);

            // exclusive
            thrust::exclusive_scan_by_key(policy,
                                          d_keys.begin(),
                                          d_keys.end(),
                                          d_input.begin(),
                                          d_output_0.begin(),
                                          T {123},
                                          ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_keys = d_keys;
                thrust::host_vector<T> h_input = d_input;

                token_helper.build_input_token(
                    "thrust::exclusive_scan_by_key",
                    h_keys.begin(),
                    h_keys.end(),
                    h_input.begin(),
                    {bwr_utils::get_scalar_token(T {123}),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::exclusive_scan_by_key(policy,
                                          d_keys.begin(),
                                          d_keys.end(),
                                          d_input.begin(),
                                          d_output_1.begin(),
                                          T {123},
                                          ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);
        }
    }
}

TYPED_TEST(ReproducibilityTests, ReduceByKey)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    using ScanOp = eepy_scan_op<thrust::plus<T>>;

    bwr_utils::TokenHelper token_helper;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Policy policy;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<int>      h_keys(size);
            thrust::default_random_engine rng(seed);
            const auto                    r = static_cast<size_t>(std::sqrt(size));
            for(size_t i = 0, k = 0, l = 0; i < size; i++)
            {
                if(l == 0)
                {
                    l = 1 + rng() % r;
                    ++k;
                }
                --l;
                h_keys[i] = k;
            }
            thrust::device_vector<int> d_keys = h_keys;

            thrust::host_vector<T> h_vals
                = get_random_data<T>(size, static_cast<T>(-100), static_cast<T>(100), seed);

            Vector d_vals = h_vals;

            Vector d_keys_output_0(size);
            Vector d_vals_output_0(size);
            Vector d_keys_output_1(size);
            Vector d_vals_output_1(size);

            thrust::reduce_by_key(policy,
                                  d_keys.begin(),
                                  d_keys.end(),
                                  d_vals.begin(),
                                  d_keys_output_0.begin(),
                                  d_vals_output_0.begin(),
                                  thrust::equal_to<T> {},
                                  ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_keys = d_keys;
                thrust::host_vector<T> h_vals = d_vals;

                token_helper.build_input_token(
                    "thrust::reduce_by_key",
                    h_keys.begin(),
                    h_keys.end(),
                    h_vals.begin(),
                    {bwr_utils::get_functor_token<T>("thrust::equal_to"),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::reduce_by_key(policy,
                                  d_keys.begin(),
                                  d_keys.end(),
                                  d_vals.begin(),
                                  d_keys_output_1.begin(),
                                  d_vals_output_1.begin(),
                                  thrust::equal_to<T> {},
                                  ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_keys_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.begin(), d_keys.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_keys_output_0, d_keys_output_1);
            assert_reproducible(d_vals_output_0, d_vals_output_1);
        }
    }
}

TYPED_TEST(ReproducibilityTests, TransformScan)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    using ScanOp = eepy_scan_op<thrust::plus<T>>;

    bwr_utils::TokenHelper token_helper;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Policy policy;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_input = get_random_data<T>(
                size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
            Vector d_input = h_input;

            Vector d_output_0(size);
            Vector d_output_1(size);

            thrust::transform_inclusive_scan(policy,
                                             d_input.begin(),
                                             d_input.end(),
                                             d_output_0.begin(),
                                             thrust::negate<T>(),
                                             ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_input;

                token_helper.build_input_token(
                    "thrust::transform_inclusive_scan",
                    h_data.begin(),
                    h_data.end(),
                    {bwr_utils::get_functor_token<T>("thrust::negate"),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::transform_inclusive_scan(policy,
                                             d_input.begin(),
                                             d_input.end(),
                                             d_output_1.begin(),
                                             thrust::negate<T>(),
                                             ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);

            thrust::transform_exclusive_scan(policy,
                                             d_input.begin(),
                                             d_input.end(),
                                             d_output_0.begin(),
                                             thrust::negate<T>(),
                                             (T)11,
                                             ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_input;

                token_helper.build_input_token(
                    "thrust::transform_exclusive_scan",
                    h_data.begin(),
                    h_data.end(),
                    {bwr_utils::get_functor_token<T>("thrust::negate"),
                     bwr_utils::get_scalar_token((T) 11),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::transform_exclusive_scan(policy,
                                             d_input.begin(),
                                             d_input.end(),
                                             d_output_1.begin(),
                                             thrust::negate<T>(),
                                             (T)11,
                                             ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);

            // in-place scans
            d_output_0 = d_input;
            d_output_1 = d_input;
            thrust::transform_inclusive_scan(policy,
                                             d_output_0.begin(),
                                             d_output_0.end(),
                                             d_output_0.begin(),
                                             thrust::negate<T>(),
                                             ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_input_token(
                    "thrust::transform_inclusive_scan",
                    h_data.begin(),
                    h_data.end(),
                    {bwr_utils::get_functor_token<T>("thrust::negate"),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::transform_inclusive_scan(policy,
                                             d_output_1.begin(),
                                             d_output_1.end(),
                                             d_output_1.begin(),
                                             thrust::negate<T>(),
                                             ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);

            d_output_0 = d_input;
            d_output_1 = d_input;
            thrust::transform_exclusive_scan(policy,
                                             d_output_0.begin(),
                                             d_output_0.end(),
                                             d_output_0.begin(),
                                             thrust::negate<T>(),
                                             (T)11,
                                             ScanOp(false));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_input_token(
                    "thrust::transform_exclusive_scan",
                    h_data.begin(),
                    h_data.end(),
                    {bwr_utils::get_functor_token<T>("thrust::negate"),
                     bwr_utils::get_scalar_token((T) 11),
                     bwr_utils::get_functor_token<T>("thrust::plus")}
                );
            }

            thrust::transform_exclusive_scan(policy,
                                             d_output_1.begin(),
                                             d_output_1.end(),
                                             d_output_1.begin(),
                                             thrust::negate<T>(),
                                             (T)11,
                                             ScanOp(true));

            if (inter_run_bwr::enabled)
            {
                thrust::host_vector<T> h_data = d_output_1;

                token_helper.build_output_token(h_data.begin(), h_data.size());
                check_bwr_match(token_helper);
            }

            assert_reproducible(d_output_0, d_output_1);
        }
    }
}
