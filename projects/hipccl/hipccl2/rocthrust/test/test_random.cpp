/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/random/detail/normal_distribution_base.h>

#include <cmath>
#include <limits>
#include <sstream>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template <typename Engine>
struct ValidateEngine
{
    __host__ __device__ ValidateEngine(const typename Engine::result_type value_10000)
        : m_value_10000(value_10000)
    {
    }

    __host__ __device__ bool operator()(void) const
    {
        Engine e;
        e.discard(9999);

        // get the 10Kth result
        return e() == m_value_10000;
    }

    const typename Engine::result_type m_value_10000;
}; // end ValidateEngine

template <typename Engine, bool trivial_min = (Engine::min == 0)>
struct ValidateEngineMin
{
    __host__ __device__ bool operator()(void) const
    {
        Engine e;

        bool result = true;

        for(int i = 0; i < 10000; ++i)
        {
            result &= (e() >= Engine::min);
        }

        return result;
    }
}; // end ValidateEngineMin

template <typename Engine>
struct ValidateEngineMin<Engine, true>
{
    __host__ __device__ bool operator()(void) const
    {
        return true;
    }
};

template <typename Engine>
struct ValidateEngineMax
{
    __host__ __device__ bool operator()(void) const
    {
        Engine e;

        bool result = true;

        for(int i = 0; i < 10000; ++i)
        {
            result &= (e() <= Engine::max);
        }

        return result;
    }
}; // end ValidateEngineMax

template <typename Engine>
struct ValidateEngineEqual
{
    __host__ __device__ bool operator()(void) const
    {
        bool result = true;

        // test from default constructor
        Engine e0, e1;
        result &= (e0 == e1);

        // advance engines
        e0.discard(10000);
        e1.discard(10000);
        result &= (e0 == e1);

        // test from identical seeds
        Engine e2(13), e3(13);
        result &= (e2 == e3);

        // test different seeds aren't equal
        Engine e4(7), e5(13);
        result &= !(e4 == e5);

        // test reseeding engine to the same seed causes equality
        e4.seed(13);
        result &= (e4 == e5);

        return result;
    }
};

template <typename Engine>
struct ValidateEngineUnequal
{
    __host__ __device__ bool operator()(void) const
    {
        bool result = true;

        // test from default constructor
        Engine e0, e1;
        result &= !(e0 != e1);

        // advance engines
        e0.discard(1000);
        e1.discard(1000);
        result &= !(e0 != e1);

        // test from identical seeds
        Engine e2(13), e3(13);
        result &= !(e2 != e3);

        // test different seeds aren't equal
        Engine e4(7), e5(13);
        result &= (e4 != e5);

        // test reseeding engine to the same seed causes equality
        e4.seed(13);
        result &= !(e4 != e5);

        // test different discards causes inequality
        Engine e6(13), e7(13);
        e6.discard(500);
        e7.discard(1000);
        result &= (e6 != e7);

        return result;
    }
};

template <typename Distribution, typename Engine>
struct ValidateDistributionMin
{
    typedef Engine random_engine;

    __host__ __device__ ValidateDistributionMin(const Distribution& dd)
        : d(dd)
    {
    }

    __host__ __device__ bool operator()(void)
    {
        Engine e;

        bool result = true;

        for(int i = 0; i < 10000; ++i)
        {
            result &= (d(e) >= d.min());
        }

        return result;
    }

    Distribution d;
};

template <typename Distribution, typename Engine>
struct ValidateDistributionMax
{
    typedef Engine random_engine;

    __host__ __device__ ValidateDistributionMax(const Distribution& dd)
        : d(dd)
    {
    }

    __host__ __device__ bool operator()(void)
    {
        Engine e;

        bool result = true;

        for(int i = 0; i < 10000; ++i)
        {
            result &= (d(e) <= d.max());
        }

        return result;
    }

    Distribution d;
};

template <typename Distribution>
struct ValidateDistributionEqual
{
    __host__ __device__ bool operator()(void) const
    {
        return d0 == d1;
    }

    Distribution d0, d1;
};

template <typename Distribution>
struct ValidateDistributionUnqual
{
    __host__ __device__ bool operator()(void) const
    {
        return d0 != d1;
    }

    Distribution d0, d1;
};

TEST(RandomTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

template <typename Engine, thrust::detail::uint64_t value_10000>
void TestEngineValidation(void)
{
    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), ValidateEngine<Engine>(value_10000));

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), ValidateEngine<Engine>(value_10000));

    ASSERT_EQ(true, d[0]);
}

template <typename Engine>
void TestEngineMax(void)
{
    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), ValidateEngineMax<Engine>());

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), ValidateEngineMax<Engine>());

    ASSERT_EQ(true, d[0]);
}

template <typename Engine>
void TestEngineMin(void)
{
    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), ValidateEngineMin<Engine>());

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), ValidateEngineMin<Engine>());

    ASSERT_EQ(true, d[0]);
}

template <typename Engine>
void TestEngineSaveRestore(void)
{
    // create a default engine
    Engine e0;

    // run it for a while
    e0.discard(10000);

    // save it
    std::stringstream ss;
    ss << e0;

    // run it a while longer
    e0.discard(10000);

    // restore old state
    Engine e1;
    ss >> e1;

    // run e1 a while longer
    e1.discard(10000);

    // both should return the same result

    ASSERT_EQ(e0(), e1());
}

template <typename Engine>
void TestEngineEqual(void)
{
    ValidateEngineEqual<Engine> f;

    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), f);

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), f);

    ASSERT_EQ(true, d[0]);
}

template <typename Engine>
void TestEngineUnequal(void)
{
    ValidateEngineUnequal<Engine> f;

    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), f);

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), f);

    ASSERT_EQ(true, d[0]);
}

TEST(RandomTests, TestRanlux24BaseValidation)
{
    typedef thrust::random::ranlux24_base Engine;

    TestEngineValidation<Engine, 7937952u>();
}

TEST(RandomTests, TestRanlux24BaseMin)
{
    typedef thrust::random::ranlux24_base Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestRanlux24BaseMax)
{
    typedef thrust::random::ranlux24_base Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestRanlux24BaseSaveRestore)
{
    typedef thrust::random::ranlux24_base Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestRanlux24BaseEqual)
{
    typedef thrust::random::ranlux24_base Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestRanlux24BaseUnequal)
{
    typedef thrust::random::ranlux24_base Engine;

    TestEngineUnequal<Engine>();
}

TEST(RandomTests, TestRanlux48BaseValidation)
{
    typedef thrust::random::ranlux48_base Engine;

    TestEngineValidation<Engine, 192113843633948ull>();
}

TEST(RandomTests, TestRanlux48BaseMin)
{
    typedef thrust::random::ranlux48_base Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestRanlux48BaseMax)
{
    typedef thrust::random::ranlux48_base Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestRanlux48BaseSaveRestore)
{
    typedef thrust::random::ranlux48_base Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestRanlux48BaseEqual)
{
    typedef thrust::random::ranlux48_base Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestRanlux48BaseUnequal)
{
    typedef thrust::random::ranlux48_base Engine;

    TestEngineUnequal<Engine>();
}

TEST(RandomTests, TestMinstdRandValidation)
{
    typedef thrust::random::minstd_rand Engine;

    TestEngineValidation<Engine, 399268537u>();
}

TEST(RandomTests, TestMinstdRandMin)
{
    typedef thrust::random::minstd_rand Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestMinstdRandMax)
{
    typedef thrust::random::minstd_rand Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestMinstdRandSaveRestore)
{
    typedef thrust::random::minstd_rand Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestMinstdRandEqual)
{
    typedef thrust::random::minstd_rand Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestMinstdRandUnequal)
{
    typedef thrust::random::minstd_rand Engine;

    TestEngineUnequal<Engine>();
}

TEST(RandomTests, TestMinstdRand0Validation)
{
    typedef thrust::random::minstd_rand0 Engine;

    TestEngineValidation<Engine, 1043618065u>();
}

TEST(RandomTests, TestMinstdRand0Min)
{
    typedef thrust::random::minstd_rand0 Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestMinstdRand0Max)
{
    typedef thrust::random::minstd_rand0 Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestMinstdRand0SaveRestore)
{
    typedef thrust::random::minstd_rand0 Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestMinstdRand0Equal)
{
    typedef thrust::random::minstd_rand0 Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestMinstdRand0Unequal)
{
    typedef thrust::random::minstd_rand0 Engine;

    TestEngineUnequal<Engine>();
}

TEST(RandomTests, TestTaus88Validation)
{
    typedef thrust::random::taus88 Engine;

    TestEngineValidation<Engine, 3535848941ull>();
}

TEST(RandomTests, TestTaus88Min)
{
    typedef thrust::random::taus88 Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestTaus88Max)
{
    typedef thrust::random::taus88 Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestTaus88SaveRestore)
{
    typedef thrust::random::taus88 Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestTaus88Equal)
{
    typedef thrust::random::taus88 Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestTaus88Unequal)
{
    typedef thrust::random::taus88 Engine;

    TestEngineUnequal<Engine>();
}

TEST(RandomTests, TestRanlux24Validation)
{
    typedef thrust::random::ranlux24 Engine;

    TestEngineValidation<Engine, 9901578>();
}

TEST(RandomTests, TestRanlux24Min)
{
    typedef thrust::random::ranlux24 Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestRanlux24Max)
{
    typedef thrust::random::ranlux24 Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestRanlux24SaveRestore)
{
    typedef thrust::random::ranlux24 Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestRanlux24Equal)
{
    typedef thrust::random::ranlux24 Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestRanlux24Unequal)
{
    typedef thrust::random::ranlux24 Engine;

    TestEngineUnequal<Engine>();
}

TEST(RandomTests, TestRanlux48Validation)
{
    typedef thrust::random::ranlux48 Engine;

    TestEngineValidation<Engine, 88229545517833ull>();
}

TEST(RandomTests, TestRanlux48Min)
{
    typedef thrust::random::ranlux48 Engine;

    TestEngineMin<Engine>();
}

TEST(RandomTests, TestRanlux48Max)
{
    typedef thrust::random::ranlux48 Engine;

    TestEngineMax<Engine>();
}

TEST(RandomTests, TestRanlux48SaveRestore)
{
    typedef thrust::random::ranlux48 Engine;

    TestEngineSaveRestore<Engine>();
}

TEST(RandomTests, TestRanlux48Equal)
{
    typedef thrust::random::ranlux48 Engine;

    TestEngineEqual<Engine>();
}

TEST(RandomTests, TestRanlux48Unequal)
{
    typedef thrust::random::ranlux48 Engine;

    TestEngineUnequal<Engine>();
}

template <typename Distribution, typename Validator>
void ValidateDistributionCharacteristic(void)
{
    typedef typename Validator::random_engine Engine;

    // test default-constructed Distribution

    // test host
    thrust::host_vector<bool> h(1);
    thrust::generate(h.begin(), h.end(), Validator(Distribution()));

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::device_vector<bool> d(1);
    thrust::generate(d.begin(), d.end(), Validator(Distribution()));

    ASSERT_EQ(true, d[0]);

    // test distribution & engine with comparable ranges
    // only do this if they have the same result_type
    if(thrust::detail::is_same<typename Distribution::result_type,
                               typename Engine::result_type>::value)
    {
        // test Distribution with same range as engine

        // test host
        thrust::generate(h.begin(), h.end(), Validator(Distribution(Engine::min, Engine::max)));

        ASSERT_EQ(true, h[0]);

        // test device
        thrust::generate(d.begin(), d.end(), Validator(Distribution(Engine::min, Engine::max)));

        ASSERT_EQ(true, d[0]);

        // test Distribution with smaller range than engine

        // test host
        typename Distribution::result_type engine_range = Engine::max - Engine::min;
        thrust::generate(
            h.begin(), h.end(), Validator(Distribution(engine_range / 3, (2 * engine_range) / 3)));

        ASSERT_EQ(true, h[0]);

        // test device
        thrust::generate(
            d.begin(), d.end(), Validator(Distribution(engine_range / 3, (2 * engine_range) / 3)));

        ASSERT_EQ(true, d[0]);
    }

    // test Distribution with a very small range

    // test host
    thrust::generate(h.begin(), h.end(), Validator(Distribution(1, 6)));

    ASSERT_EQ(true, h[0]);

    // test device
    thrust::generate(d.begin(), d.end(), Validator(Distribution(1, 6)));

    ASSERT_EQ(true, d[0]);
}

template <typename Distribution>
void TestDistributionSaveRestore(void)
{
    // create a default distribution
    Distribution d0(7, 13);

    // save it
    std::stringstream ss;
    ss << d0;

    // restore old state
    Distribution d1;
    ss >> d1;

    ASSERT_EQ(d0, d1);
}

TEST(RandomTests, TestUniformIntDistributionMin)
{
    typedef thrust::random::uniform_int_distribution<int>          int_dist;
    typedef thrust::random::uniform_int_distribution<unsigned int> uint_dist;

    ValidateDistributionCharacteristic<int_dist,
                                       ValidateDistributionMin<int_dist, thrust::minstd_rand>>();
    ValidateDistributionCharacteristic<uint_dist,
                                       ValidateDistributionMin<uint_dist, thrust::minstd_rand>>();
}

TEST(RandomTests, TestUniformIntDistributionMax)
{
    typedef thrust::random::uniform_int_distribution<int>          int_dist;
    typedef thrust::random::uniform_int_distribution<unsigned int> uint_dist;

    ValidateDistributionCharacteristic<int_dist,
                                       ValidateDistributionMax<int_dist, thrust::minstd_rand>>();
    ValidateDistributionCharacteristic<uint_dist,
                                       ValidateDistributionMax<uint_dist, thrust::minstd_rand>>();
}

TEST(RandomTests, TestUniformIntDistributionSaveRestore)
{
    typedef thrust::random::uniform_int_distribution<int>          int_dist;
    typedef thrust::random::uniform_int_distribution<unsigned int> uint_dist;

    TestDistributionSaveRestore<int_dist>();
    TestDistributionSaveRestore<uint_dist>();
}

TEST(RandomTests, TestUniformRealDistributionMin)
{
    typedef thrust::random::uniform_real_distribution<float>  float_dist;
    typedef thrust::random::uniform_real_distribution<double> double_dist;

    ValidateDistributionCharacteristic<float_dist,
                                       ValidateDistributionMin<float_dist, thrust::minstd_rand>>();
    ValidateDistributionCharacteristic<double_dist,
                                       ValidateDistributionMin<double_dist, thrust::minstd_rand>>();
}

TEST(RandomTests, TestUniformRealDistributionMax)
{
    typedef thrust::random::uniform_real_distribution<float>  float_dist;
    typedef thrust::random::uniform_real_distribution<double> double_dist;

    ValidateDistributionCharacteristic<float_dist,
                                       ValidateDistributionMax<float_dist, thrust::minstd_rand>>();
    ValidateDistributionCharacteristic<double_dist,
                                       ValidateDistributionMax<double_dist, thrust::minstd_rand>>();
}

TEST(RandomTests, TestUniformRealDistributionSaveRestore)
{
    typedef thrust::random::uniform_real_distribution<float>  float_dist;
    typedef thrust::random::uniform_real_distribution<double> double_dist;

    TestDistributionSaveRestore<float_dist>();
    TestDistributionSaveRestore<double_dist>();
}

TEST(RandomTests, TestNormalDistributionMin)
{
    typedef thrust::random::normal_distribution<float>  float_dist;
    typedef thrust::random::normal_distribution<double> double_dist;

    ValidateDistributionCharacteristic<float_dist,
                                       ValidateDistributionMin<float_dist, thrust::minstd_rand>>();
    ValidateDistributionCharacteristic<double_dist,
                                       ValidateDistributionMin<double_dist, thrust::minstd_rand>>();
}

TEST(RandomTests, TestNormalDistributionMax)
{
    typedef thrust::random::normal_distribution<float>  float_dist;
    typedef thrust::random::normal_distribution<double> double_dist;

    ValidateDistributionCharacteristic<float_dist,
                                       ValidateDistributionMax<float_dist, thrust::minstd_rand>>();
    ValidateDistributionCharacteristic<double_dist,
                                       ValidateDistributionMax<double_dist, thrust::minstd_rand>>();
}

TEST(RandomTests, TestNormalDistributionSaveRestore)
{
    typedef thrust::random::normal_distribution<float>  float_dist;
    typedef thrust::random::normal_distribution<double> double_dist;

    TestDistributionSaveRestore<float_dist>();
    TestDistributionSaveRestore<double_dist>();
}

TEST(RandomTests, erfcinvFunction)
{
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::nan("undefined");

    double double_inputs[8]
        = {-3.0,
           0.0,
           0.0000001,
           0.3,
           0.7,
           1.20,
           2.0,
           3.0}; //This values are those returned by the nvidia's double erfcinv(double a)
    double double_expected_outputs[8]
        = {nan, inf, 3.76656, 0.732869, 0.272463, -0.179143, -inf, nan};

    for(int i = 0; i < 8; i++)
    {
        double input  = double_inputs[i];
        double output = erfcinv(input);

        if(std::isnan(output))
        {
            ASSERT_EQ(std::isnan(output), std::isnan(double_expected_outputs[i]));
        }
        else if((output == inf) || (output == -inf))
        {
            ASSERT_EQ(output, double_expected_outputs[i]);
        }
        else
        {
            EXPECT_NEAR(double_expected_outputs[i], output, 0.01);
        }
    }

    float inf_f = std::numeric_limits<float>::infinity();
    float nan_f = std::nanf("undefined");

    float float_inputs[8]
        = {-3.0f,
           0.0f,
           0.0000001f,
           0.3f,
           0.7f,
           1.20f,
           2.0f,
           3.0f}; //This values are those returned by the nvidia's float erfcinv(float a)
    float float_expected_outputs[8]
        = {nan_f, inf_f, 3.76656, 0.732869, 0.272463, -0.179144, -inf_f, nan_f};

    for(int i = 0; i < 8; i++)
    {
        float input  = float_inputs[i];
        float output = erfcinv(input);

        if(std::isnan(output))
        {
            ASSERT_EQ(std::isnan(output), std::isnan(float_expected_outputs[i]));
        }
        else if((output == inf_f) || (output == -inf_f))
        {
            ASSERT_EQ(output, float_expected_outputs[i]);
        }
        else
        {
            EXPECT_NEAR(float_expected_outputs[i], output, 0.01);
        }
    }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
