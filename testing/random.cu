/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <unittest/unittest.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <sstream>

template<typename Engine>
  struct ValidateEngine
{
  THRUST_HOST_DEVICE ValidateEngine(const typename Engine::result_type value_10000)
      : m_value_10000(value_10000)
  {}

  THRUST_HOST_DEVICE bool operator()(void) const
  {
    Engine e;
    e.discard(9999);

    // get the 10Kth result
    return e() == m_value_10000;
  }

  const typename Engine::result_type m_value_10000;
}; // end ValidateEngine


template<typename Engine,
         bool trivial_min = (Engine::min == 0)>
  struct ValidateEngineMin
{
  THRUST_HOST_DEVICE bool operator()(void) const
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

template<typename Engine>
  struct ValidateEngineMin<Engine,true>
{
  THRUST_HOST_DEVICE bool operator()(void) const
  {
    return true;
  }
};


template<typename Engine>
  struct ValidateEngineMax
{
  THRUST_HOST_DEVICE bool operator()(void) const
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


template<typename Engine>
  struct ValidateEngineEqual
{
  THRUST_HOST_DEVICE bool operator()(void) const
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


template<typename Engine>
  struct ValidateEngineUnequal
{
  THRUST_HOST_DEVICE bool operator()(void) const
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


template<typename Distribution, typename Engine>
  struct ValidateDistributionMin
{
  using random_engine = Engine;

  THRUST_HOST_DEVICE ValidateDistributionMin(const Distribution& dd)
      : d(dd)
  {}

  THRUST_HOST_DEVICE bool operator()(void)
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


template<typename Distribution, typename Engine>
  struct ValidateDistributionMax
{
  using random_engine = Engine;

  THRUST_HOST_DEVICE ValidateDistributionMax(const Distribution& dd)
      : d(dd)
  {}

  THRUST_HOST_DEVICE bool operator()(void)
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


template<typename Distribution>
  struct ValidateDistributionEqual
{
  THRUST_HOST_DEVICE bool operator()(void) const
  {
    return d0 == d1;
  }

  Distribution d0, d1;
};


template<typename Distribution>
  struct ValidateDistributionUnqual
{
  THRUST_HOST_DEVICE bool operator()(void) const
  {
    return d0 != d1;
  }

  Distribution d0, d1;
};


template<typename Engine, std::uint64_t value_10000>
void TestEngineValidation(void)
{
  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngine<Engine>(value_10000));

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngine<Engine>(value_10000));

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineMax(void)
{
  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngineMax<Engine>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngineMax<Engine>());

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineMin(void)
{
  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), ValidateEngineMin<Engine>());

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), ValidateEngineMin<Engine>());

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
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

  ASSERT_EQUAL(e0(), e1());
}


template<typename Engine>
void TestEngineEqual(void)
{
  ValidateEngineEqual<Engine> f;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), f);

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), f);

  ASSERT_EQUAL(true, d[0]);
}


template<typename Engine>
void TestEngineUnequal(void)
{
  ValidateEngineUnequal<Engine> f;

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), f);

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), f);

  ASSERT_EQUAL(true, d[0]);
}

void TestRanlux24BaseValidation(void)
{
  using Engine = thrust::random::ranlux24_base;

  TestEngineValidation<Engine,7937952u>();
}
DECLARE_UNITTEST(TestRanlux24BaseValidation);


void TestRanlux24BaseMin(void)
{
  using Engine = thrust::random::ranlux24_base;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseMin);


void TestRanlux24BaseMax(void)
{
  using Engine = thrust::random::ranlux24_base;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseMax);


void TestRanlux24BaseSaveRestore(void)
{
  using Engine = thrust::random::ranlux24_base;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseSaveRestore);


void TestRanlux24BaseEqual(void)
{
  using Engine = thrust::random::ranlux24_base;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseEqual);


void TestRanlux24BaseUnequal(void)
{
  using Engine = thrust::random::ranlux24_base;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestRanlux24BaseUnequal);


void TestRanlux48BaseValidation(void)
{
  using Engine = thrust::random::ranlux48_base;

  TestEngineValidation<Engine,192113843633948ull>();
}
DECLARE_UNITTEST(TestRanlux48BaseValidation);


void TestRanlux48BaseMin(void)
{
  using Engine = thrust::random::ranlux48_base;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseMin);


void TestRanlux48BaseMax(void)
{
  using Engine = thrust::random::ranlux48_base;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseMax);


void TestRanlux48BaseSaveRestore(void)
{
  using Engine = thrust::random::ranlux48_base;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseSaveRestore);


void TestRanlux48BaseEqual(void)
{
  using Engine = thrust::random::ranlux48_base;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestRanlux48BaseEqual);


#if defined(__INTEL_COMPILER) && 1800 >= __INTEL_COMPILER
void TestRanlux48BaseUnequal(void)
{
    // ICPC has a known failure with this test.
    // See nvbug 200414000.
    KNOWN_FAILURE;
}
#else
void TestRanlux48BaseUnequal(void)
{
  using Engine = thrust::random::ranlux48_base;

  TestEngineUnequal<Engine>();
}
#endif
DECLARE_UNITTEST(TestRanlux48BaseUnequal);


void TestMinstdRandValidation(void)
{
  using Engine = thrust::random::minstd_rand;

  TestEngineValidation<Engine,399268537u>();
}
DECLARE_UNITTEST(TestMinstdRandValidation);


void TestMinstdRandMin(void)
{
  using Engine = thrust::random::minstd_rand;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandMin);


void TestMinstdRandMax(void)
{
  using Engine = thrust::random::minstd_rand;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandMax);


void TestMinstdRandSaveRestore(void)
{
  using Engine = thrust::random::minstd_rand;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandSaveRestore);


void TestMinstdRandEqual(void)
{
  using Engine = thrust::random::minstd_rand;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandEqual);


void TestMinstdRandUnequal(void)
{
  using Engine = thrust::random::minstd_rand;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestMinstdRandUnequal);


void TestMinstdRand0Validation(void)
{
  using Engine = thrust::random::minstd_rand0;

  TestEngineValidation<Engine,1043618065u>();
}
DECLARE_UNITTEST(TestMinstdRand0Validation);


void TestMinstdRand0Min(void)
{
  using Engine = thrust::random::minstd_rand0;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Min);


void TestMinstdRand0Max(void)
{
  using Engine = thrust::random::minstd_rand0;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Max);


void TestMinstdRand0SaveRestore(void)
{
  using Engine = thrust::random::minstd_rand0;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0SaveRestore);


void TestMinstdRand0Equal(void)
{
  using Engine = thrust::random::minstd_rand0;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Equal);


void TestMinstdRand0Unequal(void)
{
  using Engine = thrust::random::minstd_rand0;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestMinstdRand0Unequal);


void TestTaus88Validation(void)
{
  using Engine = thrust::random::taus88;

  TestEngineValidation<Engine,3535848941ull>();
}
DECLARE_UNITTEST(TestTaus88Validation);


void TestTaus88Min(void)
{
  using Engine = thrust::random::taus88;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestTaus88Min);


void TestTaus88Max(void)
{
  using Engine = thrust::random::taus88;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestTaus88Max);


void TestTaus88SaveRestore(void)
{
  using Engine = thrust::random::taus88;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestTaus88SaveRestore);


void TestTaus88Equal(void)
{
  using Engine = thrust::random::taus88;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestTaus88Equal);


void TestTaus88Unequal(void)
{
  using Engine = thrust::random::taus88;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestTaus88Unequal);


void TestRanlux24Validation(void)
{
  using Engine = thrust::random::ranlux24;

  TestEngineValidation<Engine,9901578>();
}
DECLARE_UNITTEST(TestRanlux24Validation);


void TestRanlux24Min(void)
{
  using Engine = thrust::random::ranlux24;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux24Min);


void TestRanlux24Max(void)
{
  using Engine = thrust::random::ranlux24;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux24Max);


void TestRanlux24SaveRestore(void)
{
  using Engine = thrust::random::ranlux24;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux24SaveRestore);


void TestRanlux24Equal(void)
{
  using Engine = thrust::random::ranlux24;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestRanlux24Equal);


void TestRanlux24Unequal(void)
{
  using Engine = thrust::random::ranlux24;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestRanlux24Unequal);



void TestRanlux48Validation(void)
{
  using Engine = thrust::random::ranlux48;

  TestEngineValidation<Engine,88229545517833ull>();
}
DECLARE_UNITTEST(TestRanlux48Validation);


void TestRanlux48Min(void)
{
  using Engine = thrust::random::ranlux48;

  TestEngineMin<Engine>();
}
DECLARE_UNITTEST(TestRanlux48Min);


void TestRanlux48Max(void)
{
  using Engine = thrust::random::ranlux48;

  TestEngineMax<Engine>();
}
DECLARE_UNITTEST(TestRanlux48Max);


void TestRanlux48SaveRestore(void)
{
  using Engine = thrust::random::ranlux48;

  TestEngineSaveRestore<Engine>();
}
DECLARE_UNITTEST(TestRanlux48SaveRestore);


void TestRanlux48Equal(void)
{
  using Engine = thrust::random::ranlux48;

  TestEngineEqual<Engine>();
}
DECLARE_UNITTEST(TestRanlux48Equal);


void TestRanlux48Unequal(void)
{
  using Engine = thrust::random::ranlux48;

  TestEngineUnequal<Engine>();
}
DECLARE_UNITTEST(TestRanlux48Unequal);


THRUST_DISABLE_MSVC_WARNING_BEGIN(4305) // truncation warning
template<typename Distribution, typename Validator>
  void ValidateDistributionCharacteristic(void)
{
  using Engine = typename Validator::random_engine;

  // test default-constructed Distribution

  // test host
  thrust::host_vector<bool> h(1);
  thrust::generate(h.begin(), h.end(), Validator(Distribution()));

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::device_vector<bool> d(1);
  thrust::generate(d.begin(), d.end(), Validator(Distribution()));

  ASSERT_EQUAL(true, d[0]);


  // test distribution & engine with comparable ranges
  // only do this if they have the same result_type
  if(thrust::detail::is_same<typename Distribution::result_type, typename Engine::result_type>::value)
  {
    // test Distribution with same range as engine

    // test host
    thrust::generate(h.begin(), h.end(), Validator(
        Distribution(Engine::min, Engine::max)
    ));

    ASSERT_EQUAL(true, h[0]);

    // test device
    thrust::generate(d.begin(), d.end(), Validator(
        Distribution(Engine::min, Engine::max)
    ));

    ASSERT_EQUAL(true, d[0]);

    // test Distribution with smaller range than engine

    // test host
    typename Distribution::result_type engine_range = Engine::max - Engine::min;
    thrust::generate(h.begin(), h.end(), Validator(Distribution(engine_range/3, (2 * engine_range)/3)));

    ASSERT_EQUAL(true, h[0]);

    // test device
    thrust::generate(d.begin(), d.end(), Validator(Distribution(engine_range/3, (2 * engine_range)/3)));

    ASSERT_EQUAL(true, d[0]);
  }


  // test Distribution with a very small range

  // test host
  thrust::generate(h.begin(), h.end(), Validator(Distribution(1,6)));

  ASSERT_EQUAL(true, h[0]);

  // test device
  thrust::generate(d.begin(), d.end(), Validator(Distribution(1,6)));

  ASSERT_EQUAL(true, d[0]);
}
THRUST_DISABLE_MSVC_WARNING_END(4305)


template<typename Distribution>
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

  ASSERT_EQUAL(d0, d1);
}


void TestUniformIntDistributionMin(void)
{
  using int_dist  = thrust::random::uniform_int_distribution<int>;
  using uint_dist = thrust::random::uniform_int_distribution<unsigned int>;

  ValidateDistributionCharacteristic<int_dist,  ValidateDistributionMin<int_dist,  thrust::minstd_rand> >();
  ValidateDistributionCharacteristic<uint_dist, ValidateDistributionMin<uint_dist, thrust::minstd_rand> >();
}
DECLARE_UNITTEST(TestUniformIntDistributionMin);


void TestUniformIntDistributionMax(void)
{
  using int_dist  = thrust::random::uniform_int_distribution<int>;
  using uint_dist = thrust::random::uniform_int_distribution<unsigned int>;

  ValidateDistributionCharacteristic<int_dist,  ValidateDistributionMax<int_dist,  thrust::minstd_rand> >();
  ValidateDistributionCharacteristic<uint_dist, ValidateDistributionMax<uint_dist, thrust::minstd_rand> >();
}
DECLARE_UNITTEST(TestUniformIntDistributionMax);


void TestUniformIntDistributionSaveRestore(void)
{
  using int_dist  = thrust::random::uniform_int_distribution<int>;
  using uint_dist = thrust::random::uniform_int_distribution<unsigned int>;

  TestDistributionSaveRestore<int_dist>();
  TestDistributionSaveRestore<uint_dist>();
}
DECLARE_UNITTEST(TestUniformIntDistributionSaveRestore);


void TestUniformRealDistributionMin(void)
{
  using float_dist  = thrust::random::uniform_real_distribution<float>;
  using double_dist = thrust::random::uniform_real_distribution<double>;

  ValidateDistributionCharacteristic<float_dist,  ValidateDistributionMin<float_dist,  thrust::minstd_rand> >();
  ValidateDistributionCharacteristic<double_dist, ValidateDistributionMin<double_dist, thrust::minstd_rand> >();
}
DECLARE_UNITTEST(TestUniformRealDistributionMin);


void TestUniformRealDistributionMax(void)
{
  using float_dist  = thrust::random::uniform_real_distribution<float>;
  using double_dist = thrust::random::uniform_real_distribution<double>;

  ValidateDistributionCharacteristic<float_dist,  ValidateDistributionMax<float_dist,  thrust::minstd_rand> >();
  ValidateDistributionCharacteristic<double_dist, ValidateDistributionMax<double_dist, thrust::minstd_rand> >();
}
DECLARE_UNITTEST(TestUniformRealDistributionMax);


void TestUniformRealDistributionSaveRestore(void)
{
  using float_dist  = thrust::random::uniform_real_distribution<float>;
  using double_dist = thrust::random::uniform_real_distribution<double>;

  TestDistributionSaveRestore<float_dist>();
  TestDistributionSaveRestore<double_dist>();
}
DECLARE_UNITTEST(TestUniformRealDistributionSaveRestore);


void TestNormalDistributionMin(void)
{
  using float_dist  = thrust::random::normal_distribution<float>;
  using double_dist = thrust::random::normal_distribution<double>;

  ValidateDistributionCharacteristic<float_dist,  ValidateDistributionMin<float_dist,  thrust::minstd_rand> >();
  ValidateDistributionCharacteristic<double_dist, ValidateDistributionMin<double_dist, thrust::minstd_rand> >();
}
DECLARE_UNITTEST(TestNormalDistributionMin);


void TestNormalDistributionMax(void)
{
  using float_dist  = thrust::random::normal_distribution<float>;
  using double_dist = thrust::random::normal_distribution<double>;

  ValidateDistributionCharacteristic<float_dist,  ValidateDistributionMax<float_dist,  thrust::minstd_rand> >();
  ValidateDistributionCharacteristic<double_dist, ValidateDistributionMax<double_dist, thrust::minstd_rand> >();
}
DECLARE_UNITTEST(TestNormalDistributionMax);


void TestNormalDistributionSaveRestore(void)
{
  using float_dist  = thrust::random::normal_distribution<float>;
  using double_dist = thrust::random::normal_distribution<double>;

  TestDistributionSaveRestore<float_dist>();
  TestDistributionSaveRestore<double_dist>();
}
DECLARE_UNITTEST(TestNormalDistributionSaveRestore);

