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

void TestAssertEqual(void)
{
  ASSERT_EQUAL(0, 0);
  ASSERT_EQUAL(1, 1);
  ASSERT_EQUAL(-15.0f, -15.0f);
}
DECLARE_UNITTEST(TestAssertEqual);

void TestAssertLEqual(void)
{
  ASSERT_LEQUAL(0, 1);
  ASSERT_LEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertLEqual);

void TestAssertGEqual(void)
{
  ASSERT_GEQUAL(1, 0);
  ASSERT_GEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertGEqual);

void TestAssertLess(void)
{
  ASSERT_LESS(0, 1);
}
DECLARE_UNITTEST(TestAssertLess);

void TestAssertGreater(void)
{
  ASSERT_GREATER(1, 0);
}
DECLARE_UNITTEST(TestAssertGreater);

void TestTypeName(void)
{
  ASSERT_EQUAL(unittest::type_name<char>(), "char");
  ASSERT_EQUAL(unittest::type_name<signed char>(), "signed char");
  ASSERT_EQUAL(unittest::type_name<unsigned char>(), "unsigned char");
  ASSERT_EQUAL(unittest::type_name<int>(), "int");
  ASSERT_EQUAL(unittest::type_name<float>(), "float");
  ASSERT_EQUAL(unittest::type_name<double>(), "double");
}
DECLARE_UNITTEST(TestTypeName);
