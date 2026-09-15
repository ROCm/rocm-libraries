/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TEST(UtilsTesterTest, TestAssertEqual)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(0, 0);
  ASSERT_EQ(1, 1);
  ASSERT_EQ(-15.0f, -15.0f);
}

TEST(UtilsTesterTest, TestAssertLEqual)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_LE(0, 1);
  ASSERT_LE(0, 0);
}

TEST(UtilsTesterTest, TestAssertGEqual)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_GE(1, 0);
  ASSERT_GE(0, 0);
}

TEST(UtilsTesterTest, TestAssertLess)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_LE(0, 1);
}

TEST(UtilsTesterTest, TestAssertGreater)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_GT(1, 0);
}

TEST(UtilsTesterTest, TestTypeName)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(type_name<char>(), "char");
  ASSERT_EQ(type_name<signed char>(), "signed char");
  ASSERT_EQ(type_name<unsigned char>(), "unsigned char");
  ASSERT_EQ(type_name<int>(), "int");
  ASSERT_EQ(type_name<float>(), "float");
  ASSERT_EQ(type_name<double>(), "double");
}
