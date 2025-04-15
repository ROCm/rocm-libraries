/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "pow2_bins_gtest.hpp"

using namespace rocshmem;

TEST_F(Pow2BinsTestFixture, alloc_0_bytes) {
  char* c_ptr{nullptr};
  size_t size{0};
  strat_.alloc(&c_ptr, size);
  ASSERT_EQ(c_ptr, nullptr);
}

TEST_F(Pow2BinsTestFixture, alloc_1_byte) {
  char* c_ptr{nullptr};
  size_t size{1};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  size_t min_size{256};
  auto bins{strat_.get_bins()};
  auto bin{(*bins)[min_size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_128_bytes) {
  char* c_ptr{nullptr};
  size_t size{128};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  size_t min_size{256};
  auto bins{strat_.get_bins()};
  auto bin{(*bins)[min_size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_256_bytes) {
  char* c_ptr{nullptr};
  size_t size{256};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  auto bins{strat_.get_bins()};
  auto bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_512_bytes) {
  char* c_ptr{nullptr};
  size_t size{512};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  auto bins{strat_.get_bins()};
  auto bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_513_bytes) {
  char* c_ptr{nullptr};
  size_t size{513};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  size_t min_size{1024};
  auto bins{strat_.get_bins()};
  auto bin{(*bins)[min_size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_4KB) {
  char* c_ptr{nullptr};
  size_t size{4096};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  auto bins{strat_.get_bins()};
  auto bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_128KB) {
  char* c_ptr{nullptr};
  size_t size{131072};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  auto bins{strat_.get_bins()};
  auto bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 1);
}

TEST_F(Pow2BinsTestFixture, alloc_1GB) {
  char* c_ptr{nullptr};
  size_t size{1 << 30};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  auto bins{strat_.get_bins()};
  auto bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 0);
}

TEST_F(Pow2BinsTestFixture, alloc_256_bytes_X2_free_256_bytes_X2) {
  char* c_ptr_1{nullptr};
  char* c_ptr_2{nullptr};
  size_t size{256};
  strat_.alloc(&c_ptr_1, size);
  ASSERT_NE(c_ptr_1, nullptr);
  strat_.alloc(&c_ptr_2, size);
  ASSERT_NE(c_ptr_2, nullptr);

  auto bins{strat_.get_bins()};
  auto& bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 0);

  strat_.free(c_ptr_1);
  strat_.free(c_ptr_2);

  ASSERT_EQ(bin.size(), 2);
}

TEST_F(Pow2BinsTestFixture, alloc_1GB_free_1GB) {
  char* c_ptr{nullptr};
  size_t size{1 << 30};
  strat_.alloc(&c_ptr, size);
  ASSERT_NE(c_ptr, nullptr);

  auto bins{strat_.get_bins()};
  auto& bin{(*bins)[size]};
  ASSERT_EQ(bin.size(), 0);

  strat_.free(c_ptr);

  ASSERT_EQ(bin.size(), 1);
}
