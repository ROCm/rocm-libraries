/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Named CMS kernel speedup factors (x100).
 *
 * Speedup is measured relative to the baseline latency model. Latency correction is
 * main_loop_efficiency = 1 / speedup.
 */
namespace cms_speedup {
constexpr uint8_t X100 = 100;
constexpr uint8_t X105 = 105;
constexpr uint8_t X110 = 110;
constexpr uint8_t X115 = 115;
constexpr uint8_t X120 = 120;
constexpr uint8_t X123 = 123;
constexpr uint8_t X126 = 126;
}  // namespace cms_speedup

inline constexpr double cms_efficiency(uint8_t speedup_x100) {
  return 100.0 / static_cast<double>(speedup_x100);
}

struct cms_kernel_entry_t {
  data_type_t mi_dtype;
  transpose_t trans_a;
  transpose_t trans_b;
  uint16_t m;
  uint16_t n;
  uint16_t k;
  uint8_t speedup_x100;
};

struct cms_kernel_table_view_t {
  hardware_t::architecture_t arch;
  const cms_kernel_entry_t* begin;
  const cms_kernel_entry_t* end;
};

template <size_t N>
constexpr cms_kernel_table_view_t make_cms_kernel_table(
    hardware_t::architecture_t arch, const std::array<cms_kernel_entry_t, N>& entries) {
  return {arch, entries.data(), entries.data() + N};
}

namespace cms_tables {

inline constexpr std::array<cms_kernel_entry_t, 38> gfx950 = {{
    // BF16 NT
    {data_type_t::BFloat16, transpose_t::N, transpose_t::T, 160, 256, 64, cms_speedup::X120},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::T, 192, 256, 64, cms_speedup::X110},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::T, 208, 256, 64, cms_speedup::X120},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::T, 256, 160, 64, cms_speedup::X120},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::T, 256, 192, 64, cms_speedup::X120},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::T, 256, 256, 64, cms_speedup::X115},
    // BF16 NN
    {data_type_t::BFloat16, transpose_t::N, transpose_t::N, 160, 256, 64, cms_speedup::X110},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::N, 208, 256, 64, cms_speedup::X110},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::N, 256, 192, 64, cms_speedup::X100},
    {data_type_t::BFloat16, transpose_t::N, transpose_t::N, 256, 256, 64, cms_speedup::X105},
    // BF16 TN
    {data_type_t::BFloat16, transpose_t::T, transpose_t::N, 160, 256, 64, cms_speedup::X110},
    {data_type_t::BFloat16, transpose_t::T, transpose_t::N, 192, 256, 64, cms_speedup::X105},
    {data_type_t::BFloat16, transpose_t::T, transpose_t::N, 256, 96, 64, cms_speedup::X110},
    {data_type_t::BFloat16, transpose_t::T, transpose_t::N, 256, 192, 64, cms_speedup::X110},
    {data_type_t::BFloat16, transpose_t::T, transpose_t::N, 256, 224, 64, cms_speedup::X105},
    {data_type_t::BFloat16, transpose_t::T, transpose_t::N, 256, 256, 64, cms_speedup::X105},
    // BF16 TT
    {data_type_t::BFloat16, transpose_t::T, transpose_t::T, 256, 256, 64, cms_speedup::X110},
    // FP16 NT
    {data_type_t::Half, transpose_t::N, transpose_t::T, 192, 320, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::N, transpose_t::T, 208, 256, 64, cms_speedup::X120},
    {data_type_t::Half, transpose_t::N, transpose_t::T, 256, 128, 64, cms_speedup::X120},
    {data_type_t::Half, transpose_t::N, transpose_t::T, 256, 192, 64, cms_speedup::X120},
    {data_type_t::Half, transpose_t::N, transpose_t::T, 256, 256, 64, cms_speedup::X115},
    // FP16 NN
    {data_type_t::Half, transpose_t::N, transpose_t::N, 128, 256, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::N, transpose_t::N, 160, 256, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::N, transpose_t::N, 256, 160, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::N, transpose_t::N, 192, 256, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::N, transpose_t::N, 256, 192, 64, cms_speedup::X100},
    {data_type_t::Half, transpose_t::N, transpose_t::N, 256, 256, 64, cms_speedup::X105},
    // FP16 TN
    {data_type_t::Half, transpose_t::T, transpose_t::N, 160, 256, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::T, transpose_t::N, 192, 256, 64, cms_speedup::X105},
    {data_type_t::Half, transpose_t::T, transpose_t::N, 256, 96, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::T, transpose_t::N, 256, 192, 64, cms_speedup::X110},
    {data_type_t::Half, transpose_t::T, transpose_t::N, 256, 224, 64, cms_speedup::X105},
    {data_type_t::Half, transpose_t::T, transpose_t::N, 256, 256, 64, cms_speedup::X105},
    // FP16 TT
    {data_type_t::Half, transpose_t::T, transpose_t::T, 256, 256, 64, cms_speedup::X110},
    // TF32 NN
    {data_type_t::XFloat32, transpose_t::N, transpose_t::N, 192, 256, 32, cms_speedup::X123},
    // TF32 TN
    {data_type_t::XFloat32, transpose_t::T, transpose_t::N, 128, 256, 32, cms_speedup::X126},
    {data_type_t::XFloat32, transpose_t::T, transpose_t::N, 192, 256, 32, cms_speedup::X123},
}};

// Register new architectures by adding a table here.
inline constexpr std::array<cms_kernel_table_view_t, 1> all = {
    make_cms_kernel_table(hardware_t::architecture_t::gfx950, gfx950),
};

constexpr size_t entry_count() {
  size_t count = 0;
  for (const auto& table : all) count += static_cast<size_t>(table.end - table.begin);
  return count;
}

}  // namespace cms_tables

}  // namespace origami
