// -*- C++ -*-

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __GPU___UTILITY_TRANSFER_H__
#define __GPU___UTILITY_TRANSFER_H__

#include <type_traits>

namespace gpu {

struct transferToDevice_t { explicit transferToDevice_t() = default; };
// extern const transferToDevice_t transferToDevice;

struct transferToHost_t { explicit transferToHost_t() = default; };
// extern const transferToHost_t transferToHost;

// Transfer from host to device. Invokes the copy-transfer or move-transfer constructor and returns an rvalue.
template<typename _Tp, bool = std::is_trivially_copyable_v<std::remove_reference_t<_Tp> /* false */>
[[nodiscard]] std::remove_reference_t<_Tp>
forward_or_transfer(_Tp&& __t, transferToDevice_t) noexcept
{ return _Tp(std::forward<_Tp>(__t), transferToDevice_t()); }

// Transfer from device to host. Invokes the copy-transfer or move-transfer constructor and returns an rvalue.
template<typename _Tp, bool = std::is_trivially_copyable_v<std::remove_reference_t<_Tp> /* false */>
[[nodiscard]] std::remove_reference_t<_Tp>
forward_or_transfer(_Tp&& __t, transferToHost_t) noexcept
{ return _Tp(std::forward<_Tp>(__t), transferToHost_t()); }

// Forward only, no transfer
template<typename _Tp>
[[nodiscard]] _Tp&&
forward_or_transfer<_Tp, true>(_Tp&& __t, transferToDevice_t) noexcept
{ return std::forward<_Tp>(__t); }

// Forward only, no transfer
template<typename _Tp>
[[nodiscard]] _Tp&&
forward_or_transfer<_Tp, true>(_Tp&& __t, transferToHost_t) noexcept
{ return std::forward<_Tp>(__t); }

}

#endif // __GPU___UTILITY_TRANSFER_H__
