// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../../detail/config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSize
>
struct block_scan_warp_scan
{
    // Select warp size
    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

    // typedef of warp_scan primitive that will be used to perform warp-level scans
    // on input values.
    using warp_scan = ::rocprim::warp_scan<T, warp_size>;
    // typedef of warp_scan primtive that will be used to get prefix values for
    // each warp (scanned carry-outs from warps before it)
    using warp_prefix_scan = ::rocprim::warp_scan<T, detail::next_power_of_two(warps_no)>;

    struct storage_type
    {
        T warp_scan_results[warps_no];
        union
        {
            typename warp_scan::storage_type wscan;
            typename warp_prefix_scan::storage_type wprefix_scan;
        };
    };

    template<class BinaryFunction>
    void inclusive_scan(T input, T& output, BinaryFunction scan_op) [[hc]]
    {
        tile_static storage_type storage;
        return this->inclusive_scan(input, output, storage, scan_op);
    }

    template<class BinaryFunction>
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op) [[hc]]
    {
        // Perform warp scan
        warp_scan().inclusive_scan(
            input, output, storage.wscan, scan_op
        );

        // Save the warp reduction result, that is the scan result
        // for last element in each warp
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int lane_id = ::rocprim::lane_id();
        if(lane_id == warp_size - 1)
        {
            storage.warp_scan_results[warp_id] = output;
        }
        ::rocprim::syncthreads();

        // // Scan the warp reduction results
        if(lane_id < warps_no)
        {
            auto warp_prefix = storage.warp_scan_results[lane_id];
            warp_prefix_scan().inclusive_scan(
                warp_prefix, warp_prefix, storage.wprefix_scan, scan_op
            );
            storage.warp_scan_results[lane_id] = warp_prefix;
        }
        ::rocprim::syncthreads();

        // Calculate the final scan result for every thread
        if(warp_id != 0)
        {
            auto warp_prefix = storage.warp_scan_results[warp_id - 1];
            output = scan_op(warp_prefix, output);
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_
