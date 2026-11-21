/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// rocsparse_sparse_to_sparse: formats outside the dispatch table (SELL and beyond) are
// rejected before it is indexed, intermediate CSR arrays are sized with the index types
// stored on that descriptor and oversized counts are rejected before any allocation,
// a null target value array is a structural conversion when the route supports it, and
// an empty conversion can be captured.

#include "unit_test_device_conv_common.hpp"

#include "rocsparse_sparse_to_sparse.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace convut;

namespace
{
    using ConvS2s = ConvTest;

    constexpr auto st_ni   = rocsparse_status_not_implemented;
    constexpr auto idx32   = i32;
    constexpr auto idx64   = rocsparse_indextype_i64;
    constexpr auto base    = base0;
    constexpr auto f32_r   = f32;
    constexpr auto f64_c   = rocsparse_datatype_f64_c;
    constexpr auto alg     = rocsparse_sparse_to_sparse_alg_default;
    constexpr auto stage_a = rocsparse_sparse_to_sparse_stage_analysis;
    constexpr auto stage_c = rocsparse_sparse_to_sparse_stage_compute;

    // Enumerators that index the conversion table are exactly the values below SELL.
    constexpr rocsparse_format k_formats[] = {rocsparse_format_coo,
                                              rocsparse_format_coo_aos,
                                              rocsparse_format_csr,
                                              rocsparse_format_csc,
                                              rocsparse_format_ell,
                                              rocsparse_format_bell,
                                              rocsparse_format_bsr,
                                              rocsparse_format_sell};

    static_assert(sizeof(k_formats) / sizeof(k_formats[0])
                  == static_cast<size_t>(rocsparse_format_sell) + 1);
    static_assert(k_formats[0] == rocsparse_format_coo);
    static_assert(k_formats[2] == rocsparse_format_csr);
    static_assert(k_formats[6] == rocsparse_format_bsr);
    static_assert(k_formats[7] == rocsparse_format_sell);

    const char* fmt_name(rocsparse_format format)
    {
        switch(format)
        {
        case rocsparse_format_coo:
            return "coo";
        case rocsparse_format_coo_aos:
            return "coo_aos";
        case rocsparse_format_csr:
            return "csr";
        case rocsparse_format_csc:
            return "csc";
        case rocsparse_format_ell:
            return "ell";
        case rocsparse_format_bell:
            return "bell";
        case rocsparse_format_bsr:
            return "bsr";
        case rocsparse_format_sell:
            return "sell";
        }
        return "invalid";
    }

    bool format_unsupported(rocsparse_format format)
    {
        return format == rocsparse_format_bell || format == rocsparse_format_sell;
    }

    std::string idx_name(rocsparse_indextype type)
    {
        return type == idx64 ? "i64" : "i32";
    }

    size_t idx_width(rocsparse_indextype type)
    {
        return type == idx64 ? sizeof(int64_t) : sizeof(int32_t);
    }

    // A failed case can leave a HIP error behind that the next kernel launch reports.
    void clear_hip_error()
    {
        (void)hipGetLastError();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    using Bytes = DevBuf<unsigned char>;

    // Sets every allocated byte of `buf` to `value`.
    void fill_bytes(Bytes& buf, unsigned char value)
    {
        std::vector<unsigned char> host = buf.read();
        std::fill(host.begin(), host.end(), value);
        ASSERT_EQ(hipMemcpy(buf.get(), host.data(), host.size(), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    // Every allocated byte of `buf` is `value`.
    bool bytes_are(const Bytes& buf, unsigned char value)
    {
        const std::vector<unsigned char> host = buf.read();
        return std::all_of(
            host.begin(), host.end(), [&](unsigned char byte) { return byte == value; });
    }

    void upload_idx(void* dst, rocsparse_indextype type, const std::vector<int64_t>& host)
    {
        if(host.empty() || dst == nullptr)
        {
            return;
        }
        if(type == idx64)
        {
            ASSERT_EQ(
                hipMemcpy(dst, host.data(), host.size() * sizeof(int64_t), hipMemcpyHostToDevice),
                hipSuccess);
            return;
        }
        std::vector<int32_t> narrow(host.size());
        for(size_t i = 0; i < host.size(); ++i)
        {
            narrow[i] = static_cast<int32_t>(host[i]);
        }
        ASSERT_EQ(
            hipMemcpy(dst, narrow.data(), narrow.size() * sizeof(int32_t), hipMemcpyHostToDevice),
            hipSuccess);
    }

    std::vector<int64_t> download_idx(const void* src, rocsparse_indextype type, size_t n)
    {
        std::vector<int64_t> host(n);
        if(n == 0 || src == nullptr)
        {
            return host;
        }
        if(type == idx64)
        {
            EXPECT_EQ(hipMemcpy(host.data(), src, n * sizeof(int64_t), hipMemcpyDeviceToHost),
                      hipSuccess);
            return host;
        }
        std::vector<int32_t> narrow(n);
        EXPECT_EQ(hipMemcpy(narrow.data(), src, n * sizeof(int32_t), hipMemcpyDeviceToHost),
                  hipSuccess);
        for(size_t i = 0; i < n; ++i)
        {
            host[i] = narrow[i];
        }
        return host;
    }

    std::vector<float> download_f(const void* src, size_t n)
    {
        std::vector<float> host(n);
        if(n != 0 && src != nullptr)
        {
            EXPECT_EQ(hipMemcpy(host.data(), src, n * sizeof(float), hipMemcpyDeviceToHost),
                      hipSuccess);
        }
        return host;
    }

    void upload_f(void* dst, const std::vector<float>& host)
    {
        if(!host.empty() && dst != nullptr)
        {
            ASSERT_EQ(
                hipMemcpy(dst, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
        }
    }

    struct Entry
    {
        int64_t row;
        int64_t col;
        float   val;
    };

    bool operator==(const Entry& a, const Entry& b)
    {
        return a.row == b.row && a.col == b.col && a.val == b.val;
    }

    bool operator<(const Entry& a, const Entry& b)
    {
        if(a.row != b.row)
        {
            return a.row < b.row;
        }
        if(a.col != b.col)
        {
            return a.col < b.col;
        }
        return a.val < b.val;
    }

    // 2x2 with one empty corner. Column 0 holds two entries so a widened index is
    // actually stored, not only a pointer. Row-major order: coo2csr requires it.
    std::vector<Entry> sample_matrix()
    {
        return {{0, 0, 1.f}, {0, 1, 2.f}, {1, 0, 3.f}};
    }

    void expect_entries(const std::vector<Entry>& got, const std::string& label)
    {
        auto expect = sample_matrix();
        auto seen   = got;
        std::sort(expect.begin(), expect.end());
        std::sort(seen.begin(), seen.end());
        ASSERT_EQ(seen.size(), expect.size()) << label;
        for(size_t i = 0; i < expect.size(); ++i)
        {
            EXPECT_EQ(seen[i].row, expect[i].row) << label << " row " << i;
            EXPECT_EQ(seen[i].col, expect[i].col) << label << " col " << i;
            EXPECT_EQ(seen[i].val, expect[i].val) << label << " val " << i;
        }
    }

    std::vector<Entry> entries_from_coo(const std::vector<int64_t>& rows,
                                        const std::vector<int64_t>& cols,
                                        const std::vector<float>&   vals)
    {
        std::vector<Entry> out;
        const size_t       n = std::min(rows.size(), std::min(cols.size(), vals.size()));
        out.reserve(n);
        for(size_t i = 0; i < n; ++i)
        {
            out.push_back({rows[i], cols[i], vals[i]});
        }
        return out;
    }

    std::vector<Entry> entries_from_csx(const std::vector<int64_t>& ptr,
                                        const std::vector<int64_t>& ind,
                                        const std::vector<float>&   vals,
                                        bool                        columns,
                                        const std::string&          label)
    {
        std::vector<Entry> out;
        if(ptr.size() < 2 || ptr.front() != 0 || ptr.back() < 0
           || static_cast<size_t>(ptr.back()) > ind.size()
           || static_cast<size_t>(ptr.back()) > vals.size())
        {
            ADD_FAILURE() << label << " bad pointer array";
            return out;
        }
        for(size_t major = 0; major + 1 < ptr.size(); ++major)
        {
            if(ptr[major] > ptr[major + 1])
            {
                ADD_FAILURE() << label << " pointer decreased";
                return {};
            }
            for(int64_t k = ptr[major]; k < ptr[major + 1]; ++k)
            {
                const int64_t row
                    = columns ? ind[static_cast<size_t>(k)] : static_cast<int64_t>(major);
                const int64_t col
                    = columns ? static_cast<int64_t>(major) : ind[static_cast<size_t>(k)];
                out.push_back({row, col, vals[static_cast<size_t>(k)]});
            }
        }
        return out;
    }

    // Owns the capture stream and graph across an early return.
    struct CaptureGuard
    {
        rocsparse_handle handle;
        hipStream_t      stream = nullptr;
        hipGraph_t       graph  = nullptr;
        hipGraphExec_t   exec   = nullptr;

        explicit CaptureGuard(rocsparse_handle h)
            : handle(h)
        {
        }

        ~CaptureGuard()
        {
            hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
            if(stream != nullptr && hipStreamIsCapturing(stream, &capturing) == hipSuccess
               && capturing == hipStreamCaptureStatusActive)
            {
                (void)hipStreamEndCapture(stream, &graph);
            }
            if(exec != nullptr)
            {
                (void)hipGraphExecDestroy(exec);
            }
            if(graph != nullptr)
            {
                (void)hipGraphDestroy(graph);
            }
            (void)rocsparse_set_stream(handle, nullptr);
            if(stream != nullptr)
            {
                (void)hipStreamDestroy(stream);
            }
        }

        CaptureGuard(const CaptureGuard&) = delete;
        CaptureGuard& operator=(const CaptureGuard&) = delete;
    };

    // --- SELL / every format -------------------------------------------------

    constexpr unsigned char k_sentinel = 0xA5;

    struct TinyMat
    {
        rocsparse_format format;
        Spmat            descr;
        Bytes            a;
        Bytes            b;
        Bytes            c;

        TinyMat(rocsparse_format f, size_t na, size_t nb, size_t nc)
            : format(f)
            , a(na)
            , b(nb)
            , c(nc)
        {
        }
    };

    void upload_i32(void* dst, std::initializer_list<int32_t> values)
    {
        std::vector<int32_t> host(values);
        ASSERT_EQ(hipMemcpy(dst, host.data(), host.size() * sizeof(int32_t), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    std::unique_ptr<TinyMat> make_tiny(rocsparse_format format, bool source)
    {
        // 1x1, one stored entry at (0, 0) with value 2, or a sentinel-filled shell.
        std::unique_ptr<TinyMat> mat;
        const auto               fill_source = [&](auto&& upload) {
            if(source)
            {
                upload();
            }
            else
            {
                fill_bytes(mat->a, k_sentinel);
                fill_bytes(mat->b, k_sentinel);
                fill_bytes(mat->c, k_sentinel);
            }
        };
        switch(format)
        {
        case rocsparse_format_coo:
        {
            mat = std::make_unique<TinyMat>(format, 4, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0});
                upload_i32(mat->b.get(), {0});
                upload_f(mat->c.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_coo_descr(&mat->descr,
                                                 1,
                                                 1,
                                                 1,
                                                 mat->a.get(),
                                                 mat->b.get(),
                                                 mat->c.get(),
                                                 idx32,
                                                 base,
                                                 f32_r),
                      st_ok);
            break;
        }
        case rocsparse_format_coo_aos:
        {
            mat = std::make_unique<TinyMat>(format, 8, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0, 0});
                upload_f(mat->b.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_coo_aos_descr(
                          &mat->descr, 1, 1, 1, mat->a.get(), mat->b.get(), idx32, base, f32_r),
                      st_ok);
            break;
        }
        case rocsparse_format_csr:
        {
            mat = std::make_unique<TinyMat>(format, 8, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0, 1});
                upload_i32(mat->b.get(), {0});
                upload_f(mat->c.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_csr_descr(&mat->descr,
                                                 1,
                                                 1,
                                                 1,
                                                 mat->a.get(),
                                                 mat->b.get(),
                                                 mat->c.get(),
                                                 idx32,
                                                 idx32,
                                                 base,
                                                 f32_r),
                      st_ok);
            break;
        }
        case rocsparse_format_csc:
        {
            mat = std::make_unique<TinyMat>(format, 8, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0, 1});
                upload_i32(mat->b.get(), {0});
                upload_f(mat->c.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_csc_descr(&mat->descr,
                                                 1,
                                                 1,
                                                 1,
                                                 mat->a.get(),
                                                 mat->b.get(),
                                                 mat->c.get(),
                                                 idx32,
                                                 idx32,
                                                 base,
                                                 f32_r),
                      st_ok);
            break;
        }
        case rocsparse_format_ell:
        {
            mat = std::make_unique<TinyMat>(format, 4, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0});
                upload_f(mat->b.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_ell_descr(
                          &mat->descr, 1, 1, mat->a.get(), mat->b.get(), 1, idx32, base, f32_r),
                      st_ok);
            break;
        }
        case rocsparse_format_bsr:
        {
            mat = std::make_unique<TinyMat>(format, 8, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0, 1});
                upload_i32(mat->b.get(), {0});
                upload_f(mat->c.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_bsr_descr(&mat->descr,
                                                 1,
                                                 1,
                                                 1,
                                                 dir_r,
                                                 1,
                                                 mat->a.get(),
                                                 mat->b.get(),
                                                 mat->c.get(),
                                                 idx32,
                                                 idx32,
                                                 base,
                                                 f32_r),
                      st_ok);
            break;
        }
        case rocsparse_format_bell:
        {
            mat = std::make_unique<TinyMat>(format, 4, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0});
                upload_f(mat->b.get(), {2.f});
            });
            EXPECT_EQ(
                rocsparse_create_bell_descr(
                    &mat->descr, 1, 1, dir_r, 1, 1, mat->a.get(), mat->b.get(), idx32, base, f32_r),
                st_ok);
            break;
        }
        case rocsparse_format_sell:
        {
            mat = std::make_unique<TinyMat>(format, 8, 4, 4);
            fill_source([&] {
                upload_i32(mat->a.get(), {0, 1});
                upload_i32(mat->b.get(), {0});
                upload_f(mat->c.get(), {2.f});
            });
            EXPECT_EQ(rocsparse_create_sell_descr(&mat->descr,
                                                  1,
                                                  1,
                                                  1,
                                                  1,
                                                  1,
                                                  mat->a.get(),
                                                  mat->b.get(),
                                                  mat->c.get(),
                                                  idx32,
                                                  idx32,
                                                  base,
                                                  f32_r),
                      st_ok);
            break;
        }
        }
        return mat;
    }

    void refresh_pointers(TinyMat& mat)
    {
        switch(mat.format)
        {
        case rocsparse_format_coo:
            EXPECT_EQ(rocsparse_coo_set_pointers(mat.descr, mat.a.get(), mat.b.get(), mat.c.get()),
                      st_ok);
            break;
        case rocsparse_format_coo_aos:
            EXPECT_EQ(rocsparse_coo_aos_set_pointers(mat.descr, mat.a.get(), mat.b.get()), st_ok);
            break;
        case rocsparse_format_csr:
            EXPECT_EQ(rocsparse_csr_set_pointers(mat.descr, mat.a.get(), mat.b.get(), mat.c.get()),
                      st_ok);
            break;
        case rocsparse_format_csc:
            EXPECT_EQ(rocsparse_csc_set_pointers(mat.descr, mat.a.get(), mat.b.get(), mat.c.get()),
                      st_ok);
            break;
        case rocsparse_format_ell:
            EXPECT_EQ(rocsparse_ell_set_pointers(mat.descr, mat.a.get(), mat.b.get()), st_ok);
            break;
        case rocsparse_format_bsr:
            EXPECT_EQ(rocsparse_bsr_set_pointers(mat.descr, mat.a.get(), mat.b.get(), mat.c.get()),
                      st_ok);
            break;
        case rocsparse_format_bell:
        case rocsparse_format_sell:
            break;
        }
    }

    void expect_untouched(const TinyMat& mat, const std::string& label)
    {
        EXPECT_TRUE(bytes_are(mat.a, k_sentinel)) << label << " array a";
        EXPECT_TRUE(bytes_are(mat.b, k_sentinel)) << label << " array b";
        EXPECT_TRUE(bytes_are(mat.c, k_sentinel)) << label << " array c";
    }

    void expect_one_entry(const TinyMat& mat, const std::string& label)
    {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << label;
        int64_t nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_nnz(mat.descr, &nnz), st_ok) << label;
        if(mat.format == rocsparse_format_ell)
        {
            // One row stores ell_width columns; this matrix's width is 1.
            EXPECT_EQ(nnz, 1) << label;
            const auto col = download_idx(mat.a.get(), idx32, 1);
            const auto val = download_f(mat.b.get(), 1);
            ASSERT_EQ(col.size(), 1u);
            EXPECT_EQ(col[0], 0) << label;
            EXPECT_EQ(val[0], 2.f) << label;
            return;
        }
        EXPECT_EQ(nnz, 1) << label;
        if(mat.format == rocsparse_format_coo)
        {
            const auto row = download_idx(mat.a.get(), idx32, 1);
            const auto col = download_idx(mat.b.get(), idx32, 1);
            const auto val = download_f(mat.c.get(), 1);
            EXPECT_EQ(row[0], 0) << label;
            EXPECT_EQ(col[0], 0) << label;
            EXPECT_EQ(val[0], 2.f) << label;
        }
        else if(mat.format == rocsparse_format_coo_aos)
        {
            const auto ind = download_idx(mat.a.get(), idx32, 2);
            const auto val = download_f(mat.b.get(), 1);
            EXPECT_EQ(ind[0], 0) << label;
            EXPECT_EQ(ind[1], 0) << label;
            EXPECT_EQ(val[0], 2.f) << label;
        }
        else
        {
            const auto ptr = download_idx(mat.a.get(), idx32, 2);
            const auto ind = download_idx(mat.b.get(), idx32, 1);
            const auto val = download_f(mat.c.get(), 1);
            EXPECT_EQ(ptr[0], 0) << label;
            EXPECT_EQ(ptr[1], 1) << label;
            EXPECT_EQ(ind[0], 0) << label;
            EXPECT_EQ(val[0], 2.f) << label;
        }
    }

    // Size of the buffer passed to an execution call that is expected to fail. It is
    // not queried first, since the query rejects the same descriptors.
    constexpr size_t k_dummy_buffer_bytes = 256;

    // One call to either public entry point on a fresh permissive descriptor: the
    // buffer size query, or the conversion itself with a dummy buffer.
    rocsparse_status call_stage(rocsparse_handle                 handle,
                                rocsparse_spmat_descr            source,
                                rocsparse_spmat_descr            target,
                                rocsparse_sparse_to_sparse_stage stage,
                                bool                             query)
    {
        S2sDescr         descr;
        rocsparse_status status
            = rocsparse_create_sparse_to_sparse_descr(&descr, source, target, alg);
        if(status != st_ok)
        {
            return status;
        }
        status = rocsparse_sparse_to_sparse_permissive(descr);
        if(status != st_ok)
        {
            return status;
        }
        if(query)
        {
            size_t bytes = 0;
            return rocsparse_sparse_to_sparse_buffer_size(
                handle, descr, source, target, stage, &bytes);
        }
        Bytes scratch(k_dummy_buffer_bytes);
        return rocsparse_sparse_to_sparse(
            handle, descr, source, target, stage, k_dummy_buffer_bytes, scratch.get());
    }

    void expect_rejected(rocsparse_handle      handle,
                         rocsparse_spmat_descr source,
                         rocsparse_spmat_descr target,
                         TinyMat&              written,
                         const std::string&    label)
    {
        const auto check
            = [&](rocsparse_sparse_to_sparse_stage stage, bool query, const char* name) {
                  fill_bytes(written.a, k_sentinel);
                  fill_bytes(written.b, k_sentinel);
                  fill_bytes(written.c, k_sentinel);
                  EXPECT_EQ(call_stage(handle, source, target, stage, query), st_ni)
                      << label << " " << name;
                  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << label << " " << name;
                  expect_untouched(written, label + " " + name);
              };
        check(stage_a, true, "buffer_size analysis");
        check(stage_c, true, "buffer_size compute");
        check(stage_a, false, "analysis");
        check(stage_c, false, "compute");
    }

    // --- index widening ------------------------------------------------------

    struct HostIdx
    {
        rocsparse_indextype type;
        Bytes               mem;
        explicit HostIdx(rocsparse_indextype t, const std::vector<int64_t>& host)
            : type(t)
            , mem(std::max<size_t>(host.size(), 1) * idx_width(t))
        {
            upload_idx(mem.get(), t, host);
        }
    };

    // Runs both stages. On success, *column_bytes is the size of the intermediate CSR
    // column allocation after the analysis buffer-size query (0 when there is none).
    rocsparse_status run_permissive(rocsparse_handle      handle,
                                    rocsparse_spmat_descr source,
                                    rocsparse_spmat_descr target,
                                    size_t*               column_bytes)
    {
        S2sDescr descr;
        clear_hip_error();
        auto status = rocsparse_create_sparse_to_sparse_descr(&descr, source, target, alg);
        if(status != st_ok)
        {
            return status;
        }
        status = rocsparse_sparse_to_sparse_permissive(descr);
        if(status != st_ok)
        {
            return status;
        }
        for(auto stage : {stage_a, stage_c})
        {
            size_t bytes = 0;
            status       = rocsparse_sparse_to_sparse_buffer_size(
                handle, descr, source, target, stage, &bytes);
            if(status != st_ok)
            {
                return status;
            }
            if(stage == stage_a && column_bytes != nullptr)
            {
                // hipMemPtrGetInfo reports the requested size, so a column array sized
                // with a narrower index type is visible even when the allocator pads it.
                column_bytes[0]                          = 0;
                const rocsparse_spmat_descr intermediate = descr.get()->m_intermediate;
                if(intermediate != nullptr && intermediate->col_data != nullptr)
                {
                    EXPECT_EQ(hipMemPtrGetInfo(intermediate->col_data, column_bytes), hipSuccess);
                }
            }
            Bytes scratch(bytes);
            status = rocsparse_sparse_to_sparse(
                handle, descr, source, target, stage, bytes, scratch.get());
            if(status != st_ok)
            {
                return status;
            }
        }
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st_ok;
    }

    void check_coo_csc(rocsparse_handle    handle,
                       rocsparse_indextype src_type,
                       rocsparse_indextype col_ptr_type,
                       rocsparse_indextype row_ind_type)
    {
        const auto           entries = sample_matrix();
        std::vector<int64_t> rows, cols;
        std::vector<float>   vals;
        for(const auto& entry : entries)
        {
            rows.push_back(entry.row);
            cols.push_back(entry.col);
            vals.push_back(entry.val);
        }
        HostIdx       row_buf(src_type, rows);
        HostIdx       col_buf(src_type, cols);
        DevBuf<float> val_buf(vals);
        // Column pointers and row indices are filled by the conversion.
        HostIdx       csc_ptr(col_ptr_type, std::vector<int64_t>(3, 0));
        HostIdx       csc_row(row_ind_type, std::vector<int64_t>(entries.size(), 0));
        DevBuf<float> csc_val(std::vector<float>(entries.size(), 0.f));

        Spmat source, target;
        ASSERT_EQ(rocsparse_create_coo_descr(&source,
                                             2,
                                             2,
                                             static_cast<int64_t>(entries.size()),
                                             row_buf.mem.get(),
                                             col_buf.mem.get(),
                                             val_buf.get(),
                                             src_type,
                                             base,
                                             f32_r),
                  st_ok);
        ASSERT_EQ(rocsparse_create_csc_descr(&target,
                                             2,
                                             2,
                                             static_cast<int64_t>(entries.size()),
                                             csc_ptr.mem.get(),
                                             csc_row.mem.get(),
                                             csc_val.get(),
                                             col_ptr_type,
                                             row_ind_type,
                                             base,
                                             f32_r),
                  st_ok);
        const std::string label = "coo " + idx_name(src_type) + " -> csc ptr "
                                  + idx_name(col_ptr_type) + " row " + idx_name(row_ind_type);
        SCOPED_TRACE(label);
        size_t     column_bytes = 0;
        const auto status       = run_permissive(handle, source, target, &column_bytes);
        EXPECT_EQ(status, st_ok);
        if(status != st_ok)
        {
            return;
        }
        // Intermediate CSR column type for COO -> CSC is the CSC row-index type.
        EXPECT_EQ(column_bytes, entries.size() * idx_width(row_ind_type));
        const auto ptr = download_idx(csc_ptr.mem.get(), col_ptr_type, 3);
        const auto ind = download_idx(csc_row.mem.get(), row_ind_type, entries.size());
        const auto got = download_f(csc_val.get(), entries.size());
        expect_entries(entries_from_csx(ptr, ind, got, true, label), label);
    }

    void check_coo_bsr(rocsparse_handle    handle,
                       rocsparse_indextype src_type,
                       rocsparse_indextype row_ptr_type,
                       rocsparse_indextype col_ind_type)
    {
        const auto           entries = sample_matrix();
        std::vector<int64_t> rows, cols;
        std::vector<float>   vals;
        for(const auto& entry : entries)
        {
            rows.push_back(entry.row);
            cols.push_back(entry.col);
            vals.push_back(entry.val);
        }
        HostIdx       row_buf(src_type, rows);
        HostIdx       col_buf(src_type, cols);
        DevBuf<float> val_buf(vals);
        HostIdx       bsr_ptr(row_ptr_type, std::vector<int64_t>(3, 0));
        HostIdx       bsr_col(col_ind_type, std::vector<int64_t>(entries.size(), 0));
        DevBuf<float> bsr_val(std::vector<float>(entries.size(), 0.f));
        Spmat         source, target;
        ASSERT_EQ(rocsparse_create_coo_descr(&source,
                                             2,
                                             2,
                                             static_cast<int64_t>(entries.size()),
                                             row_buf.mem.get(),
                                             col_buf.mem.get(),
                                             val_buf.get(),
                                             src_type,
                                             base,
                                             f32_r),
                  st_ok);
        ASSERT_EQ(rocsparse_create_bsr_descr(&target,
                                             2,
                                             2,
                                             static_cast<int64_t>(entries.size()),
                                             dir_r,
                                             1,
                                             bsr_ptr.mem.get(),
                                             bsr_col.mem.get(),
                                             bsr_val.get(),
                                             row_ptr_type,
                                             col_ind_type,
                                             base,
                                             f32_r),
                  st_ok);
        const std::string label = "coo " + idx_name(src_type) + " -> bsr ptr "
                                  + idx_name(row_ptr_type) + " col " + idx_name(col_ind_type);
        SCOPED_TRACE(label);
        size_t     column_bytes = 0;
        const auto status       = run_permissive(handle, source, target, &column_bytes);
        EXPECT_EQ(status, st_ok);
        if(status != st_ok)
        {
            return;
        }
        // Intermediate CSR column type for COO -> BSR is the BSR column-index type.
        EXPECT_EQ(column_bytes, entries.size() * idx_width(col_ind_type));
        const auto ptr = download_idx(bsr_ptr.mem.get(), row_ptr_type, 3);
        const auto ind = download_idx(bsr_col.mem.get(), col_ind_type, entries.size());
        const auto got = download_f(bsr_val.get(), entries.size());
        expect_entries(entries_from_csx(ptr, ind, got, false, label), label);
    }

    void check_aos_csc(rocsparse_handle    handle,
                       rocsparse_indextype src_type,
                       rocsparse_indextype col_ptr_type,
                       rocsparse_indextype row_ind_type)
    {
        const auto           entries = sample_matrix();
        std::vector<int64_t> ind;
        std::vector<float>   vals;
        for(const auto& entry : entries)
        {
            ind.push_back(entry.row);
            ind.push_back(entry.col);
            vals.push_back(entry.val);
        }
        HostIdx       aos(src_type, ind);
        DevBuf<float> aval(vals);
        HostIdx       csc_ptr(col_ptr_type, std::vector<int64_t>(3, 0));
        HostIdx       csc_row(row_ind_type, std::vector<int64_t>(entries.size(), 0));
        DevBuf<float> csc_val(std::vector<float>(entries.size(), 0.f));
        Spmat         source, target;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&source,
                                                 2,
                                                 2,
                                                 static_cast<int64_t>(entries.size()),
                                                 aos.mem.get(),
                                                 aval.get(),
                                                 src_type,
                                                 base,
                                                 f32_r),
                  st_ok);
        ASSERT_EQ(rocsparse_create_csc_descr(&target,
                                             2,
                                             2,
                                             static_cast<int64_t>(entries.size()),
                                             csc_ptr.mem.get(),
                                             csc_row.mem.get(),
                                             csc_val.get(),
                                             col_ptr_type,
                                             row_ind_type,
                                             base,
                                             f32_r),
                  st_ok);
        const std::string label = "coo_aos " + idx_name(src_type) + " -> csc ptr "
                                  + idx_name(col_ptr_type) + " row " + idx_name(row_ind_type);
        SCOPED_TRACE(label);
        size_t     column_bytes = 0;
        const auto status       = run_permissive(handle, source, target, &column_bytes);
        EXPECT_EQ(status, st_ok);
        if(status != st_ok)
        {
            return;
        }
        EXPECT_EQ(column_bytes, entries.size() * idx_width(row_ind_type));
        const auto ptr = download_idx(csc_ptr.mem.get(), col_ptr_type, 3);
        const auto row = download_idx(csc_row.mem.get(), row_ind_type, entries.size());
        const auto got = download_f(csc_val.get(), entries.size());
        expect_entries(entries_from_csx(ptr, row, got, true, label), label);
    }

    void check_csc_coo(rocsparse_handle    handle,
                       rocsparse_indextype col_ptr_type,
                       rocsparse_indextype row_ind_type,
                       rocsparse_indextype coo_type)
    {
        // CSC of sample_matrix(), rows within a column sorted.
        const std::vector<int64_t> ptr     = {0, 2, 3};
        const std::vector<int64_t> row_ind = {0, 1, 0};
        const std::vector<float>   vals    = {1.f, 3.f, 2.f};
        HostIdx                    csc_ptr(col_ptr_type, ptr);
        HostIdx                    csc_row(row_ind_type, row_ind);
        DevBuf<float>              csc_val(vals);
        HostIdx                    coo_row(coo_type, std::vector<int64_t>(3, 0));
        HostIdx                    coo_col(coo_type, std::vector<int64_t>(3, 0));
        DevBuf<float>              coo_val(std::vector<float>(3, 0.f));
        Spmat                      source, target;
        ASSERT_EQ(rocsparse_create_csc_descr(&source,
                                             2,
                                             2,
                                             3,
                                             csc_ptr.mem.get(),
                                             csc_row.mem.get(),
                                             csc_val.get(),
                                             col_ptr_type,
                                             row_ind_type,
                                             base,
                                             f32_r),
                  st_ok);
        ASSERT_EQ(rocsparse_create_coo_descr(&target,
                                             2,
                                             2,
                                             3,
                                             coo_row.mem.get(),
                                             coo_col.mem.get(),
                                             coo_val.get(),
                                             coo_type,
                                             base,
                                             f32_r),
                  st_ok);
        const std::string label = "csc ptr " + idx_name(col_ptr_type) + " row "
                                  + idx_name(row_ind_type) + " -> coo " + idx_name(coo_type);
        SCOPED_TRACE(label);
        size_t     column_bytes = 0;
        const auto status       = run_permissive(handle, source, target, &column_bytes);
        EXPECT_EQ(status, st_ok);
        if(status != st_ok)
        {
            return;
        }
        // Intermediate CSR column type for CSC is the CSC row-index type.
        EXPECT_EQ(column_bytes, static_cast<size_t>(3) * idx_width(row_ind_type));
        const auto rows = download_idx(coo_row.mem.get(), coo_type, 3);
        const auto cols = download_idx(coo_col.mem.get(), coo_type, 3);
        const auto got  = download_f(coo_val.get(), 3);
        expect_entries(entries_from_coo(rows, cols, got), label);
    }

    // --- symbolic ------------------------------------------------------------

    struct SymbolicPair
    {
        rocsparse_format source;
        rocsparse_format target;
    };

    // A null target value pointer is a symbolic conversion between the COO, COO AoS,
    // CSR, and CSC formats, including an identity copy, and in an ELL or BSR identity
    // copy. The index arrays are still written. The other ELL and BSR routes store
    // values unconditionally and are not listed.
    constexpr SymbolicPair k_symbolic[] = {
        {rocsparse_format_coo_aos, rocsparse_format_coo},
        {rocsparse_format_coo_aos, rocsparse_format_csr},
        {rocsparse_format_coo, rocsparse_format_coo_aos},
        {rocsparse_format_coo, rocsparse_format_csr},
        {rocsparse_format_csr, rocsparse_format_coo},
        {rocsparse_format_csr, rocsparse_format_csc},
        {rocsparse_format_csc, rocsparse_format_csr},
        {rocsparse_format_csc, rocsparse_format_coo},
        {rocsparse_format_csc, rocsparse_format_coo_aos},
        {rocsparse_format_csr, rocsparse_format_coo_aos},
        {rocsparse_format_coo, rocsparse_format_csc},
        {rocsparse_format_coo_aos, rocsparse_format_csc},
        {rocsparse_format_coo, rocsparse_format_coo},
        {rocsparse_format_csr, rocsparse_format_csr},
        {rocsparse_format_csc, rocsparse_format_csc},
        {rocsparse_format_coo_aos, rocsparse_format_coo_aos},
        {rocsparse_format_ell, rocsparse_format_ell},
        {rocsparse_format_bsr, rocsparse_format_bsr},
    };

    // nnz 0 accepts a null value pointer. set_pointers does not, so the value pointer
    // is installed only by the constructor and is not replaced after analysis.
    void create_shell(rocsparse_format format,
                      int64_t          nnz,
                      void*            primary,
                      void*            secondary,
                      void*            values,
                      Spmat&           descr)
    {
        if(format == rocsparse_format_coo)
        {
            ASSERT_EQ(rocsparse_create_coo_descr(
                          &descr, 1, 1, nnz, primary, secondary, values, idx32, base, f32_r),
                      st_ok);
        }
        else if(format == rocsparse_format_coo_aos)
        {
            ASSERT_EQ(rocsparse_create_coo_aos_descr(
                          &descr, 1, 1, nnz, primary, values, idx32, base, f32_r),
                      st_ok);
        }
        else if(format == rocsparse_format_csr)
        {
            ASSERT_EQ(rocsparse_create_csr_descr(
                          &descr, 1, 1, nnz, primary, secondary, values, idx32, idx32, base, f32_r),
                      st_ok);
        }
        else if(format == rocsparse_format_csc)
        {
            ASSERT_EQ(rocsparse_create_csc_descr(
                          &descr, 1, 1, nnz, primary, secondary, values, idx32, idx32, base, f32_r),
                      st_ok);
        }
        else if(format == rocsparse_format_ell)
        {
            ASSERT_EQ(
                rocsparse_create_ell_descr(&descr, 1, 1, primary, values, nnz, idx32, base, f32_r),
                st_ok);
        }
        else if(format == rocsparse_format_bsr)
        {
            ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                                 1,
                                                 1,
                                                 nnz,
                                                 dir_r,
                                                 1,
                                                 primary,
                                                 secondary,
                                                 values,
                                                 idx32,
                                                 idx32,
                                                 base,
                                                 f32_r),
                      st_ok);
        }
    }

    void run_symbolic(rocsparse_handle handle, const SymbolicPair& pair, bool with_values)
    {
        // Source holds entry (0, 0) = 3. Target indices start as 0xA5. A symbolic target
        // is created with nnz 0 and a null value pointer, which must stay null.
        clear_hip_error();
        const bool    src_aos = pair.source == rocsparse_format_coo_aos;
        const bool    tgt_aos = pair.target == rocsparse_format_coo_aos;
        Bytes         src_primary(src_aos || pair.source != rocsparse_format_coo ? 8 : 4);
        Bytes         src_secondary(4);
        const float   source_value = 3.f;
        DevBuf<float> source_values(std::vector<float>{source_value});
        DevBuf<float> numeric_values(1);

        if(pair.source == rocsparse_format_coo)
        {
            upload_i32(src_primary.get(), {0});
            upload_i32(src_secondary.get(), {0});
        }
        else if(src_aos)
        {
            upload_i32(src_primary.get(), {0, 0});
        }
        else
        {
            upload_i32(src_primary.get(), {0, 1});
            upload_i32(src_secondary.get(), {0});
        }
        Spmat source;
        create_shell(
            pair.source, 1, src_primary.get(), src_secondary.get(), source_values.get(), source);
        ASSERT_NE(source.get(), nullptr);

        Bytes dst_primary(tgt_aos || pair.target != rocsparse_format_coo ? 8 : 4);
        Bytes dst_secondary(4);
        Bytes dst_extra(4);
        fill_bytes(dst_primary, k_sentinel);
        fill_bytes(dst_secondary, k_sentinel);
        fill_bytes(dst_extra, k_sentinel);
        void* target_values = with_values ? numeric_values.get() : nullptr;
        Spmat target;
        // Numeric targets are created at the final nnz. A symbolic target has a null
        // value pointer, which the constructors accept only when nnz is 0; analysis
        // then stores the real nnz and leaves the value pointer null.
        create_shell(pair.target,
                     with_values ? 1 : 0,
                     dst_primary.get(),
                     dst_secondary.get(),
                     target_values,
                     target);
        ASSERT_NE(target.get(), nullptr);

        const std::string label = std::string(fmt_name(pair.source)) + " -> "
                                  + fmt_name(pair.target)
                                  + (with_values ? " numeric" : " symbolic");
        SCOPED_TRACE(label);

        S2sDescr descr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&descr, source, target, alg), st_ok);
        ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(descr), st_ok);
        size_t bytes = 0;
        ASSERT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_a, &bytes),
            st_ok);
        Bytes analysis(bytes);
        ASSERT_EQ(rocsparse_sparse_to_sparse(
                      handle, descr, source, target, stage_a, bytes, analysis.get()),
                  st_ok);
        ASSERT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_c, &bytes),
            st_ok);
        Bytes      compute(bytes);
        const auto status = rocsparse_sparse_to_sparse(
            handle, descr, source, target, stage_c, bytes, compute.get());
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const auto expect_values_kept = [&] {
            EXPECT_EQ(download_f(source_values.get(), 1)[0], source_value) << label;
            void* values = &values;
            EXPECT_EQ(rocsparse_spmat_get_values(target, &values), st_ok);
            if(with_values)
            {
                EXPECT_EQ(values, static_cast<void*>(numeric_values.get())) << label;
            }
            else
            {
                EXPECT_EQ(values, nullptr) << label;
            }
        };
        expect_values_kept();

        EXPECT_EQ(status, st_ok);
        if(status != st_ok)
        {
            return;
        }
        int64_t nnz = -1;
        EXPECT_EQ(rocsparse_spmat_get_nnz(target, &nnz), st_ok);
        EXPECT_EQ(nnz, 1);
        if(pair.target == rocsparse_format_coo)
        {
            const auto row = download_idx(dst_primary.get(), idx32, 1);
            const auto col = download_idx(dst_secondary.get(), idx32, 1);
            EXPECT_EQ(row[0], 0);
            EXPECT_EQ(col[0], 0);
        }
        else if(tgt_aos)
        {
            const auto ind = download_idx(dst_primary.get(), idx32, 2);
            EXPECT_EQ(ind[0], 0);
            EXPECT_EQ(ind[1], 0);
        }
        else if(pair.target == rocsparse_format_csr || pair.target == rocsparse_format_bsr)
        {
            const auto ptr = download_idx(dst_primary.get(), idx32, 2);
            const auto col = download_idx(dst_secondary.get(), idx32, 1);
            EXPECT_EQ(ptr[0], 0);
            EXPECT_EQ(ptr[1], 1);
            EXPECT_EQ(col[0], 0);
        }
        else if(pair.target == rocsparse_format_ell)
        {
            const auto col = download_idx(dst_primary.get(), idx32, 1);
            EXPECT_EQ(col[0], 0);
        }
        else if(pair.target == rocsparse_format_csc)
        {
            const auto ptr = download_idx(dst_primary.get(), idx32, 2);
            const auto row = download_idx(dst_secondary.get(), idx32, 1);
            EXPECT_EQ(ptr[0], 0);
            EXPECT_EQ(ptr[1], 1);
            EXPECT_EQ(row[0], 0);
        }
        if(with_values)
        {
            const auto got = download_f(numeric_values.get(), 1);
            EXPECT_EQ(got[0], 3.f) << label;
        }
    }

    // --- capture -------------------------------------------------------------

    struct CaptureCase
    {
        rocsparse_format source;
        rocsparse_format target;
        int64_t          m;
        int64_t          n;
        int64_t          nnz; // 0, or 1 for the entry (0, 0) = 4
        bool             alias; // identity whose target shares the source arrays
        bool             values; // false: null target value array
    };

    constexpr size_t k_capture_bytes = 64;

    void create_capture_descr(rocsparse_format format,
                              int64_t          m,
                              int64_t          n,
                              int64_t          nnz,
                              void*            a,
                              void*            b,
                              void*            c,
                              Spmat&           descr)
    {
        switch(format)
        {
        case rocsparse_format_coo:
            ASSERT_EQ(rocsparse_create_coo_descr(&descr, m, n, nnz, a, b, c, idx32, base, f32_r),
                      st_ok);
            break;
        case rocsparse_format_coo_aos:
            ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr, m, n, nnz, a, c, idx32, base, f32_r),
                      st_ok);
            break;
        case rocsparse_format_csr:
            ASSERT_EQ(
                rocsparse_create_csr_descr(&descr, m, n, nnz, a, b, c, idx32, idx32, base, f32_r),
                st_ok);
            break;
        default:
            FAIL() << "no capture descriptor for " << fmt_name(format);
        }
    }

    void run_capture(rocsparse_handle handle, const CaptureCase& cc)
    {
        const std::string label = std::string(fmt_name(cc.source)) + " -> " + fmt_name(cc.target)
                                  + " m " + std::to_string(cc.m) + " n " + std::to_string(cc.n)
                                  + " nnz " + std::to_string(cc.nnz) + (cc.alias ? " alias" : "")
                                  + (cc.values ? "" : " symbolic");
        SCOPED_TRACE(label);
        clear_hip_error();

        Bytes src_a(k_capture_bytes);
        Bytes src_b(k_capture_bytes);
        Bytes src_c(k_capture_bytes);
        Bytes dst_a(k_capture_bytes);
        Bytes dst_b(k_capture_bytes);
        Bytes dst_c(k_capture_bytes);
        if(cc.source == rocsparse_format_csr)
        {
            std::vector<int64_t> ptr(static_cast<size_t>(cc.m + 1), cc.nnz);
            ptr[0] = 0;
            upload_idx(src_a.get(), idx32, ptr);
        }
        if(cc.nnz > 0)
        {
            upload_f(src_c.get(), {4.f});
        }

        Bytes* const out[]
            = {cc.alias ? &src_a : &dst_a, cc.alias ? &src_b : &dst_b, cc.alias ? &src_c : &dst_c};
        const auto reset_target = [&] {
            if(!cc.alias)
            {
                fill_bytes(dst_a, k_sentinel);
                fill_bytes(dst_b, k_sentinel);
                fill_bytes(dst_c, k_sentinel);
            }
        };
        const auto snapshot = [&] {
            std::vector<std::vector<unsigned char>> bytes;
            for(const Bytes* mem : out)
            {
                bytes.push_back(mem->read());
            }
            return bytes;
        };
        reset_target();

        Spmat source, target;
        create_capture_descr(
            cc.source, cc.m, cc.n, cc.nnz, src_a.get(), src_b.get(), src_c.get(), source);
        create_capture_descr(cc.target,
                             cc.m,
                             cc.n,
                             0,
                             out[0]->get(),
                             out[1]->get(),
                             cc.values ? out[2]->get() : nullptr,
                             target);
        ASSERT_NE(source.get(), nullptr);
        ASSERT_NE(target.get(), nullptr);

        CaptureGuard guard(handle);
        ASSERT_EQ(hipStreamCreate(&guard.stream), hipSuccess);
        ASSERT_EQ(rocsparse_set_stream(handle, guard.stream), st_ok);

        S2sDescr descr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&descr, source, target, alg), st_ok);
        ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(descr), st_ok);
        size_t bytes = 0;
        ASSERT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_a, &bytes),
            st_ok);
        Bytes analysis(bytes);
        ASSERT_EQ(rocsparse_sparse_to_sparse(
                      handle, descr, source, target, stage_a, bytes, analysis.get()),
                  st_ok);
        ASSERT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_c, &bytes),
            st_ok);
        Bytes compute(bytes);

        // Eager result, then the same compute captured and replayed.
        ASSERT_EQ(rocsparse_sparse_to_sparse(
                      handle, descr, source, target, stage_c, bytes, compute.get()),
                  st_ok);
        ASSERT_EQ(hipStreamSynchronize(guard.stream), hipSuccess);
        const auto eager = snapshot();

        int64_t nnz = -1;
        EXPECT_EQ(rocsparse_spmat_get_nnz(target, &nnz), st_ok);
        EXPECT_EQ(nnz, cc.nnz);
        if(cc.target == rocsparse_format_csr)
        {
            const auto ptr = download_idx(out[0]->get(), idx32, static_cast<size_t>(cc.m + 1));
            EXPECT_EQ(ptr.front(), 0);
            EXPECT_EQ(ptr.back(), cc.nnz);
        }
        if(cc.nnz > 0)
        {
            const auto got = download_f(out[2]->get(), 1);
            EXPECT_EQ(got[0], 4.f);
        }

        reset_target();
        ASSERT_EQ(hipStreamBeginCapture(guard.stream, hipStreamCaptureModeGlobal), hipSuccess);
        const auto compute_status = rocsparse_sparse_to_sparse(
            handle, descr, source, target, stage_c, bytes, compute.get());
        const auto end_status = hipStreamEndCapture(guard.stream, &guard.graph);
        EXPECT_EQ(compute_status, st_ok);
        EXPECT_EQ(end_status, hipSuccess);
        if(compute_status != st_ok || end_status != hipSuccess)
        {
            return;
        }
        ASSERT_EQ(hipGraphInstantiate(&guard.exec, guard.graph, nullptr, nullptr, 0), hipSuccess);
        ASSERT_EQ(hipGraphLaunch(guard.exec, guard.stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(guard.stream), hipSuccess);
        EXPECT_EQ(snapshot(), eager);
    }

    constexpr auto coo = rocsparse_format_coo;
    constexpr auto aos = rocsparse_format_coo_aos;
    constexpr auto csr = rocsparse_format_csr;

    constexpr CaptureCase k_capture[] = {
        // Identity.
        {coo, coo, 1, 1, 0, false, true},
        {coo, coo, 1, 1, 0, true, true},
        {coo, coo, 1, 1, 1, false, true},
        {coo, coo, 1, 1, 1, true, true},
        {csr, csr, 1, 1, 0, false, true},
        {csr, csr, 1, 1, 0, true, true},
        {csr, csr, 1, 1, 1, false, true},
        {csr, csr, 1, 1, 1, true, true},
        // Empty conversions between formats.
        {coo, csr, 1, 1, 0, false, true},
        {csr, coo, 1, 1, 0, false, true},
        {coo, aos, 1, 1, 0, false, true},
        {aos, coo, 1, 1, 0, false, true},
        {aos, coo, 1, 1, 0, false, false},
        {aos, csr, 1, 1, 0, false, true},
        {aos, csr, 1, 1, 0, false, false},
        // No rows or no columns.
        {coo, coo, 0, 3, 0, false, true},
        {coo, coo, 3, 0, 0, false, true},
        {csr, csr, 0, 3, 0, false, true},
        {csr, csr, 3, 0, 0, false, true},
        {csr, csr, 0, 0, 0, false, true},
        {csr, csr, 0, 3, 0, true, true},
        {coo, csr, 0, 3, 0, false, true},
        {coo, csr, 3, 0, 0, false, true},
        {csr, coo, 3, 0, 0, false, true},
        // Nonempty conversions between formats.
        {coo, csr, 1, 1, 1, false, true},
        {csr, coo, 1, 1, 1, false, true},
    };

    // --- count overflow ------------------------------------------------------

    void expect_count_rejected(rocsparse_handle      handle,
                               rocsparse_spmat_descr source,
                               rocsparse_spmat_descr target,
                               const std::string&    label)
    {
        SCOPED_TRACE(label);
        clear_hip_error();
        S2sDescr descr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&descr, source, target, alg), st_ok);
        ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(descr), st_ok);
        size_t bytes = 99;
        EXPECT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_a, &bytes),
            rocsparse_status_invalid_size);
        EXPECT_EQ(bytes, static_cast<size_t>(0));
        // The intermediate descriptor is created after its arrays are allocated.
        EXPECT_EQ(descr.get()->m_intermediate, nullptr);
    }

    // --- BSR expansion -------------------------------------------------------

    constexpr int64_t k_wide_block_dim = 16;
    constexpr int64_t k_wide_block_nnz = k_wide_block_dim * k_wide_block_dim;

    // A 1 x nb BSR matrix whose only block is in the last block column, and a COO
    // target with int64_t indices for its expansion.
    struct WideBsr
    {
        HostIdx       ptr;
        HostIdx       col;
        DevBuf<float> val;
        HostIdx       coo_row;
        HostIdx       coo_col;
        DevBuf<float> coo_val;
        Spmat         source;
        Spmat         target;

        WideBsr(rocsparse_indextype type, int64_t nb)
            : ptr(type, {0, 1})
            , col(type, {nb - 1})
            , val(std::vector<float>(k_wide_block_nnz, 1.f))
            , coo_row(idx64, std::vector<int64_t>(k_wide_block_nnz, 0))
            , coo_col(idx64, std::vector<int64_t>(k_wide_block_nnz, 0))
            , coo_val(k_wide_block_nnz)
        {
            EXPECT_EQ(rocsparse_create_bsr_descr(&source,
                                                 1,
                                                 nb,
                                                 1,
                                                 dir_r,
                                                 k_wide_block_dim,
                                                 ptr.mem.get(),
                                                 col.mem.get(),
                                                 val.get(),
                                                 type,
                                                 type,
                                                 base,
                                                 f32_r),
                      st_ok);
            EXPECT_EQ(rocsparse_create_coo_descr(&target,
                                                 k_wide_block_dim,
                                                 k_wide_block_dim * nb,
                                                 k_wide_block_nnz,
                                                 coo_row.mem.get(),
                                                 coo_col.mem.get(),
                                                 coo_val.get(),
                                                 idx64,
                                                 base,
                                                 f32_r),
                      st_ok);
        }

        // The target columns are the block_dim columns of the last block column.
        void expect_last_columns(int64_t nb) const
        {
            const auto got = download_idx(coo_col.mem.get(), idx64, k_wide_block_nnz);
            ASSERT_FALSE(got.empty());
            EXPECT_EQ(*std::min_element(got.begin(), got.end()), k_wide_block_dim * (nb - 1));
            EXPECT_EQ(*std::max_element(got.begin(), got.end()), k_wide_block_dim * nb - 1);
        }
    };

    TEST_F(ConvS2s, SellSourceEachStageWritesNothing)
    {
        auto source = make_tiny(rocsparse_format_sell, true);
        auto target = make_tiny(rocsparse_format_csr, false);
        ASSERT_TRUE(source && target);
        expect_rejected(handle, source->descr, target->descr, *target, "sell -> csr");
    }

    TEST_F(ConvS2s, SellTargetEachStageWritesNothing)
    {
        auto source = make_tiny(rocsparse_format_csr, true);
        auto target = make_tiny(rocsparse_format_sell, false);
        ASSERT_TRUE(source && target);
        expect_rejected(handle, source->descr, target->descr, *target, "csr -> sell");
    }

    TEST_F(ConvS2s, FormatOutsideTableWritesNothing)
    {
        // The public API cannot build these descriptors, so the format is set directly.
        for(const int value : {static_cast<int>(rocsparse_format_sell) + 1, -1})
        {
            for(const bool on_source : {true, false})
            {
                const std::string label = std::string(on_source ? "source" : "target") + " format "
                                          + std::to_string(value);
                SCOPED_TRACE(label);
                clear_hip_error();
                auto source = make_tiny(rocsparse_format_csr, true);
                auto target = make_tiny(rocsparse_format_csr, false);
                ASSERT_TRUE(source && target);
                Spmat& poked        = on_source ? source->descr : target->descr;
                poked.get()->format = static_cast<rocsparse_format>(value);
                expect_rejected(handle, source->descr, target->descr, *target, label);
                poked.get()->format = rocsparse_format_csr;
            }
        }
    }

    TEST_F(ConvS2s, FormatSweepEveryEnum)
    {
        for(rocsparse_format source_format : k_formats)
        {
            for(rocsparse_format target_format : k_formats)
            {
                const std::string label
                    = std::string(fmt_name(source_format)) + " -> " + fmt_name(target_format);
                SCOPED_TRACE(label);
                clear_hip_error();
                auto source = make_tiny(source_format, true);
                auto target = make_tiny(target_format, false);
                ASSERT_TRUE(source && target);
                ASSERT_NE(source->descr.get(), nullptr);
                ASSERT_NE(target->descr.get(), nullptr);
                if(format_unsupported(source_format) || format_unsupported(target_format))
                {
                    expect_rejected(handle, source->descr, target->descr, *target, label);
                    continue;
                }
                fill_bytes(target->a, 0);
                fill_bytes(target->b, 0);
                fill_bytes(target->c, 0);
                refresh_pointers(*target);
                S2sDescr descr;
                ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                              &descr, source->descr, target->descr, alg),
                          st_ok);
                ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(descr), st_ok);
                for(auto stage : {stage_a, stage_c})
                {
                    size_t bytes = 0;
                    ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                                  handle, descr, source->descr, target->descr, stage, &bytes),
                              st_ok)
                        << label;
                    Bytes scratch(bytes);
                    ASSERT_EQ(rocsparse_sparse_to_sparse(handle,
                                                         descr,
                                                         source->descr,
                                                         target->descr,
                                                         stage,
                                                         bytes,
                                                         scratch.get()),
                              st_ok)
                        << label;
                    if(stage == stage_a)
                    {
                        refresh_pointers(*target);
                    }
                }
                expect_one_entry(*target, label);
            }
        }
    }

    TEST_F(ConvS2s, IndexWidthThroughIntermediate)
    {
        constexpr rocsparse_indextype types[] = {idx32, idx64};
        for(auto src : types)
        {
            for(auto a : types)
            {
                for(auto b : types)
                {
                    check_coo_csc(handle, src, a, b);
                    check_coo_bsr(handle, src, a, b);
                    check_aos_csc(handle, src, a, b);
                    check_csc_coo(handle, a, b, src);
                }
            }
        }
    }

    TEST_F(ConvS2s, SymbolicNullTargetValues)
    {
        for(const auto& pair : k_symbolic)
        {
            run_symbolic(handle, pair, true);
            run_symbolic(handle, pair, false);
        }
    }

    TEST_F(ConvS2s, IntermediateCountOverflow)
    {
        // Each case has one count whose byte size, or one product of counts, does not fit.
        // The host pointers are never read: the count is rejected before any allocation.
        int dummy = 0;
        {
            // Columns: 2^61 int64 entries.
            constexpr int64_t nnz
                = static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(int64_t)) + 1;
            Spmat source, target;
            ASSERT_EQ(rocsparse_create_coo_descr(
                          &source, 4, 1LL << 60, nnz, &dummy, &dummy, &dummy, idx32, base, f32_r),
                      st_ok);
            ASSERT_EQ(
                rocsparse_create_csc_descr(
                    &target, 4, 1LL << 60, nnz, &dummy, &dummy, &dummy, idx64, idx64, base, f32_r),
                st_ok);
            expect_count_rejected(handle, source, target, "column bytes");
        }
        {
            // Values: 2^60 complex double entries; the int32 column bytes fit.
            constexpr int64_t nnz
                = static_cast<int64_t>(std::numeric_limits<size_t>::max() / (2 * sizeof(double)))
                  + 1;
            Spmat source, target;
            ASSERT_EQ(rocsparse_create_coo_descr(
                          &source, 4, 1LL << 60, nnz, &dummy, &dummy, &dummy, idx32, base, f64_c),
                      st_ok);
            ASSERT_EQ(
                rocsparse_create_csc_descr(
                    &target, 4, 1LL << 60, nnz, &dummy, &dummy, &dummy, idx64, idx32, base, f64_c),
                st_ok);
            expect_count_rejected(handle, source, target, "value bytes");
        }
        {
            // Row pointers: 2^61 + 1 int64 entries.
            constexpr int64_t rows = 1LL << 61;
            Spmat             source, target;
            ASSERT_EQ(rocsparse_create_coo_descr(
                          &source, rows, 2, 1, &dummy, &dummy, &dummy, idx64, base, f32_r),
                      st_ok);
            ASSERT_EQ(rocsparse_create_csc_descr(
                          &target, rows, 2, 1, &dummy, &dummy, &dummy, idx64, idx64, base, f32_r),
                      st_ok);
            expect_count_rejected(handle, source, target, "row pointer bytes");
        }
        {
            // BSR: 2^62 blocks of 3 x 3 entries.
            constexpr int64_t brows = 1LL << 61;
            constexpr int64_t bnnz  = 1LL << 62;
            Spmat             source, target;
            ASSERT_EQ(rocsparse_create_bsr_descr(&source,
                                                 brows,
                                                 2,
                                                 bnnz,
                                                 dir_r,
                                                 3,
                                                 &dummy,
                                                 &dummy,
                                                 &dummy,
                                                 idx64,
                                                 idx64,
                                                 base,
                                                 f32_r),
                      st_ok);
            ASSERT_EQ(rocsparse_create_coo_descr(
                          &target, 3 * brows, 6, 0, &dummy, &dummy, &dummy, idx64, base, f32_r),
                      st_ok);
            expect_count_rejected(handle, source, target, "bsr entry count");
        }

        // A valid conversion still succeeds after the rejected ones.
        auto source = make_tiny(rocsparse_format_csr, true);
        auto target = make_tiny(rocsparse_format_csc, false);
        ASSERT_TRUE(source && target);
        EXPECT_EQ(run_permissive(handle, source->descr, target->descr, nullptr), st_ok);
        expect_one_entry(*target, "csr -> csc after overflow");
    }

    TEST_F(ConvS2s, BsrExpansionFitsIndexType)
    {
        // The intermediate CSR matrix keeps the int32_t source column type, so an
        // expansion past the int32_t range is rejected before anything is allocated.
        constexpr int64_t too_wide = 1LL << 28;
        {
            WideBsr bsr(idx32, too_wide);
            expect_count_rejected(handle, bsr.source, bsr.target, "i32 2^28 block columns");
        }
        {
            // One block in 2^28 block rows. The host pointers are never read: the
            // expansion is rejected before any allocation.
            int   dummy = 0;
            Spmat source, target;
            ASSERT_EQ(rocsparse_create_bsr_descr(&source,
                                                 too_wide,
                                                 1,
                                                 1,
                                                 dir_r,
                                                 k_wide_block_dim,
                                                 &dummy,
                                                 &dummy,
                                                 &dummy,
                                                 idx32,
                                                 idx32,
                                                 base,
                                                 f32_r),
                      st_ok);
            ASSERT_EQ(rocsparse_create_coo_descr(&target,
                                                 k_wide_block_dim * too_wide,
                                                 k_wide_block_dim,
                                                 k_wide_block_nnz,
                                                 &dummy,
                                                 &dummy,
                                                 &dummy,
                                                 idx64,
                                                 base,
                                                 f32_r),
                      st_ok);
            expect_count_rejected(handle, source, target, "i32 2^28 block rows");
        }
        {
            // The widest int32_t expansion still converts.
            constexpr int64_t widest = std::numeric_limits<int32_t>::max() / k_wide_block_dim;
            WideBsr           bsr(idx32, widest);
            SCOPED_TRACE("i32 widest");
            ASSERT_EQ(run_permissive(handle, bsr.source, bsr.target, nullptr), st_ok);
            bsr.expect_last_columns(widest);
        }
        {
            // int64_t indices hold the expansion that int32_t cannot.
            WideBsr bsr(idx64, too_wide);
            SCOPED_TRACE("i64 2^28 block columns");
            ASSERT_EQ(run_permissive(handle, bsr.source, bsr.target, nullptr), st_ok);
            bsr.expect_last_columns(too_wide);
        }
    }

    TEST_F(ConvS2s, EmptyEllThroughIntermediate)
    {
        // A 0-row ELL still needs a one-element intermediate CSR row pointer.
        // csr_set_pointers rejects a null one, including when m is 0.
        Spmat source, target;
        ASSERT_EQ(
            rocsparse_create_ell_descr(&source, 0, 3, nullptr, nullptr, 0, idx32, base, f32_r),
            st_ok);
        ASSERT_EQ(rocsparse_create_coo_descr(
                      &target, 0, 3, 0, nullptr, nullptr, nullptr, idx32, base, f32_r),
                  st_ok);
        S2sDescr descr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(&descr, source, target, alg), st_ok);
        ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(descr), st_ok);
        size_t bytes = 0;
        ASSERT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_a, &bytes),
            st_ok);
        Bytes analysis(bytes);
        ASSERT_EQ(rocsparse_sparse_to_sparse(
                      handle, descr, source, target, stage_a, bytes, analysis.get()),
                  st_ok);
        ASSERT_EQ(
            rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, stage_c, &bytes),
            st_ok);
        Bytes compute(bytes);
        ASSERT_EQ(rocsparse_sparse_to_sparse(
                      handle, descr, source, target, stage_c, bytes, compute.get()),
                  st_ok);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    TEST_F(ConvS2s, CaptureMatchesEager)
    {
        for(const auto& cc : k_capture)
        {
            run_capture(handle, cc);
        }
    }
}
