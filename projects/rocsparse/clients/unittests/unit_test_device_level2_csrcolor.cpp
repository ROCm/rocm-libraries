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

//
// Device unit tests of the graph coloring rocsparse_Xcsrcolor with a fraction of the
// nodes to color below one: the remaining nodes receive distinct new colors. The
// coloring is checked for what the documentation promises: it is a proper coloring
// of the graph (no two adjacent nodes share a color), the colors lie in
// [0, ncolors) and the optional reordering is a permutation.
//
#include "unit_test_device_level2_common.hpp"

using namespace level2;

namespace
{
    // Adjacency (without self loops) of a w x h grid graph with 4 neighbours and,
    // optionally, one extra long edge per node.
    host_csr<float> make_grid_graph(int32_t w, int32_t h, bool long_edges, int base)
    {
        const int32_t                  m = w * h;
        std::vector<std::set<int32_t>> adj(m);
        auto                           edge = [&](int32_t a, int32_t b) {
            if(a != b)
            {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        };
        for(int32_t y = 0; y < h; ++y)
            for(int32_t x = 0; x < w; ++x)
            {
                const int32_t i = y * w + x;
                if(x + 1 < w)
                    edge(i, i + 1);
                if(y + 1 < h)
                    edge(i, i + w);
                if(long_edges)
                    edge(i, (i * 7 + 13) % m);
            }
        host_csr<float> a;
        a.m    = m;
        a.n    = m;
        a.base = 0;
        a.ptr.push_back(0);
        for(int32_t i = 0; i < m; ++i)
        {
            for(int32_t j : adj[i])
            {
                a.ind.push_back(j);
                a.val.push_back(1.0f);
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    template <typename T>
    void run_coloring(rocsparse_handle       handle,
                      const host_csr<float>& graph,
                      double                 fraction,
                      bool                   with_reordering)
    {
        using R = typename real_of<T>::type;
        SCOPED_TRACE(::testing::Message()
                     << "csrcolor m=" << graph.m << " nnz=" << graph.nnz() << " base=" << graph.base
                     << " fraction=" << fraction << " reordering=" << with_reordering);
        const rocsparse_int m = graph.m;

        std::vector<T> val(graph.val.size());
        for(size_t k = 0; k < val.size(); ++k)
            val[k] = rocsparse_ut::scalar<T>(graph.val[k]);

        device_vector<rocsparse_int> drow(convert_vector<rocsparse_int>(graph.ptr)),
            dcol(convert_vector<rocsparse_int>(graph.ind));
        device_vector<T>             dval(val);
        device_vector<rocsparse_int> dcoloring(std::vector<rocsparse_int>(m, -1)),
            dreordering(std::vector<rocsparse_int>(m, -1));
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dcoloring.ptr && dreordering.ptr);

        mat_descr descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr.d, to_base(graph.base)),
                  rocsparse_status_success);
        mat_info info;
        ASSERT_TRUE(info.d);

        // fraction_to_color and ncolors live in host memory.
        const R       frac    = static_cast<R>(fraction);
        rocsparse_int ncolors = 0;
        ASSERT_EQ(csrcolor<T>(handle,
                              m,
                              graph.nnz(),
                              descr.d,
                              dval.ptr,
                              drow.ptr,
                              dcol.ptr,
                              &frac,
                              &ncolors,
                              dcoloring.ptr,
                              with_reordering ? dreordering.ptr : nullptr,
                              info.d),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<rocsparse_int> colors = to_host(dcoloring);
        ASSERT_GE(ncolors, 1);
        ASSERT_LE(ncolors, m);
        for(int32_t i = 0; i < m; ++i)
        {
            ASSERT_GE(colors[i], 0) << "node " << i;
            ASSERT_LT(colors[i], ncolors) << "node " << i;
        }
        // Proper coloring: adjacent nodes have different colors.
        for(int32_t i = 0; i < m; ++i)
            for(int32_t k = graph.ptr[i] - graph.base; k < graph.ptr[i + 1] - graph.base; ++k)
            {
                const int32_t j = graph.ind[k] - graph.base;
                if(j != i)
                    ASSERT_NE(colors[i], colors[j]) << "nodes " << i << " and " << j;
            }

        if(with_reordering)
        {
            std::vector<rocsparse_int> perm = to_host(dreordering);
            std::sort(perm.begin(), perm.end());
            for(int32_t i = 0; i < m; ++i)
                ASSERT_EQ(perm[i], i) << "reordering is not a permutation";
        }
    }

    class L2DeviceCsrcolor : public rocsparse_ut::HandleTest
    {
    };

    TEST_F(L2DeviceCsrcolor, partial_fraction_float)
    {
        const auto grid = make_grid_graph(40, 40, false, 0);
        for(const double fraction : {0.05, 0.3, 0.6, 0.9, 1.0})
        {
            run_coloring<float>(handle, grid, fraction, false);
            if(HasFailure())
                return;
        }
        const auto rough = make_grid_graph(33, 30, true, 1);
        for(const double fraction : {0.1, 0.5})
        {
            run_coloring<float>(handle, rough, fraction, true);
            if(HasFailure())
                return;
        }
    }

    TEST_F(L2DeviceCsrcolor, partial_fraction_other_precisions)
    {
        const auto grid = make_grid_graph(40, 40, false, 0);
        run_coloring<double>(handle, grid, 0.5, false);
        run_coloring<rocsparse_float_complex>(handle, grid, 0.5, true);
        run_coloring<rocsparse_double_complex>(handle, grid, 0.5, false);
    }

}
