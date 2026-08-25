/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include <vector>

bool valid_permutation(int m, const int* permutation)
{
    std::vector<int> check(m, 0);

    for(int i = 0; i < m; i++)
    {
        const int val = permutation[i];
        if(val < 0 || val >= m)
        {
            // value is out of bounds
            return false;
        }

        if(check[val] > 0)
        {
            // duplicate value
            return false;
        }

        check[val]++;
    }

    return true;
}

bool valid_coloring(int        m,
                    const int* csr_ptr,
                    const int* csr_ind,
                    int        num_colors,
                    const int* size_colors,
                    const int* permutation)
{
    /*
    *   Create Inverse Permutation
    */
    std::vector<int> p_inverse(m);
    for(int i = 0; i < m; i++)
    {
        p_inverse[permutation[i]] = i;
    }

    std::vector<int> mark(m, 0);

    int perm_in = 0;
    for(int i = 0; i < num_colors; i++)
    {
        // mark nodes in the color group
        for(int j = perm_in; j < perm_in + size_colors[i]; j++)
        {
            const int check_node = p_inverse[j];
            mark[check_node]     = 1;
        }

        // check the node's adjacent nodes
        for(int j = perm_in; j < perm_in + size_colors[i]; j++)
        {
            const int parent_node = p_inverse[j];

            for(int k = csr_ptr[parent_node]; k < csr_ptr[parent_node + 1]; k++)
            {
                const int adj_node = csr_ind[k];

                // invalid coloring if adjacent node is in the same color
                if(mark[adj_node] != 0 && adj_node != parent_node)
                {
                    return false;
                }
            }
        }

        // unmark nodes in the color group
        for(int j = perm_in; j < perm_in + size_colors[i]; j++)
        {
            const int check_node = p_inverse[j];
            mark[check_node]     = 0;
        }
        perm_in += size_colors[i];
    }

    return true;
}
