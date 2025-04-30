// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <numeric>
#include <unordered_set>

#include "array_validator.h"
#include "increment.h"

// Check a 2D array for collisions.
// The 2D case can be determined via a number-theoretic argument.
bool valid_length_stride_2d(const size_t l0, const size_t l1, const size_t s0, const size_t s1)
{
    if(s0 == s1)
        return false;
    const auto c = std::lcm(s0, s1);
    return !((s0 * (l0 - 1) >= c) && (s1 * (l1 - 1) >= c));
}

// Compare a 1D direction with a multi-index hyperface for collisions.
bool valid_length_stride_1d_multi(const unsigned int        idx,
                                  const std::vector<size_t> l,
                                  const std::vector<size_t> s,
                                  const int                 verbose)
{
    size_t              l0{0}, s0{0};
    std::vector<size_t> l1{}, s1{};
    for(unsigned int i = 0; i < l.size(); ++i)
    {
        if(i == idx)
        {
            l0 = l[i];
            s0 = s[i];
        }
        else
        {
            l1.push_back(l[i]);
            s1.push_back(s[i]);
        }
    }

    if(verbose > 4)
    {
        std::cout << "l0: " << l0 << "\ts0: " << s0 << std::endl;
    }

    // We only need to go to the maximum pointer offset for (l1,s1).
    const auto max_offset
        = std::accumulate(l1.begin(), l1.end(), (size_t)1, std::multiplies<size_t>())
          - std ::inner_product(l1.begin(), l1.end(), s1.begin(), (size_t)0);
    std::unordered_set<size_t> a0{};
    for(size_t i = 1; i < l0; ++i)
    {
        const auto val = i * s0;
        if(val <= max_offset)
            a0.insert(val);
        else
            break;
    }

    if(verbose > 5)
    {
        std::cout << "a0:";
        for(auto i : a0)
            std::cout << " " << i;
        std::cout << std::endl;

        std::cout << "l1:";
        for(auto i : l1)
            std::cout << " " << i;
        std::cout << std::endl;

        std::cout << "s1:";
        for(auto i : s1)
            std::cout << " " << i;
        std::cout << std::endl;
    }

    // TODO: this can be multi-threaded, since find(...) is thread-safe.
    std::vector<size_t> index(l1.size());
    std::fill(index.begin(), index.end(), 0);
    do
    {
        const int i = std::inner_product(index.begin(), index.end(), s1.begin(), (size_t)0);
        if(i > 0 && (i % s0 == 0))
        {
            // TODO: use an ordered set and binary search
            if(verbose > 6)
                std::cout << i << std::endl;
            if(a0.find(i) != a0.end())
            {
                if(verbose > 4)
                {
                    std::cout << "l0: " << l0 << "\ts0: " << s0 << std::endl;
                    std::cout << "l1:";
                    for(const auto li : l1)
                        std::cout << " " << li;
                    std::cout << " s1:";
                    for(const auto si : s1)
                        std::cout << " " << si;
                    std::cout << std::endl;
                    std::cout << "Found duplicate: " << i << std::endl;
                }
                return false;
            }
        }
    } while(increment_rowmajor(index, l1));

    return true;
}

// Compare a hyperface with another hyperface for collisions.
bool valid_length_stride_multi_multi(const std::vector<size_t> l0,
                                     const std::vector<size_t> s0,
                                     const std::vector<size_t> l1,
                                     const std::vector<size_t> s1)
{
    std::unordered_set<size_t> a0{};

    const auto max_offset
        = std::accumulate(l1.begin(), l1.end(), (size_t)1, std::multiplies<size_t>())
          - std::inner_product(l1.begin(), l1.end(), s1.begin(), (size_t)0);
    std::vector<size_t> index0(l0.size()); // TODO: check this
    std::fill(index0.begin(), index0.end(), 0);
    do
    {
        const auto i = std::inner_product(index0.begin(), index0.end(), s0.begin(), (size_t)0);
        if(i > max_offset)
            a0.insert(i);
    } while(increment_rowmajor(index0, l0));

    std::vector<size_t> index1(l1.size());
    std::fill(index1.begin(), index1.end(), 0);
    do
    {
        const auto i = std::inner_product(index1.begin(), index1.end(), s1.begin(), (size_t)0);
        if(i > 0)
        {
            // TODO: use an ordered set and binary search
            if(a0.find(i) != a0.end())
            {

                return false;
            }
        }
    } while(increment_rowmajor(index1, l1));

    return true;
}

bool valid_length_stride_3d(const std::vector<size_t>& l,
                            const std::vector<size_t>& s,
                            const int                  verbose)
{
    // Check that 2D faces are valid:
    if(!valid_length_stride_2d(l[0], l[1], s[0], s[1]))
        return false;
    if(!valid_length_stride_2d(l[0], l[2], s[0], s[2]))
        return false;
    if(!valid_length_stride_2d(l[1], l[2], s[1], s[2]))
        return false;

    // If the 2D faces are valid, check an axis vs a face for collisions:
    bool invalid = false;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int idx = 0; idx < 3; ++idx)
    {
        if(!valid_length_stride_1d_multi(idx, l, s, verbose))
        {
#ifdef _OPENMP
#pragma omp cancel for
#endif
            invalid = true;
        }
    }
    if(invalid)
        return false;
    return true;
}

bool valid_length_stride_4d(const std::vector<size_t>& l,
                            const std::vector<size_t>& s,
                            const int                  verbose)
{
    if(l.size() != 4)
    {
        throw std::runtime_error("Incorrect dimensions for valid_length_stride_4d");
    }

    // Check that 2D faces are valid:
    for(int idx0 = 0; idx0 < 3; ++idx0)
    {
        for(int idx1 = idx0 + 1; idx1 < 4; ++idx1)
        {
            if(!valid_length_stride_2d(l[idx0], l[idx1], s[idx0], s[idx1]))
                return false;
        }
    }

    bool invalid = false;
    // Check that 1D vs 3D faces are valid:
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int idx0 = 0; idx0 < 4; ++idx0)
    {
        if(!valid_length_stride_1d_multi(idx0, l, s, verbose))
        {
#ifdef _OPENMP
#pragma omp cancel for
#endif
            invalid = true;
        }
    }
    if(invalid)
        return false;

    // Check that 2D vs 2D faces are valid:

    // First, get all the permutations
    std::vector<std::vector<size_t>> perms;
    std::vector<size_t>              v(l.size());
    std::fill(v.begin(), v.begin() + 2, 0);
    std::fill(v.begin() + 2, v.end(), 1);
    do
    {
        perms.push_back(v);
        if(verbose > 3)
        {
            std::cout << "v:";
            for(const auto i : v)
            {
                std::cout << " " << i;
            }
            std::cout << "\n";
        }
    } while(std::next_permutation(v.begin(), v.end()));

    // Then loop over all of the permutations.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(size_t iperm = 0; iperm < perms.size(); ++iperm)
    {
        std::vector<size_t> l0(2);
        std::vector<size_t> s0(2);
        std::vector<size_t> l1(2);
        std::vector<size_t> s1(2);
        for(size_t i = 0; i < l.size(); ++i)
        {
            if(perms[iperm][i] == 0)
            {
                l0.push_back(l[i]);
                s0.push_back(s[i]);
            }
            else
            {
                l1.push_back(l[i]);
                s1.push_back(s[i]);
            }
        }

        if(verbose > 3)
        {
            std::cout << "\tl0:";
            for(const auto i : l0)
            {
                std::cout << " " << i;
            }
            std::cout << "\n";
            std::cout << "\ts0:";
            for(const auto i : s0)
            {
                std::cout << " " << i;
            }
            std::cout << "\n";
            std::cout << "\tl1:";
            for(const auto i : l1)
            {
                std::cout << " " << i;
            }
            std::cout << "\n";
            std::cout << "\ts1:";
            for(const auto i : s1)
            {
                std::cout << " " << i;
            }
            std::cout << "\n";
        }

        if(!valid_length_stride_multi_multi(l0, s0, l1, s1))
        {
#ifdef _OPENMP
#pragma omp cancel for
#endif
            invalid = true;
        }
    }
    if(invalid)
        return false;

    return true;
}

bool valid_length_stride_generald(const std::vector<size_t> l,
                                  const std::vector<size_t> s,
                                  const int                 verbose)
{
    if(verbose > 2)
    {
        std::cout << "checking dimension " << l.size() << std::endl;
    }

    // Recurse on d-1 hyper-faces:
    for(unsigned int idx = 0; idx < l.size(); ++idx)
    {
        std::vector<size_t> l0{};
        std::vector<size_t> s0{};
        for(size_t i = 0; i < l.size(); ++i)
        {
            if(i != idx)
            {
                l0.push_back(l[i]);
                s0.push_back(s[i]);
            }
        }
        if(!array_valid(l0, s0, verbose))
            return false;
    }

    // Handle the 1D vs (N-1) case:
    for(unsigned int idx = 0; idx < l.size(); ++idx)
    {
        if(!valid_length_stride_1d_multi(idx, l, s, verbose))
            return false;
    }

    for(size_t dim0 = 2; dim0 <= l.size() / 2; ++dim0)
    {
        const size_t dim1 = l.size() - dim0;
        if(verbose > 2)
            std::cout << "dims: " << dim0 << " " << dim1 << std::endl;

        // We iterate over all permutations of an array of length l.size() which contains dim0 zeros
        // and dim1 ones.  We start with {0, ..., 0, 1, ... 1} to guarantee that we hit all the
        // possibilities.

        // First, get all the permutations
        std::vector<std::vector<size_t>> perms;
        std::vector<size_t>              v(l.size());
        std::fill(v.begin(), v.begin() + dim1, 0);
        std::fill(v.begin() + dim1, v.end(), 1);
        do
        {
            perms.push_back(v);
            if(verbose > 3)
            {
                std::cout << "v:";
                for(const auto i : v)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
            }

        } while(std::next_permutation(v.begin(), v.end()));

        bool invalid = false;
        // Then loop over all of the permutations.
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t iperm = 0; iperm < perms.size(); ++iperm)
        {
            std::vector<size_t> l0(dim0);
            std::vector<size_t> s0(dim0);
            std::vector<size_t> l1(dim1);
            std::vector<size_t> s1(dim1);

            for(size_t i = 0; i < l.size(); ++i)
            {
                if(v[i] == 0)
                {
                    l0.push_back(l[i]);
                    s0.push_back(s[i]);
                }
                else
                {
                    l1.push_back(l[i]);
                    s1.push_back(s[i]);
                }
            }

            if(verbose > 3)
            {
                std::cout << "\tl0:";
                for(const auto i : l0)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
                std::cout << "\ts0:";
                for(const auto i : s0)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
                std::cout << "\tl1:";
                for(const auto i : l1)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
                std::cout << "\ts1:";
                for(const auto i : s1)
                {
                    std::cout << " " << i;
                }
                std::cout << "\n";
            }

            if(!valid_length_stride_multi_multi(l0, s0, l1, s1))
            {
#ifdef _OPENMP
#pragma omp cancel for
#endif
                invalid = true;
            }
        }
        if(invalid)
            return false;
    }

    return true;
}

bool sort_by_stride(const std::pair<size_t, size_t>& ls0, const std::pair<size_t, size_t>& ls1)
{
    return ls0.second < ls1.second;
}

bool array_valid(const std::vector<size_t>& length,
                 const std::vector<size_t>& stride,
                 const int                  verbose)
{
    if(length.size() != stride.size())
        return false;

    // If a length is 1, then the stride is irrelevant.
    // If a length is > 1, then the corresponding stride must be > 1.
    std::vector<size_t> l{}, s{};
    for(unsigned int i = 0; i < length.size(); ++i)
    {
        if(length[i] > 1)
        {
            if(stride[i] == 0)
                return false;
            l.push_back(length[i]);
            s.push_back(stride[i]);
        }
    }

    if(length.size() > 1)
    {
        // Check happy path.
        bool                                   happy_path = true;
        std::vector<std::pair<size_t, size_t>> ls;
        for(size_t idx = 0; idx < length.size(); ++idx)
        {
            ls.push_back(std::pair(length[idx], stride[idx]));
        }
        std::sort(ls.begin(), ls.end(), sort_by_stride);

        if(verbose > 2)
        {
            for(size_t idx = 0; idx < ls.size(); ++idx)
            {
                std::cout << ls[idx].first << "\t" << ls[idx].second << "\n";
            }
        }

        for(size_t idx = 1; idx < ls.size(); ++idx)
        {
            if(ls[idx].second < ls[idx - 1].first * ls[idx - 1].second)
            {
                happy_path = false;
                break;
            }
        }
        if(happy_path)
        {
            if(verbose > 2)
            {
                std::cout << "happy path\n";
            }
            return true;
        }
    }

    switch(l.size())
    {
    case 0:
        return true;
        break;
    case 1:
        return s[0] != 0;
        break;
    case 2:
    {
        return valid_length_stride_2d(l[0], l[1], s[0], s[1]);
        break;
    }
    case 3:
    {
        return valid_length_stride_3d(l, s, verbose);
        break;
    }
    case 4:
    {
        return valid_length_stride_4d(l, s, verbose);
        break;
    }
    default:
        return valid_length_stride_generald(l, s, verbose);
        return true;
    }

    return true;
}
