// Copyright (C) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef EXAMPLEUTILS_H
#define EXAMPLEUTILS_H

std::ostream& operator<<(std::ostream& stream, hipDoubleComplex c)
{
    stream << "(" << c.x << "," << c.y << ")";
    return stream;
}

// Increment the index (column-major) for looping over arbitrary dimensional loops with
// dimensions length.
template <class T1, class T2>
bool increment_cm(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(unsigned int idim = 0; idim < length.size(); ++idim)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            break;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

// Output a formatted general-dimensional array with given length and stride in batches
// separated by dist, in column-major order.
template <class Tdata, class Tint1, class Tint2>
void printbuffer_cm(const std::vector<Tdata>& data,
                    const std::vector<Tint1>& length,
                    const std::vector<Tint2>& stride,
                    const size_t              nbatch,
                    const size_t              dist)
{
    for(size_t b = 0; b < nbatch; b++)
    {
        std::vector<size_t> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const auto i = std::inner_product(index.begin(), index.end(), stride.begin(), b * dist);
            assert(i >= 0);
            assert(i < data.size());

            std::cout << data[i] << " ";

            for(size_t idx = 0; idx < index.size(); ++idx)
            {
                if(index[idx] == (length[idx] - 1))
                {
                    std::cout << "\n";
                }
                else
                {
                    break;
                }
            }
        } while(increment_cm(index, length));
        std::cout << std::endl;
    }
}

// Check that an multi-dimensional array of complex values with dimensions length
// and straide stride, with nbatch copies separated by dist is Hermitian-symmetric.
// Column-major version.
template <class Tcomplex, class Tint1, class Tint2>
bool check_symmetry_cm(const std::vector<Tcomplex>& data,
                       const std::vector<Tint1>&    length_cm,
                       const std::vector<Tint2>&    stride_cm,
                       const size_t                 nbatch,
                       const size_t                 dist,
                       const bool                   verbose = true)
{
    bool issymmetric = true;
    for(size_t b = 0; b < nbatch; b++)
    {
        std::vector<size_t> index(length_cm.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            bool skip = false;

            std::vector<size_t> negindex(index.size());
            for(size_t idx = 0; idx < index.size(); ++idx)
            {
                if(index[0] > length_cm[0] / 2)
                {
                    skip = true;
                    break;
                }
                negindex[idx] = (length_cm[idx] - index[idx]) % length_cm[idx];
            }
            if(negindex[0] > length_cm[0] / 2)
            {
                skip = true;
            }

            if(!skip)
            {
                const auto i
                    = std::inner_product(index.begin(), index.end(), stride_cm.begin(), b * dist);
                const auto j = std::inner_product(
                    negindex.begin(), negindex.end(), stride_cm.begin(), b * dist);
                if((data[i].x != data[j].x) or (data[i].y != -data[j].y))
                {
                    if(verbose)
                    {
                        std::cout << "(";
                        std::string separator;
                        for(auto val : index)
                        {
                            std::cout << separator << val;
                            separator = ",";
                        }
                        std::cout << ")->";
                        std::cout << i << "\t";
                        std::cout << "(";
                        separator = "";
                        for(auto val : negindex)
                        {
                            std::cout << separator << val;
                            separator = ",";
                        }
                        std::cout << ")->";
                        std::cout << j << ":\t";
                        std::cout << data[i] << " " << data[j];
                        std::cout << "\tnot conjugate!" << std::endl;
                    }
                    issymmetric = false;
                }
            }

        } while(increment_cm(index, length_cm));
    }
    return issymmetric;
}

#endif /* EXAMPLEUTILS_H */
