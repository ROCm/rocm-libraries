/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_TEST_UNIT_MEMORY_2DARRAY_HPP
#define ROCWMMA_TEST_UNIT_MEMORY_2DARRAY_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rocwmma
{
    namespace test
    {
        namespace references
        {
            template <class T>
            class RegisterBank
            {
            private:
                std::vector<std::vector<T>> array;
                int                         height;
                int                         width;

            public:
                RegisterBank<T>(int height, int width);
                RegisterBank<T>(std::vector<std::vector<T>> const& array);
                RegisterBank<T>();

                int  getHeight() const;
                int  getWidth() const;
                T    get(int h, int w) const;
                void print() const;
                void copyTo(T* data) const;

                void setData(T const* data);
                void gatherOddEven();
                void rotateR(int group, int distance, std::pair<int, int> range);
                void zipLoHi(int group);
                void concatHorizon(RegisterBank const& m);

                void unpackLo1();
                void unpackLo2();
                void unpackLo4();
                void unpackLo8();
                void unpackLo16();
                void unpackLo32();
                void unpackHi1();
                void unpackHi2();
                void unpackHi4();
                void unpackHi8();
                void unpackHi16();
                void unpackHi32();
                void unpackLoHi1();
                void unpackLoHi2();
                void unpackLoHi4();
                void unpackLoHi8();
                void unpackLoHi16();
                void unpackLoHi32();
            };

            template <class T>
            RegisterBank<T>::RegisterBank(int height, int width)
            {
                this->height = height;
                this->width  = width;
                this->array  = std::vector<std::vector<T>>(height, std::vector<T>(width));
            }

            template <class T>
            RegisterBank<T>::RegisterBank(std::vector<std::vector<T>> const& array)
            {
                this->height = array.size();
                this->width  = array.size() > 0 ? array[0].size() : 0;
                this->array  = array;
            }

            template <class T>
            RegisterBank<T>::RegisterBank()
            {
                height = 0;
                width  = 0;
            }

            template <class T>
            int RegisterBank<T>::getHeight() const
            {
                return height;
            }

            template <class T>
            int RegisterBank<T>::getWidth() const
            {
                return width;
            }

            template <class T>
            T RegisterBank<T>::get(int h, int w) const
            {
                if(!(h >= 0 && h < height && w >= 0 && w < width))
                    throw std::invalid_argument("Index out of bounds.");

                return array[h][w];
            }

            template <class T>
            void RegisterBank<T>::print() const
            {
                printf("matrix\n");
                for(auto&& row : array)
                {
                    for(auto v : row)
                    {
                        printf("%f, ", (double)v);
                    }
                    printf("\n");
                }
            }

            template <class T>
            void RegisterBank<T>::setData(T const* data)
            {
                size_t offset = 0;
                for(auto&& row : array)
                {
                    std::copy(data + offset, data + offset + row.size(), row.begin());
                    offset += row.size();
                }
            }

            template <class T>
            void RegisterBank<T>::copyTo(T* data) const
            {
                size_t offset = 0;
                for(auto&& row : array)
                {
                    std::copy(row.begin(), row.end(), data + offset);
                    offset += row.size();
                }
            }

            template <class T>
            void RegisterBank<T>::gatherOddEven()
            {
                for(auto& row : array)
                {
                    std::vector<T> odd(row.size() / 2);
                    for(int i = 0; i < row.size(); i++)
                    {
                        if(i % 2 == 0)
                        {
                            row[i / 2] = row[i];
                        }
                        else
                        {
                            odd[i / 2] = row[i];
                        }
                    }
                    std::copy(odd.begin(), odd.end(), row.begin() + row.size() / 2);
                }
            }

            template <class T>
            void RegisterBank<T>::rotateR(int group, int distance, std::pair<int, int> range)
            {
                assert(array.size() % group == 0 && distance <= group);
                decltype(array) temp(array);

                for(int i = 0; i < temp.size() / group; i++)
                {
                    auto start = temp.begin() + i * group;
                    std::rotate(start, start + (group - distance), start + group);
                }

                for(int i = 0; i < array.size(); i++)
                {
                    std::copy(temp[i].begin() + range.first,
                              temp[i].begin() + range.second,
                              array[i].begin() + range.first);
                }
            }

            template <class T>
            void RegisterBank<T>::zipLoHi(int group)
            {
                bool selectLo = true;
                for(int i = 0; i < height; i++)
                {
                    if((i / group) % 2 == 1)
                    {
                        std::copy(array[i].begin() + width / 2, array[i].end(), array[i].begin());
                    }
                    array[i].resize(width / 2);
                }
                width /= 2;
            }

            template <class T>
            void RegisterBank<T>::concatHorizon(RegisterBank const& m)
            {
                assert(height == m.getHeight());
                for(int i = 0; i < height; i++)
                {
                    array[i].resize(width + m.getWidth());
                    std::copy(m.array[i].begin(), m.array[i].end(), array[i].begin() + width);
                }
                width += m.getWidth();
            }

            template <class T>
            void RegisterBank<T>::unpackLo1()
            {
                gatherOddEven();
                rotateR(16, 1, {width / 2, width});
                zipLoHi(1);
            }

            template <class T>
            void RegisterBank<T>::unpackLo2()
            {
                gatherOddEven();
                rotateR(16, 2, {width / 2, width});
                zipLoHi(2);
            }

            template <class T>
            void RegisterBank<T>::unpackLo4()
            {
                gatherOddEven();
                rotateR(16, 4, {width / 2, width});
                zipLoHi(4);
            }

            template <class T>
            void RegisterBank<T>::unpackLo8()
            {
                gatherOddEven();
                rotateR(16, 8, {width / 2, width});
                zipLoHi(8);
            }

            template <class T>
            void RegisterBank<T>::unpackLo16()
            {
                gatherOddEven();
                rotateR(32, 16, {width / 2, width});
                zipLoHi(16);
            }

            template <class T>
            void RegisterBank<T>::unpackLo32()
            {
                gatherOddEven();
                rotateR(64, 32, {width / 2, width});
                zipLoHi(32);
            }

            template <class T>
            void RegisterBank<T>::unpackHi1()
            {
                gatherOddEven();
                rotateR(16, 15, {0, width / 2});
                zipLoHi(1);
            }

            template <class T>
            void RegisterBank<T>::unpackHi2()
            {
                gatherOddEven();
                rotateR(16, 14, {0, width / 2});
                zipLoHi(2);
            }

            template <class T>
            void RegisterBank<T>::unpackHi4()
            {
                gatherOddEven();
                rotateR(16, 12, {0, width / 2});
                zipLoHi(4);
            }

            template <class T>
            void RegisterBank<T>::unpackHi8()
            {
                gatherOddEven();
                rotateR(16, 8, {0, width / 2});
                zipLoHi(8);
            }

            template <class T>
            void RegisterBank<T>::unpackHi16()
            {
                gatherOddEven();
                rotateR(32, 16, {0, width / 2});
                zipLoHi(16);
            }

            template <class T>
            void RegisterBank<T>::unpackHi32()
            {
                gatherOddEven();
                rotateR(64, 32, {0, width / 2});
                zipLoHi(32);
            }

            template <class T>
            void RegisterBank<T>::unpackLoHi1()
            {
                RegisterBank<T> lo(array);
                RegisterBank<T> hi(array);
                lo.unpackLo1();
                hi.unpackHi1();
                lo.concatHorizon(hi);
                array = lo.array;
            }

            template <class T>
            void RegisterBank<T>::unpackLoHi2()
            {
                RegisterBank<T> lo(array);
                RegisterBank<T> hi(array);
                lo.unpackLo2();
                hi.unpackHi2();
                lo.concatHorizon(hi);
                array = lo.array;
            }

            template <class T>
            void RegisterBank<T>::unpackLoHi4()
            {
                RegisterBank<T> lo(array);
                RegisterBank<T> hi(array);
                lo.unpackLo4();
                hi.unpackHi4();
                lo.concatHorizon(hi);
                array = lo.array;
            }

            template <class T>
            void RegisterBank<T>::unpackLoHi8()
            {
                RegisterBank<T> lo(array);
                RegisterBank<T> hi(array);
                lo.unpackLo8();
                hi.unpackHi8();
                lo.concatHorizon(hi);
                array = lo.array;
            }

            template <class T>
            void RegisterBank<T>::unpackLoHi16()
            {
                RegisterBank<T> lo(array);
                RegisterBank<T> hi(array);
                lo.unpackLo16();
                hi.unpackHi16();
                lo.concatHorizon(hi);
                array = lo.array;
            }

            template <class T>
            void RegisterBank<T>::unpackLoHi32()
            {
                RegisterBank<T> lo(array);
                RegisterBank<T> hi(array);
                lo.unpackLo32();
                hi.unpackHi32();
                lo.concatHorizon(hi);
                array = lo.array;
            }
        }
    }
}
#endif //ROCWMMA_TEST_UNIT_MEMORY_2DARRAY_HPP
