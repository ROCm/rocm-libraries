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

#ifndef ROCWMMA_DEVICE_VECTOR_ITERATOR_HPP
#define ROCWMMA_DEVICE_VECTOR_ITERATOR_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool bcastTest()
    {
        bool err = false;

        // Vector data unused
        //non_native_vector_base<DataT, VecSize> vec{static_cast<DataT>(5.0f)};
        HIP_vector_type<DataT, VecSize> vec{static_cast<DataT>(5.0f)};

        for(int i = 0; i < VecSize; i++)
        {
            //err |= (vec.d[i] != static_cast<DataT>(5.0f));
            err |= (vec.data[i] != static_cast<DataT>(5.0f));
        }

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorIndexTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;

        // Check iterator range
        auto it = makeVectorIterator(vec).begin();
        err     = err || (VecSize != it.range());

        // Check that the index increments properly
        for(uint32_t i = 0; i < it.range(); i++, it++)
        {
            err = err || (i != it.index());
        }

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              uint32_t StrideSize,
              typename std::enable_if<(StrideSize >= 1)>::type* = nullptr>
    ROCWMMA_DEVICE static inline bool iteratorStrideIndexTest(VecT<DataT, VecSize> const& vec)
    {
        bool err = false;

        // Iterate over vector in strides of half
        auto it  = makeVectorIterator<StrideSize>(vec).begin();
        auto end = makeVectorIterator<StrideSize>(vec).end();

        // Check iterator range
        err = err || (VecSize != (it.range() * StrideSize));

        // Check that the index increments properly
        for(uint32_t i = 0; i < it.range(); i++, it++)
        {
            err = err || (i != it.index());
        }

        // Should have reached the end
        err = err || (it != end);

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              uint32_t StrideSize,
              typename std::enable_if<(StrideSize == 0)>::type* = nullptr>
    ROCWMMA_DEVICE static inline bool iteratorStrideIndexTest(VecT<DataT, VecSize> const& vec)
    {
        return false;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorStrideIndexTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;

        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 2>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 4>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 8>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 16>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 32>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 64>(vec);
        err = err || iteratorStrideIndexTest<DataT, VecSize, VecSize / 128>(vec);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorValueTest()
    {
        bool err = false;

        // Init data as linear values
        VecT<DataT, VecSize> vec;
        for(uint32_t i = 0; i < VecSize; i++)
        {
            vec.data[i] = static_cast<DataT>(i);
        }

        // Iterate over vector
        auto it = makeVectorIterator(vec).begin();

        err = err || (VecSize != it.range());
        for(uint32_t i = 0; i < it.range(); i++, it++)
        {
            // 0th element check as the iterator stride = 1
            err = err || (vec.data[i] != (*it).data[0]);
        }

        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              uint32_t StrideSize,
              typename std::enable_if<(StrideSize >= 1)>::type* = nullptr>
    ROCWMMA_DEVICE static inline bool iteratorStrideValueTest(VecT<DataT, VecSize> const& vec)
    {
        bool err = false;
        auto it  = makeVectorIterator<StrideSize>(vec).begin();

        // Check range over stride
        err = err || (VecSize != (it.range() * StrideSize));

        // Check values over iteration
        for(uint32_t i = 0; i < it.range(); i++, it++)
        {
            for(uint32_t j = 0; j < StrideSize; j++)
            {
                err = err || (vec.data[i * StrideSize + j] != (*it).data[j]);
            }
        }
        return err;
    }

    template <typename DataT,
              uint32_t VecSize,
              uint32_t StrideSize,
              typename std::enable_if<(StrideSize == 0)>::type* = nullptr>
    ROCWMMA_DEVICE static inline bool iteratorStrideValueTest(VecT<DataT, VecSize> const& v)
    {
        return false;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorStrideValueTest()
    {
        bool err = false;

        // Init data as linear values
        VecT<DataT, VecSize> vec;
        for(uint32_t i = 0; i < VecSize; i++)
        {
            vec.data[i] = static_cast<DataT>(i);
        }

        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 2>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 4>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 8>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 16>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 32>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 64>(vec);
        err = err || iteratorStrideValueTest<DataT, VecSize, VecSize / 128>(vec);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorSubVectorTypeTest()
    {
        bool err = false;

        // Uninitialized vec
        VecT<DataT, VecSize> vec;

        // Iterate over vector in strides of half
        constexpr uint32_t iterSize = std::max(VecSize / 2u, 1u);
        auto               it       = makeVectorIterator<iterSize>(vec).begin();

        return err || (!std::is_same<std::decay_t<decltype(*it)>, VecT<DataT, iterSize>>::value);
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorBeginTest()
    {
        bool err = false;

        // Uninitialized vec
        VecT<DataT, VecSize> vec;

        // Iterate over vector in strides of 1
        auto it = makeVectorIterator(vec).begin();

        // Begins at idx 0
        err = err || (it.index() != 0u);

        // Is valid
        err = err || (it.valid() != true);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorEndTest()
    {
        bool err = false;

        // Uninitialized vec
        VecT<DataT, VecSize> vec;

        // Iterate over vector in strides of 1
        auto it = makeVectorIterator(vec).end();

        // Begins at idx VecSize
        err = err || (it.index() != VecSize);

        // Is not valid
        err = err || (it.valid() != false);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorItTest()
    {
        bool err = false;

        // Uninitialized vec
        VecT<DataT, VecSize> vec;

        // Iterate over vector in strides of 1
        const uint32_t idx = std::min(VecSize - 1u, 2u);
        auto           it  = makeVectorIterator(vec).it(idx);

        // Begins at idx
        err = err || (it.index() != idx);

        // Is valid
        err = err || (it.valid() != true);

        // Iterate over vector in strides of 1
        auto it1 = makeVectorIterator(vec).it(VecSize + 1u);

        // Begins at idx VecSize + 1
        err = err || (it1.index() != (VecSize + 1u));

        // Is not valid
        err = err || (it1.valid() != false);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorPostIncDecTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;

        auto it = makeVectorIterator(vec).begin();

        auto it1 = it++;

        // Check that indices are not same
        err = err || (it.index() == it1.index());

        // Check index difference is +1
        err = err || ((it.index() - it1.index()) != 1);

        auto it2 = it--;

        // Check that indices are not same
        err = err || (it.index() == it2.index());

        // Check index difference is -1
        err = err || ((it.index() - it2.index()) != -1);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorPreIncDecTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;

        auto it = makeVectorIterator(vec).begin();

        // As a copy
        auto it1 = ++it;

        // Check that indices are same
        err = err || (it.index() != it1.index());

        // Check index difference is 0
        err = err || ((it.index() - it1.index()) != 0);

        // As a reference
        auto& it2 = --it;

        // Check that indices are same
        err = err || (it.index() != it2.index());

        // Check that indices not the same
        err = err || (it1.index() == it2.index());

        // Check index difference is 0
        err = err || ((it.index() - it2.index()) != 0);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorPlusMinusTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;

        auto it = makeVectorIterator(vec).begin();

        auto it1 = it + 5;

        // Check that index has not changed
        err = err || (it.index() == 5);

        // Check that new index has changed
        err = err || (it1.index() != 5);

        // Check that indices are not same
        err = err || (it.index() == it1.index());

        // Check index difference is -5
        err = err || ((it.index() - it1.index()) != -5);

        auto it2 = it1 - 2;

        // Check that old index has not changed
        err = err || (it1.index() == 3);

        // Check that old index has not changed
        err = err || (it2.index() != 3);

        // Check that indices are not same
        err = err || (it1.index() == it2.index());

        // Check index difference is +2
        err = err || ((it1.index() - it2.index()) != 2);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorPlusMinusEqTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;

        auto it = makeVectorIterator(vec).begin();

        // As a copy
        auto it1 = it += 5;

        // Check that index has changed
        err = err || (it.index() != 5);

        // Check that indices are same
        err = err || (it.index() != it1.index());

        // Check index difference is 0
        err = err || ((it.index() - it1.index()) != 0);

        // As a reference
        auto& it2 = it -= 2;

        // Check that index has changed
        err = err || (it.index() != 3);

        // Check that indices are same
        err = err || (it.index() != it2.index());

        // Check that copy is different than ref
        err = err || (it1.index() == it2.index());

        // Check that copy is different than ref
        err = err || ((it1.index() - it2.index()) != 2);

        // Check index difference is 0
        err = err || ((it.index() - it2.index()) != 0);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorEqTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;
        VecT<DataT, VecSize> vec1;

        auto it  = makeVectorIterator(vec).begin();
        auto it1 = makeVectorIterator(vec).it(0);

        auto it2 = makeVectorIterator(vec1).begin();
        auto it3 = makeVectorIterator(vec1).end();
        auto it4 = makeVectorIterator(vec1).it(VecSize);

        // Check idx = 0 and begin on same vec
        err = err || (it != it1);

        // Check that begin on diff vec is not same
        err = err || (it == it2);

        // Check that begin is not end
        err = err || (it2 == it3);

        // Check idx = VecSize is end
        err = err || (it3 != it4);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE static inline bool iteratorRangeBasedForTest()
    {
        bool err = false;

        // Vector data unused
        VecT<DataT, VecSize> vec;
        auto                 count = 0u;
        for(auto const& l : makeVectorIterator(vec))
        {
            count++;
        }

        for(auto& l : makeVectorIterator(vec))
        {
            count++;
        }

        return err || (count != (VecSize * 2));
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_KERNEL void vectorIteratorTest(uint32_t     m,
                                           uint32_t     n,
                                           DataT const* in,
                                           DataT*       out,
                                           uint32_t     ld,
                                           DataT        param1,
                                           DataT        param2)
    {
        __shared__ int result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        err = err ? err : bcastTest<DataT, VecSize>();

        err = err ? err : iteratorIndexTest<DataT, VecSize>();

        err = err ? err : iteratorStrideIndexTest<DataT, VecSize>();

        err = err ? err : iteratorStrideValueTest<DataT, VecSize>();

        err = err ? err : iteratorSubVectorTypeTest<DataT, VecSize>();

        err = err ? err : iteratorBeginTest<DataT, VecSize>();

        err = err ? err : iteratorEndTest<DataT, VecSize>();

        err = err ? err : iteratorItTest<DataT, VecSize>();

        err = err ? err : iteratorPostIncDecTest<DataT, VecSize>();

        err = err ? err : iteratorPreIncDecTest<DataT, VecSize>();

        err = err ? err : iteratorPlusMinusTest<DataT, VecSize>();

        err = err ? err : iteratorPlusMinusEqTest<DataT, VecSize>();

        err = err ? err : iteratorEqTest<DataT, VecSize>();

        err = err ? err : iteratorRangeBasedForTest<DataT, VecSize>();

        // Reduce error count
        atomicAdd(&result, (int)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_VECTOR_ITERATOR_HPP
