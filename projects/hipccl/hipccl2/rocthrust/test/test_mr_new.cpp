/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/mr/new.h>
#include <thrust/fill.h>

#include "test_header.hpp"

template<typename MemoryResource>
void TestAlignment(MemoryResource memres, std::size_t size, std::size_t alignment)
{
    void * ptr = memres.do_allocate(size, alignment);
    ASSERT_EQ(reinterpret_cast<std::size_t>(ptr) % alignment, 0u);

    char * char_ptr = reinterpret_cast<char *>(ptr);
    thrust::fill(char_ptr, char_ptr + size, 0);

    memres.do_deallocate(ptr, size, alignment);
}

static const std::size_t MinTestedSize = 32;
static const std::size_t MaxTestedSize = 8 * 1024;
static const std::size_t TestedSizeStep = 1;

static const std::size_t MinTestedAlignment = 16;
static const std::size_t MaxTestedAlignment = 4 * 1024;
static const std::size_t TestedAlignmentShift = 1;

TEST(MrAlignmentTests, TestNewDeleteResourceAlignedAllocation)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    for (std::size_t size = MinTestedSize; size <= MaxTestedSize; size += TestedSizeStep)
    {
        for (std::size_t alignment = MinTestedAlignment; alignment <= MaxTestedAlignment;
            alignment <<= TestedAlignmentShift)
        {
            TestAlignment(thrust::mr::new_delete_resource(), size, alignment);
        }
    }
}
