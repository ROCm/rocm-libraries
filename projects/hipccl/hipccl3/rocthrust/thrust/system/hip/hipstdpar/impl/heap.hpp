/*
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

/*! \file thrust/system/hip/hipstdpar/impl/heap.hpp
 *  \brief <tt>Heap operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

// rocThrust includes

// STL includes

namespace std
{
    // BEGIN IS_HEAP
    // TODO: UNIMPLEMENTED IN THRUST
    // END IS_HEAP

    // BEGIN IS_HEAP_UNTIL
    // TODO: UNIMPLEMENTED IN THRUST
    // END IS_HEAP_UNTIL
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
