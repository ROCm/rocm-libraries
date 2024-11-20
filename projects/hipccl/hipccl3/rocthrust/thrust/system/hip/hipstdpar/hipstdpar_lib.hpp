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

/*! \file thrust/system/hip/hipstdpar/hipstdpar_lib.hpp
 *  \brief Forwarding header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

    // Interposed allocations
#if defined(__HIPSTDPAR_INTERPOSE_ALLOC__)
    #include "impl/interpose_allocations.hpp"
#endif
    // Parallel STL algorithms
    #include "impl/batch.hpp"
    #include "impl/copy.hpp"
    #include "impl/generation.hpp"
    #include "impl/heap.hpp"
    #include "impl/lexicographical_comparison.hpp"
    #include "impl/merge.hpp"
    #include "impl/min_max.hpp"
    #include "impl/numeric.hpp"
    #include "impl/order_changing.hpp"
    #include "impl/partitioning.hpp"
    #include "impl/removing.hpp"
    #include "impl/search.hpp"
    #include "impl/set.hpp"
    #include "impl/sorting.hpp"
    #include "impl/swap.hpp"
    #include "impl/transformation.hpp"
    #include "impl/uninitialized.hpp"
    
#endif
