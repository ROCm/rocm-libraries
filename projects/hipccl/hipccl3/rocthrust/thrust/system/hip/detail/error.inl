/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <thrust/system/hip/detail/guarded_hip_runtime_api.h>
#include <thrust/system/hip/error.h>

THRUST_BEGIN_NS
namespace system
{

error_code make_error_code(hip_rocprim::errc::errc_t e)
{
    return error_code(static_cast<int>(e), hip_category());
} // end make_error_code()

error_condition make_error_condition(hip_rocprim::errc::errc_t e)
{
    return error_condition(static_cast<int>(e), hip_category());
} // end make_error_condition()

namespace hip_rocprim
{
namespace detail
{
    class hip_error_category : public error_category
    {
    public:
        inline hip_error_category(void) {}

        inline virtual const char* name(void) const
        {
            return "hip";
        }

        inline virtual std::string message(int ev) const
        {      
            char const* const unknown_str  = "unknown error";
            char const* const unknown_name = "hipErrorUnknown";
            char const* c_str  = ::hipGetErrorString(static_cast<hipError_t>(ev));
            char const* c_name = ::hipGetErrorName(static_cast<hipError_t>(ev));
            return std::string(c_name ? c_name : unknown_name)
              + ": " + (c_str ? c_str : unknown_str);
        }

        inline virtual error_condition default_error_condition(int ev) const
        {
            using namespace hip_rocprim::errc;

            if(ev < (::hipErrorMissingConfiguration - 1) /* ::hipErrorApiFailureBase */)
            {
                return make_error_condition(static_cast<errc_t>(ev));
            }

            return system_category().default_error_condition(ev);
        }
    }; // end hip_error_category

} // end detail

} // end namespace hip_rocprim

const error_category& hip_category(void)
{
    static const thrust::system::hip_rocprim::detail::hip_error_category result;
    return result;
}

} // end namespace system
THRUST_END_NS
