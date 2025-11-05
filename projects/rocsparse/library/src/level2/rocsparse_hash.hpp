#include "rocsparse_csrsv.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_csrsv_solve_kernel.hpp"
#include "rocsparse_utility.hpp"

inline uint32_t rocsparse_array_hash(const void* data, size_t size)
{
    void* buffer = malloc(size);
    hipMemcpy(buffer, data, size, hipMemcpyDefault);
    const unsigned char* byte_data = static_cast<const unsigned char*>(buffer);
    uint32_t             hash      = 0x811C9DC5; // FNV_prime_32
    for(size_t i = 0; i < size; ++i)
    {
        hash ^= byte_data[i];
        hash *= 0x01000193; // FNV_offset_basis_32
    }
    free(buffer);
    return hash;
}
