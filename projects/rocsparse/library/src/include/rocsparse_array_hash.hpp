#include <hip/hip_runtime.h>
//
inline uint32_t rocsparse_array_hash(const void* data, size_t size)
{
    void* buffer = malloc(size);
    std::ignore = hipMemcpy(buffer, data, size, hipMemcpyDefault);
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
