#include <iostream>

int main()
{
    for(int x = 0; x < 128; ++x)
    {
        // std::cout << "asm volatile(\"ds_read_b32 v" << (x + 1) << ", %0 offset:" << (x * 0)
        //           << "\" : : \"v\"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));\n";
        std::cout << "asm volatile(\"ds_read_b32 v" << (x + 1) << ", %0 offset:" << (x * 0)
                  << "\" : : \"v\"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));\n";
    }
}