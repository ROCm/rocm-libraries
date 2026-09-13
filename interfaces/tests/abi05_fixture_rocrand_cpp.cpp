#include <cstdio>
#include <rocrand/rocrand.hpp>

extern "C" const char* abi05_force_cpp_error(int status) {
    static rocrand_cpp::error err(static_cast<rocrand_status>(status));
    return err.what();
}
