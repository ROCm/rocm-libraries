// rocPRIM should have at least minimal support for this configuration,
// because projects already depend on it, so regression should be avoided.
// Some algorithms won't work with `half` in this setting e.g. histogram,
// but it should at least compile.

#define __HIP_NO_HALF_CONVERSIONS__ 1
#define __HIP_NO_HALF_OPERATORS__ 1

#include <rocprim/rocprim.hpp>

int main() {}
