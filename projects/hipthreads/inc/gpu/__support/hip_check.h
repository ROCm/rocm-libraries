#ifndef __GPU___SUPPORT_HIP_CHECK_H__
#define __GPU___SUPPORT_HIP_CHECK_H__

#include <string>
#include <stdexcept>

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define __LIBGPU_HIP_CHECK__(cmd)                                                                                      \
    {                                                                                                                  \
        hipError_t error = cmd;                                                                                        \
        if (error != hipSuccess) {                                                                                     \
            throw std::runtime_error(std::string("[" __FILE__ ":" STRINGIZE(__LINE__) "] ") +                          \
                                                 hipGetErrorString(error));                                            \
        }                                                                                                              \
    }

#endif // __GPU___SUPPORT_HIP_CHECK_H__
