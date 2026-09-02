#if defined(__has_include)
#if __has_include(<cxxabi.h>)
#define MIOPEN_HAS_CXXABI_H
#endif
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#define MIOPEN_HAS_CXXABI_H
#endif

#if defined(MIOPEN_HAS_CXXABI_H)
#include <cstdlib>
#include <cxxabi.h>
#endif

#include <memory>

#include <miopen/demangle.hpp>

namespace miopen {

#if defined(MIOPEN_HAS_CXXABI_H)

using DemangledNamePtr = std::unique_ptr<char, decltype(&free)>;

std::string demangle(const char* name)
{
    std::string s;
    DemangledNamePtr demangled_name(abi::__cxa_demangle(name, nullptr, nullptr, nullptr), free);

    if(demangled_name)
    {
        s = demangled_name.get();
    }
    else
    {
        s = name;
    }

    return s;
}

#else

std::string demangle(const char* name) { return name; }

#endif

} // namespace miopen
