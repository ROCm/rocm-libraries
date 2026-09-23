#ifndef DEMANGLE_HPP
#define DEMANGLE_HPP

#include <string>

#include <miopen/export.h>

namespace miopen {

MIOPEN_EXPORT std::string demangle(const char* name);

} // namespace miopen

#endif // DEMANGLE_HPP
