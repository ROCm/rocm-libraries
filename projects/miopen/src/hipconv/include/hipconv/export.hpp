#pragma once

// hipconv links statically, so on Windows its API carries no decoration.
//
// __declspec on a static library's symbols is inert, and the amdgcn device pass does not
// support the attribute, so it warns once per declaration in every HIP translation unit
// that includes the public header.
#ifdef _WIN32
#define HIPCONV_API
#else
#define HIPCONV_API __attribute__((visibility("default")))
#endif
