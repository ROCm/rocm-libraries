#pragma once

// Exports an internal declaration that tests call, when libhipconv is a shared library.
#ifdef HIPCONV_EXPORT_INTERNALS
#define HIPCONV_INTERNAL_API __attribute__((visibility("default")))
#else
#define HIPCONV_INTERNAL_API
#endif
