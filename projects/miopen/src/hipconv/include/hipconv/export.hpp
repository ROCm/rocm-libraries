#pragma once

#ifdef _WIN32
#ifdef HIPCONV_BUILDING_LIBRARY
#define HIPCONV_API __declspec(dllexport)
#else
#define HIPCONV_API __declspec(dllimport)
#endif
#else
#define HIPCONV_API __attribute__((visibility("default")))
#endif
