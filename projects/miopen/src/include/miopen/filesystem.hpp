// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_MIOPEN_FILESYSTEM_HPP_
#define GUARD_MIOPEN_FILESYSTEM_HPP_

#include <string>
#include <string_view>

// clang-format off
#if defined(CPPCHECK)
  #define MIOPEN_HAS_FILESYSTEM 1
  #define MIOPEN_HAS_FILESYSTEM_TS 1
#elif defined(_WIN32)
  #if _MSC_VER >= 1920
    #define MIOPEN_HAS_FILESYSTEM 1
    #define MIOPEN_HAS_FILESYSTEM_TS 0
  #elif _MSC_VER >= 1900
    #define MIOPEN_HAS_FILESYSTEM 0
    #define MIOPEN_HAS_FILESYSTEM_TS 1
  #else
    #define MIOPEN_HAS_FILESYSTEM 0
    #define MIOPEN_HAS_FILESYSTEM_TS 0
  #endif
#elif defined(__has_include)
  #if __has_include(<filesystem>) && __cplusplus >= 201703L
    #define MIOPEN_HAS_FILESYSTEM 1
  #else
    #define MIOPEN_HAS_FILESYSTEM 0
  #endif
  #if __has_include(<experimental/filesystem>) && __cplusplus >= 201103L
    #define MIOPEN_HAS_FILESYSTEM_TS 1
  #else
    #define MIOPEN_HAS_FILESYSTEM_TS 0
  #endif
#else
  #define MIOPEN_HAS_FILESYSTEM 0
  #define MIOPEN_HAS_FILESYSTEM_TS 0
#endif
// clang-format on

#if MIOPEN_HAS_FILESYSTEM
#include <filesystem>
#elif MIOPEN_HAS_FILESYSTEM_TS
#include <experimental/filesystem>
#else
#error "No filesystem include available"
#endif

namespace miopen {

#if MIOPEN_HAS_FILESYSTEM
namespace fs = ::std::filesystem;
#elif MIOPEN_HAS_FILESYSTEM_TS
namespace fs = ::std::experimental::filesystem;
#endif

} // namespace miopen

inline std::string operator+(const std::string_view s, const miopen::fs::path& path)
{
    return path.string().insert(0, s);
}

inline std::string operator+(const miopen::fs::path& path, const std::string_view s)
{
    return path.string().append(s);
}

#if !MIOPEN_HAS_FILESYSTEM && MIOPEN_HAS_FILESYSTEM_TS

#include <climits>
#include <cstdlib>
#include <vector>

#if defined(__linux__) || defined(__linux) || defined(linux)

#define MAX_PATH_LENGTH PATH_MAX

#define get_full_path(relative_path_std_string, result_std_vector) \
    realpath(relative_path_std_string.c_str(), result_std_vector.data())

#elif defined(_WIN32) // defined(__linux__) || defined(__linux) || defined(linux)

#define MAX_PATH_LENGTH _MAX_PATH

#define get_full_path(relative_path_std_string, result_std_vector) \
    _fullpath(                                                     \
        result_std_vector.data(), relative_path_std_string.c_str(), result_std_vector.size() - 1)

#else // defined(__linux__) || defined(__linux) || defined(linux)
#error "Function weakly_canonical() not implemented for the current platform!"
#endif // defined(__linux__) || defined(__linux) || defined(linux)

namespace miopen {

inline fs::path weakly_canonical(const fs::path& path)
{
    std::vector<char> result(MAX_PATH_LENGTH + 1, '\0');
    const std::string p{path.is_relative() ? (fs::current_path() / path).string() : path.string()};
    const char* retval = get_full_path(p, result);
    return (retval == nullptr) ? path : fs::path{result.data()};
}

} // namespace miopen

#else  // !MIOPEN_HAS_FILESYSTEM && MIOPEN_HAS_FILESYSTEM_TS
namespace miopen {
inline fs::path weakly_canonical(const fs::path& path) { return fs::weakly_canonical(path); }
} // namespace miopen
#endif // !MIOPEN_HAS_FILESYSTEM && MIOPEN_HAS_FILESYSTEM_TS

namespace miopen {

#ifdef _WIN32
constexpr std::string_view executable_postfix{".exe"};
constexpr std::string_view library_prefix{""};
constexpr std::string_view dynamic_library_postfix{".dll"};
constexpr std::string_view static_library_postfix{".lib"};
constexpr std::string_view object_file_postfix{".obj"};
#else
constexpr std::string_view executable_postfix{""};
constexpr std::string_view library_prefix{"lib"};
constexpr std::string_view dynamic_library_postfix{".so"};
constexpr std::string_view static_library_postfix{".a"};
constexpr std::string_view object_file_postfix{".o"};
#endif

inline fs::path make_executable_name(const fs::path& path)
{
    return path.parent_path() / (path.filename() + executable_postfix);
}

inline fs::path make_dynamic_library_name(const fs::path& path)
{
    return path.parent_path() / (library_prefix + path.filename() + dynamic_library_postfix);
}

inline fs::path make_object_file_name(const fs::path& path)
{
    return path.parent_path() / (path.filename() + object_file_postfix);
}

inline fs::path make_static_library_name(const fs::path& path)
{
    return path.parent_path() / (library_prefix + path.filename() + static_library_postfix);
}

struct FsPathHash
{
    std::size_t operator()(const fs::path& path) const { return fs::hash_value(path); }
};
} // namespace miopen

#endif // GUARD_MIOPEN_FILESYSTEM_HPP_
