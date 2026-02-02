// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef UNIQUE_PATH_H
#define UNIQUE_PATH_H

#include <filesystem>
#include <system_error>

namespace fs = ::std::filesystem;

namespace miopen {

namespace detail {

fs::path unique_path(const fs::path& model, std::error_code* ec = nullptr);

}

inline fs::path unique_path(const fs::path& model = "%%%%-%%%%-%%%%-%%%%")
{
    return detail::unique_path(model);
}

inline fs::path unique_path(std::error_code& ec)
{
    return detail::unique_path("%%%%-%%%%-%%%%-%%%%", &ec);
}

inline fs::path unique_path(const fs::path& model, std::error_code& ec)
{
    return detail::unique_path(model, &ec);
}

} // namespace miopen

#endif // UNIQUE_PATH_H
