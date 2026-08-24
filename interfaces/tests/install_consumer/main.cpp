// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "rocm/interfaces/loader.h"

int main(int argc, char** argv) {
    try {
        if (argc != 3) return 2;
#if defined(_WIN32)
        _putenv_s("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", argv[2]);
        _putenv_s("ROCM_INTERFACES_ROCBLAS_PROVIDER_MANIFEST", argv[1]);
#else
        setenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", argv[2], 1);
        setenv("ROCM_INTERFACES_ROCBLAS_PROVIDER_MANIFEST", argv[1], 1);
#endif
        rocm::interfaces::ProviderRegistry registry;
        registry.load_manifest(argv[1]);
        auto lease = registry.select(ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE, 0, 1);
        if (!lease || lease->provider_id() != "system-rocblas-bridge") return 3;
        rocm::interfaces::ProviderRegistry narrow_registry;
        const std::filesystem::path narrow_manifest =
            std::filesystem::path(argv[1]).parent_path() / "rocblas-narrow-v2-system.json";
        narrow_registry.load_manifest(narrow_manifest);
        auto narrow_lease = narrow_registry.select(ROCM_INTERFACES_DOMAIN_BLAS_V2, 0, 1);
        if (!narrow_lease || narrow_lease->provider_id() != "system-rocblas-narrow-v2") return 5;
        size_t version_size = 0;
        if (rocblas_get_version_string_size(&version_size) != rocblas_status_success ||
            !version_size)
            return 4;
        std::cout << "installed provider manifest selected " << lease->provider_id() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
