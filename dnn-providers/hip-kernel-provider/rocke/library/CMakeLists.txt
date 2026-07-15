# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocke/library: the SDPA/MHA product.
#   - api/     : C-api / hipDNN provider plugin (former rocke-client). Library is
#                the provider for now; preserves the "rocke-client" engine name
#                and ROCKE_ENGINE_ID until the provider is split out.
# The library Python (kernels/, builders/, dispatch/) is BUILD-TIME ONLY and is
# intentionally not installed.
add_subdirectory(api)

# aot/: build-time producer for provider-owned rocKE AOT artifacts (loose HSACO +
# sidecars), packed into per-arch .kpack + bundle manifests and installed beside
# the provider plugin data. Python + CMake only; no compiled targets.
add_subdirectory(aot)

if(TARGET rocke_client AND TARGET rocke_client_aot_artifacts)
    add_dependencies(rocke_client rocke_client_aot_artifacts)
endif()
if(TARGET rocke_client_integration_tests AND TARGET rocke_client_aot_artifacts)
    add_dependencies(rocke_client_integration_tests rocke_client_aot_artifacts)
endif()

# The GPU-tier launch test in rocke_client_tests reads the per-arch bundle
# (rocke_client_<arch>.{json,kpack}) that the kpack packing copies beside the
# plugin, so the test binary must not build before those artifacts exist. Wired
# here, after both subdirs, because rocke_client_aot_kpack is defined in aot/,
# which is added after api/ (where rocke_client_tests is defined).
if(TARGET rocke_client_tests AND TARGET rocke_client_aot_kpack)
    add_dependencies(rocke_client_tests rocke_client_aot_kpack)
endif()
