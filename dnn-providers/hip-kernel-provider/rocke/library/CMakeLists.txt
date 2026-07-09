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
