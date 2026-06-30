# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocke project umbrella:
#   - library/  : the SDPA product (C-api provider plugin + build-time Python).
#                 Added first to mirror the historical rocke-client-before-rocKE
#                 ordering; both complete before src/ (added by the parent) reads
#                 the staged CTest entries.
#   - platform/ : the authoring SDK + engine (former rocKE/).
add_subdirectory(library)
add_subdirectory(platform)
