# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Enroll an architecture after publishing its first variant's archive and lock.
# A listed architecture installs every variant that has a committed
# baseline_lock.json, so publishing another variant does not edit this file.
# Tools that read these lists get (operation, architecture) pairs; a pair's
# archives are tracked by reference_bundles/<operation>/<arch>/*.tar.gz.dvc.
# Every operation package needs a list; one with nothing published uses
# set(ROCKE_PUBLISHED_<OP>_ARCHITECTURES "") because a bare set() unsets it.
set(ROCKE_PUBLISHED_SDPA_ARCHITECTURES gfx942 gfx950 gfx1151 gfx1201)
set(ROCKE_PUBLISHED_CONV_ARCHITECTURES gfx942 gfx950)
