# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Named packaging probes. Loaded by hkp_load_packaging_probes() under
# HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES; see HkpPackagingProbes.cmake for the
# arguments and descriptor-packaging/README.md, "Packaging probes", for usage.
#
# Architectures need no line here. After this file, hkp_load_packaging_probes() declares
# one automatic probe, hkp-probe-<arch>, for every explicit `arch` entry on a KDP or UKD
# under the production descriptor root that is not one of the build's GPU_TARGETS. A new
# architecture, or a new pack under one, is probed with no edit to this file.
#
# Wildcard descriptors (no `arch`, or an empty list) name no architecture and get no
# automatic probe. That is safe: a wildcard ships for whatever the build targets, so
# every lane's own production pack, and its GPU test lanes, exercise it on every PR.
# A generic target in the generic target table (gfx11-generic) is probed through its
# first member in table order (gfx1100): the packer compiles the generic once and merges
# it into each requested member's folder, so the probe for the member covers it. A build
# that targets any member of the generic (gfx1151) already compiles it and gets no probe
# for it. A TheRock family name is skipped with a STATUS line: no build packs for it. A
# `*-generic` absent from the table fails configure, as it fails the packer.
#
# Without UKDS a probe packs, per compile group of every KDP shipping for its ARCH, one
# UKD, or for rocke enough UKDs that each value of each varying spec field is packed
# once; combinations of values are not guaranteed. This file is for named UKDS probes
# only: an integration that needs specific UKDs (a combination the sweep may miss)
# packed on every PR adds
#     hkp_add_packaging_probe(ARCH <gfx> NAME <integration> UKDS <ukd-name>...)
# Automatic probes are named by their bare arch, so a NAME equal to an automatically
# probed arch is declared twice and fails configure.
