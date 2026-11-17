# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compatibility alias for the former ``Tensile`` package name."""

from tensilelite._namespace_bridge import install_alias


install_alias(alias=__name__, canonical="tensilelite")
