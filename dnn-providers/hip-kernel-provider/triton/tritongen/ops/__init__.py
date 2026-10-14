# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Registry of op contracts: op name -> its ``ops/<op>.py`` module."""

from . import rmsnorm

OPS = {rmsnorm.OP: rmsnorm}
