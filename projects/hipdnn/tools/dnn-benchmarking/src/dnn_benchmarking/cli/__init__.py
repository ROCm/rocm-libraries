# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CLI module for dnn-benchmarking."""

from .main import main
from .parser import create_parser

__all__ = ["main", "create_parser"]
