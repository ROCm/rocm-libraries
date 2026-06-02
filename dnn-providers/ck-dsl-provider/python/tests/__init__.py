# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Provider-owned, torch-free test suite for the CK DSL provider.

These tests deliberately live OUTSIDE the DSL's
``projects/composablekernel/python/test/test_ck_dsl.py`` (which imports
torch) so the provider's SDPA-fwd generation coverage can run with torch
absent. Nothing in this package may ``import torch``.
"""
