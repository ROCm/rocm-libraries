# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reference provider implementations.

Import providers here to register them with the ReferenceProviderRegistry.
"""

from .pytorch_provider import PyTorchReferenceProvider

__all__ = ["PyTorchReferenceProvider"]
