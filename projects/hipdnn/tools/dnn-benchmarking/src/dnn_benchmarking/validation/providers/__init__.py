# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reference provider registration.

Providers are registered lazily by import path so importing this package (and
hence ``dnn_benchmarking.validation``) does not pull in a provider's optional
or heavy dependencies -- e.g. torch via the PyTorch provider -- until that
provider is actually requested from the registry.
"""

from ..reference_provider import ReferenceProviderRegistry

ReferenceProviderRegistry.register_lazy(
    "cpu_plugin",
    "dnn_benchmarking.validation.providers.cpu_plugin_provider",
    "CPUPluginReferenceProvider",
)
ReferenceProviderRegistry.register_lazy(
    "pytorch",
    "dnn_benchmarking.validation.providers.pytorch_provider",
    "PyTorchReferenceProvider",
)
