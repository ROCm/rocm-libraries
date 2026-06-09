# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Abstract base class and registry for reference computation providers.

Reference providers compute expected outputs for hipDNN graph operations,
enabling validation of GPU execution against known-correct implementations.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ReferenceOutput:
    """Output from a reference provider for a single tensor.

    Attributes:
        data: The computed output data as numpy array.
        tensor_uid: UID of the tensor in the graph.
        metadata: Optional provider-specific metadata.
    """

    data: "np.ndarray"
    tensor_uid: int
    metadata: Optional[Dict[str, Any]] = None


class ReferenceProvider(ABC):
    """Abstract base for reference computation backends.

    Implementations compute reference outputs for hipDNN graphs using
    alternative backends (e.g., PyTorch, CPU reference plugin).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display and logging."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can be used.

        Returns:
            True if the provider's dependencies are available.
        """
        ...

    @abstractmethod
    def compute_reference(
        self,
        graph_json: Dict[str, Any],
        input_data: Dict[int, "np.ndarray"],
    ) -> Dict[int, ReferenceOutput]:
        """Compute reference outputs for given graph and inputs.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            input_data: Mapping of tensor UID to input numpy arrays.

        Returns:
            Mapping of output tensor UID to ReferenceOutput.

        Raises:
            NotImplementedError: If provider is not available.
            ValueError: If graph contains unsupported operations.
        """
        ...

    def supports_graph(self, graph_json: Dict[str, Any]) -> bool:
        """Check if provider supports all operations in graph.

        Args:
            graph_json: The graph as a parsed JSON dictionary.

        Returns:
            True if all operations are supported.
        """
        return True  # Default: assume supported, let compute_reference fail if not


class ReferenceProviderRegistry:
    """Registry of available reference providers.

    Allows dynamic registration and lookup of provider implementations.
    """

    _providers: Dict[str, Type[ReferenceProvider]] = {}
    # name -> (module_path, class_name) for providers imported on first use.
    _lazy: Dict[str, Tuple[str, str]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class eagerly.

        Args:
            name: Name to register the provider under.

        Returns:
            Decorator function.

        Example:
            @ReferenceProviderRegistry.register("pytorch")
            class PyTorchReferenceProvider(ReferenceProvider):
                ...
        """

        def decorator(provider_cls: Type[ReferenceProvider]) -> Type[ReferenceProvider]:
            cls._providers[name] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def register_lazy(cls, name: str, module: str, attr: str) -> None:
        """Register a provider by import path, resolved on first use.

        The provider module -- and its dependencies, e.g. torch -- is imported
        only when the provider is first requested, not when the registry or the
        providers package is imported. This keeps importing ``validation`` free
        of optional heavy dependencies.

        Args:
            name: Name to register the provider under.
            module: Importable module path containing the provider class.
            attr: Provider class name within that module.
        """
        cls._lazy[name] = (module, attr)

    @classmethod
    def _resolve(cls, name: str) -> Optional[Type[ReferenceProvider]]:
        """Return the provider class for ``name``, importing it if lazy.

        Returns ``None`` if the name is not registered. Propagates
        ImportError/AttributeError if a lazy provider's module or class
        cannot be loaded (e.g. the PyTorch provider when torch is absent).
        """
        if name in cls._providers:
            return cls._providers[name]
        spec = cls._lazy.get(name)
        if spec is None:
            return None
        module, attr = spec
        provider_cls = getattr(importlib.import_module(module), attr)
        cls._providers[name] = provider_cls  # cache so the import happens once
        return provider_cls

    @classmethod
    def get_provider(cls, name: str, **kwargs: Any) -> ReferenceProvider:
        """Get instance of named provider.

        Args:
            name: Registered provider name.
            **kwargs: Arguments passed to provider constructor.

        Returns:
            Provider instance.

        Raises:
            ValueError: If provider name is not registered.
        """
        provider_cls = cls._resolve(name)
        if provider_cls is None:
            available = ", ".join(cls.list_registered()) or "(none)"
            raise ValueError(
                f"Unknown reference provider: '{name}'. Available: {available}"
            )
        return provider_cls(**kwargs)

    @classmethod
    def list_registered(cls) -> List[str]:
        """List names of all registered providers (lazy ones included)."""
        names = list(cls._lazy.keys())
        for name in cls._providers:
            if name not in cls._lazy:
                names.append(name)
        return names

    @classmethod
    def list_available(cls) -> List[str]:
        """List names of providers that are currently usable.

        Resolves each registered provider (importing lazy ones) and checks
        is_available(). A provider whose module fails to import -- e.g. the
        PyTorch provider without torch installed -- is treated as unavailable.

        Returns:
            List of provider names where is_available() returns True.
        """
        available = []
        for name in cls.list_registered():
            try:
                provider_cls = cls._resolve(name)
                if provider_cls is not None and provider_cls().is_available():
                    available.append(name)
            except Exception:
                # Module import or instantiation failed -> not available.
                pass
        return available
