# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Abstract base class and registry for reference computation providers.

Reference providers compute expected outputs for hipDNN graph operations,
enabling validation of GPU execution against known-correct implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

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

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class.

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
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys()) or "(none)"
            raise ValueError(
                f"Unknown reference provider: '{name}'. Available: {available}"
            )
        return cls._providers[name](**kwargs)

    @classmethod
    def list_registered(cls) -> List[str]:
        """List names of all registered providers."""
        return list(cls._providers.keys())

    @classmethod
    def list_available(cls) -> List[str]:
        """List names of providers that are currently usable.

        Checks is_available() on each registered provider.

        Returns:
            List of provider names where is_available() returns True.
        """
        available = []
        for name, provider_cls in cls._providers.items():
            try:
                instance = provider_cls()
                if instance.is_available():
                    available.append(name)
            except Exception:
                # Provider failed to instantiate, not available
                pass
        return available
