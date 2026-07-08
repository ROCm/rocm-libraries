# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Backend abstraction for benchmark solution selection and optimization strategies.

Provides a pluggable interface for different solution enumeration and optimization backends:
- TensileBackend: Exhaustive fork parameter enumeration
- DuctileBackend: Genetic algorithm-based parameter optimization

Each backend implements its own candidate generation and evaluation loop,
calling out to shared helpers provided by BenchmarkProblems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Tuple


class OptimizationBackend(ABC):
    """Abstract base class for optimization backends.
    
    A backend orchestrates the solution optimization strategy (exhaustive, GA, etc.)
    by implementing run(). The run() method handles its own
    candidate generation and evaluation loop, calling shared helper functions
    provided by the caller.
    """

    @abstractmethod
    def run(self, 
            backend_config: Dict[str, Any],
            benchmark_config: Dict[str, Any],
            benchmark_runner: Callable[..., Tuple[str, int]],
            cacheValid: bool = False,
            buildOnly: bool = False) -> None:
        """Execute the backend's optimization strategy.
        
        The backend orchestrates its own candidate generation loop, converts candidates
        to solutions, and benchmarks them. Each backend handles solution generation 
        using helper functions from BenchmarkProblems.
        
        Args:
            backend_config: Configuration dict containing backend-specific settings.
                    
            benchmark_config: Configuration dict containing backend-specific step settings.
                          Required keys: forkParams, constantParams, paramGroups,
                          customKernels, internalSupportParams, customKernelWildcard,
                          ForkParameters, problemType, assembler, debugConfig, isaInfoMap,
                          sourcePath
                          
            benchmark_runner: Callable that benchmarks a list of solutions and returns results.
                            Signature: (solutions, isCached=False, buildOnly=False)
                                     -> (resultsFileName, returncode)
                            
            cacheValid: If True, matching cache data is available for this step
            buildOnly: If True, skip benchmarking (only build/compile kernels)
        
        Returns:
            None
        """
        pass

    @abstractmethod
    def supports_solution_pool(self) -> bool:
        """Check if this backend supports loading solutions from a solution pool.
        
        Returns:
            True if the backend can load solutions from pool files, False otherwise.
        """
        pass


class BackendFactory:
    """Factory for registering and instantiating benchmark backends."""
    
    _backends: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, backend_class: type) -> None:
        """Register a backend class.
        
        Args:
            name: String identifier for the backend (e.g., 'tensile', 'ductile')
            backend_class: Class that inherits from OptimizationBackend
        """
        if not issubclass(backend_class, OptimizationBackend):
            raise TypeError(f"{backend_class} must inherit from OptimizationBackend")
        cls._backends[name] = backend_class

    @classmethod
    def create(cls, name: str) -> OptimizationBackend:
        """Create an instance of a registered backend.
        
        Args:
            name: String identifier of the backend to create
            
        Returns:
            New instance of the backend class
            
        Raises:
            ValueError: If backend name is not registered
        """
        if name == "ductile" and name not in cls._backends:
            from .ductile_backend import DuctileBackend
            cls.register("ductile", DuctileBackend)

        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}. Registered backends: {list(cls._backends.keys())}")
        return cls._backends[name]()

    @classmethod
    def get_available_backends(cls) -> List[str]:
        """Get list of registered backend names.
        
        Returns:
            List of available backend identifiers
        """
        return list(cls._backends.keys())
