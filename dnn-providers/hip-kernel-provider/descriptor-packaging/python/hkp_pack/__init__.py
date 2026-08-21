"""Build-time hip UKD -> compile -> prune -> kpack packaging for the hip-kernel-provider.

hkp = Hip Kernel-provider Packaging; the hkp_ prefix (package, CLI, and CMake
hkp_/HKP_ symbols) marks internal parts of this kpack-packaging module.

Consumes a flat authored source folder (KDPs with inline hip-form UKDs, by-Id
generic descriptors, and HIP sources), compiles each kernel via hipcc --genco
per targeted arch, prunes each per-arch intermediate to what that arch needs,
packs the code objects into a per-arch rocm_kpack archive, and rewrites the UKDs
into self-describing kpack form (library/toc_key/symbol/sha256 + provenance).
No manifest is emitted. Provider-internal; no public API.
"""

from .errors import HkpPackError
from .pipeline import ArchResult, run_pipeline

__all__ = ["HkpPackError", "ArchResult", "run_pipeline"]
