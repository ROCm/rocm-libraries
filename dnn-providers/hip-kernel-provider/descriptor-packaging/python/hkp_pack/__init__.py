"""Build-time UKD -> compile/pack -> prune -> kpack packaging for the hip-kernel-provider.

hkp = Hip Kernel-provider Packaging; the hkp_ prefix (package, CLI, and CMake
hkp_/HKP_ symbols) marks internal parts of this kpack-packaging module.

Consumes a flat authored source folder (KDPs with inline UKDs, by-Id generic
descriptors, and HIP sources), compiles each hip kernel via hipcc --genco and each
rocKE kernel via comgr per targeted arch, prunes each per-arch intermediate to what
that arch needs, packs the code objects into a per-arch rocm_kpack archive, and
rewrites the UKDs into self-describing kpack form (library/toc_key/symbol/sha256 +
provenance).
No manifest is emitted. Provider-internal; no public API.

A UKD may instead name a prebuilt code object (`kernel_source` kind `hsaco`):
its descriptor-relative file is packed as-is, with no compile, into the same
archive, and its signature is read from the object's AMDGPU metadata, as for a
compiled object. The packer does not check the object's format or target
processor; hsaco kernels declare `arch` and are tested on device
(descriptor-packaging/README.md, "hsaco kernels and arch").
"""

from .errors import HkpPackError
from .pipeline import ArchResult, run_pipeline

__all__ = ["HkpPackError", "ArchResult", "run_pipeline"]
