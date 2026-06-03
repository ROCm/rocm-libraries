#!/usr/bin/env python3
# Build spike/mp2/frozen_src/: the exact conv codegen closure (ck modules + shims)
# to freeze into the embed binary. Runs under CPython as a build step.
#   - captures the closure by importing the conv codegen path against the mp1 bundle
#   - copies those modules + ALL shims (embed port lacks functools/itertools/etc.)
#   - embeds arch_specs.json as a Python module (frozen modules have no filesystem)
#   - patches arch/target.py off __file__/open/json onto the embedded dict
import json
import os
import shutil
import sys

# Paths are supplied by the CMake freeze pipeline (build_embed.sh) via the environment:
#   BUNDLE_DIR - transformed bundle (ck_dsl + ck_dsl_provider) from build_bundle.py
#   SHIMS_DIR  - vendored shims/ (functools/itertools/etc. the embed port lacks)
#   FROZEN_DIR - output dir for the frozen module set
BUNDLE = os.environ["BUNDLE_DIR"]
SHIMS = os.environ["SHIMS_DIR"]
FROZEN = os.environ["FROZEN_DIR"]

# 1. Capture the real provider-entry closure. Import ck_dsl_provider.compile_service
#    + helpers.compile (pulls compile_service, helpers/compile, runtime/comgr, conv,
#    etc.) and run a lower() to pull lazy core/arch imports. We do NOT run comgr here
#    (just need the import set), so build_frozen needs no libamd_comgr.
sys.path.insert(0, BUNDLE)
sys.path.insert(0, SHIMS)
import ck_dsl_provider.compile_service  # noqa: E402,F401
from ck_dsl.helpers.compile import compile_kernel  # noqa: E402,F401
from ck_dsl.instances.common.conv_implicit_gemm import (  # noqa: E402
    ImplicitGemmConvSpec,
    ConvProblem,
    build_implicit_gemm_conv,
)
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402

spec = ImplicitGemmConvSpec(
    problem=ConvProblem(
        N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
    ),
    tile_m=64,
    tile_n=64,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_tile_m=32,
    warp_tile_n=32,
    warp_tile_k=16,
)
_ = lower_kernel_to_llvm(build_implicit_gemm_conv(spec, arch="gfx950"), arch="gfx950")

# Also capture the compile_smoke (elementwise) closure so the smoke path is
# frozen alongside conv. compile_smoke uses ElementwiseSpec/build_elementwise;
# lower (not compile) to pull its lazy imports without needing comgr here.
from ck_dsl.instances.common.elementwise import (  # noqa: E402
    ElementwiseSpec,
    build_elementwise,
)

_smoke_spec = ElementwiseSpec(
    op="copy", dtype="f16", block_size=64, vec=2, name="ck_dsl_provider_smoke_copy"
)
_ = lower_kernel_to_llvm(build_elementwise(_smoke_spec), arch="gfx950")

# hip_module is the launch-side ctypes layer; the native-backed comgr.py we write
# below does not import it, so exclude it from the frozen set.
EXCLUDE = {"ck_dsl/runtime/hip_module.py"}
ck_files = sorted(
    rel
    for m in sys.modules.values()
    if getattr(m, "__file__", None) and m.__file__.startswith(BUNDLE + "/")
    for rel in [m.__file__[len(BUNDLE) + 1 :]]
    if rel not in EXCLUDE
)

# 2. Reset frozen tree; copy ck closure + ALL shims.
if os.path.exists(FROZEN):
    shutil.rmtree(FROZEN)
os.makedirs(FROZEN)
for rel in ck_files:
    dst = os.path.join(FROZEN, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(BUNDLE, rel), dst)
for shim in os.listdir(SHIMS):
    if shim.endswith(".py"):
        shutil.copy(os.path.join(SHIMS, shim), os.path.join(FROZEN, shim))

# 3. Embed arch_specs.json as a Python module.
with open(os.path.join(BUNDLE, "ck_dsl/core/arch/data/arch_specs.json")) as f:
    doc = json.load(f)
with open(os.path.join(FROZEN, "ck_dsl/core/arch/_arch_specs_embedded.py"), "w") as f:
    f.write(
        "# Generated: arch_specs.json embedded for the frozen (no-filesystem) build.\n"
    )
    f.write("DOC = " + repr(doc) + "\n")

# 4. Patch target.py off __file__/open/json onto the embedded dict.
tp = os.path.join(FROZEN, "ck_dsl/core/arch/target.py")
s = open(tp).read()
# NB: match the source verbatim (black formats string literals with double
# quotes); a silent no-match here would leave a dangling _EMBEDDED_DOC.
s = s.replace(
    '_DATA_FILE = Path(__file__).parent / "data" / "arch_specs.json"',
    "from ._arch_specs_embedded import DOC as _EMBEDDED_DOC\n"
    '_DATA_FILE = Path("arch_specs.json")  # name only; data is embedded',
)
s = s.replace(
    "    with open(str(_DATA_FILE)) as fh:\n        doc = json.load(fh)\n",
    "    doc = _EMBEDDED_DOC\n",
)
s = s.replace("import json\n", "")  # json.load patched out; embed has no json module
open(tp, "w").write(s)
# Assert on the IMPORT, not just the name: the doc=_EMBEDDED_DOC replace alone
# would satisfy a bare-name check even if the import injection silently failed.
patched = open(tp).read()
assert (
    "from ._arch_specs_embedded import DOC as _EMBEDDED_DOC" in patched
), "target.py arch_specs patch failed (import not injected -- source format drift?)"
assert "json.load" not in patched, "target.py still references json.load after patch"

# 4b. Swap ck_dsl's runtime/comgr.py backend to the native `comgr` module. This is
#     the actual ck_dsl change for Arch A: comgr.py keeps its interface
#     (build_hsaco_from_llvm_ir -> (hsaco, timings)) but its body calls the native
#     C++ comgr instead of ctypes -> libamd_comgr. ck_dsl's flow is otherwise intact.
os.makedirs(os.path.join(FROZEN, "ck_dsl/runtime"), exist_ok=True)
open(os.path.join(FROZEN, "ck_dsl/runtime/__init__.py"), "w").write(
    "# trimmed for MicroPython bundle\n"
)
open(os.path.join(FROZEN, "ck_dsl/runtime/comgr.py"), "w").write(
    "# Native-backed comgr for the embed build: same interface as the ctypes\n"
    "# original, but the backend is the C++ `comgr` module exposed to the interpreter.\n\n\n"
    "class ComgrError(RuntimeError):\n    pass\n\n\n"
    "class ComgrTimings:\n"
    "    def __init__(self):\n"
    "        self.bc = 0.0\n        self.relocatable = 0.0\n        self.executable = 0.0\n\n"
    "    @property\n    def total(self):\n        return self.bc + self.relocatable + self.executable\n\n\n"
    "def build_hsaco_from_llvm_ir(ir_text, isa='amdgcn-amd-amdhsa--gfx950', options=None):\n"
    "    import comgr\n"
    "    hsaco = comgr.build_hsaco(ir_text, isa, list(options or ['-O3']))\n"
    "    return hsaco, ComgrTimings()\n"
)

# 5. Entry module the C host imports + calls. Mirrors ck_dsl's flow: lower to IR,
#    then call comgr (exposed as a native module) -> returns the HSACO bytes, like
#    runtime/comgr.py would. So the C host gets a HSACO, not IR text.
with open(os.path.join(FROZEN, "ckdsl_entry.py"), "w") as f:
    f.write(
        "# Calls the PROVIDER's real entry point, ck_dsl_provider.compile_service.compile,\n"
        "# with the conv payload the C++ ConvImplicitGemmPayload would emit. compile()\n"
        "# runs the full ck_dsl flow (build -> compile_kernel -> native-backed comgr) and\n"
        "# returns the artifact dict; we hand back its hsaco bytes.\n"
        "from ck_dsl_provider.compile_service import compile as _provider_compile\n\n\n"
        "_PAYLOAD = {\n"
        "    'problem': {'N': 8, 'Hi': 56, 'Wi': 56, 'C': 64, 'K': 64, 'R': 3, 'S': 3,\n"
        "                'sH': 1, 'sW': 1, 'pH': 1, 'pW': 1, 'dH': 1, 'dW': 1},\n"
        "    'tile_m': 64, 'tile_n': 64, 'tile_k': 64, 'warp_m': 2, 'warp_n': 2,\n"
        "    'warp_tile_m': 32, 'warp_tile_n': 32, 'warp_tile_k': 16,\n"
        "}\n\n\n"
        "def compile_conv():\n"
        "    art = _provider_compile('conv_implicit_gemm', _PAYLOAD, 'gfx950')\n"
        "    return art['hsaco']\n"
    )

n = sum(len(fs) for _, _, fs in os.walk(FROZEN))
print(
    "frozen_src: %d ck modules + shims, %d files total at %s"
    % (len(ck_files), n, FROZEN)
)
