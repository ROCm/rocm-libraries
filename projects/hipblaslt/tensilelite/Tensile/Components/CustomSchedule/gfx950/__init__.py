# Explicit per-schedule import list for gfx950.
#
# Each `import` below is a SIDE-EFFECT import: importing the module runs the
# `@RegisterSchedule` decorator at the top of the file, which appends to
# `dispatch._SCHEDULE_REGISTRY` and `dispatch._SCHEDULE_METADATA`. The
# registration order below MUST match the original CustomSchedule.py source
# ordering — `hasCustomSchedule` performs a linear scan with first-FOUND-wins
# / first-UNSUPPORTED-aborts semantics, so the order is load-bearing.
#
# DO NOT switch this list to a `pkgutil`/`importlib` walk: implicit ordering
# would silently change semantics, and tools that prune unused imports would
# break registration. Keep it explicit.

# Aliases need their callee imported FIRST so the bare-name binding exists
# when the alias's `from ._other import _get_schedule_other` line runs.
# Python handles this transparently because each per-schedule file imports
# its alias dependency at module-load time; we just keep the original
# registration order from CustomSchedule.py.

from . import _256x96x64_16bit          # noqa: F401
from . import _256x96x64_16bit_DPLB     # noqa: F401
from . import _192x256x64_16bit         # noqa: F401
from . import _256x192x64_16bit         # noqa: F401
from . import _256x256x128_8bit         # noqa: F401
from . import _256x256x64_16bit         # noqa: F401
from . import _160x256x64_16bit         # noqa: F401
from . import _96x256x64_16bit          # noqa: F401
from . import _256x160x64_16bit         # noqa: F401
from . import _256x240x64_16bit         # noqa: F401
from . import _256x208x64_16bit         # noqa: F401
from . import _192x128x64_16bit         # noqa: F401
from . import _224x128x64_16bit         # noqa: F401
from . import _224x256x64_16bit         # noqa: F401
from . import _192x320x64_16bit         # noqa: F401
from . import _256x224x64_16bit         # noqa: F401
from . import _320x192x64_16bit         # noqa: F401
from . import _352x192x64_16bit         # noqa: F401
from . import _240x256x64_16bit         # noqa: F401
from . import _208x256x64_16bit         # noqa: F401
from . import _128x224x64_16bit         # noqa: F401  alias -> _224x128x64_16bit
from . import _128x192x64_16bit         # noqa: F401
from . import _128x192x32_TF32          # noqa: F401
from . import _192x256x32_TF32          # noqa: F401
from . import _256x192x32_TF32          # noqa: F401
from . import _256x256x32_TF32          # noqa: F401
from . import _192x128x32_TF32          # noqa: F401
from . import _128x128x32_TF32          # noqa: F401
from . import _128x128x32_TF32_plr1     # noqa: F401
from . import _128x128x64_TF32          # noqa: F401
from . import _128x256x32_TF32          # noqa: F401
from . import _128x160x64_TF32          # noqa: F401
from . import _256x128x32_TF32          # noqa: F401
from . import _64x128x64_TF32           # noqa: F401
from . import _128x64x64_TF32           # noqa: F401  alias -> _64x128x64_TF32
from . import _160x128x64_TF32          # noqa: F401  alias -> _128x160x64_TF32
from . import _128x256x64_16bit         # noqa: F401
from . import _224x320x64_16bit         # noqa: F401
