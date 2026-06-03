import sys

sys.path.insert(
    0, "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1/shims"
)
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ES:
    op: str = field()  # required, but explicit value -> visible in __dict__
    dtype: str = "f16"
    block_size: int = 256
    vec: int = 8
    name: str = "x"


# (1) keyword construction (the provider's style)
e = ES(op="copy", block_size=64)
print("kw  :", e.op, e.dtype, e.block_size, e.vec, e.name)

# (2) positional construction (common inside ck_dsl): is arg0 -> op, or misordered?
try:
    e2 = ES("copy")
    print("pos0->op?", e2.op == "copy", "| block_size=", e2.block_size)
except Exception as ex:
    print("pos err:", type(ex).__name__, ex)

# (3) required enforcement: omitting op should error
try:
    ES(dtype="f16")
    print("required NOT enforced (op defaulted)")
except Exception as ex:
    print("required ok:", type(ex).__name__)

print("replace:", replace(e, vec=4).vec, "| eq:", ES(op="copy") == ES(op="copy"))
