# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""hipTensor monokernels (gfx950, EXPERIMENTAL: not reached by any dispatcher).

Two kernel styles, both on the universal-GEMM bodies:

* :mod:`.persistent_contraction` (+ :mod:`.contraction_plan`): ONE persistent
  launch per contraction -- operand pack tasks, the persistent GEMM tile loop
  (``spec._mono`` hooks) and a permuted alpha / beta epilogue with an optional
  row-max reduction. Its schedule is checked with
  :mod:`rocke.helpers.task_schedule`.
* :mod:`.task_queue` (+ :mod:`.inline`, :mod:`.queue_contraction`,
  :mod:`.queue_plan`, :mod:`.permute_gather`, :mod:`.transpose_lds`): ONE
  persistent launch drains a ticket queue holding every tile of every op of a
  plan of small / medium contractions (inlined stand-alone bodies, per-op
  dependency counters).

Python engine builders only; the emitted IR uses mirrored ops.
"""

from .contraction_plan import ContractionPlan, plan_contraction  # noqa: F401
from .persistent_contraction import (  # noqa: F401
    OperandPackSpec,
    PersistentContractionHooks,
    PersistentContractionKnobs,
    PersistentContractionSpec,
    build_operand_pack,
    build_persistent_contraction,
    persistent_body_spec,
    persistent_contraction_hooks,
)
from .task_queue import (  # noqa: F401
    QueueKnobs,
    QueueOp,
    QueuePlan,
    TaskQueueSpec,
    build_task_queue,
    queue_counter_words,
)
