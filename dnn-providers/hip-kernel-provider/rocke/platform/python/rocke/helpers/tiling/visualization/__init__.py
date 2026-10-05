# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.visualization -- the public layout-visualization surface.

Import the components, recipes, adapters, and the colour model from HERE, not the submodules:

    from rocke.helpers.tiling.visualization import MmaTee, Pipeline, flow_load_phase, field_inputs

matplotlib is imported LAZILY (only when a view actually renders), so importing this package -- or the
base ``rocke.helpers.tiling`` package, which re-exports the text `describe` -- does NOT pull a plotting
stack. The drawing primitives (`draw_*`), the register->LDS conflict drawers (owned by `lds_conflict`),
the encoding helpers, and `_canvas` are INTERNAL and stay behind their submodules on purpose.
"""

from __future__ import annotations

# -- colour model (hue/shade for any custom render) --
from .layout_render import (
    ACCENTS,
    NACC,
    accent_tint,
)

# -- cell-field components + their grouping objects --
from .layout_render import (
    CellGroup,
    LdsBankView,
    LogicalGroup,
    LogicalTileComponent,
    MmaTee,
    RegGroup,
    RegisterFileComponent,
)

# -- the pipeline/flow spine + the one-shot render entry --
from .layout_render import (
    FlowStage,
    LabelMutationError,
    Pipeline,
    WaveStrip,
    render_coalescing,
    render_coalescing_compare,
    render_views,
    transform_note,
)

# -- primitive flow recipes (single-hop workflows) --
from .layout_render import (
    flow_kloop_operand,
    flow_lds_to_register,
    flow_mem_to_register,
    flow_wave_mma,
)

# -- kernel-pipeline PHASE recipes + the descriptor->viz adapters --
from .kernel_stages import (
    classify_epilogue,
    coop_forward_map,
    field_inputs,
    flow_epilogue_phase,
    flow_lds_load_placement,
    flow_lds_store_placement,
    flow_load_phase,
    flow_mma_phase,
    lds_inputs,
)

# -- text reflection of an encoding (no matplotlib) --
from .layout_visualizer import describe, render_forward_map, render_inverse_map

__all__ = [
    # colour model
    "accent_tint", "ACCENTS", "NACC",
    # components
    "RegisterFileComponent", "RegGroup",
    "LogicalTileComponent", "LogicalGroup",
    "LdsBankView", "CellGroup", "MmaTee",
    # pipeline spine + render entry
    "FlowStage", "Pipeline", "WaveStrip", "transform_note", "render_views", "render_coalescing",
    "LabelMutationError",
    # Level-0 block diagram (selection-flow entry point)
    "block_diagram",
    # primitive flow recipes
    "flow_mem_to_register", "flow_lds_to_register", "flow_wave_mma", "flow_kloop_operand",
    # phase recipes
    "flow_load_phase", "flow_mma_phase", "flow_epilogue_phase",
    "flow_lds_store_placement", "flow_lds_load_placement",
    # descriptor -> viz adapters
    "field_inputs", "lds_inputs", "coop_forward_map", "classify_epilogue",
    # text reflection
    "describe", "render_forward_map", "render_inverse_map",
]
