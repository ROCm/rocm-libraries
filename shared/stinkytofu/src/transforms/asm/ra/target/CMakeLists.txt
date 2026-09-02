# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# One file per ISA triple. A new architecture's rules are a new source here plus
# an anchor in AllocationRulesRegistry::registerAll(); nothing above this
# directory changes.
target_sources(stinkytofu_objs PRIVATE
    Gfx1250AllocationRules.cpp
)
