# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ULM1 (`UseLoopModel`) writer path.

`KernelWriter` and `KernelWriterAssembly` keep only the seams; the bodies behind them are free
functions here, taking the writer -- the shape `Components/Subtile` uses.  Living outside those
modules is also what makes every `Tensile/LoopModel` and `Tensile/Lowering` import a plain
module-level one.

`Theta` holds the cached schedule, `Registers` the Valu rings, `Program` the one GIR Program,
`Emit` the three emission forks, and `TdmRegion` the descriptor primitives GIR places.  Nothing
is re-exported here, so importing one module never drags in another's dependencies.
"""
