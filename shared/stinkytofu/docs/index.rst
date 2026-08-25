.. meta::
  :description: StinkyTofu is an LLVM-inspired pass-based IR optimizer for AMD GPU assembly kernels.
  :keywords: StinkyTofu, ROCm, documentation, hipBLASLt, TensileLite

*************************
StinkyTofu documentation
*************************

StinkyTofu is an LLVM-inspired pass-based IR optimizer for AMD GPU assembly kernels. It is used by hipBLASLt/TensileLite to schedule and optimize generated GPU code for gfx1250.

StinkyTofu uses a two-level IR: a high-level, architecture-agnostic Logical IR and a low-level, architecture-specific Asm IR, connected by a pass pipeline covering DAG scheduling, wait-count insertion, dead code elimination, redundant mov elimination, and peephole optimization.

The public repository for StinkyTofu is located at `https://github.com/ROCm/rocm-libraries/tree/develop/shared/stinkytofu <https://github.com/ROCm/rocm-libraries/tree/develop/shared/stinkytofu>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: User guides

    * :doc:`Global parameters <user/global-parameters>`
    * :doc:`IR converter <user/ir-converter>`
    * :doc:`Assembly emitter <user/asm-emitter>`
    * :doc:`Virtual registers <user/virtual-registers>`
    * :doc:`Error codes <user/error-codes>`

  .. grid-item-card:: Developer guides

    * :doc:`Architecture overview <developer/architecture>`
    * :doc:`Adding instructions <developer/adding-instructions>`
    * :doc:`Adding a GPU architecture <developer/adding-architecture>`
    * :doc:`Pattern grammar reference <developer/pattern-grammar>`

  .. grid-item-card:: Reference

    * :doc:`API reference <reference/api-reference>`
    * :doc:`Known issues <known-issues>`

  .. grid-item-card:: About

    * :doc:`License <license>`

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
