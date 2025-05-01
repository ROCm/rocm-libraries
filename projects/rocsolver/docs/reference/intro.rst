.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _intro:

********************************************************************
Introduction to rocSOLVER API
********************************************************************

.. note::
    The rocSOLVER library is in the early stages of active development. New features are being continuously added,
    with new functionality documented at each `release of the ROCm platform <https://rocm.docs.amd.com/en/latest/release.html>`_.

Currently implemented functionality
====================================

The following tables summarize the functionality implemented for the different supported precisions in rocSOLVER's latest release.
All LAPACK and LAPACK-like main functions include *_batched* and *_strided_batched* versions. For a complete description of the listed
routines, please see the corresponding reference guides.

LAPACK auxiliary functions
----------------------------

.. csv-table:: Vector and matrix manipulations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_lacgv <lacgv>`, x, x, x, x
    :ref:`rocsolver_laswp <laswp>`, x, x, x, x
    :ref:`rocsolver_lauum <lauum>`, x, x, x, x

.. csv-table:: Householder reflections
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_larfg <larfg>`, x, x, x, x
    :ref:`rocsolver_larf <larf>`, x, x, x, x
    :ref:`rocsolver_larft <larft>`, x, x, x, x
    :ref:`rocsolver_larfb <larfb>`, x, x, x, x

.. csv-table:: Givens (plane) rotations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_lasr <lasr>`, x, x, x, x

.. csv-table:: Bidiagonal forms
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_labrd <labrd>`, x, x, x, x
    :ref:`rocsolver_bdsqr <bdsqr>`, x, x, x, x
    :ref:`rocsolver_bdsvdx <bdsvdx>`, x, x, ,

.. csv-table:: Tridiagonal forms
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_sterf <sterf>`, x, x, ,
    :ref:`rocsolver_stebz <stebz>`, x, x, ,
    :ref:`rocsolver_latrd <latrd>`, x, x, x, x
    :ref:`rocsolver_steqr <steqr>`, x, x, x, x
    :ref:`rocsolver_stedc <stedc>`, x, x, x, x
    :ref:`rocsolver_stein <stein>`, x, x, x, x

.. csv-table:: Symmetric matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_lasyf <lasyf>`, x, x, x, x

.. csv-table:: Orthonormal matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_org2r <org2r>`, x, x, ,
    :ref:`rocsolver_orgqr <orgqr>`, x, x, ,
    :ref:`rocsolver_orgl2 <orgl2>`, x, x, ,
    :ref:`rocsolver_orglq <orglq>`, x, x, ,
    :ref:`rocsolver_org2l <org2l>`, x, x, ,
    :ref:`rocsolver_orgql <orgql>`, x, x, ,
    :ref:`rocsolver_orgbr <orgbr>`, x, x, ,
    :ref:`rocsolver_orgtr <orgtr>`, x, x, ,
    :ref:`rocsolver_orm2r <orm2r>`, x, x, ,
    :ref:`rocsolver_ormqr <ormqr>`, x, x, ,
    :ref:`rocsolver_orml2 <orml2>`, x, x, ,
    :ref:`rocsolver_ormlq <ormlq>`, x, x, ,
    :ref:`rocsolver_orm2l <orm2l>`, x, x, ,
    :ref:`rocsolver_ormql <ormql>`, x, x, ,
    :ref:`rocsolver_ormbr <ormbr>`, x, x, ,
    :ref:`rocsolver_ormtr <ormtr>`, x, x, ,

.. csv-table:: Unitary matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_ung2r <ung2r>`, , , x, x
    :ref:`rocsolver_ungqr <ungqr>`, , , x, x
    :ref:`rocsolver_ungl2 <ungl2>`, , , x, x
    :ref:`rocsolver_unglq <unglq>`, , , x, x
    :ref:`rocsolver_ung2l <ung2l>`, , , x, x
    :ref:`rocsolver_ungql <ungql>`, , , x, x
    :ref:`rocsolver_ungbr <ungbr>`, , , x, x
    :ref:`rocsolver_ungtr <ungtr>`, , , x, x
    :ref:`rocsolver_unm2r <unm2r>`, , , x, x
    :ref:`rocsolver_unmqr <unmqr>`, , , x, x
    :ref:`rocsolver_unml2 <unml2>`, , , x, x
    :ref:`rocsolver_unmlq <unmlq>`, , , x, x
    :ref:`rocsolver_unm2l <unm2l>`, , , x, x
    :ref:`rocsolver_unmql <unmql>`, , , x, x
    :ref:`rocsolver_unmbr <unmbr>`, , , x, x
    :ref:`rocsolver_unmtr <unmtr>`, , , x, x

LAPACK main functions
----------------------------

.. csv-table:: Triangular factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_potf2 <potf2>`, x, x, x, x
    :ref:`rocsolver_potrf <potrf>`, x, x, x, x
    :ref:`rocsolver_getf2 <getf2>`, x, x, x, x
    :ref:`rocsolver_getrf <getrf>`, x, x, x, x
    :ref:`rocsolver_sytf2 <sytf2>`, x, x, x, x
    :ref:`rocsolver_sytrf <sytrf>`, x, x, x, x

.. csv-table:: Orthogonal factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_geqr2 <geqr2>`, x, x, x, x
    :ref:`rocsolver_geqrf <geqrf>`, x, x, x, x
    :ref:`rocsolver_gerq2 <gerq2>`, x, x, x, x
    :ref:`rocsolver_gerqf <gerqf>`, x, x, x, x
    :ref:`rocsolver_gelq2 <gelq2>`, x, x, x, x
    :ref:`rocsolver_gelqf <gelqf>`, x, x, x, x
    :ref:`rocsolver_geql2 <geql2>`, x, x, x, x
    :ref:`rocsolver_geqlf <geqlf>`, x, x, x, x

.. csv-table:: Problem and matrix reductions
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_sytd2 <sytd2>`, x, x, ,
    :ref:`rocsolver_sytrd <sytrd>`, x, x, ,
    :ref:`rocsolver_sygs2 <sygs2>`, x, x, ,
    :ref:`rocsolver_sygst <sygst>`, x, x, ,
    :ref:`rocsolver_hetd2 <hetd2>`, , , x, x
    :ref:`rocsolver_hetrd <hetrd>`, , , x, x
    :ref:`rocsolver_hegs2 <hegs2>`, , , x, x
    :ref:`rocsolver_hegst <hegst>`, , , x, x
    :ref:`rocsolver_gebd2 <gebd2>`, x, x, x, x
    :ref:`rocsolver_gebrd <gebrd>`, x, x, x, x

.. csv-table:: Linear-systems solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_trtri <trtri>`, x, x, x, x
    :ref:`rocsolver_getri <getri>`, x, x, x, x
    :ref:`rocsolver_getrs <getrs>`, x, x, x, x
    :ref:`rocsolver_gesv <gesv>`, x, x, x, x
    :ref:`rocsolver_potri <potri>`, x, x, x, x
    :ref:`rocsolver_potrs <potrs>`, x, x, x, x
    :ref:`rocsolver_posv <posv>`, x, x, x, x

.. csv-table:: Least-square solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_gels <gels>`, x, x, x, x

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_syev <syev>`, x, x, ,
    :ref:`rocsolver_syevd <syevd>`, x, x, ,
    :ref:`rocsolver_syevx <syevx>`, x, x, ,
    :ref:`rocsolver_sygv <sygv>`, x, x, ,
    :ref:`rocsolver_sygvd <sygvd>`, x, x, ,
    :ref:`rocsolver_sygvx <sygvx>`, x, x, ,
    :ref:`rocsolver_heev <heev>`, , , x, x
    :ref:`rocsolver_heevd <heevd>`, , , x, x
    :ref:`rocsolver_heevx <heevx>`, , , x, x
    :ref:`rocsolver_hegv <hegv>`, , , x, x
    :ref:`rocsolver_hegvd <hegvd>`, , , x, x
    :ref:`rocsolver_hegvx <hegvx>`, , , x, x

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_gesvd <gesvd>`, x, x, x, x
    :ref:`rocsolver_gesvdx <gesvdx>`, x, x, x, x

LAPACK-like functions
----------------------------

.. csv-table:: Triangular factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_getf2_npvt <getf2_npvt>`, x, x, x, x
    :ref:`rocsolver_getrf_npvt <getrf_npvt>`, x, x, x, x
    :ref:`rocsolver_geblttrf_npvt <geblttrf_npvt>`, x, x, x, x

.. csv-table:: Linear-systems solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_getri_npvt <getri_npvt>`, x, x, x, x
    :ref:`rocsolver_getri_outofplace <getri_outofplace>`, x, x, x, x
    :ref:`rocsolver_getri_npvt_outofplace <getri_npvt_outofplace>`, x, x, x, x
    :ref:`rocsolver_geblttrs_npvt <geblttrs_npvt>`, x, x, x, x

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_syevj <syevj>`, x, x, ,
    :ref:`rocsolver_sygvj <sygvj>`, x, x, ,
    :ref:`rocsolver_heevj <heevj>`, , , x, x
    :ref:`rocsolver_hegvj <hegvj>`, , , x, x
    :ref:`rocsolver_syevdj <syevdj>`, x, x, ,
    :ref:`rocsolver_sygvdj <sygvdj>`, x, x, ,
    :ref:`rocsolver_heevdj <heevdj>`, , , x, x
    :ref:`rocsolver_hegvdj <hegvdj>`, , , x, x
    :ref:`rocsolver_syevdx <syevdx>`, x, x, ,
    :ref:`rocsolver_sygvdx <sygvdx>`, x, x, ,
    :ref:`rocsolver_heevdx <heevdx>`, , , x, x
    :ref:`rocsolver_hegvdx <hegvdx>`, , , x, x

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_gesvdj <gesvdj>`, x, x, x, x


Re-factorization and direct solvers
----------------------------------------

.. csv-table:: Triangular re-factorization
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_csrrf_sumlu <rfsumlu>`, x, x, ,
    :ref:`rocsolver_csrrf_splitlu <rfsplitlu>`, x, x, ,
    :ref:`rocsolver_csrrf_refactlu <rfrefactlu>`, x, x, ,
    :ref:`rocsolver_csrrf_refactchol <rfrefactchol>`, x, x, ,

.. csv-table:: Direct solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_csrrf_solve <rfsolve>`, x, x, ,

