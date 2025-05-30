.. meta::
  :description: rocSPARSE conversion functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, conversion functions

.. _rocsparse_conversion_functions_:

********************************************************************
Sparse conversion functions
********************************************************************

This module contains all sparse conversion routines.

The sparse conversion routines describe operations on a matrix in sparse format to obtain a matrix in a different sparse format.

rocsparse_csr2coo()
-------------------

.. doxygenfunction:: rocsparse_csr2coo

rocsparse_coo2csr()
-------------------

.. doxygenfunction:: rocsparse_coo2csr

rocsparse_csr2csc_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_csr2csc_buffer_size

rocsparse_csr2csc()
-------------------

.. doxygenfunction:: rocsparse_scsr2csc
  :outline:
.. doxygenfunction:: rocsparse_dcsr2csc
  :outline:
.. doxygenfunction:: rocsparse_ccsr2csc
  :outline:
.. doxygenfunction:: rocsparse_zcsr2csc

rocsparse_gebsr2gebsc_buffer_size()
-----------------------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsc_buffer_size

rocsparse_gebsr2gebsc()
-----------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsc

rocsparse_csr2ell_width()
-------------------------

.. doxygenfunction:: rocsparse_csr2ell_width

rocsparse_csr2ell()
-------------------

.. doxygenfunction:: rocsparse_scsr2ell
  :outline:
.. doxygenfunction:: rocsparse_dcsr2ell
  :outline:
.. doxygenfunction:: rocsparse_ccsr2ell
  :outline:
.. doxygenfunction:: rocsparse_zcsr2ell

rocsparse_ell2csr_nnz()
-----------------------

.. doxygenfunction:: rocsparse_ell2csr_nnz

rocsparse_ell2csr()
-------------------

.. doxygenfunction:: rocsparse_sell2csr
  :outline:
.. doxygenfunction:: rocsparse_dell2csr
  :outline:
.. doxygenfunction:: rocsparse_cell2csr
  :outline:
.. doxygenfunction:: rocsparse_zell2csr

rocsparse_csr2hyb()
-------------------

.. doxygenfunction:: rocsparse_scsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_dcsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_ccsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_zcsr2hyb

rocsparse_hyb2csr_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_hyb2csr_buffer_size

rocsparse_hyb2csr()
-------------------

.. doxygenfunction:: rocsparse_shyb2csr
  :outline:
.. doxygenfunction:: rocsparse_dhyb2csr
  :outline:
.. doxygenfunction:: rocsparse_chyb2csr
  :outline:
.. doxygenfunction:: rocsparse_zhyb2csr

rocsparse_bsr2csr()
-------------------

.. doxygenfunction:: rocsparse_sbsr2csr
  :outline:
.. doxygenfunction:: rocsparse_dbsr2csr
  :outline:
.. doxygenfunction:: rocsparse_cbsr2csr
  :outline:
.. doxygenfunction:: rocsparse_zbsr2csr

rocsparse_gebsr2csr()
---------------------

.. doxygenfunction:: rocsparse_sgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2csr

rocsparse_gebsr2gebsr_buffer_size()
-----------------------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsr_buffer_size

rocsparse_gebsr2gebsr_nnz()
---------------------------

.. doxygenfunction:: rocsparse_gebsr2gebsr_nnz

rocsparse_gebsr2gebsr()
-----------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsr


rocsparse_csr2bsr_nnz()
-----------------------

.. doxygenfunction:: rocsparse_csr2bsr_nnz

rocsparse_csr2bsr()
-------------------

.. doxygenfunction:: rocsparse_scsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_dcsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_ccsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_zcsr2bsr

rocsparse_csr2gebsr_nnz()
-------------------------

.. doxygenfunction:: rocsparse_csr2gebsr_nnz

rocsparse_csr2gebsr_buffer_size()
---------------------------------

.. doxygenfunction:: rocsparse_scsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsr2gebsr_buffer_size

rocsparse_csr2gebsr()
---------------------

.. doxygenfunction:: rocsparse_scsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_dcsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_ccsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_zcsr2gebsr

rocsparse_csr2csr_compress()
----------------------------

.. doxygenfunction:: rocsparse_scsr2csr_compress
  :outline:
.. doxygenfunction:: rocsparse_dcsr2csr_compress
  :outline:
.. doxygenfunction:: rocsparse_ccsr2csr_compress
  :outline:
.. doxygenfunction:: rocsparse_zcsr2csr_compress

rocsparse_inverse_permutation()
-------------------------------

.. doxygenfunction:: rocsparse_inverse_permutation

rocsparse_create_identity_permutation()
---------------------------------------

.. doxygenfunction:: rocsparse_create_identity_permutation

rocsparse_csrsort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_csrsort_buffer_size

rocsparse_csrsort()
-------------------

.. doxygenfunction:: rocsparse_csrsort

rocsparse_cscsort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_cscsort_buffer_size

rocsparse_cscsort()
-------------------

.. doxygenfunction:: rocsparse_cscsort

rocsparse_coosort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_coosort_buffer_size

rocsparse_coosort_by_row()
--------------------------

.. doxygenfunction:: rocsparse_coosort_by_row

rocsparse_coosort_by_column()
-----------------------------

.. doxygenfunction:: rocsparse_coosort_by_column

rocsparse_nnz_compress()
------------------------

.. doxygenfunction:: rocsparse_snnz_compress
  :outline:
.. doxygenfunction:: rocsparse_dnnz_compress
  :outline:
.. doxygenfunction:: rocsparse_cnnz_compress
  :outline:
.. doxygenfunction:: rocsparse_znnz_compress

rocsparse_nnz()
---------------

.. doxygenfunction:: rocsparse_snnz
  :outline:
.. doxygenfunction:: rocsparse_dnnz
  :outline:
.. doxygenfunction:: rocsparse_cnnz
  :outline:
.. doxygenfunction:: rocsparse_znnz


rocsparse_dense2csr()
---------------------

.. doxygenfunction:: rocsparse_sdense2csr
  :outline:
.. doxygenfunction:: rocsparse_ddense2csr
  :outline:
.. doxygenfunction:: rocsparse_cdense2csr
  :outline:
.. doxygenfunction:: rocsparse_zdense2csr


rocsparse_dense2csc()
---------------------

.. doxygenfunction:: rocsparse_sdense2csc
  :outline:
.. doxygenfunction:: rocsparse_ddense2csc
  :outline:
.. doxygenfunction:: rocsparse_cdense2csc
  :outline:
.. doxygenfunction:: rocsparse_zdense2csc


rocsparse_dense2coo()
---------------------

.. doxygenfunction:: rocsparse_sdense2coo
  :outline:
.. doxygenfunction:: rocsparse_ddense2coo
  :outline:
.. doxygenfunction:: rocsparse_cdense2coo
  :outline:
.. doxygenfunction:: rocsparse_zdense2coo


rocsparse_csr2dense()
---------------------

.. doxygenfunction:: rocsparse_scsr2dense
  :outline:
.. doxygenfunction:: rocsparse_dcsr2dense
  :outline:
.. doxygenfunction:: rocsparse_ccsr2dense
  :outline:
.. doxygenfunction:: rocsparse_zcsr2dense


rocsparse_csc2dense()
---------------------

.. doxygenfunction:: rocsparse_scsc2dense
  :outline:
.. doxygenfunction:: rocsparse_dcsc2dense
  :outline:
.. doxygenfunction:: rocsparse_ccsc2dense
  :outline:
.. doxygenfunction:: rocsparse_zcsc2dense

rocsparse_coo2dense()
---------------------

.. doxygenfunction:: rocsparse_scoo2dense
  :outline:
.. doxygenfunction:: rocsparse_dcoo2dense
  :outline:
.. doxygenfunction:: rocsparse_ccoo2dense
  :outline:
.. doxygenfunction:: rocsparse_zcoo2dense

rocsparse_prune_dense2csr_buffer_size()
---------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_buffer_size

rocsparse_prune_dense2csr_nnz()
-------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_nnz
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_nnz

rocsparse_prune_dense2csr()
---------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr

rocsparse_prune_csr2csr_buffer_size()
-------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_buffer_size

rocsparse_prune_csr2csr_nnz()
-----------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_nnz
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_nnz

rocsparse_prune_csr2csr()
-------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr

rocsparse_prune_dense2csr_by_percentage_buffer_size()
-----------------------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_by_percentage_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_by_percentage_buffer_size

rocsparse_prune_dense2csr_nnz_by_percentage()
---------------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_nnz_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_nnz_by_percentage

rocsparse_prune_dense2csr_by_percentage()
-----------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_by_percentage

rocsparse_prune_csr2csr_by_percentage_buffer_size()
---------------------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_by_percentage_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_by_percentage_buffer_size

rocsparse_prune_csr2csr_nnz_by_percentage()
-------------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_nnz_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_nnz_by_percentage

rocsparse_prune_csr2csr_by_percentage()
---------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_by_percentage

rocsparse_rocsparse_bsrpad_value()
----------------------------------

.. doxygenfunction:: rocsparse_sbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_dbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_cbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_zbsrpad_value
