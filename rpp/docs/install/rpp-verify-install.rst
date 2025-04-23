.. meta::
  :description: Verifying ROCm Performance Primitives installations 
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, verifuing

********************************************************************
Verifying the ROCm Performance Primitives installation
********************************************************************

After installation, verify that all the ROCm Performance Primitives (RPP) files have been copied to the right locations:

* Libraries: ``/opt/rocm/lib``
* Header files: ``/opt/rocm/include/rpp``
* Samples: ``/opt/rocm/share/rpp``
* Documentation: ``/opt/rocm/share/doc/rpp``

You can verify your installation using the CTest module. You will need to install the `test suite prerequisites <https://github.com/ROCm/rpp/blob/develop/utilities/test_suite/README.md>`_ before building and running the tests.

.. code-block:: shell

    mkdir rpp-test
    cd rpp-test
    cmake /opt/rocm/share/rpp/test/
    ctest -VV

To test RPP's functionality, run ``testALLScript.sh``:

.. tab-set::
 
  .. tab-item:: HIP

    .. code:: shell

        cd rpp/utilities/rpp-unittests/HIP_NEW
        ./testAllScript.sh
  
  .. tab-item:: OpenCL

    .. code:: shell

        cd rpp/utilities/rpp-unittests/OCL_NEW
        ./testAllScript.sh


  .. tab-item:: CPU-only

    .. code:: shell
      
        cd rpp/utilities/rpp-unittests/HOST_NEW
        ./testAllScript.sh
    