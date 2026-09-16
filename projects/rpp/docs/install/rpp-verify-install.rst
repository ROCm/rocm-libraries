.. meta::
  :description: Verifying ROCm Performance Primitives installations
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, verifying

********************************************************************
Verifying the ROCm Performance Primitives installation
********************************************************************

After installation, verify that all the ROCm Performance Primitives (RPP) files have been copied to the right locations:

* Libraries: ``/opt/rocm/lib``
* Header files: ``/opt/rocm/include/rpp``
* Samples: ``/opt/rocm/share/rpp``
* Documentation: ``/opt/rocm/share/doc/rpp``

.. note::

    Packages built by TheRock install into a version-scoped prefix such as ``/opt/rocm/core-<major>.<minor>`` to support side-by-side installations. The ``/opt/rocm`` paths listed above remain valid because they are maintained as ``update-alternatives`` symlinks into the active prefix.

You can verify your installation using the CTest module. You will need to install the `test suite prerequisites <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rpp/utilities/test_suite/README.md>`_ before building and running the tests.

.. code-block:: shell

    mkdir rpp-test
    cd rpp-test
    cmake /opt/rocm/share/rpp/test/
    ctest -VV

To test RPP's full functionality, refer to the `RPP test suite <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rpp/utilities/test_suite/README.md>`_ for instructions on running image, voxel, audio, and miscellaneous augmentation tests.
