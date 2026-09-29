.. meta::
  :description: Call an RPP tensor augmentation from C++
  :keywords: RPP, ROCm, tensor, augmentation, rppt_brightness, rppCreate, HIP, HOST

*************************************
Run an RPP tensor augmentation
*************************************

Tensor augmentations in ROCm Performance Primitives (RPP) take a source buffer, a destination buffer, and tensor descriptors. They also take per-image parameter tensors, a region of interest, an RPP handle, and a backend.

The brightness example in `ROCm/rocm-examples <https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/RPP/brightness>`_ runs that sequence on the HIP backend. ``rppt_brightness()`` scales each pixel by a per-image ``alpha`` and adds a per-image ``beta``.

Get the example
===============

Clone the examples repository and change to the brightness directory:

.. code:: shell

    git clone https://github.com/ROCm/rocm-examples.git
    cd rocm-examples/Libraries/RPP/brightness

The RPP examples build on Linux only. They need a ROCm installation with RPP, OpenCV for image reading and writing, and OpenMP. OpenCV isn't part of ROCm, so the build skips the RPP examples when it's missing.

Describe the tensors
====================

``RpptDesc`` holds the layout, data type, batch size, extents, and strides for a batch. The example declares a source and destination descriptor, then fills them in three steps:

.. code-block:: cpp

    #include <rpp.h>

    RpptDesc src_desc, dst_desc;
    RpptDescPtr src_desc_ptr = &src_desc;
    RpptDescPtr dst_desc_ptr = &dst_desc;

    set_descriptor_layout(src_desc_ptr, dst_desc_ptr, layout_type,
                          output_format_toggle);
    set_descriptor_data_type(input_bit_depth, src_desc_ptr, dst_desc_ptr);
    set_descriptor_dims_and_strides(src_desc_ptr, num_images, max_height, max_width,
                                    input_channels, offset_in_bytes);

Those three functions are in ``Common/rpp_utils.hpp`` in the examples repository. They're example code, not RPP API. ``set_descriptor_layout()`` sets ``layout`` to ``RpptLayout::NHWC`` for a packed batch and ``RpptLayout::NCHW`` for a planar one. ``set_descriptor_data_type()`` maps the bit depth to ``RpptDataType``. ``set_descriptor_dims_and_strides()`` sets ``numDims``, ``offsetInBytes``, ``n``, ``h``, ``w``, ``c``, and the four strides, and it rounds the width up to a multiple of 8.

Allocate device buffers and pinned parameter tensors
====================================================

On the HIP backend, the source and destination buffers must be HIP device memory. The example sizes each one from its descriptor, then copies the decoded image data across:

.. code-block:: cpp

    const Rpp64u io_buffer_size = static_cast<Rpp64u>(src_desc_ptr->h) *
                                  src_desc_ptr->w * src_desc_ptr->c * num_images;
    const Rpp64u input_buffer_size =
        io_buffer_size * get_size_of_data_type(src_desc_ptr->dataType) +
        src_desc_ptr->offsetInBytes;

    void* d_input;
    void* d_output;
    hipMalloc(&d_input, input_buffer_size);
    hipMalloc(&d_output, output_buffer_size);
    hipMemcpy(d_input, input, input_buffer_size, hipMemcpyHostToDevice);

``get_size_of_data_type()`` is another helper from ``Common/rpp_utils.hpp``. It returns the byte size for the ``RpptDataType`` in the descriptor. ``output_buffer_size`` is computed the same way from ``dst_desc_ptr``.

RPP reads the ROI and the parameter tensors from the device, so the example allocates them as pinned host memory with ``hipHostMalloc()``. Its comment states that ordinary pageable memory doesn't work here:

.. code-block:: cpp

    RpptROI* roi_tensor_ptr_src;
    hipHostMalloc(&roi_tensor_ptr_src, num_images * sizeof(RpptROI));
    RpptRoiType roi_type_src = RpptRoiType::XYWH;

    Rpp32f* alpha_tensor;
    Rpp32f* beta_tensor;
    hipHostMalloc(&alpha_tensor, num_images * sizeof(Rpp32f));
    hipHostMalloc(&beta_tensor, num_images * sizeof(Rpp32f));

    for (int i = 0; i < num_images; i++) {
        roi_tensor_ptr_src[i].xywhROI.xy.x = 0;
        roi_tensor_ptr_src[i].xywhROI.xy.y = 0;
        roi_tensor_ptr_src[i].xywhROI.roiWidth = image.cols;
        roi_tensor_ptr_src[i].xywhROI.roiHeight = image.rows;
        alpha_tensor[i] = alpha;
        beta_tensor[i] = beta;
    }

The example keeps a second ROI tensor and an ``RpptImagePatch`` array for the destination. Those describe the output images for the OpenCV writer, and the augmentation call doesn't use them.

Create a handle and run the augmentation
========================================

``rppCreate()`` allocates the handle and its memory for one batch. On HIP, pass ``0`` for the thread count and a ``hipStream_t``:

.. code-block:: cpp

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, num_images, 0, stream, backend);

The augmentation call takes the buffers, the descriptors, the parameter tensors, the ROI and its type, the handle, and the backend. The call is asynchronous on HIP, so synchronize before you read the destination buffer:

.. code-block:: cpp

    rppt_brightness(d_input, src_desc_ptr, d_output, dst_desc_ptr, alpha_tensor,
                    beta_tensor, roi_tensor_ptr_src, roi_type_src, handle, backend);
    hipDeviceSynchronize();

``rppt_brightness()`` returns ``RppStatus``, and ``rppCreate()`` returns ``rppStatus_t``. The example checks both through its own ``RPP_CHECK`` macro, which exits when the status isn't ``RPP_SUCCESS`` or ``rppStatusSuccess``.

Release the handle with the same backend you created it with, then free the pinned and device allocations:

.. code-block:: cpp

    rppDestroy(handle, backend);
    hipHostFree(roi_tensor_ptr_src);
    hipHostFree(alpha_tensor);
    hipHostFree(beta_tensor);
    hipFree(d_input);
    hipFree(d_output);
    hipStreamDestroy(stream);

Build and run the example
=========================

The example directory ships a Makefile that compiles ``main.cpp`` with ``hipcc`` into an executable named ``rpp_brightness``:

.. code:: shell

    make
    ./rpp_brightness

With no options, the example reads the images in ``Libraries/RPP/data/images`` and writes to ``./output``. The default ``alpha`` is ``1.75`` and the default ``beta`` is ``50``. ``--layout`` selects PKD3, PLN3, or PLN1, and ``--bit-depth`` selects the input and output data types:

.. code:: shell

    ./rpp_brightness --input image_folder --output output_folder --alpha 1.75 --beta 50

Set ``ROCM_PATH`` if ROCm isn't installed in ``/opt/rocm``.

Link RPP
========

Before you build against RPP, :doc:`install it <../install/rpp-install>` or :doc:`build it from source <../install/rpp-build>`.

The top-level ``Libraries/RPP/CMakeLists.txt`` tries ``find_package(rpp QUIET)`` first. When no package configuration is present, it locates ``librpp`` with ``find_library()`` and the headers with ``find_path()``, then creates an imported target named ``rpp``. Each example links the library directly and adds the RPP include directory:

.. code-block:: cmake

    target_link_libraries(${example_name} PRIVATE
        -lrpp
        ${OpenCV_LIBS}
        ${OMP_LIBRARY}
    )

    target_include_directories(${example_name} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        "${ROCM_PATH}/include/rpp"
    )

The Makefile build does the same thing with ``-isystem $(ROCM_PATH)/include``, ``-I $(ROCM_PATH)/include/rpp``, ``-L $(ROCM_PATH)/lib``, and ``-lrpp``.

Other tensor APIs follow the same pattern: descriptors, per-image parameter tensors, ROI, handle, and backend. The :doc:`supported functionalities <../reference/rpp-supported-functionalities>` tables list which operations exist on HOST and HIP. The :doc:`example outputs <../reference/rpp-supported-func-and-var-examples>` page shows sample input and output images.
