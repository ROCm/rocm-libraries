.. meta::
  :description: hipFFT documentation and API reference library
  :keywords: FFT, hipFFT, rocFFT, ROCm, API, documentation

.. _hipfft-overview:

********************************************************************
hipFFT overview
********************************************************************

hipFFT is a GPU FFT marshalling library that supports
either :doc:`rocFFT <rocfft:index>` or NVIDIA CUDA `cuFFT`_ as the backend.

hipFFT exports an interface that does not require the client to
change, regardless of the chosen backend. It sits between the
application and the backend FFT library, marshalling inputs into the
backend and results back to the application.

=====================
Basic hipFFT usage
=====================

To use hipFFT, follow this step-by-step process:

#. Create a transform plan for the FFT.

   To create a plan, use the functions :cpp:func:`hipfftPlan1d`, :cpp:func:`hipfftPlan2d`, or :cpp:func:`hipfftPlan3d`,
   depending on the dimensions of the FFT.

   For a 1D FFT, use the following code:

   .. code-block:: cpp

      hipfftHandle plan;
      hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);

   For higher-dimension plans, use :cpp:func:`hipfftPlan2d` or :cpp:func:`hipfftPlan3d`.

#. Allocate a work buffer (optional)

   hipFFT generally handles memory allocation internally, so work buffers aren't explicitly required.
   However, to manually manage memory, you can still
   allocate buffers before execution. You might want to do this, for example,
   if you have multiple plans that need work buffers and you want them to share a single buffer.
   Otherwise, each plan will allocate its own work memory, which might be wasteful.

#. Execute the plan

   To execute the FFT computation, use :cpp:func:`hipfftExecC2C`, :cpp:func:`hipfftExecR2C`, or :cpp:func:`hipfftExecC2R`,
   depending on the type of transform. You can reuse the same plan for multiple executions,
   changing the data pointers as necessary.

   .. code-block:: cpp

      hipfftExecC2C(plan, x, x, HIPFFT_FORWARD);

#. Destroy the plan

   After you are done with the plan, destroy it to free the associated resources:
     
   .. code-block:: cpp

      hipfftDestroy(plan);

#. Free any device memory (if applicable)

   If you allocated any buffers for storing input/output data or intermediate results, free them using ``hipFree``:

   .. code-block:: cpp

      hipFree(x);

#. Terminate the library

   No specific cleanup function is required for hipFFT, but ensure that any HIP memory is freed
   and the HIP runtime is cleaned up appropriately after all computations are done.

The following code sample illustrates how to apply these steps:

.. code-block:: cpp

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "hip/hip_vector_types.h"
   #include "hipfft/hipfft.h"

   int main()
   {
      hipfftHandle plan;
      size_t N = 16;
      size_t Nbytes = N * sizeof(hipfftComplex);

      // Create HIP device buffer
      hipfftComplex *x;
      hipMalloc(&x, Nbytes);

      // Initialize data
      std::vector<hipfftComplex> cx(N);
      for (size_t i = 0; i < N; i++)
      {
         cx[i].x = 1;
         cx[i].y = -1;
      }

      // Copy data to device
      hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);

      // Create hipFFT plan
      hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);

      // Execute plan
      hipfftExecC2C(plan, x, x, HIPFFT_FORWARD);

      // Wait for execution to finish
      hipDeviceSynchronize();

      // Copy result back to host
      std::vector<hipfftComplex> y(N);
      hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

      // Print results
      for (size_t i = 0; i < N; i++)
      {
         std::cout << y[i].x << ", " << y[i].y << std::endl;
      }

      // Free device buffer
      hipFree(x);

      // Destroy plan
      hipfftDestroy(plan);

      return 0;
   }

========================
Advanced hipFFT usage
========================

For transforms that require advanced input layouts, use the :cpp:func:`hipfftPlanMany` function, setting these parameters:

*  ``int rank``: The number of dimensions for the FFT (1D, 2D, or 3D).
*  ``int* n``: Array specifying the size of the FFT in each dimension.
*  ``int* inembed``: The dimensions of the input data layout in memory.
*  ``int istride``: Stride between elements in the input array.
*  ``int idist``: Distance between consecutive FFTs in the input array.
*  ``int* onembed``: The dimensions of the output data layout in memory.
*  ``int ostride``: Stride between elements in the output array.
*  ``int odist``: Distance between consecutive FFTs in the output array.
*  ``hipfftType type``: Type of FFT (for example, ``HIPFFT_C2C`` or ``HIPFFT_R2C``).
*  ``int batch``: Number of FFTs to compute in parallel.

Here's an example of a 2D single-precision real-to-complex transform using the hipFFT advanced interface:

.. code-block:: cpp

   #include <complex>
   #include <hipfft/hipfft.h>
   #include <iostream>
   #include <vector>
   #include <hip/hip_runtime_api.h>

   int main()
   {
      // Define the parameters for the 2D FFT
      int rank    = 2;            // Rank of the transform (2D FFT)
      int n[2]    = {4, 5};       // Dimensions of the FFT (4 rows, 5 columns)
      int howmany = 3;            // Number of transforms to compute (batch size)

      // Derived parameters for handling real-to-complex output
      int n1_complex_elements      = n[1] / 2 + 1; // Number of complex elements in the last dimension
      int n1_padding_real_elements = n1_complex_elements * 2; // Adjusted real elements to account for padding

      // Strides and distances
      int istride    = 1; // Stride between elements in input
      int ostride    = istride; // Stride between elements in output
      int inembed[2] = {n[0], n1_padding_real_elements}; // Input layout
      int onembed[2] = {n[0], n1_complex_elements};      // Output layout
      int idist      = istride * inembed[0] * inembed[1]; // Distance between batches in input
      int odist      = ostride * onembed[0] * onembed[1]; // Distance between batches in output

      // Print the layout parameters
      std::cout << "n: " << n[0] << " " << n[1] << "\n"
               << "howmany: " << howmany << "\n"
               << "istride: " << istride << "\tostride: " << ostride << "\n"
               << "inembed: " << inembed[0] << " " << inembed[1] << "\n"
               << "onembed: " << onembed[0] << " " << onembed[1] << "\n"
               << "idist: " << idist << "\todist: " << odist << "\n"
               << std::endl;

      // Initialize input data
      std::vector<float> data(howmany * idist); // Allocate space for batched input
      const auto total_bytes = data.size() * sizeof(decltype(data)::value_type);

      std::cout << "input:\n";
      std::fill(data.begin(), data.end(), 0.0); // Fill data with zeros
      for(int ibatch = 0; ibatch < howmany; ++ibatch)
      {
         for(int i = 0; i < n[0]; i++) // Loop over rows
         {
               for(int j = 0; j < n[1]; j++) // Loop over columns
               {
                  // Calculate the position in the input array
                  const auto pos = ibatch * idist + istride * (i * inembed[1] + j);
                  data[pos]      = i + ibatch + j; // Populate data with unique values for clarity
               }
         }
      }

      // Print the input data for each batch
      for(int ibatch = 0; ibatch < howmany; ++ibatch)
      {
         std::cout << "batch: " << ibatch << "\n";
         for(int i = 0; i < n[0]; i++)
         {
               for(int j = 0; j < n[1]; j++)
               {
                  const auto pos = ibatch * idist + istride * (i * inembed[1] + j);
                  std::cout << data[pos] << " ";
               }
               std::cout << "\n";
         }
         std::cout << "\n";
      }
      std::cout << std::endl;

      // Create the hipFFT plan for batched 2D real-to-complex transforms
      hipfftHandle hipForwardPlan;
      hipfftResult hipfft_rt = hipfftPlanMany(&hipForwardPlan,
                                             rank,
                                             n,
                                             inembed,
                                             istride,
                                             idist,
                                             onembed,
                                             ostride,
                                             odist,
                                             HIPFFT_R2C, // Transform type (real-to-complex)
                                             howmany);   // Number of transforms in the batch
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("failed to create plan");

      // Allocate GPU memory for input and output
      hipfftReal* gpu_data;
      hipError_t hip_rt = hipMalloc((void**)&gpu_data, total_bytes);
      if(hip_rt != hipSuccess)
         throw std::runtime_error("hipMalloc failed");

      // Copy input data to the GPU
      hip_rt = hipMemcpy(gpu_data, (void*)data.data(), total_bytes, hipMemcpyHostToDevice);
      if(hip_rt != hipSuccess)
         throw std::runtime_error("hipMemcpy failed");

      // Execute the FFT on the GPU
      hipfft_rt = hipfftExecR2C(hipForwardPlan, gpu_data, (hipfftComplex*)gpu_data);
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("failed to execute plan");

      // Copy the output data back to the host
      hip_rt = hipMemcpy((void*)data.data(), gpu_data, total_bytes, hipMemcpyDeviceToHost);
      if(hip_rt != hipSuccess)
         throw std::runtime_error("hipMemcpy failed");

      // Display the output data
      std::cout << "output:\n";
      const std::complex<float>* output = (std::complex<float>*)data.data();
      for(int ibatch = 0; ibatch < howmany; ++ibatch)
      {
         std::cout << "batch: " << ibatch << "\n";
         for(int i = 0; i < n[0]; i++)
         {
               for(int j = 0; j < n1_complex_elements; j++)
               {
                  const auto pos = ibatch * odist + ostride * (i * onembed[1] + j);
                  std::cout << output[pos] << " ";
               }
               std::cout << "\n";
         }
         std::cout << "\n";
      }
      std::cout << std::endl;

      // Clean up resources
      hipfftDestroy(hipForwardPlan); // Destroy the FFT plan

      hip_rt = hipFree(gpu_data); // Free the GPU memory
      if(hip_rt != hipSuccess)
         throw std::runtime_error("hipFree failed");

      return 0;
   }

======================
Overlapping input data
======================

There are signal processing tasks, such as sliding window FFTs, 
where overlapping data can improve computational efficiency. 
Care must be taken to ensure proper memory management and alignment when using 
overlapping input layouts.  

The following example demonstrates the use of overlapping input data by configuring 
the ``inembed``, ``istride``, and ``idist`` parameters in the :cpp:func:`hipfftMakePlanMany` 
function. Set these parameters to create a memory layout where portions of 
the input data are reused across multiple FFT batches: 

*  ``inembed`` specifies the physical layout of the input data in memory, with 
   extra padding to accommodate overlapping rows (for example, ``2240``).

*  ``istride`` ensures continuous reading of data within each row (if set to ``1``).

*  ``idist`` defines the distance between the starting points of consecutive batches 
   (for example, ``432``), which is smaller than the total memory implied by 
   ``xformSz`` and ``inembed``.


.. code-block:: cpp

   #include <hipfft/hipfft.h>
   #include <hip/hip_runtime.h>
   #include <cstdio>
   #include <cstdlib>
   #include <vector>
   #include <iostream>
   #include <complex>

   int main()
   {
      std::cout << "hipFFT 2D batched complex-to-complex transform example\n";

      // FFT configuration
      int rank = 2;
      int xformSz[2] = {512, 512};      // 2D FFT size: 512x512
      int inEmbed[2] = {512, 2240};     // Input data layout
      int onEmbed[2] = {512, 512};      // Output data layout
      int istride = 1, ostride = 1;     // Stride for input and output
      int idist = 432, odist = 262144;  // Batch distance for input and output
      int batch = 5;                    // Number of FFTs to compute in parallel

      // Calculate input and output sizes in bytes
      // Input data sequences are all within an array of inEmbed[0] x inEmbed[1] complex
      // floating-point values (of elementary stride istride):
      size_t inSize = istride * inEmbed[0] * inEmbed[1] * sizeof(std::complex<float>);
      // Output data sequences are made of odist complex floating-point values, stored
      // contiguously without overlap:
      size_t outSize = odist * batch * sizeof(std::complex<float>);

      // Initialize HIP and hipFFT resources
      hipSetDevice(0);
      hipStream_t stream;
      hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

      hipfftHandle handleF;
      if (hipfftPlanMany(&handleF, rank, xformSz, inEmbed, istride, idist, onEmbed, ostride, odist, HIPFFT_C2C, batch) != HIPFFT_SUCCESS)
      {
         std::cerr << "Failed to create hipFFT plan" << std::endl;
         return EXIT_FAILURE;
      }

      hipfftSetStream(handleF, stream);

      // Allocate device memory
      std::complex<float>* miTD; // Input buffer
      std::complex<float>* miFD; // Output buffer
      if (hipMalloc(&miTD, inSize) != hipSuccess || hipMalloc(&miFD, outSize) != hipSuccess)
      {
         std::cerr << "hipMalloc failed" << std::endl;
         return EXIT_FAILURE;
      }

      // Initialize input data on the host
      std::vector<std::complex<float>> inputData(istride * inEmbed[0] * inEmbed[1], {0.0f, 0.0f});
      for (int ibatch = 0; ibatch < batch; ++ibatch)
      {
         for (int i = 0; i < xformSz[0]; ++i)
         {
               for (int j = 0; j < xformSz[1]; ++j)
               {
                  int pos = ibatch * idist + istride * (i * inEmbed[1] + j);
                  inputData[pos] = std::complex<float>(i + j, ibatch);
               }
         }
      }

      // Copy input data to device
      if (hipMemcpy(miTD, inputData.data(), inSize, hipMemcpyHostToDevice) != hipSuccess)
      {
         std::cerr << "hipMemcpy failed" << std::endl;
         return EXIT_FAILURE;
      }

      // Execute FFT
      if (hipfftExecC2C(handleF, reinterpret_cast<hipfftComplex*>(miTD), reinterpret_cast<hipfftComplex*>(miFD), HIPFFT_FORWARD) != HIPFFT_SUCCESS)
      {
         std::cerr << "Failed to execute hipFFT" << std::endl;
         return EXIT_FAILURE;
      }

      // Synchronize stream
      hipStreamSynchronize(stream);

      // Copy results back to host
      std::vector<std::complex<float>> outputData(odist * batch);
      if (hipMemcpy(outputData.data(), miFD, outSize, hipMemcpyDeviceToHost) != hipSuccess)
      {
         std::cerr << "hipMemcpy failed" << std::endl;
         return EXIT_FAILURE;
      }

      // Display results
      std::cout << "Output data:\n";
      for (int ibatch = 0; ibatch < batch; ++ibatch)
      {
         std::cout << "Batch " << ibatch << ":\n";
         for (int i = 0; i < xformSz[0]; ++i)
         {
               for (int j = 0; j < xformSz[1]; ++j)
               {
                  int pos = ibatch * odist + ostride * (i * onEmbed[1] + j);
                  std::cout << outputData[pos] << " ";
               }
               std::cout << "\n";
         }
         std::cout << "\n";
      }

      // Clean up resources
      hipfftDestroy(handleF);
      hipStreamDestroy(stream);
      hipFree(miTD);
      hipFree(miFD);

      return EXIT_SUCCESS;
   }

=================
Multi-GPU example
=================

The following example illustrates how to compute a 2D double-precision complex-to-complex transform across two GPUs using the hipFFT library.
The following concepts and API calls are used:

*  ``hipfftXt``: This API lets users execute FFTs across multiple GPUs by managing multi-GPU plans.
   ``hipfftXt`` provides an extended version of the hipFFT API to handle GPU-specific operations, such as memory allocation
   and execution across multiple devices. For more details, see the :doc:`API reference <../reference/fft-api-usage>`.

*  :cpp:func:`hipfftCreate`: Creates a hipFFT plan that contains the FFT configuration. This plan is used to configure
   the FFT transform operation.

*  ``hipStreamCreate``: Creates a stream for managing GPU work concurrently. This enables execution of the multi-GPU plan
   in parallel on multiple GPUs. For more details, see :doc:`HIP <hip:index>`.
  
*  :cpp:func:`hipfftXtSetGPUs`: Assigns the GPUs (in this case, two GPUs) to the hipFFT plan,
   enabling multi-GPU computation for the FFT.
  
*  :cpp:func:`hipfftMakePlan2d`: Creates a 2D FFT plan for the specified input/output size (``Nx``, ``Ny``), specifying
   the transform type (complex-to-complex in this case).
  
*  :cpp:func:`hipfftXtMalloc`: Allocates memory on the GPUs for storing the FFT input and output data.
  
*  :cpp:func:`hipfftXtMemcpy`: Copies data between the host and GPU memory, supporting both host-to-device and
   device-to-host operations.
  
*  :cpp:func:`hipfftXtExecDescriptor`: Executes the FFT operation based on the input descriptor ``desc``,
   which holds the input data and transform configuration.
  
*  :cpp:func:`hipfftXtFree`: Frees the memory allocated for the input/output descriptors after the computation is completed.

For detailed API usage, see :ref:`hipfft-api-usage`.

.. code-block:: cpp

   #include <complex>
   #include <iostream>
   #include <vector>
   #include <hipfft/hipfft.h>
   #include <hipfft/hipfftXt.h>
   #include <hip/hip_runtime_api.h>

   int main()
   {
      // Define FFT dimensions
      const int Nx = 512;
      const int Ny = 512;
      int direction = HIPFFT_FORWARD; // forward = -1, backward = 1

      // Initialize input data (complex numbers) for FFT computation
      int verbose = 0;
      std::vector<std::complex<double>> cinput(Nx * Ny);
      for(size_t i = 0; i < Nx * Ny; i++)
      {
         cinput[i] = i;  // Initialize the data with some values
      }

      // Optionally, print the input data
      if(verbose)
      {
         std::cout << "Input:\n";
         for(int i = 0; i < Nx; i++)
         {
               for(int j = 0; j < Ny; j++)
               {
                  int pos = i * Ny + j;
                  std::cout << cinput[pos] << " ";
               }
               std::cout << "\n";
         }
         std::cout << std::endl;
      }

      // Specify the GPUs you want to use for multi-GPU setup
      std::vector<int> gpus = {0, 1};  // Use GPU 0 and GPU 1

      // Create a multi-GPU plan
      hipLibXtDesc* desc; // Input descriptor for the Xt format
      hipfftHandle plan;  // Plan handle

      // Create the FFT plan
      if(hipfftCreate(&plan) != HIPFFT_SUCCESS)
         throw std::runtime_error("failed to create plan");

      // Create a GPU stream and assign it to the plan for asynchronous operations
      hipStream_t stream{};
      if(hipStreamCreate(&stream) != hipSuccess)
         throw std::runtime_error("hipStreamCreate failed.");
      if(hipfftSetStream(plan, stream) != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftSetStream failed.");

      // Assign GPUs to the plan (this is where multi-GPU is specified)
      hipfftResult hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftXtSetGPUs failed.");

      // Make the 2D plan for FFT (this defines the 2D FFT using the specified dimensions)
      size_t workSize[gpus.size()];
      hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize);
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftMakePlan2d failed.");

      // Allocate memory for input data on the GPUs (Xt format handles the data distribution)
      hipfftXtSubFormat_t format = HIPFFT_XT_FORMAT_INPLACE;
      hipfft_rt = hipfftXtMalloc(plan, &desc, format);  // Allocate memory for the descriptor
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftXtMalloc failed.");

      // Copy the input data to the GPUs (device memory)
      hipfft_rt = hipfftXtMemcpy(plan,
                                 reinterpret_cast<void*>(desc),
                                 reinterpret_cast<void*>(cinput.data()),
                                 HIPFFT_COPY_HOST_TO_DEVICE);  // Copy from host to device
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftXtMemcpy failed.");

      // Execute the FFT computation using the Xt descriptor
      hipfft_rt = hipfftXtExecDescriptor(plan, desc, desc, direction);
      if(hipfft_rt != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftXtExecDescriptor failed.");

      // Optionally, print the output data (copy the results back to the host)
      if(verbose)
      {
         // Copy the output data back to the host
         hipfft_rt = hipfftXtMemcpy(plan,
                                    reinterpret_cast<void*>(cinput.data()),
                                    reinterpret_cast<void*>(desc),
                                    HIPFFT_COPY_DEVICE_TO_HOST);  // Copy from device to host
         if(hipfft_rt != HIPFFT_SUCCESS)
               throw std::runtime_error("hipfftXtMemcpy D2H failed.");

         std::cout << "Output:\n";
         for(size_t i = 0; i < Nx; i++)
         {
               for(size_t j = 0; j < Ny; j++)
               {
                  auto pos = i * Ny + j;
                  std::cout << cinput[pos] << " ";  // Print the output FFT results
               }
               std::cout << "\n";
         }
         std::cout << std::endl;
      }

      // Clean up memory and resources
      if(hipfftXtFree(desc) != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftXtFree failed.");

      if(hipfftDestroy(plan) != HIPFFT_SUCCESS)
         throw std::runtime_error("hipfftDestroy failed.");

      if(hipStreamDestroy(stream) != hipSuccess)
         throw std::runtime_error("hipStreamDestroy failed.");

      return 0;
   }


.. _rocFFT: https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html
.. _cuFFT: https://developer.nvidia.com/cufft
