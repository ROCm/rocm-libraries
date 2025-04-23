# Chapter 6: Using with C++ API

This chapter explains how to create a pipeline and add augmentations using C++ APIs directly. The Python APIs also call these C++ APIs internally using the Python pybind utility as explained in the section Installing rocAL Python Package.

## 6.1 C++ Common APIs

The following sections list the commonly used C++ APIs.

## 6.1.1 rocalCreate

Use: To create the pipeline

Returns: The context for the pipeline

Arguments:

* RocalProcessMode: Defines whether rocal data loading should be on the CPU or [GPU](https://github.com/ROCm/rocAL/blob/master/rocAL/include/api/rocal_api_types.h#L91)

```c++
RocalProcessMode::ROCAL_PROCESS_GPU
RocalProcessMode::ROCAL_PROCESS_CPU
```

* RocalTensorOutputType: Defines whether the output of rocal tensor is FP32 or [FP16](https://github.com/ROCm/rocAL/blob/master/rocAL/include/api/rocal_api_types.h#L124)

```c++
RocalTensorOutputType::ROCAL_FP32
RocalTensorOutputType::ROCAL_FP16
```

[Example](https://github.com/ROCm/rocAL/blob/master/rocAL/include/api/rocal_api.h#L41):

```c++
extern "C"  RocalContext  ROCAL_API_CALL rocalCreate(size_t batch_size, RocalProcessMode affinity, int gpu_id = 0, size_t cpu_thread_count = 1, size_t prefetch_queue_depth = 3, RocalTensorOutputType output_tensor_data_type = RocalTensorOutputType::ROCAL_FP32);
```

## 6.1.2 rocalVerify

Use: To verify the graph for all the inputs and outputs

Returns: A status code indicating the success or failure

[Example](https://github.com/ROCm/rocAL/blob/master/rocAL/include/api/rocal_api.h#L47):

```c++
extern "C"  RocalStatus ROCAL_API_CALL rocalVerify(RocalContext context);
```

## 6.1.3 rocalRun

Use: To process and run the built and verified graph

Returns: A status code indicating the success or failure

[Example](https://github.com/ROCm/rocAL/blob/master/rocAL/include/api/rocal_api.h#L52):

```c++
extern "C"  RocalStatus  ROCAL_API_CALL rocalRun(RocalContext context);
```

## 6.1.4 rocalRelease

Use: To free all the resources allocated during the graph creation process

Returns: A status code indicating the success or failure

[Example](https://github.com/ROCm/rocAL/blob/master/rocAL/include/api/rocal_api.h#L57):

```c++
extern "C"  RocalStatus  ROCAL_API_CALL rocalRelease(RocalContext rocal_context);
```

## 6.1.5 Image Augmentation Using C++ API

The example below shows how to create a pipeline, read JPEG images, perform certain augmentations on them, and show the output using OpenCV by utilizing C++ [APIs](https://github.com/ROCm/rocAL/blob/develop/tests/cpp_api/image_augmentation/image_augmentation.cpp#L103).

```c++
Auto handle = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1);
input1 = rocalJpegFileSource(handle, folderPath1,  color_format, shard_count, false, shuffle, false, ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height, dec_type);

image0 = rocalResize(handle, input1, resize_w, resize_h, true);

RocalImage image1 = rocalRain(handle, image0, false);

RocalImage image11 = rocalFishEye(handle, image1, false);


rocalRotate(handle, image11, true, rand_angle);


// Creating successive blur nodes to simulate a deep branch of augmentations
RocalImage image2 = rocalCropResize(handle, image0, resize_w, resize_h, false, rand_crop_area);;
for(int i = 0 ; i < aug_depth; i++) {
    image2 = rocalBlurFixed(handle, image2, 17.25, (i == (aug_depth -1)) ? true:false );
}
// Calling the API to verify and build the augmentation graph
if(rocalVerify(handle) != ROCAL_OK)
{
    std::cout << "Could not verify the augmentation graph" << std::endl;
    return -1;
}

while (!rocalIsEmpty(handle)) {
    if(rocalRun(handle) != 0)
        break;
}
```

To see a sample image augmentation application in C++, click [here](https://github.com/ROCm/rocAL/tree/develop/tests/cpp_api/image_augmentation). 
