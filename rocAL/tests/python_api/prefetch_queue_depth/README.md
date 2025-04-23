## Test

This application briefs how to configure and pass the prefetch_queue_depth pipeline arugument.
This  test  aims to evaluate the prefetch queue depth support and the perfomance boost up.

## Prefetch queue depth

* The rocAL pipeline allows the buffering of one or more batches of data.
* This can be achieved by configuring the prefetch_queue_depth pipeline argument which sets the depth of buffer in both load and output routine.
* Configuring this paramter controls the buffer that keeps the decoded image batch ready for processing and the processed image batch ready for user.
* The default prefetch depth is 2.
* Depending on the machine configuration decreasing or increasing   prefetch_queue_depth helps in achieving better performance.

## Running the app

`python3 prefetch_queue_depth.py  <path to the dataset> <cpu/gpu> <batch_size> <prefetch_queue_depth>`

## Example

* Run with 5000 images from COCO2017 Val dataset and batch size 128 on AMD Eng Sample: 100-000000248-08_35/21_N Processor with nproc - 128.

prefetch_queue_depth as 2

```bash
OK: loaded 82 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  434458 milli seconds
```

prefetch_queue_depth as 4

```bash
OK: loaded 82 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  433953 milli seconds
```

prefetch_queue_depth as 8

```bash
OK: loaded 82 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  420856 milli seconds
```
