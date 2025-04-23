# rocPyDecode samples

## Prerequisites
* [rocDecode C/C++ Library](https://github.com/ROCm/rocDecode)
* [rocPyDecode installed](../README.md#rocpydecode-install)
* [DLPack](https://pypi.org/project/dlpack/)
* [pytorch for ROCm](https://pytorch.org/get-started/locally/)
* [Python HIP](https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/0_install.html)

The torch python sample requires pytorch for ROCm, which can be installed as follow:

- If using bare-metal, `sudo` access is needed.
```bash
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
    sudo reboot 
```

- If using a docker environment or any system with `root` access, no need for reboot.
```bash
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
```

The performance sample requires python HIP, which can be installed as follows:

```
python3 -m pip install --upgrade pip
python3 -m pip install -i https://test.pypi.org/simple hip-python
```

## videodecode.py

This sample demuxes & decode frames from a video file, and optionally saves the frames to a file. \
To run this python sample script, you need to provide input video file full path name, other arguments are optional.

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                       : Show detail help message and exit
-i INPUT, --input INPUT                          : Input File Path - required
-o OUTPUT, --output OUTPUT                       : Output File Path - optional
-d DEVICE, --device DEVICE                       : GPU device ID - optional, default - 0
-m MEM_TYPE, --mem_type MEM_TYPE                 : Memory Type of output surface - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped - optional, default 0
-z ZERO_LATENCY, --zero_latency ZERO_LATENCY     : Force zero latency - [options: yes,no], default: no
-crop CROP_RECT, --crop_rect CROP_RECT           : Crop rectangle (left, top, right, bottom) - optional, default: None (no cropping)
-s SEEK, --seek SEEK                             : seek this number of frames, optional, default: no seek
-sm SEEK_MODE, --seek_mode SEEK_MODE             : seek mode, 0 - by exact frame number, 1 - by previous key frame, optional, default: 1 - by previous key frame
-sc SEEK_CRITERIA, --seek_criteria SEEK_CRITERIA : seek criteria, 0 - by frame number, 1 - by time stamp, optional, default: 0 - by frame number
-resize RESIZE_DIM RESIZE_DIM, --resize_dim RESIZE_DIM RESIZE_DIM : Width & Height of new resized frame, optional, default: no resizing
```

## videodecodeperf.py

This sample demuxes & decode frames from a video file on multiple processes. User can define the number of parallel jobs to observe performance scaling. \
To run this python sample script, you need to provide input video file full path name, other arguments are optional.



### Arguments

The following are full list of arguments that can be passed to the sample.

```bash
-h, --help                                                  : Show detail help message and exit
-i INPUT, --input INPUT                                     : Input File Path - required
-d DEVICE, --device DEVICE                                  : GPU device ID - optional, default - 0
-m MEM_TYPE, --mem_type MEM_TYPE                            : Memory Type of output surface - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped - optional, default 3
-t Number of Processes, --num_process Number of Processes   : Number of Processes - optional, default 4
```

## videodecodergb.py

This sample demuxes & decode frames from a video file, and convert it to an rgb frame. Optionally you can save the rgb frames to a file. The output is the rgb frame with the format the user specifies with -of or as default 'rgb' if not specified. To run this python sample script, you need to provide input video file full path name, other arguments are optional.

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                    : Show detail help message and exit
-i INPUT, --input INPUT                       : Input File Path - required
-o OUTPUT, --output OUTPUT                    : Output File Path - optional
-d DEVICE, --device DEVICE                    : GPU device ID - optional, default - 0
-m MEM_TYPE, --mem_type MEM_TYPE              : Memory Type of output surface - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped - optional, default 0
-z ZERO_LATENCY, --zero_latency ZERO_LATENCY  : Force zero latency - [options: yes,no], default: no
-crop CROP_RECT, --crop_rect CROP_RECT        : Crop rectangle (left, top, right, bottom) - optional, default: None (no cropping)
-of RGB_FORMAT, --rgb_format RGB_FORMAT       : Rgb Format to use - 1:bgr, 3:rgb, converts decoded YUV frame to Tensor in RGB format, optional, default: 3
```

## videodecodergb.py

This sample demuxes & decode frames from a video file, and convert it to an rgb frame. Optionally you can save the rgb frames to a file. The output is the rgb frame with the format the user specifies with -of or as default 'rgb' if not specified. To run this python sample script, you need to provide input video file full path name, other arguments are optional.

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                    : Show detail help message and exit
-i INPUT, --input INPUT                       : Input File Path - required
-o OUTPUT, --output OUTPUT                    : Output File Path - optional
-d DEVICE, --device DEVICE                    : GPU device ID - optional, default - 0
-m MEM_TYPE, --mem_type MEM_TYPE              : Memory Type of output surface - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped - optional, default 0
-z ZERO_LATENCY, --zero_latency ZERO_LATENCY  : Force zero latency - [options: yes,no], default: no
-crop CROP_RECT, --crop_rect CROP_RECT        : Crop rectangle (left, top, right, bottom) - optional, default: None (no cropping)
-of RGB_FORMAT, --rgb_format RGB_FORMAT       : Rgb Format to use - 1:bgr, 3:rgb, converts decoded YUV frame to Tensor in RGB format, optional, default: 3
```

## videodecodetorch.py

This sample demuxes & decode frames from a video file, and convert it to pytorch tensor via DLPack. Optionally you can save it to a file. \
To run this python sample script, you need to provide input video file full path name, other arguments are optional.

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                    : Show detail help message and exit
-i INPUT, --input INPUT                       : Input File Path - required
-o OUTPUT, --output OUTPUT                    : Output File Path - optional
-d DEVICE, --device DEVICE                    : GPU device ID - optional, default - 0
-m MEM_TYPE, --mem_type MEM_TYPE              : Memory Type of output surface - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped - optional, default 0
-z ZERO_LATENCY, --zero_latency ZERO_LATENCY  : Force zero latency - [options: yes,no], default: no
-crop CROP_RECT, --crop_rect CROP_RECT        : Crop rectangle (left, top, right, bottom) - optional, default: None (no cropping)
```

## videodecodetorch_yuv.py

This sample demuxes & decode frames from a video file, and convert it to pytorch tensor via DLPack, each component in a separate tensor. Optionally you can save it to a file, specifying saving the chroma plane U/V with -y no, otherwise will save the luma plane. \
To run this python sample script, you need to provide input video file full path name, other arguments are optional.

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                    : Show detail help message and exit
-i INPUT, --input INPUT                       : Input File Path - required
-o OUTPUT, --output OUTPUT                    : Output File Path - optional
-d DEVICE, --device DEVICE                    : GPU device ID - optional, default - 0
-m MEM_TYPE, --mem_type MEM_TYPE              : Memory Type of output surface - 0: Internal 1: dev_copied 2: host_copied 3: MEM not mapped - optional, default 0
-z ZERO_LATENCY, --zero_latency ZERO_LATENCY  : Force zero latency - [options: yes,no], default: no
-crop CROP_RECT, --crop_rect CROP_RECT        : Crop rectangle (left, top, right, bottom) - optional, default: None (no cropping)
```

## videodecodetorch_resnet50.py

This sample demuxes & decode frames from a video file, converts each frame to pytorch tensor via DLPack. \
It resizes the frame to match resnet50 model, and feed it to the model inference, printing out 5 possible predictions with precision ratio, for each frame it decodes. \
To run this python sample script, you need to provide input video file full path name, other argument is optional.

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                    : Show detail help message and exit
-i INPUT, --input INPUT                       : Input File Path - required
-d DEVICE, --device DEVICE                    : GPU device ID - optional, default - 0
```

## videodecode_mpi.py

This sample uses MPI to allow scalability while decoding video files. It scale the decoding by distributing it to local cores, or to what the user provide as MPI parameters. To run this python sample script, you need to provide input video file full path name, other arguments are optional. Proceed the python command with mpirun or mpiexec -n X, where X is the count of the processes/cores to use, or mpirun --hostfile filename, where filename is the file that contains all other external nodes information, please refer to MPI documentation. This example uses MPI to distribute the decoding task, it also uses MPI-IO to allow each node to save its portion of the task in the output file utilizing the built it IO mechanism MPI offers.

### Prerequisite

* Install mpi4py and av:
    * sudo apt-get install -y python3-mpi4py
    * sudo pip3 install av

### Example of shell command
This command will invoke the mpi to run the python example on 4 cores (locally), and saves the decoded frames into file named outputfile.yuv
* *mpiexec* -n 4 *python3* samples/videodecode_mpi.py **-i** /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-AV1.mp4 **-o** outputfile.yuv

### Arguments
The following are full list of arguments that can be passed to the sample.
```bash
-h, --help                                    : Show detail help message and exit
-i INPUT, --input INPUT                       : Input File Path - required
-o OUTPUT, --output OUTPUT                    : Output File Path - optional
```
