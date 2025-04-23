# Video decode batch sample

This sample decodes multiple files using multiple threads, using the rocDecode library. The input is a directory of files and an input number of threads. The maximum number of threads is capped to 64.
If the number of files is higher than the number of threads requested by the user, the files are distributed to the threads in a round robin fashion. 
If the number of files is lesser than the number of threads requested by the user, the number of threads created will be equal to the number of files.

## Prerequisites:

* Install [rocDecode](../../README.md#build-and-install-instructions)

* [FFMPEG](https://ffmpeg.org/about.html)

    * On `Ubuntu`

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```
  
    * On `RHEL`/`SLES` - install ffmpeg development packages manually or use [rocDecode-setup.py](../../rocDecode-setup.py) script

## Build

```shell
mkdir video_decode_batch && cd video_decode_batch
cmake ../
make -j
```

## Run

```shell
./videodecodebatch -i <directory containing input video files [required]> 
                                   -t <number of threads [optional - default:4]>
                                   -d <Device ID (>= 0) [optional - default:0]>
```