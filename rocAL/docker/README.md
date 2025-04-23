# rocAL Docker

Docker is a set of platform as a service (PaaS) products that use OS-level virtualization to deliver software in packages called containers. [Read More](https://github.com/ROCm/MIVisionX/wiki/Docker)

## Build - dockerfiles

```shell
sudo docker build --build-arg {ARG_1_NAME}={ARG_1_VALUE} [--build-arg {ARG_2_NAME}={ARG_2_VALUE}] -f {DOCKER_FILE_NAME}.dockerfile -t {DOCKER_IMAGE_NAME} .
```

## ARG options

* [Pytorch docker](https://hub.docker.com/r/rocm/pytorch/tags): 

```
PYTORCH_VERSION:                    rocm/pytorch docker tag
ROCAL_PYTHON_VERSION_SUGGESTED:     Python version if required for rocal_pybind
```

* [Tensorflow docker](https://hub.docker.com/r/rocm/tensorflow/tags): 

```
TENSORFLOW_VERSION:                 rocm/tensorflow docker tag
ROCAL_PYTHON_VERSION_SUGGESTED:     Python version if required for rocal_pybind
```

