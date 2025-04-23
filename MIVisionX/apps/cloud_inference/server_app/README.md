# MIVisionX Inference Server

This Sample Inference Server supports:
* convert and maintain a database of pre-trained CAFFE models using [Model Compiler](https://github.com/ROCm/MIVisionX/tree/master/model_compiler/README.md#neural-net-model-compiler--optimizer)
* allow multiple TCP/IP client connections for inference work submissions
* multi-GPU high-throughput live streaming batch scheduler

Command-line usage:
````
  inference_server_app  [-p     <port>                           default:26262]
                        [-b     <batch size>                     default:64]
                        [-n     <model compiler path>            default:/opt/rocm/libexec/mivisionx/model_compiler/python]
                        [-fp16  <ON:1 or OFF:0>                  default:0]
                        [-w     <server working directory>       default:~/]
                        [-t     <num cpu decoder threads [2-64]> default:1]
                        [-gpu   <comma separated list of GPUs>]
                        [-q     <max pending batches>]
                        [-s     <local shadow folder full path>]
````

Make sure that all executables and libraries are in `PATH` and `LD_LIBRARY_PATH` environment variables.
````
% export PATH=$PATH:/opt/rocm/bin
% export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
````

The `inference_server_app` works with [Client Application](../client_app/README.md).
* Execute `inference_server_app` on the server machine with Radeon Instinct GPUs
* Execute `Client Application` on one or more workstations: connect to the server and classify images using any pre-trained neural network
