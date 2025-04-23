# NFNet Inference with MIGraphX

## NFNet
NFNet: Normalizer-Free Nets. An image recognition model that can be trained without batch normalization layers. It instead uses gradient clipping algorithm to provide same affects of BatchNorm.

<ins>**Summary:**</ins>
- SOTA on ImageNet (86.5% top-1 w/o extra data)
- Up to 8.7x faster to train than EfficientNets to a given accuracy
- Normalizer-free (no BatchNorm)

**Paper**: https://arxiv.org/pdf/2102.06171.pdf

**Colab notebook**: https://github.com/deepmind/deepmind-research/tree/master/nfnets

### Why not batch norm?

Batch normalization has three significant practical disadvantages:
1. It is an expensive computational primitive, which incurs memory overhead and significantly increases the time required to evaluate the gradient in some networks.
2. It introduces a discrepancy between the behavior of the model during training and at inference time, introducing hidden hyper-parameters that have to be tuned.
3. Last and most important point, batch normalization breaks the independence between training examples in the minibatch (batch size matters with batch norm, distributed training becomes extremely cumbersome).

Instead:

- Authors provide Adaptive Gradient Clipping (AGC), which clips gradients based on the unit-wise ratio of gradient norms to parameter norms, and they demonstrate that AGC allows them to train normalizer-free networks with larger batch sizes and stronger data augmentations.
- They design a family of Normalizer-Free ResNets, called NFNets, which set new state-of-the-art validation accuracies on ImageNet for a range of training latencies. Their NFNet-F1 model achieves similar accuracy to EfficientNet-B7 while being 8.7× faster to train, and their largest model sets a new overall state of the art without extra data of 86.5% top-1 accuracy.
- They show that NFNets achieve substantially higher validation accuracies than batch-normalized networks when fine-tuning on ImageNet after pre-training on a large private dataset of 300 million labelled images. Their best model achieves 89.2% top-1 accuracy after fine-tuning.

## Inference with MIGraphX using NFNet ONNX Model

There is no ONNX model released for NFNet, as of June 2021, however a PyTorch model is available at:
https://github.com/rwightman/pytorch-image-models. 
We provide an in-house produced and optimized ONNX model, which can be parsed and compiled using MIGraphX for AMD GPUs. The ONNX model file can be fetched using the Jupyter notebook we provide.

### Requirements:
1) AMD GPU system with ROCm installed.
2) Jupyter notebook library.

### How to use NFNet for image recognition:
Please utilize the notebook example provided:
1) Install jupyter notebook to your environment if not already installed:
```
https://jupyter.org/install
```
2) Connect to your jupyter server and utilize `nfnet_inference.ipynb` notebook file.

### How to compare MIGraphX to ONNX Runtime for NFNet ONNX model:
First install requirements:
```
pip3 install -r requirements_nfnet.txt
```

On your terminal, invoke:
```
python3 ort_comparison.py
````
