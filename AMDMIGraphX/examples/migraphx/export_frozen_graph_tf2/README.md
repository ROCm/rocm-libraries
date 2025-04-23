# Exporting Frozen Graphs in Tensorflow 2 

## Description
This example demonstrates how to export a frozen graph protobuf in Tensorflow 2.X that can be used as input for MIGraphX. The method for accomplishing this has changed from Tensorflow 1.X. Please refer to [export_frozen_graphs_tf1]() if you are not yet using Tensorflow 2. 

## How to Use this Example
If you do not already have Jupyter Notebooks installed, please refer to this [page](https://jupyter.org/install) for instructions. 

Once Jupyter Notebooks is installed, you can navigate to this directory and issue the command:

```
$ jupyter notebook
```

From the browser window that is launched, click on `example.ipynb`
You should now be able to run the notebook from your browser.

To use this on your own models you wish to save, simply edit the first cell to include any additional libraries and modify `MODEL_NAME` and `model` to the model of your choosing. Additionally, training and fine-tuning can be performed before moving on to cells 2 and beyond.
 
