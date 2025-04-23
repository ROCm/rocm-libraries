# BERT-SQuAD Example with MIGraphX
Question answering with BERT using MIGraphX optimizations on ROCm platform.

There are two ways to run the example:
1) Install MIGraphX and Jupyter notebook to your system and then utilize `BERT-Squad.ipynb` notebook file.
2) Install MIGraphx to your system and follow the steps executing the python script `bert-squad-migraphx.py`.

# Steps
1) Install MIGraphX to your environment. Please follow the steps to build MIGraphX given at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX
2) Upgrade your pip3 to latest version
```
pip3 install --upgrade pip 
```
3) Install the requirements file
```
pip3 install -r requirements_bertsquad.txt
```
4) Install `unzip` and fetch the uncased file (vocabulary):
```
apt-get install unzip
wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```
5) Get BERT ONNX model (bertsquad-10.onnx):
```
wget https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx
```
6) Run the inference, it will compile and run the model on three questions and small data provided in `inputs.json`:
```
python3 bert-squad-migraphx.py
```
## References
This example utilizes the following notebook :notebook: and applies it to MIGraphX:
https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/BERT-Squad.ipynb
