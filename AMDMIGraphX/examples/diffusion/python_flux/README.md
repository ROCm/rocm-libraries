## Setup

Make sure python interpreter can find migraphx. Default location:
```
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

Install dependencies
```
pip install -r torch_requirements.txt -r requirements.txt
```

Login to Huggingface:
```
huggingface-cli login
```

## Generate Image
```
python3 txt2img.py -p "A cat holding a sign that says hello world"
```

## Benchmark
Ex. 10 full executions:
```
python3 txt2img.py -b 10 --bf16
```