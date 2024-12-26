# DLCV: Don't Learn Computer Vision

## Environment Setup
Follow these steps to set up your environment:

1. Setup LLaVA
```bash
$ git clone https://github.com/haotian-liu/LLaVA.git
$ cd LLaVA

$ conda create -n llava python=3.10 -y
$ conda activate llava
$ pip install --upgrade pip  # enable PEP 660 support
$ pip install -e .
```
2. Setup Depth-Anything-v2
```bash
# Please ensure you are in the llava env

$ pip install gradio_imageslider
$ pip install gradio==4.29.0
$ pip install matplotlib
$ pip install opencv-python
$ pip install torch
$ pip install torchvision
```
3. Setup YOLOv11
```bash
# Please ensure you are in the llava env

$ pip install ultralytics
```
4. Setup CLIP
```bash
# Please ensure you are in the llava env

$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
5. Setup FAISS
```bash
# Please ensure you are in the llava env

$ pip install faiss-gpu
```

6. Install gdown
```bash
# Please ensure you are in the llava env

$ pip install gdown
```

## Model Checkpoint Download
```bash
# Please ensure you are in the llava env

$ python download.py
```


## Inference
```bash
# Please ensure you are in the llava env
# Run this command to reproduce our result 

$ bash inference_RAG.sh
```