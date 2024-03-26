# Let's Study Classification ! 💻
<div align="left">
  <img src="https://img.shields.io/badge/Python-007396?style=flat&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white">
</div>

<br/>

2024 Vision Lab Seminar Repository.  
We will study classification, GAN or *whatever you want* . . . 😙

**Team mate** : [Hoju](https://github.com/shinhoju), [Nayoung](https://github.com/skdud940), [Jaeho](https://github.com/Kimjaeho9)  

## Goal #1
VGGNet, ResNet 구조 이해 / PyTorch 구현
### VGGNet
[ArXiv](https://arxiv.org/abs/1409.1556)  
1. Getting Started
``` bash
conda create -n cls python=3.8 -y
conda activate cls

# you should check CUDA version
conda install pytorch, torchvision -c pytorch
pip install -r requirements.txt
```
2. Train/Test
``` bash
python train.py
```

## Goal #2
ResNet 모델을 활용한 Kaggle 데이터셋 학습
### Dataset
[Kaggle: 6 Human Emotions](https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier)

### Model Structure
- ResNet-28, 50, 101 구현

### Train
1. No augmentation
2. Strong augmentation
    - 기본 방식 [참고](https://velog.io/@xpelqpdj0422/3.-%EC%9E%98-%EB%A7%8C%EB%93%A0-Augmentation-%EC%9D%B4%EB%AF%B8%EC%A7%80-100%EC%9E%A5-%EC%95%88-%EB%B6%80%EB%9F%BD%EB%8B%A4)
    - MixUp, Mosaic 기법
3. Scheduler & optimizer
    - Learning rate scheduler [참고](https://gaussian37.github.io/dl-pytorch-lr_scheduler/)
    - Optimizer: SGD, Adam, AdamW
    - Hyper parameter tuning

### Test
🚧
