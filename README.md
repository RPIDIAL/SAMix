# SAMix

[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

This repository includes open-source codes, detailed experimental results and full references of our MICCAI 2023 work 

[*Spectral Adversarial MixUp for Few-Shot Unsupervised Domain Adaptation*](https://arxiv.org/abs/2309.01207).


## Overview

<p align='center'><img src="image/overview.png" width=90% height=90% /></p>

The figure above summarizes our algorithm comparisons framework, *Spectral Adversarial MixUp*.
- First, our method computes [DoDiSS map](https://github.com/DIAL-RPI/Spectral-Adversarial-Data-Augmentation/tree/main/AAAIcodeSubmissoin__model_sensitivity_map). Thie algorithm is a variant of our AAAI-23 paper [*When Neural Networks Fail to Generalize? A Model Sensitivity Perspective*](https://arxiv.org/abs/2212.00850).
- Then, we run [SAMix](https://github.com/DIAL-RPI/Spectral-Adversarial-Data-Augmentation/tree/main/AAAIcodeSubmissoin__SADA) to generate the augmented data and train the model with the JS-div regularization.


## Prerequisites

- Python 3.8
- PyTorch 1.8.1+
- A computing device with GPU

## Getting started

### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/products/distribution)
- Install [CUDA 11.0+](https://developer.nvidia.com/cuda-11.0-download-archive)
- Install [PyTorch 1.8.1+](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 1.8.1](http://pytorch.org/)

### Dataset & Preparation

The two medical image datasets used in our work are publicly available.

The __Fundus__ datasets include the [REFUGE](https://refuge.grand-challenge.org/) and the [RIM](https://github.com/miag-ull/rim-one-dl). We follow the [DoFE](https://arxiv.org/pdf/2010.06208.pdf) to preprocess the datasets.

The __Camelyon__ dataset is a subset from the [WILDS](https://wilds.stanford.edu/datasets/). This dataset is already preprocessed by the challenge.


### Measure the sensitivity map of an ERM model

Train and evaluate an ERM model by

```bash
python ./AAAIcodeSubmissoin__model_sensitivity_map/train_ERM.py
```

The model will be saved in `./AAAIcodeSubmissoin__model_sensitivity_map/save_dir`.


Measure the DoDiSS map of an ERM model by

```bash
python ./AAAIcodeSubmissoin__model_sensitivity_map/model_sensitivity_map.py
```

### Train model with Spectral Adversarial Data Augmentation (SADA)

Train and evaluate the models with SADA by

```bash
python ./AAAIcodeSubmissoin__SADA/train_SADA.py
```

The key __SADA data augmentation module__ is in 

```
./AAAIcodeSubmissoin__SADA/sada.py
```

Augmentation settings for all datasets:

- `--epsilon` iteration of the checkpoint to load. #Default: 0.2
- `--step_size` step size of the adversarial attack on the amplitude spectrum. #Default: 0.08
- `--num_steps` batch size of the attack steps. #Default: 5
- `--tau` settings for the early stop acceleration. #Default: 1
- `--random_init` if or not initializing amplitude spertrums with random perturbations. #Default: True
- `--randominit_type` type of random init type. #Default: 'normal'
- `--criterion` loss functions to attack. #Default: 'ce', choices=['ce', 'kl']
- `--direction` neg: standard attack || pos:inverse attack. #Default: neg, choices=['pos','neg']

## Citation

Please cite these papers in your publications if it helps your research:

```bibtex
@article{zhang2022spectral,
  title={Spectral Adversarial MixUp for Few-Shot Unsupervised Domain Adaptation},
  author={Zhang, Jiajin and Chao, Hanqing and Dhurandhar, Amit and Chen, Pin-Yu and Tajer, Ali and Xu, Yangyang and Yan, Pingkun},
  journal={https://arxiv.org/abs/2309.01207},
  year={2023}
}
```

## Acknowledgement

We would like to thank the authors we cited in our paper for sharing their codes.




