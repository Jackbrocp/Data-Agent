# [ICML 2026] 🎉 Data Agent: Learning to Select Data via End-to-End Dynamic Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2603.07433-b31b1b.svg)](https://arxiv.org/abs/2603.07433)
[![Conference](https://img.shields.io/badge/ICML-2026-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the official implementation of **Data Agent**, an end-to-end dynamic data selection framework that learns to select training data adaptively during model optimization.


## News

- **2026/03**: Paper released on arXiv.
- **2026/04**: Accepted by ICML 2026.
- **2026/05**: Code released.

## Overview

The overall framework of **Data Agent** is shown below.

<p align="center">
  <img src="assets/Fig2.png" width="75%">
</p>

<p align="center">
  <em>Figure 1. Overview of Data Agent. A PPO-based agent learns to dynamically select informative training data during model optimization.</em>
</p>

## Installation

Clone this repository:

```bash
git clone https://github.com/Jackbrocp/Data-Agent.git
cd Data-Agent
```

### Requirements

```text
Python >= 3.8
PyTorch
Torchvision
NumPy
```

Please install the PyTorch version that matches your CUDA environment.

## Data Preparation

Create the data directory:

```bash
mkdir -p data
```

For CIFAR experiments, please download the provided files and place them under `./data/`.

- [CIFAR-10](https://drive.google.com/file/d/1j0isu0eaXBDklyMi36RTrPPYtTb3I5Kg/view?usp=sharing)
- [CIFAR-100](https://drive.google.com/file/d/1wDLfl34pCAO3-Ezw0kGr8zNUMPk1AEDH/view?usp=sharing)

If you place the data in another directory, please modify the corresponding dataset path argument.

## Usage

### Training with Data Agent

Train ResNet-18 on CIFAR-10 with a 50% selection ratio:

```bash
python train.py --dataset cifar10 --model r18 --ratio 0.5
```

Train ResNet-50 on CIFAR-100 with a 30% selection ratio:

```bash
python train.py --dataset cifar100 --model r50 --ratio 0.3
```

We also provide training logs using both datasets with a 50% selection ratio under 'logs/' for reference.

## Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@misc{yang2026dataagentlearningselect,
      title={Data Agent: Learning to Select Data via End-to-End Dynamic Optimization}, 
      author={Suorong Yang and Fangjian Su and Hai Gan and Ziqi Ye and Jie Li and Baile Xu and Furao Shen and Soujanya Poria},
      year={2026},
      eprint={2603.07433},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.07433}, 
}
```

## Contact

For questions, issues, or collaboration, please feel free to contact:

```text
sryang@smail.nju.edu.cn
```

## Acknowledgements

This codebase builds upon [InfoBatch](https://github.com/NUS-HPC-AI-Lab/InfoBatch/). We sincerely thank the authors for their excellent open-source implementation.