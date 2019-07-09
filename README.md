## Dual Averaging Method for online graph-structured sparsity

## Overview

Welcome to the repository of GraphDA! This repository is only for 
reproducing all experimental results shown in our KDD paper. To 
install it via pip, please try [sparse-learn](https://github.com/baojianzhou/sparse-learn). 
More details of GraphDA can be found in: "Zhou, Baojian, Feng Chen, and Yiming Ying. "Dual Averaging Method for Online Graph-structured Sparsity." arXiv preprint arXiv:1905.10714 (2019).".

## Preparation
Our code is based on [Openblas-0.3.1](https://github.com/xianyi/OpenBLAS/releases/tag/v0.3.1), which we already copied into our repository. Suppose you are using GNU/Linux based system or Mac, you can first goto OpenBLAS-0.3.1 folder and then make install it via the following command:
```sh
>>> cd OpenBLAS-0.3.1
>>> make && make install PREFIX=../lib
```

The lib folder under graph-da and corresponding libraries will be generated.

### Figure 1
To generate Figure 1, run the following command:
```sh
>>> python exp_logit_benchmark.py show_figure_1
```

### Figure 2
To generate Figure 2, run the following command:
```sh
>>> python exp_logit_benchmark.py show_figure_2
```

### Figure 3
To generate Figure 3, run the following command:
```sh
>>> python exp_logit_benchmark.py show_figure_2
```
