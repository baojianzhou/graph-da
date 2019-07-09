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

### Figure 3-7
To generate Figure 3-7, run the following command:
```sh
>>> python exp_logit_benchmark.py show_figure_3-7
```

### Figure 8
To generate Figure 8, run the following command:
```sh
>>> python exp_linear_mnist.py show_figure_8
```

### Figure 9-10
To generate Figure 9-10, run the following command:
```sh
>>> python exp_logit_kegg.py show_figure_9-10
```

### Figure 11
To generate Figure 11, run the following command:
```sh
>>> python exp_linear_mnist.py show_figure_11
```

### Figure 12
To generate Figure 12, run the following command:
```sh
>>> python exp_logit_kegg.py show_figure_12
```

### Table 1-4
To generate Figure 12, run the following command:
```sh
>>> python exp_logit_benchmark.py show_4_tables
```

### Exactly Reproduce the results
To reproduce those results, you need to run the following commands:
```sh
>>> python exp_logit_benchmark.py run_{fix_tr_mu,diff_tr,diff_mu,diff_s}
>>> python exp_logit_kegg.py {test_graphda,test_baselines}
>>> python exp_linear_mnist.py run_test
```
