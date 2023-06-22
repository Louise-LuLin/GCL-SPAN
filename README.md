# GCL-SPAN

This is the code for the paper "Spectral Augmentation for Self-Supervised Learning on Graphs" accepted by ICLR 2023.

## Introduction

## Requirement

Code is tested in **Python 3.10.10**. Some major requirements are listed below:
```
$pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$pip install torch_geometric
$pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
$pip install dgl
$pip install networkx
$pip install numba
```

## Run the code

## Cite

Please cite our paper if you find this repo useful for your research or development.

```
@inproceedings{lin2022spectral,
  title={Spectral Augmentation for Self-Supervised Learning on Graphs},
  author={Lin, Lu and Chen, Jinghui and Wang, Hongning},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
