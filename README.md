DF-Net
======
## DF-Net: A Direction-Aware Dual-Frequency Feature Modeling Network for Coronary Artery Segmentation

This repository provides the main network implementation of our proposed method for coronary artery segmentation in CCTA images.

## Status

Our manuscript is currently under review. To support reproducibility and facilitate academic communication, we have released the core network code in this repository.

Please note that the current release mainly includes the principal model architecture. Additional components, including training scripts, data preprocessing, evaluation protocols, and detailed experimental settings, will be gradually released after the manuscript is accepted.

## Repository Structure

```text
DF-Net/networks
├── DWT_IDWT/
│   └── Implementation of 3D discrete wavelet transform and inverse wavelet transform
│
├── DFNet.py
│   └── Main implementation of DF-Net, including Tri-Mamba-based low-frequency modeling,
│       high-frequency structural modeling, and the dual-frequency fusion framework
│
├── DWT_downsample.py
│   └── Wavelet-based downsampling and feature decomposition modules
│
├── networks_other.py
│   └── Auxiliary network implementations and baseline-related modules
│
├── utils.py
    └── Core encoder and decoder modules, including DAConv and SG-MoE decoder blocks

```
## Datasets

For applications and downloads of public datasets, please refer to the official projects:

ASOCA: https://asoca.grand-challenge.org/access/

ImageCAS: https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT
