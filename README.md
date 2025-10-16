# DC (Distill and Contrast)

Official PyTorch implementation of **D&C** in *"Distill & Contrast: A New Graph Self-Supervised Method With Approximating Nature Data Relationships"* (IEEE TKDE 2025).

## Repository Structure
This repository contains the official PyTorch implementation of the D&C method. The main file is named **DC-master**, and it contains two primary sections:

1. **DC_master_homo**: This directory contains the code for the original D&C method on homophilic graphs.
2. **DC_GraphACL_plugin**: This directory includes the integration of D&C as a plugin to the GraphACL method, along with experiments on heterophilic graphs.

## Paper Link
The full paper is available at [IEEE TKDE 2025](https://ieeexplore.ieee.org/abstract/document/10938656).

## Required Environment

This project requires the following dependencies to be installed:

- `dgl`: 0.7.1
- `networkx`: 3.2.1
- `pyg-lib`: 0.4.0+pt24cu124
- `PyGCL`: 0.1.2
- `scikit-learn`: 1.6.1
- `scipy`: 1.13.1
- `torch`: 2.4.0

All environment setup details are included in the `Env.yaml` file.

## Quick Start

To quickly run the D&C method:

- For **DC_master_homo** (homogeneous dataset), you can directly run the `rundataset.ipynb` Jupyter notebook.
- For the **heterogeneous plugin experiment**, you can run the experiment with the following command:

    ```bash
    bash run.sh
    ```

## Contact

If you have any questions or issues, feel free to contact email: [893038487@qq.com](mailto:893038487@qq.com).


## 🔄 Code from Other Projects

Part of the code in this repository is adapted from the following open-source projects:

- **GraphACL**: A simple and enhancement-free self-supervised method for graph contrastive learning.
  - GitHub link: [https://github.com/tengxiao1/GraphACL](https://github.com/tengxiao1/GraphACL)

- **PyGCL**: A PyTorch-based graph contrastive learning library that provides implementations for various graph contrastive learning algorithms.
  - GitHub link: [https://github.com/PyGCL/PyGCL](https://github.com/PyGCL/PyGCL)

- **GRACE**: A PyTorch implementation of deep graph contrastive representation learning methods.
  - GitHub link: [https://github.com/CRIPAC-DIG/GRACE](https://github.com/CRIPAC-DIG/GRACE)

Thanks to the authors of these projects for providing valuable code and inspiration.
