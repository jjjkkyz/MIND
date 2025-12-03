
# 🌟 MIND: Material Interface Generation from UDFs for Non-Manifold Surface Reconstruction

[![GitHub Stars](https://img.shields.io/github/stars/jjjkkyz/MIND?style=social)](https://github.com/jjjkkyz/MIND/stargazers)
[![License](https://img.shields.io/github/license/jjjkkyz/MIND)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%2FJournal-red)](https://arxiv.org/abs/2506.02938)

## 📖 Overview

This repository hosts the official code and resources for the paper **"MIND: Material Interface Generation from UDFs for Non-Manifold Surface Reconstruction"** published at NIPS 2025.

We propose a novel algorithm to extract non-manifold mesh from unsigned distance fields.


## ⚙️ Requirements and Installation

This project requires Python [Version] or higher.

### Prerequisites

* We need pytorch installed.
  
### Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/jjjkkyz/MIND.git
    cd MIND
    ```


2.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ``````

3.  **Install:**
    ```
    pip install -e .
    ```


## 🚀 Usage

```

    from mldf import MIND


    # some neural network
    udf_func = your_input
    resolution = 256

    # for more paremeters, see the source code
    mind_mesh_extractor = MIND(udf_func, resolution)

    mesh = mind_mesh_extractor.run()

```


## 📝 Citation

If you find this repository or our paper useful, please cite our work:

```

    @inproceedings{
        chen2025mind,
        title={{MIND}: Material Interface Generation from {UDF}s for Non-Manifold Surface Reconstruction},
        author={Xuhui Chen and Fei Hou and Wencheng Wang and Hong Qin and Ying He},
        booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
        year={2025},
        url={https://openreview.net/forum?id=4lR9OhAisI}
    }

```

---


## 📄 License

This project is licensed under the **MIT** License
