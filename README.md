# CaRaCTO

This repository contains the official Python implementation of the research papers:
- [**CaRaCTO**: *Robust Camera-Radar Extrinsic Calibration with Triple Constraint Optimization*, published at ICPRAM 2024 (Best Industrial Paper Award).](https://www.scitepress.org/Link.aspx?doi=10.5220/0012369700003654)
- [**CaRaCTO-3D**: *From Camera-Radar Calibration to Scene Reconstruction*, published in SN Computer Science 2025.](https://link.springer.com/article/10.1007/s42979-025-04355-w)

This project provides the tools to perform extrinsic calibration between a camera and a radar sensor and to use this calibration to generate dense 3D reconstructions of the environment.

## Project Overview

The primary goal of this project is to perform calibration between a camera and a radar sensor and then use this calibration to create a 3D reconstruction of the surrounding environment.

### Core Functionality

*   **Calibration:** Calculates the spatial relationship (transformation matrix) between the camera and radar sensors using optimization techniques.
*   **Depth Estimation:** Utilizes a pre-trained deep learning model (`depth-anything`) from the Hugging Face `transformers` library to estimate depth and disparity from 2D images.
*   **3D Reconstruction:** Combines the sensor data, calibration results, and depth maps to generate 3D point clouds of the scene using the Open3D library.
*   **Evaluation:** Includes evaluation of calibration methods against established baselines, such as El Natour et al.
*   **Analysis:** The project includes scripts for analyzing the reconstructed scene, such as identifying planes and measuring dimensions.

### Key Technologies

*   **Language:** Python
*   **Core Libraries:**
    *   NumPy & SciPy: For numerical operations and optimization.
    *   OpenCV (`cv2`): For image processing.
    *   PyTorch & Hugging Face `transformers`: For running the depth estimation model.
    *   Open3D: For 3D data processing and point cloud manipulation.
    *   Matplotlib: For data visualization.

## Getting Started

### Installation

```bash
    # Uses Python 3.12
    # Clone the repository to your local machine:
    git clone https://github.com/mahdichamseddine/CaRaCTO.git
    cd CaRaCTO
    
    # Install the required dependencies using `uv`
    uv sync

    # Alternatively use your favorite environment management solution
```


### Dataset

You will need to download the appropriate dataset. The path to the dataset is passed as a command-line argument to the scripts (e.g., `--dataset_path /path/to/your/dataset`).
*Data will be uploaded soon*

## Usage

The project is structured as a research repository and is run by executing individual Python scripts.

*   **Full Reconstruction Pipeline:**
    ```bash
    python caracto/reconstruction/scene_reconstruction.py --dataset_path /path/to/your/dataset
    ```

*   **Calibration Only:**
    ```bash
    python caracto/calibration/run_calibration.py --dataset_path /path/to/your/dataset
    ```

You may need to modify the `main()` functions within these files to suit your specific needs and data.

## Citing this Work

If you use this code in your research, please cite the following publications:

```
@article{chamseddine2025caracto,
    title     = {CaRaCTO-3D: From Camera-Radar Calibration to Scene Reconstruction},
    author    = {Chamseddine, Mahdi and Rambach, Jason and Stricker, Didier},
    journal   = {SN Computer Science},
    volume    = {6},
    number    = {7},
    pages     = {822},
    year      = {2025},
    publisher = {Springer},
}

@inproceedings{chamseddine2024caracto,
    title        = {CaRaCTO: Robust Camera-Radar Extrinsic Calibration with Triple Constraint Optimization},
    author       = {Chamseddine, Mahdi and Rambach, Jason R and Stricker, Didier },
    year         = 2024,
    booktitle    = {Proceedings of the 13th International Conference on Pattern Recognition Applications and Methods - ICPRAM},
    pages        = {534--545},
    organization = {INSTICC},
}
```

## Acknowledgement

This research was partially funded by the European Union as part of the project HumanTech (Grant Agreement 101058236) and the Federal Ministry of Education and Research (BMBF) of the Federal Republic of Germany as part of the research project COPPER (Grant Number 01IW24009).

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.