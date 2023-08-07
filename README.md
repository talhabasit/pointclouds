# Segmentation of a Human Arm in Point Clouds using Nuitrack

This repository contains the code for the segmentation of a human arm in point clouds using the Nuitrack SDK. The code heavily relies on [Nuitrack SDK](https://nuitrack.com/) and the [Open3D](http://www.open3d.org/) library. The code is written in Python 3.9.16 and is tested on Windows 10/11.

## Python Setup 
- Conda environment setup
    - Create a new Conda environment using the following command: `conda create -n nuitrack python=3.9.16`
    - Activate the Environment using the following command: `conda activate nuitrack`
    - Install the following packages
        - `pip install open3d`
        - `pip install numba`
        - `pip install opencv-python`
        - `pip install tqdm`
        - `pip install torch`
        - The current PyNuitrack version from the Nuitrack github repo under the link: [PyNuitrack](https://github.com/3DiVi/nuitrack-sdk/tree/master/PythonNuitrack-beta)
        - `pip install tqdm`
        - `pip install torch`

