# Human Arm Segmentation in Point Clouds using Nuitrack and PointNet++

A comprehensive pipeline for human arm segmentation in 3D point clouds using the Nuitrack SDK for skeletal tracking and data collection, combined with deep learning models for semantic segmentation. This project enables real-time 3D visualization, data labeling, and training of neural networks for arm part classification.

## Overview

This project implements a complete workflow for segmenting human arms in point clouds:
- **Data Collection**: Real-time capture of RGB-D data and skeletal joint information using Intel RealSense cameras and Nuitrack SDK
- **3D Visualization**: Live visualization of point clouds with skeletal overlay using Open3D
- **Automated Labeling**: Cylinder-based geometric labeling of arm segments (forearm vs upper arm)
- **Deep Learning**: PointNet++ implementation for semantic segmentation training and inference

## Features

- **Real-time Data Capture**: Synchronized RGB, depth, and skeletal joint data recording
- **Live 3D Visualization**: Interactive point cloud visualization with skeletal tracking
- **Geometric Labeling**: Automated arm segment labeling using cylindrical bounding regions
- **Multiple Export Formats**: Support for PLY and CSV point cloud exports
- **Neural Network Training**: PointNet++ implementation for semantic segmentation
- **Performance Optimization**: Multi-threaded data processing and Numba acceleration

## Prerequisites

### Hardware Requirements
- Intel RealSense D435i camera
- NVIDIA GPU with CUDA support (recommended for deep learning)
- Windows 10/11 or Linux

### Software Dependencies
- Python 3.9.16
- [Nuitrack SDK](https://nuitrack.com/) - Skeletal tracking library
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) - Camera drivers
- CUDA toolkit (for GPU acceleration)

## Installation

### 1. Environment Setup
```bash
# Create and activate conda environment
conda create -n nuitrack python=3.9.16
conda activate nuitrack
```

### 2. Install Python Dependencies
```bash
pip install open3d
pip install numba
pip install opencv-python
pip install tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas
```

### 3. Install Nuitrack SDK
- Download and install Nuitrack SDK from [official website](https://nuitrack.com/)
- Install PyNuitrack from the [GitHub repository](https://github.com/3DiVi/nuitrack-sdk/tree/master/PythonNuitrack-beta)

### 4. Camera Setup
- Install Intel RealSense SDK
- Connect and configure Intel RealSense D435i camera
- Verify camera calibration parameters in `utils/Calib_file.json`

## Usage

### 1. Data Collection (`nuitrack_2D.py`)
Capture RGB-D data with skeletal tracking for training data generation:

```bash
python nuitrack_2D.py
```

**Features:**
- Real-time skeletal tracking overlay on depth/color streams
- Synchronized data saving (RGB, depth, joints)
- Performance monitoring and profiling
- Multi-threaded data processing

**Controls:**
- `Space`: Toggle between depth and color visualization
- `ESC`: Exit application

### 2. Live 3D Visualization (`nuitrack_3D.py`)
Real-time 3D point cloud visualization with skeletal tracking:

```bash
python nuitrack_3D.py
```

**Features:**
- Live point cloud generation from RGB-D data
- Real-time joint tracking in 3D space
- Interactive visualization controls
- Performance-optimized rendering

**Controls:**
- `Q`: Quit application

### 3. Data Visualization (`Visualize_saved_joints.py`)
Visualize previously captured data with arm segmentation:

```bash
python Visualize_saved_joints.py
```

**Features:**
- Interactive file selection dialog
- Point cloud downsampling options
- Cylinder-based arm segmentation visualization
- Bounding box and geometric analysis

### 4. Automated Labeling (`label_and_save_pcds.py`)
Generate labeled training data from captured sessions:

```bash
python label_and_save_pcds.py
```

**Features:**
- Batch processing of captured data
- Geometric arm segmentation using cylinders
- Multiple export formats (PLY, CSV)
- RGB color option for point clouds

### 5. Deep Learning Training (`pointnet++/train.py`)
Train PointNet++ model for semantic segmentation:

```bash
cd pointnet++
python train.py --model pointnet_sem_seg --batch_size 16 --epoch 32
```

**Training Parameters:**
- `--model`: Model architecture (default: pointnet_sem_seg)
- `--batch_size`: Training batch size (default: 16)
- `--epoch`: Number of training epochs (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--npoint`: Number of points per sample (default: 4096)

## Project Structure

```
pointclouds/
├── README.md                    # This documentation
├── LICENSE                      # License file
├── .gitignore                  # Git ignore rules
│
├── nuitrack_2D.py              # 2D visualization and data capture
├── nuitrack_3D.py              # Live 3D visualization
├── Visualize_saved_joints.py   # Offline data visualization
├── label_and_save_pcds.py      # Automated labeling pipeline
│
├── pointnet++/                 # Deep learning components
│   ├── model.py               # PointNet++ model architecture
│   ├── model_msg.py           # Alternative model implementation
│   ├── train.py               # Training script
│   ├── pointnet_utils.py      # PointNet utility functions
│   └── source.md              # Model documentation
│
├── utils/                      # Utility functions and configuration
│   ├── read_calib_file.py     # Camera calibration loader
│   ├── get_cylinder.py        # Cylinder geometry functions
│   ├── cylinder_test.py       # Cylinder testing utilities
│   ├── angle.py               # Angle calculation utilities
│   ├── Calib_file.json        # Camera intrinsic parameters
│   ├── labels_key.json        # Segmentation class definitions
│   └── list_of_joints.json    # Joint mapping definitions
│
└── diagrams/                   # Documentation images and diagrams
    ├── 2d_visu.png            # 2D visualization example
    ├── 3d_visu.png            # 3D visualization example
    ├── cylinderrot.png        # Cylinder rotation demonstration
    ├── segmented.png          # Final segmentation results
    └── *.drawio               # Draw.io diagram sources
```

## Data Flow

1. **Capture Phase**: `nuitrack_2D.py` collects synchronized RGB-D and skeletal data
2. **Processing Phase**: `label_and_save_pcds.py` generates geometric labels using cylinder fitting
3. **Training Phase**: `pointnet++/train.py` trains neural networks on labeled data
4. **Visualization**: `nuitrack_3D.py` and `Visualize_saved_joints.py` provide real-time and offline visualization

## Segmentation Classes

The system classifies arm points into three categories:

| Class ID | Class Name | Description | Color |
|----------|------------|-------------|-------|
| 0 | Background | Non-arm points | White |
| 1 | Forearm | Wrist to elbow segment | Red |
| 2 | Upper Arm | Elbow to shoulder segment | Green |

## Camera Configuration

Camera intrinsic parameters are configured in `utils/Calib_file.json`:
- **Resolution**: 1280x720 (configurable)
- **Frame Rate**: 30 FPS (RGB), up to 90 FPS (Depth)
- **Depth Range**: Up to 5-7 meters
- **Preset**: High accuracy mode

## Performance Optimization

- **Multi-threading**: Concurrent data saving and processing
- **Numba JIT**: Accelerated coordinate transformations
- **Voxel Downsampling**: Configurable point cloud reduction
- **Memory Management**: Efficient array operations and caching

## Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Verify RealSense SDK installation
   - Check USB 3.0 connection
   - Update camera firmware

2. **Nuitrack License Issues**
   - Activate Nuitrack license key
   - Check SDK installation path
   - Verify device compatibility

3. **Performance Issues**
   - Reduce point cloud resolution
   - Enable voxel downsampling
   - Check CUDA installation for GPU acceleration

4. **Memory Errors**
   - Reduce batch size for training
   - Lower point cloud density
   - Monitor system memory usage

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Nuitrack SDK](https://nuitrack.com/) for skeletal tracking capabilities
- [Open3D](http://www.open3d.org/) for 3D visualization and processing
- [PointNet++](https://github.com/charlesq34/pointnet2) for the deep learning architecture
- Intel RealSense team for camera hardware and SDK

## Results Gallery

### 2D Skeletal Tracking
![2D Visualization](diagrams/2d_visu.png)

### 3D Cylinder Fitting
![Cylinder Rotation](diagrams/cylinderrot.png)

### Point Cloud Segmentation
![3D Visualization](diagrams/3dvisu_load.png)

### Final Segmentation Results
![Segmented Results](diagrams/segmented.png)