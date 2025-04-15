# 📌 Description
This project utilizes [mmpose](https://github.com/open-mmlab/mmpose) for human pose estimation (HPE). It includes dataset processing and supports Docker-based deployment.

# 📂 Project Structure

- **data/**: Contains all the data needed for the project.
  - **source/**: This directory contains all the source code and classes for the project.
    - **PoseEstimator/**: This folder includes the main class `PoseEstimator` and the `main.py` file, which serves as the entry point for the application.
    - **evaluation/**: This folder contains JSON files for predictions as well as the `evaluation.py` file, which handles the evaluation of model performance.
    - **animation/**: This folder includes keypoints of the video as well as the `animation.py` file, which is responsible for creating animations based on the detected keypoints.
  - **test_data/**: Data used for testing purposes.
  - **configs/**: Configuration files for the model.

- **mmpose/**: Folder containing the `mmpose` model.

- **Dockerfile**: A file used to create a Docker image for the project.

- **script.bat**: A script for executing tasks in a Windows environment.

- **script.sh**: A script for executing tasks in a Linux or macOS environment.

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/VadimProd/PoseEstimation.git
cd PoseEstimation
```

## Windows

Build docker-image and start it with .bat script file

```bash
script.bat docker-build
script.bat docker-start
```

## Linux/macOS

Build docker-image and start it with .sh script file

```bash
script.sh docker-build
script.sh docker-start
```

# ⚡ Launch

Run the pose estimation model using `main.py` with the following arguments:
```bash
python3 main.py \
    --method [topdown|bottomup] \
    --input <path_to_input_file> \
    --output <path_to_output_file>
