# 📌 Description
This project utilizes [mmpose](https://github.com/open-mmlab/mmpose) for human pose estimation (HPE). It includes dataset processing and supports Docker-based deployment.

# 📂 Project Structure

- **data/**: Contains all the data needed for the project.
  - **source/**: This folder includes classes, such as the `PoseEstimator` and the `main.py` file.
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
cd mmpose
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
