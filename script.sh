#!/bin/bash

# Clean files
clean() {
    echo "Cleaning files ..."
    rm -rf ./data/configs/*.py ./data/configs/*.pth
    echo "Cleaning completed"
}

# Build docker-image
docker_build() {
    echo "Building docker-image ..."
    docker build -t mmpose .
    echo "Building completed"
}

docker_start() {
    echo "Starting docker-container"
    docker run --gpus all --shm-size=8g -it -v ./data/:/mmpose/data mmpose
    echo "Starting container completed"
}

# Show help
usage() {
    echo ""
    echo "********************************************"
    echo "* MMPose Management Script                *"
    echo "* Version: 1.0                            *"
    echo "********************************************"
    echo ""
    echo "Available commands:"
    echo ""
    echo "clean         - Cleans temporary files"
    echo "                 Deletes contents of:"
    echo "                 - ./data/configs/"
    echo ""
    echo "docker-build  - Builds Docker image with:"
    echo "                 - PyTorch with CUDA support"
    echo "                 - MMPose framework"
    echo "                 - All required dependencies"
    echo ""
    echo "docker-start  - Runs Docker container with:"
    echo "                 - GPU access enabled"
    echo "                 - 8GB shared memory"
    echo "                 - ./data mounted to /mmpose/data"
    echo ""
    echo "help          - Shows this help message"
    echo ""
    echo "Example usage:"
    echo "  $0 docker-build   # Build the image"
    echo "  $0 docker-start   # Run the container"
    echo "  $0 clean          # Clean temp files"
    echo ""
}

# Check args
case "$1" in
    clean)
        clean
        ;;
    docker-build)
        docker_build
        ;;
    docker-start)
        docker_start
        ;;
    help|*)
        usage
        ;;
esac