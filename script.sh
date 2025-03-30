#!/bin/bash

# Clean files
clean() {
    echo "Cleaning files ..."
    rm -rf ./data/configs/* ./test_data/out/*
    echo "Cleaning completed"
}

# Build docker-image
docker_build() {
    echo "Building docker-image ..."
    docker build -t mmpose ./mmpose/docker/
    echo "Building completed"
}

docker_start() {
    echo "Starting docker-container"
    docker run --gpus all --shm-size=8g -it -v .\data\:/mmpose/data mmpose
    echo "Starting container completed"
}

# Show help
usage() {
    echo "Usage: $0 {clean|docker-build|help}"
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