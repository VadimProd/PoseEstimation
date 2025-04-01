@echo off
setlocal

if "%1"=="" goto usage
if "%1"=="clean" goto clean
if "%1"=="docker-build" goto docker-build
if "%1"=="docker-start" goto docker-start
if "%1"=="help" goto usage
goto usage

:clean
echo Cleaning files ...
del /s /q ".\data\configs\*.py" ".\data\configs\*.pth"
echo Cleaning completed!
exit /b

:docker-build
echo Building docker-container...
docker build -t mmpose .
echo Building completed
exit /b

:docker-start
echo Starting docker-container...
docker run --gpus all --shm-size=8g -it -v .\data\:/mmpose/data mmpose
echo Starting docker-container completed!
exit /b

:usage
echo.
echo ********************************************
echo * MMPose Management Script                *
echo * Version: 1.0                            *
echo ********************************************
echo.
echo Available commands:
echo.
echo clean        - Cleans temporary files
echo                Deletes contents of:
echo                - .\data\configs\
echo.
echo docker-build - Builds Docker image with:
echo                - PyTorch with CUDA support
echo                - MMPose framework
echo                - All required dependencies
echo.
echo docker-start - Runs Docker container with:
echo                - GPU access enabled
echo                - 8GB shared memory
echo                - ./data mounted to /mmpose/data
echo.
echo help         - Shows this help message
echo.
echo Example usage:
echo   script.bat docker-build  - Build the image
echo   script.bat docker-start  - Run the container
echo   script.bat clean         - Clean temp files
echo.
exit /b