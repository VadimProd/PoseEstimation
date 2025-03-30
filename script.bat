@echo off
setlocal

if "%1"=="" goto usage
if "%1"=="clean" goto clean
if "%1"=="docker-start" goto docker
if "%1"=="help" goto usage
goto usage

:clean
echo Cleaning files ...
del /s /q ".\data\configs\*" ".\data\test_data\out\*"
echo Cleaning completed!
exit /b

:docker
echo Запуск Docker-контейнера...
docker run -d --name my_container my_image
echo Docker-контейнер запущен!
exit /b

:usage
echo Usage: script.bat {clean|docker-start|help}
exit /b