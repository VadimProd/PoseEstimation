python demo/image_demo.py \
    data/image.jpg \
    data/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    data/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file data/vis_results.jpg \
    --draw-heatmap

    
python demo/topdown_demo_with_mmdet.py \
    data/image.jpg \
    data/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    data/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file data/vis_results.jpg \
    --draw-heatmap

# Build container
docker build -t mmpose docker/

# Start docker
docker run --gpus all --shm-size=8g -it -v .\data\:/mmpose/data mmpose
docker run -it --gpus all -p 8888:8888 mmpose/mmpose:latest jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

# Download configs
mim download mmpose --config ae_hrnet-w32_8xb24-300e_coco-512x512  --dest .
mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192  --dest .