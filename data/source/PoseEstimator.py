import cv2
import numpy as np
import time
import os
import torch
import subprocess

from tqdm import tqdm
from pathlib import Path
from typing import Union, Literal

from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.apis import init_model, inference_topdown, inference_bottomup
from mmdet.apis import init_detector, inference_detector
from mmpose.structures.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh


class PoseEstimator():
    def __init__(
        self, 
        method: Literal['topdown', 'bottomup'] = 'bottomup'
    ):
        """
        Initialize the model.
        
        Args:
            method: Detection method ('topdown' or 'bottomup')
        """
        self.config_file = ''
        self.checkpoint_file = ''
        self.det_config_file = ''
        self.det_checkpoint_file = ''

        self.method = method.lower()
        self.pose_model = None
        self.visualizer = None
        self.det_model = None
        
        self.download_configs()
        self.init_model()
    
    def process(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        Processing input data (video or image)
        
        Args:
            input_path: Path to the input file (video/image)
            output_path: Path to the output file (video/image)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            self._process_image(input_path, output_path)
        else:
            self._process_video(input_path, output_path)

    def _process_image(
        self,
        image_path: Path,
        output_path: str
    ) -> None:

        print(f"\nImage processing: {image_path.name}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise IOError(f"Failed to read image: {image_path}")
        
        processed_frame = self._process_frame(frame)
        
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"The result has been saved in: {output_path}")

    def _process_video(
        self,
        video_path: Path,
        output_path: str
    ) -> None:

        print(f"\nOpening video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video {video_path}")

        width, height, fps, total_frames = self._get_video_info(cap)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        pbar = tqdm(
            total=total_frames,
            desc="--> Video processing",
            unit=" frame",
            bar_format="{l_bar}{bar:20}{r_bar}", 
            colour="#00ff00", # green
            ncols=100,
        )
        frame_count = 0
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self._process_frame(frame)
            out.write(processed_frame)

            pbar.update(1)
        
        self._release_resources(cap, out)
        
        if output_path:
            print(f"\nThe result has been saved in: {output_path}")

    def _detect_person(
        self,
        frame: np.ndarray
    ) -> np.ndarray:

        det_result = inference_detector(self.det_model, frame)
        pred_instances = det_result.pred_instances

        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        person_bboxes = bboxes[(labels == 0) & (scores > 0.5)]
        return person_bboxes.astype(np.float32)

    def _process_frame(
        self, 
        frame: np.ndarray
    ) -> np.ndarray:

        if self.method == 'topdown':
            
            results = inference_topdown(
                self.pose_model, 
                frame, 
                bboxes=self._detect_person(frame), 
                bbox_format='xyxy'
            )
        else:
            results = inference_bottomup(self.pose_model, frame)
        
        merge_result = merge_data_samples(results)
        pred_instances = merge_result.pred_instances

        if hasattr(pred_instances, 'bboxes'):
            bboxes = pred_instances.bboxes
            print(f"After topdown: {bbox_xyxy2xywh(bboxes)}")

        vis_frame = self.visualizer.add_datasample(
            'result',
            frame,
            data_sample=PoseDataSample(pred_instances=pred_instances),
            draw_gt=True,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style='mmpose',
            show=False,
            wait_time=0,
            kpt_thr=0.5
        )

        #print(f'Coordinates: {pred_instances}')
        
        return cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    
    def _get_video_info(
        self, 
        cap
    ) -> tuple[int, int, int, int]:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video information: {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

        return width, height, fps, total_frames
    
    def _release_resources(
        self, 
        cap, 
        out
    ):
        cap.release()
        out.release()
    
    def init_model(self) -> None:
        try:
            print("Model initialization...")

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Check your GPU and PyTorch installation.")

            self.pose_model = init_model(
                self.config_file, 
                self.checkpoint_file, 
                device='cuda:0'
            )
            
            self.visualizer = PoseLocalVisualizer(
                alpha=0.8,
                line_width=2,
                radius=4
            )

            self.visualizer.set_dataset_meta(
                self.pose_model.dataset_meta,
                skeleton_style='mmpose'
            )

            print(self.det_checkpoint_file)

            self.det_model = init_detector(
                self.det_config_file,
                self.det_checkpoint_file,
                device='cuda:0'
            )
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            print("Model loaded successfully!") 

    def download_configs(self) -> None:
        try:
            configs_dir = Path("../configs")
            configs_dir.mkdir(parents=True, exist_ok=True)

            configs = [
                ["mmpose", "ae_hrnet-w32_8xb24-300e_coco-512x512.py"],
                ["mmpose", "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"],
                ["mmdet", "faster-rcnn_r50_fpn_1x_coco.py"]
            ]

            for lib, config in configs:
                if not (configs_dir / config).exists():
                    subprocess.run([
                        "mim", "download", lib,
                        "--config", config[:-3],
                        "--dest", str(configs_dir)
                    ], check=True)

            if self.method == 'bottomup':
                self.config_file = str(configs_dir) + "/ae_hrnet-w32_8xb24-300e_coco-512x512.py"
                self.checkpoint_file = str(configs_dir) + "/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"
            else:
                self.config_file = str(configs_dir) + "/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
                self.checkpoint_file = str(configs_dir) + "/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"

            self.det_config_file = str(configs_dir) + "/faster-rcnn_r50_fpn_1x_coco.py"
            self.det_checkpoint_file = str(configs_dir) + "/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

        except Exception as e:
            print(f"Error: {e}")
