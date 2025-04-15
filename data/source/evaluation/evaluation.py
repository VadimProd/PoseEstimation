import json
import cv2
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

try:
    from data.source.PoseEstimator.PoseEstimator import PoseEstimator
    from mmpose.structures.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
except ImportError:
    print(f"Module mmpose isn`t installed")


def prepare_dataset():
    with open('datasets//annotations//person_keypoints_val2017.json', 'r') as file:
        coco_data = json.load(file)

    #
    # Find images with human
    #

    images_cnt = []

    for item in coco_data['annotations']:
        if item['category_id'] == 1:
            images_cnt.append(item['image_id'])

    #
    # Delete other images
    #

    image_dir = Path("datasets//val2017")

    for img in image_dir.glob('*.jpg'):

        image_id = int(img.stem)

        if image_id not in images_cnt:
            os.remove(img)

    #
    # Get image_id in the folder Получаем ID изображений, реально присутствующих в папке
    #
    
    remaining_image_ids = {int(img.stem) for img in image_dir.glob("*.jpg")}

    #
    # Delete dublicates
    #
    
    unique_images = {}
    for img in coco_data["images"]:
        if img["id"] in remaining_image_ids:
            unique_images[img["id"]] = img

    #
    # Save filtered images
    #
    
    filtered_images = list(unique_images.values())

    #
    # Save filtered annotations
    #
    
    filtered_annotations = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in remaining_image_ids
    ]

    #
    # Build final JSON-file
    #
    
    filtered_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_data.get("categories", [])
    }

    print(f"Remaining images: {len(filtered_images)}")

    #
    # Save
    #
    
    filtered_annotations_path = "files/person_keypoints_val2017_filtered.json"
    with open(filtered_annotations_path, "w") as f:
        json.dump(filtered_coco_data, f, indent=4)

def predict_video(video_path: str, out_json: str):
    #
    # Get predictions
    #

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    estimator = PoseEstimator(method="topdown")

    preds = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preds.append(estimator.get_pred(image=frame))

        if len(preds) == 5:
            break

    cap.release()

    #
    # Transformation into a convenient form
    #

    keypoints_over_time = []

    for result in preds:
        # Get keypoints and keypoint_scores
        keypoints = result.keypoints[0]
        keypoint_scores = result.keypoint_scores[0]
        
        frame_keypoints = []
        
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            score = keypoint_scores[i]
            frame_keypoints.append([x, y, score])
        
        keypoints_over_time.append(np.array(frame_keypoints))

    keypoints_over_time_serializable = [frame.tolist() for frame in keypoints_over_time]

    with open(out_json, 'w') as json_file:
        json.dump(keypoints_over_time_serializable, json_file, indent=4)

def predict_images(images_path: str, out_json: str):
    
    estimator = PoseEstimator(method="topdown")
    image_dir = Path(images_path)

    if not image_dir.exists():
        print(f"Error: {image_dir} doesn`t exists")
        return
    
    imgs = []
    
    for img in image_dir.glob('*.jpg'):
        imgs.append(img)

    pbar = tqdm(
        total=len(imgs),
        desc="--> Predicting...",
        unit=" image",
        bar_format="{l_bar}{bar:20}{r_bar}", 
        colour="#00ff00", # green
        ncols=100,
    )
    
    coco_result = []

    for img in imgs:
        pred_instances = estimator.get_pred(image=str(img))

        bboxes = []
        if hasattr(pred_instances, 'bboxes'):
            bboxes = pred_instances.bboxes
            bboxes = bbox_xyxy2xywh(bboxes)

        for i in range(len(pred_instances.keypoints)):
            keypoints_xy = pred_instances.keypoints[i]
            keypoints_visible = pred_instances.keypoints_visible[i]
            if len(bboxes) != 0:
                bbox = [float(coord) for coord in bboxes[i]]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            
            keypoints_with_visibility = []
            for (x, y), v in zip(keypoints_xy, keypoints_visible):
                visibility = 2 if v > 0.5 else 0
                x = 0 if visibility == 0 else x
                y = 0 if visibility == 0 else y
                keypoints_with_visibility.extend([float(x), float(y), float(visibility)])
            
            score = float(pred_instances.keypoint_scores[i].mean())
            res = {
                "image_id": int(img.stem),
                "category_id": 1,
                "keypoints": keypoints_with_visibility,
                "bbox": bbox,
                "score": score
            }
            coco_result.append(res)
        pbar.update(1)

    with open(out_json, 'w') as f:
        json.dump(coco_result, f, indent=4)

def calc_metrics(type: str, coco_path: str, pred_path: str):
    coco_gt = COCO(coco_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    

    with open(pred_path, 'r') as f:
        data = json.load(f)

    ids = []
    for item in data:
        id = item['image_id']
        if not id in ids:
            ids.append(id)

    total = 0
    bad_images = []
    for id in ids:

        if type == 'bbox':
            oks_matrix = coco_eval.computeIoU(imgId=id, catId=1)
        else:
            oks_matrix = coco_eval.computeOks(imgId=id, catId=1)
        
        oks_matrix = np.array(oks_matrix)
        max_oks_per_dt = oks_matrix.max(axis=1)

        mean = np.mean(max_oks_per_dt)
        total += mean
        #print(f"Image {id}\t : \t{mean}")
        if mean < 0.5:
            bad_images.append(id)
            print(f"Image {id}\t : \t{mean}")
    print(f"Total: {total/len(ids)}")
    return bad_images

if __name__ == '__main__':
    # predict_images(
    #     images_path="../../evaluation/val2017",
    #     out_json="predictions/predictions_bottomup.json"
    # )

    # predict_video(
    #     video_path="/mmpose/data/test_data/video/video1.mp4", 
    #     out_json="video1_keypoints.json"
    # )

    bad_images = calc_metrics(
        type='keypoints', #type='keypoints', 
        coco_path="predictions/person_keypoints_val2017_filtered.json",
        pred_path="predictions/predictions_bottomup.json"
    )
    #print(f"Bad images: {bad_images}")
    print(f"Bad images cnt: {len(bad_images)}")