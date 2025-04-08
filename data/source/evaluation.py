import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PoseEstimator import PoseEstimator
from mmpose.structures.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
from pathlib import Path
from tqdm import tqdm


def main():
    
    estimator = PoseEstimator(method="topdown")
    image_dir = Path("../evaluation/val2017")

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
        pred_instances = estimator.get_pred(image_path=str(img))

        if hasattr(pred_instances, 'bboxes'):
            bboxes = pred_instances.bboxes
            bboxes = bbox_xyxy2xywh(bboxes)

        for i in range(len(pred_instances.keypoints)):
            keypoints_xy = pred_instances.keypoints[i]
            keypoints_visible = pred_instances.keypoints_visible[i]
            bbox = [float(coord) for coord in bboxes[i]]
            
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

    with open('../evaluation/predictions.json', 'w') as f:
        json.dump(coco_result, f, indent=4)

def eval():
    coco_gt = COCO("../evaluation/person_keypoints_val2017_filtered.json") # Path to COCO annotations
    coco_dt = coco_gt.loadRes('../evaluation/predictions.json')

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = coco_eval.eval

    print(results['kp_ok'])


if __name__ == '__main__':
    #main()
    eval()