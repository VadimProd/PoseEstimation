import cv2
import numpy as np
import json

skeleton = [
    (5, 6),              # плечи (left_shoulder – right_shoulder)
    
    (5, 7), (7, 9),      # левая рука: shoulder -> elbow -> wrist
    (6, 8), (8, 10),     # правая рука: shoulder -> elbow -> wrist

    (11, 12),            # бедра (left_hip – right_hip)
    
    (11, 13), (13, 15),  # левая нога: hip -> knee -> ankle
    (12, 14), (14, 16),  # правая нога: hip -> knee -> ankle

    (0, 1), (1, 3),      # нос -> левый глаз -> левое ухо
    (0, 2), (2, 4),      # нос -> правый глаз -> правое ухо

    (0, 5), (0, 6),      # нос -> плечи (обозначим "шею")
    (11, 5), (12, 6)     # бедра -> плечи (корпус)
]

# Параметры
canvas_width, canvas_height = 3840, 2160
canvas_size = (canvas_height, canvas_width, 3)
joint_radius = 4
line_thickness = 2
conf_thresh = 0.3


with open("video1_keypoints.json", 'r') as f:
    keypoints_json = json.load(f)

keypoints_over_time = []
for instance in keypoints_json:
    if isinstance(instance, dict) and 'keypoints' in instance:
        kpts = np.array(instance['keypoints']).reshape(-1, 3)
    else:
        kpts = np.array(instance).reshape(-1, 3)
    keypoints_over_time.append(kpts)

# Видео вывод
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('skeleton_fancy.mp4', fourcc, 20.0, (canvas_width, canvas_height))

for frame_idx, frame_kpts in enumerate(keypoints_over_time):
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255
    #trail.append(frame_kpts.copy())

    def get_point(index):
        if index >= len(frame_kpts):
            return None
        x, y, c = frame_kpts[index]
        if c < conf_thresh:
            return None
        return int(x), int(y)

    # Нарисовать голову (овал вокруг носа)
    nose = get_point(0)
    if nose:
        cv2.ellipse(canvas, nose, (85, 115), 0, 0, 360, (100, 100, 100), 2)

    # Нарисовать линии
    for i, j in skeleton:
        if i < len(frame_kpts) and j < len(frame_kpts):
            x1, y1, s1 = frame_kpts[i]
            x2, y2, s2 = frame_kpts[j]
            if s1 > conf_thresh and s2 > conf_thresh:
                pt1 = int(x1), int(y1)
                pt2 = int(x2), int(y2)
                cv2.line(canvas, pt1, pt2, (0, 0, 255), thickness=line_thickness)

    # Нарисовать текущие точки
    for x, y, s in frame_kpts:
        if s > conf_thresh:
            cv2.circle(canvas, (int(x), int(y)), joint_radius, (0, 255, 0), -1)


    out.write(canvas)
    preview = cv2.resize(canvas, (960, 540))
    cv2.imshow("Skeleton", preview)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()