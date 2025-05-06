import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# Связь между суставами
mask = []
# bones = [
#     (5, 6),              # плечи (left_shoulder – right_shoulder)

#     (5, 7), (7, 9),      # левая рука: shoulder -> elbow -> wrist
#     (6, 8), (8, 10),     # правая рука: shoulder -> elbow -> wrist

#     (11, 12),            # бедра (left_hip – right_hip)
    
#     (11, 13), (13, 15),  # левая нога: hip -> knee -> ankle
#     (12, 14), (14, 16),  # правая нога: hip -> knee -> ankle

#     (0, 1), (1, 3),      # нос -> левый глаз -> левое ухо
#     (0, 2), (2, 4),      # нос -> правый глаз -> правое ухо

#     (0, 5), (0, 6),      # нос -> плечи (обозначим "шею")
#     (11, 5), (12, 6)     # бедра -> плечи (корпус)
# ]

bones = [
    (5, 6),              # плечи (left_shoulder – right_shoulder)
    (5, 7), (7, 9),      # левая рука: shoulder -> elbow -> wrist
    (6, 8), (8, 10),     # правая рука: shoulder -> elbow -> wrist
    (11, 12),            # бедра (left_hip – right_hip)
    (11, 13), (13, 15),  # левая нога: hip -> knee -> ankle
    (12, 14), (14, 16),  # правая нога: hip -> knee -> ankle
    (0, 1), (1, 3),      # нос -> левый глаз -> левое ухо
    (0, 2), (2, 4),      # нос -> правый глаз -> правое ухо
    (3, 5),              # левое ухо -> левое плечо
    (4, 6),              # правое ухо -> правое плечо
    (11, 5), (12, 6)     # бедра -> плечи (корпус)
]

CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# square_size = 25.0

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# objp *= square_size

# ---------------------------
# ===== Find chessboard ===== 
# ---------------------------

objpoints = []
imgpointsL = []
imgpointsR = []

imagesL = glob.glob("calibration_images/right*.jpg")
imagesR = glob.glob("calibration_images/left*.jpg")
image_sizeL = None
image_sizeR = None

for imageL, imageR in zip(imagesL, imagesR):
    imgL = cv2.imread(imageL)
    imgR = cv2.imread(imageR)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retR and retL == True:
        objpoints.append(objp)

        # Improves the accuracy of the corners found.
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        # Draw corners
        imgL = cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
        imgR = cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)

        # cv2.imshow('img',imgL)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        if image_sizeL is None:
            image_sizeL = grayL.shape[::-1]  # ширина, высота

        if image_sizeR is None:
            image_sizeR = grayR.shape[::-1]  # ширина, высота
            
# cv2.destroyAllWindows()

"""
cv2.calibrateCamera(...) -> [
    ret:    status (true/false)
    K:      calibration matrices for the camera
    dist:   distortion parameters
    rvecs:  rotation vector
    tvecs:  camera translation vector
]
"""
retL, KL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, image_sizeL, None, None)
retR, KR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, image_sizeR, None, None)

hL, wL, channelsL = imgL.shape
hR, wR, channelsR = imgR.shape

# Calculates optimal matrices for calibration to improve output quality.
KL, rvecsL = cv2.getOptimalNewCameraMatrix(KL, distL, (wL, hL), 1, (wL, hL))
KR, rvecsR = cv2.getOptimalNewCameraMatrix(KR, distR, (wR, hR), 1, (wR, hR))

print(f"KL = {KL}\nKR = {KR}")

# ------------------------------
# ===== Stereo calibration ===== 
# ------------------------------

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(f"criteria_stereo = {(criteria_stereo)}")

retStereo, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, KL, distL, KR, distR, grayL.shape[::-1], criteria_stereo, flags
)
print(f"R = {R}\nE = {E}\nF = {F}")

# retStereo, KL, distL, KR, distR, R, t, E, F = cv2.stereoCalibrate(
#     objpoints, imgpointsL, imgpointsR, KL, distL, KR, distR, grayL.shape[::-1], criteria_stereo, flags
# )

# --------------------------------
# ===== Stereo rectification =====
# --------------------------------

rectifyScale = 1
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    KL, distL, KR, distR,
    image_sizeL, R, t,
    rectifyScale, (0,0)
)

print(f"RL = {R1}\nRR = {R2}")
print(f"T = {t}")

# ===== Stereo maps for remapping =====

map1x, map1y = cv2.initUndistortRectifyMap(KL, distL, R1, P1, image_sizeL, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(KR, distR, R2, P2, image_sizeR, cv2.CV_32FC1)

rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

"""
Выполнение калибровки камеры с помощью
Передача значения известных трехмерных точек (объектов)
и соответствующие пиксельные координаты
обнаруженные углы (imgpoints)
"""
#ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

print("✅ Calibration complete!")
# print("K =", K)
# print("dist =", dist)

# *-------------------------------------------------------------------------------------*
# | Восстановление глубины и 3D координат
# *-------------------------------------------------------------------------------------*

def get3Dv5(pts1, pts2):
    # Ректифицировать (нормализовать) точки
    pointsL_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), KL, distL, R=R1, P=P1)
    pointsR_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), KR, distR, R=R2, P=P2)

    # Триангуляция
    points4D = cv2.triangulatePoints(P1, P2, pointsL_norm, pointsR_norm)
    points3D = (points4D[:3] / points4D[3]).T

    for i, point in enumerate(points3D):
        print(f"{i}: ({point[0]}, {point[1]}, {point[2]})")
    
    # Проверка глубины
    print("Минимальная Z:", np.min(points3D[:, 2]))
    print("Максимальная Z:", np.max(points3D[:, 2]))

    return points3D

def get3Dv4(K1, K2, points1, points2, dist1=None, dist2=None):
    # Найти фундаментальную матрицу
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    if F is None:
        raise ValueError("Fundamental matrix could not be computed.")

    # Вычислить матрицу Essential
    E = K2.T @ F @ K1

    # Убираем дисторсию, если нужно
    if dist1 is not None:
        points1_norm = cv2.undistortPoints(np.expand_dims(points1, axis=1), K1, dist1)
    else:
        points1_norm = cv2.undistortPoints(np.expand_dims(points1, axis=1), K1, None)

    if dist2 is not None:
        points2_norm = cv2.undistortPoints(np.expand_dims(points2, axis=1), K2, dist2)
    else:
        points2_norm = cv2.undistortPoints(np.expand_dims(points2, axis=1), K2, None)

    points1_norm = np.squeeze(points1_norm)
    points2_norm = np.squeeze(points2_norm)

    # Восстановить относительную позу между камерами
    _, R, t, mask_pose = cv2.recoverPose(E, points1_norm, points2_norm)

    # Построить проекционные матрицы
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, t))

    # Триангуляция
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_norm.T, points2_norm.T)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # (N, 3)

    return points_3d

def get3D(K, pts1, pts2):
    global mask

    # 1. Найдём фундаментальную матрицу
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    # print(f"Маска: {mask.ravel()==1}")

    # bones2 = []
    # for i, flag in enumerate(mask):
    #     if flag:
    #         bones2.append(bones[i])

    # bones.clear()
    # bones = bones2.copy()
    # print(bones)
    # print(f"Маска: {mask.ravel()==1}")

    # 2. Получим матрицу движения
    E = K.T @ F @ K
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # 3. Построим проекционные матрицы
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    # 4. Триангуляция
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    # pts3d = (pts4d_hom / pts4d_hom[3])[:3].T  # Nx3

    # points1u = cv2.undistortPoints(src=pts1, cameraMatrix=K, R=None, P=P1)
    # points2u = cv2.undistortPoints(src=pts2, cameraMatrix=K, R=None, P=P2)
    # points4d = cv2.triangulatePoints(P1, P2, points1u, points2u)

    pts3d = (pts4d[:3, :]/pts4d[3, :]).T

    # Проверка глубины
    print("Минимальная Z:", np.min(pts3d[:, 2]))
    print("Максимальная Z:", np.max(pts3d[:, 2]))

    # Центрирование по бедрам
    # center = (pts3d[11] + pts3d[12]) / 2  # таз
    # pts3d_centered = pts3d - center

    if np.mean(pts3d[:, 2]) < 0:
        pts3d[:, 2] *= -1
    #print(pts3d)
    return pts3d

def get3D_dev(K1, K2, R1, R2, T1, T2, pts1, pts2):
       #
    # Исправление: преобразование из мировых координат. 
    # Мы хотим получить R_rel, T_rel — положение правой камеры относительно левой.
    #

    # Преобразуем в матрицы (на всякий случай)
    R1 = np.array(R1)
    T1 = np.array(T1).reshape(3, 1)
    R2 = np.array(R2)
    T2 = np.array(T2).reshape(3, 1)

    # Вычисляем относительные R и T
    R_rel = R2 @ R1.T
    T_rel = T2 - R_rel @ T1

    # Теперь можем построить P1 и P2 для триангуляции:
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))  # Левая камера - опорная
    P2 = K2 @ np.hstack((R_rel, T_rel))               # Правая камера - относительная

    #
    # End
    #

    # # Ректифицировать (нормализовать) точки
    # P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))  # Матрица для левой камеры
    # P2 = K2 @ np.hstack((R2, T2.reshape(3, 1)))         # Матрица для правой камеры

    # Триангуляция
    points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (points4D[:3] / points4D[3]).T

    for i, point in enumerate(pts_3d):
        print(f"{i}: ({point[0]}, {point[1]}, {point[2]})")
    
    # Проверка глубины
    print("Минимальная Z:", np.min(pts_3d[:, 2]))
    print("Максимальная Z:", np.max(pts_3d[:, 2]))

    return pts_3d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def draw3D(frames_3d):
    global bones

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatters = []
    lines = []

    def init():
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        return []

    def update(frame):
        ax.cla()  # очищаем сцену
        pts = frames_3d[frame]
        
        # Центрируем (например, по тазу — точка 0)
        center = (pts[11] + pts[12]) / 2
        pts = pts - center
        #pts = pts - pts[0]
        
        # ax.set_xlim(-15, 15)
        # ax.set_ylim(-15, 15)
        # ax.set_zlim(-15, 15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Frame {frame}")
        
        for indx, (i, j) in enumerate(bones):
            x = [pts[i, 0], pts[j, 0]]
            y = [pts[i, 1], pts[j, 1]]
            z = [pts[i, 2], pts[j, 2]]
            ax.plot(x, y, z, 'ro-')

        # Подпись каждой точки по индексу
        for idx, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, str(idx), fontsize=8, color='blue')
           
        return []

    ani = FuncAnimation(fig, update, frames=len(frames_3d), init_func=init, blit=False)
    plt.show()

def draw2D(frames_2d):
    global bones
    # Формат: [ [x0, y0, score0], [x1, y1, score1], ..., [x16, y16, score16] ]
    keypoints = frames_2d

    # Пример костей для COCO-схемы

    # Отрисовка
    plt.figure(figsize=(6, 8))
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x, y, c='red')
        plt.text(x + 3, y, str(i), fontsize=8)

    # Соединяем кости
    for i, j in bones:
        x = [keypoints[i][0], keypoints[j][0]]
        y = [keypoints[i][1], keypoints[j][1]]
        plt.plot(x, y, 'b-', linewidth=2)

    plt.gca().invert_yaxis()
    plt.title("2D Skeletal Keypoints (COCO-style)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

with open('warmup_keypoints_left.json', 'r') as f:
    left_keypoints_json = json.load(f)

with open('warmup_keypoints_right.json', 'r') as f:
    right_keypoints_json = json.load(f)

var = 1
frames_3d = []

if var == 1:
    start = 0
    for i in range(start, 1):#len(left_keypoints_json)):
        pts1 = np.array([np.float32(keypoint[:-1]) for keypoint in left_keypoints_json[i]], dtype=np.float32)
        pts2 = np.array([np.float32(keypoint[:-1]) for keypoint in right_keypoints_json[i]], dtype=np.float32)
    # frames_3d.append(get3Dv4(KL, KR, pts1, pts2, distL, distR))
        frames_3d.append(get3Dv5(pts1, pts2))

else:
    K1 = np.array([
        [1092.3211669921875, 0, 461.3157958984375],  # Матрица камеры левой
        [0, 1089.9815673828125, 441.7460632324219],
        [0, 0, 1]
    ])

    # Пример калибровочных параметров для правой камеры
    K2 = np.array([
        [1035.1177978515625, 0, 476.61102294921875],  # Матрица камеры правой
        [0, 1030.60693359375, 448.9285583496094],
        [0, 0, 1]
    ])

    # Поворот и трансляция между камерами (например, из данных)
    R1 = np.array([
        [0.38335646046117283, 0.9235359490864682, -0.010916728797482081], 
        [0.13268295186324477, -0.06676567303201067, -0.9889072652121829], 
        [-0.9140202724817087, 0.37765552511454026, -0.14813252797047552]
    ])

    T1 = np.array([4.086554878406053, -1.4422745200869256, 1.5420494433064695])

    R2 = np.array([
        [-0.458866846330765, 0.8885003750868004, 0.002881112471444765], 
        [0.15795120267122098, 0.08476418995008086, -0.9838020378494962], 
        [-0.874352694805233, -0.4509790633418825, -0.17923517934296662]
    ])

    T2 = np.array([3.9023938502003386, 2.1214361844482563, 1.483635123887001])

    for i in range(len(left_keypoints_json)):
        # i = 115
        pts1 = np.array([np.float32(keypoint[:-1]) for keypoint in left_keypoints_json[i]], dtype=np.float32)
        pts2 = np.array([np.float32(keypoint[:-1]) for keypoint in right_keypoints_json[i]], dtype=np.float32)

        frames_3d.append(get3D_dev(K1, K2, R1, R2, T1, T2, pts1, pts2))

draw3D(frames_3d=frames_3d)
# print(frames_3d)

#draw2D(frames_2d=pts1)


# frames_3d = [
#     [point.tolist() for point in frame]
#     for frame in frames_3d
# ]

# with open("frames_3d.json", "w") as f:
#     json.dump(frames_3d, f, indent=4)