import cv2
import glob
import json
import numpy as np

pattern_size = (9, 6)
square_size = 25.0

objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob("calibration_images/*.jpg")
image_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        if image_size is None:
            image_size = gray.shape[::-1]  # ширина, высота
    else:
        print(f"Chessboard not found in {fname}")

if len(objpoints) == 0:
    raise ValueError("❌ No chessboard found in any of the images!")

# Калибровка
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

print("✅ Calibration complete!")
print("K =", K)
print("dist =", dist)

# *-------------------------------------------------------------------------------------*
# | Восстановление глубины и 3D координат
# *-------------------------------------------------------------------------------------*

def get3D(K, pts1, pts2):

    # pts1 и pts2 — ключевые точки (Nx2) в двух кадрах
    # Здесь примерные точки, подставь свои:
    # (можно из JSON загрузить и взять [x, y] для каждого кадра)

    # pts1 = np.array([1, 2], dtype=np.float32)  # кадр t
    # pts2 = np.array([1, 2], dtype=np.float32)  # кадр t+1
    # print(pts1)

    #pts1 = np.array(keypoints_json[0], dtype=np.float32)

    # 1. Найдём фундаментальную матрицу
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)

    # 2. Получим матрицу движения
    E = K.T @ F @ K
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # 3. Построим проекционные матрицы
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    # 4. Триангуляция
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d_hom / pts4d_hom[3])[:3].T  # Nx3

    # Проверка глубины
    print("Минимальная Z:", np.min(pts3d[:, 2]))
    print("Максимальная Z:", np.max(pts3d[:, 2]))

    # Центрирование по бедрам
    center = (pts3d[11] + pts3d[12]) / 2  # таз
    pts3d_centered = pts3d - center

    # if np.mean(pts3d[:, 2]) < 0:
    #     pts3d[:, 2] *= -1
    print(pts3d)
    return pts3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

image_front = [
    [ 861.4123 ,  458.12143],
    [ 874.53375,  444.99994],
    [ 854.8515 ,  444.99994],
    [ 907.3375 ,  458.12143],
    [ 841.73004,  451.56067],
    [ 933.5805 ,  530.2897 ],
    [ 835.16925,  530.2897 ],
    [ 979.50574,  628.7009 ],
    [ 808.9263 ,  628.7009 ],
    [ 979.50574,  727.1121 ],
    [ 776.12256,  707.4299 ],
    [ 907.3375 ,  707.4299 ],
    [ 835.16925,  707.4299 ],
    [ 920.45905,  864.8879 ],
    [ 835.16925,  864.8879 ],
    [ 920.45905, 1015.7851 ],
    [ 835.16925, 1002.6636 ]
]

image_back = [
    [ 836.137  ,  421.03317],
    [ 843.4925 ,  413.67767],
    [ 850.848  ,  413.67767],
    [ 836.137  ,  428.38867],
    [ 902.3365 ,  428.38867],
    [ 806.715  ,  516.65454],
    [ 939.11395,  524.0101 ],
    [ 755.2266 ,  626.98694],
    [ 975.8914 ,  634.34247],
    [ 733.1601 ,  722.60834],
    [ 975.8914 ,  722.60834],
    [ 828.78156,  715.25287],
    [ 909.69196,  715.25287],
    [ 814.07056,  899.1402 ],
    [ 917.0474 ,  891.78467],
    [ 799.35956, 1068.3165 ],
    [ 924.40295, 1060.961  ]
]

def draw3D(frames_3d):

    # Связь между суставами
    bones = [
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatters = []
    lines = []

    def init():
        # ax.set_xlim(-200, 200)
        # ax.set_ylim(-200, 200)
        # ax.set_zlim(-200, 200)
        return []

    def update(frame):
        ax.cla()  # очищаем сцену
        pts = frames_3d[frame]
        
        # Центрируем (например, по тазу — точка 0)
        center = (pts[11] + pts[12]) / 2
        pts = pts - center
        # pts = pts - pts[0]
        
        #ax.set_xlim(-1, 1)
        #ax.set_ylim(-1, 1)
        #ax.set_zlim(0, 2)
        ax.set_title(f"Frame {frame}")
        
        for i, j in bones:
            x = [pts[i, 0], pts[j, 0]]
            y = [pts[i, 1], pts[j, 1]]
            z = [pts[i, 2], pts[j, 2]]
            ax.plot(x, y, z, 'ro-')
        
        return []

    ani = FuncAnimation(fig, update, frames=len(frames_3d), init_func=init, blit=False)
    plt.show()

def draw2D(frames_2d):
    # Формат: [ [x0, y0, score0], [x1, y1, score1], ..., [x16, y16, score16] ]
    keypoints = frames_2d

    # Пример костей для COCO-схемы
    bones = [
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

    # Отрисовка
    plt.figure(figsize=(6, 8))
    for i, (x, y, _) in enumerate(keypoints):
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

with open('video1_keypoints.json', 'r') as f:
    keypoints_json = json.load(f)

frames_3d = []
# for i in range(1):#len(keypoints_json) - 1):
#     pts1 = np.array([np.float32(keypoint[:-1]) for keypoint in keypoints_json[i]], dtype=np.float32)
#     pts2 = np.array([np.float32(keypoint[:-1]) for keypoint in keypoints_json[i + 1]], dtype=np.float32)
#     frames_3d.append(get3D(K, pts1, pts2))

# pts1 = np.array([np.float32(keypoint[:-1]) for keypoint in keypoints_json[0]], dtype=np.float32)
# pts2 = np.array([np.float32(keypoint[:-1]) for keypoint in keypoints_json[12]], dtype=np.float32)

pts1 = np.array([np.float32(keypoint[:]) for keypoint in image_front], dtype=np.float32)
pts2 = np.array([np.float32(keypoint[:]) for keypoint in image_back], dtype=np.float32)

frames_3d.append(get3D(K, pts1, pts2))

draw3D(frames_3d=frames_3d)
#draw2D(frames_2d=pts1)